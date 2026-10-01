from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from tongbench_eval.agents.protocol_gateway import (
    _anthropic_to_messages,
    _anthropic_stream_events,
    _base_chat_request,
    _chat_to_responses,
    _chat_to_anthropic,
    _response_input_to_messages,
    _responses_stream_events,
)


class AnthropicGatewayTests(unittest.TestCase):
    def test_text_image_tool_and_reasoning_roundtrip(self):
        request = {"system": "system unchanged", "messages": [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "reasoning unchanged"},
                {"type": "text", "text": "assistant text unchanged"},
                {"type": "tool_use", "id": "call_1", "name": "select_option", "input": {"option_id": "B"}},
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": [
                {"type": "text", "text": "feedback unchanged"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "aGVsbG8="}},
            ]}]},
        ]}
        messages = _anthropic_to_messages(request)
        self.assertEqual([m["role"] for m in messages], ["system", "assistant", "tool"])
        self.assertEqual(messages[0]["content"], "system unchanged")
        self.assertEqual(messages[1]["content"], "assistant text unchanged")
        self.assertEqual(messages[1]["reasoning_content"], "reasoning unchanged")
        self.assertEqual(json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]), {"option_id": "B"})
        self.assertEqual(messages[2]["tool_call_id"], "call_1")
        self.assertEqual(messages[2]["content"], [
            {"type": "text", "text": "feedback unchanged"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aGVsbG8="}},
        ])

    def test_effort_override_and_tool_choice(self):
        with patch.dict(os.environ, {"TONGBENCH_GATEWAY_REASONING_EFFORT": "high"}):
            request = _base_chat_request({"tool_choice": {"type": "auto"}}, [], "ZHIPU/GLM-5.3-Flash")
        self.assertEqual(request["reasoning_effort"], "high")
        self.assertTrue(request["enable_thinking"])
        self.assertEqual(request["tool_choice"], "auto")

    def test_stream_has_block_lifecycle_and_exact_usage(self):
        response = {"choices": [{"message": {
            "reasoning_content": "thought", "content": "text",
            "tool_calls": [{"id": "call_1", "function": {"name": "select_option", "arguments": '{"option_id":"B"}'}}],
        }, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 123, "completion_tokens": 45}}
        events = _anthropic_stream_events(response, "glm")
        self.assertTrue(all(name == event["type"] for name, event in events))
        self.assertEqual(events[0][0], "message_start")
        self.assertEqual(events[-1][0], "message_stop")
        self.assertEqual(events[0][1]["message"]["usage"]["input_tokens"], 123)
        self.assertEqual(events[-2][1]["usage"]["output_tokens"], 45)
        self.assertEqual(events[-2][1]["delta"]["stop_reason"], "tool_use")
        thinking_starts = [
            event for name, event in events
            if name == "content_block_start" and event["content_block"]["type"] == "thinking"
        ]
        self.assertEqual(len(thinking_starts), 1)
        for index in range(3):
            names = [name for name, e in events if e.get("index") == index]
            self.assertEqual(names, ["content_block_start", "content_block_delta", "content_block_stop"])
        self.assertEqual(_chat_to_anthropic({"choices": [{"finish_reason": "stop"}]}, "glm", {})["stop_reason"], "end_turn")

    def test_malformed_tool_arguments_fail_without_rewriting(self):
        data = {"choices": [{"message": {"tool_calls": [{"function": {"arguments": "broken"}}]}}]}
        with self.assertRaises(json.JSONDecodeError):
            _chat_to_anthropic(data, "glm", {})

    def test_responses_stream_opens_and_closes_items_before_deltas(self):
        response = {"choices": [{"message": {
            "reasoning_content": "reason first",
            "content": "choose one",
            "tool_calls": [{"id": "call_1", "function": {"name": "select_option", "arguments": '{"option_id":"B"}'}}],
        }, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 5, "completion_tokens": 6}}
        events = _responses_stream_events(response, "glm")
        names = [name for name, _ in events]
        self.assertLess(names.index("response.output_item.added"), names.index("response.output_text.delta"))
        self.assertIn("response.function_call_arguments.delta", names)
        self.assertIn("response.reasoning_text.delta", names)
        self.assertEqual(names[-1], "response.completed")
        active = set()
        for name, body in events:
            item_id = body.get("item_id")
            if name == "response.output_item.added":
                active.add(body["item"]["id"])
            elif name in {"response.output_text.delta", "response.function_call_arguments.delta"}:
                self.assertIn(item_id, active)
            elif name == "response.output_item.done":
                active.discard(body["item"]["id"])
        self.assertFalse(active)

    def test_responses_reasoning_and_tool_history_roundtrip(self):
        upstream = {"choices": [{"message": {
            "reasoning_content": "inspect the new task",
            "content": "",
            "tool_calls": [{
                "id": "call_keep",
                "type": "function",
                "function": {"name": "select_option", "arguments": '{"option_id":"C"}'},
            }],
        }, "finish_reason": "tool_calls"}]}
        output = _chat_to_responses(upstream, "ZHIPU/GLM-5.3-Flash")["output"]
        messages = _response_input_to_messages({
            "instructions": "system unchanged",
            "input": [
                *output,
                {"type": "function_call_output", "call_id": "call_keep", "output": "feedback unchanged"},
            ],
        })
        self.assertEqual([message["role"] for message in messages], ["system", "assistant", "tool"])
        self.assertEqual(messages[1]["reasoning_content"], "inspect the new task")
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call_keep")
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["name"], "select_option")
        self.assertEqual(messages[2]["tool_call_id"], "call_keep")
        self.assertEqual(messages[2]["content"], "feedback unchanged")

    def test_responses_preserves_image_detail_and_strict_tool_schema(self):
        payload = {
            "model": "untrusted-caller-model",
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "look"},
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "high"},
            ]}],
            "tools": [{
                "type": "function",
                "name": "select_option",
                "description": "select exactly one",
                "parameters": {"type": "object", "properties": {"option_id": {"type": "string"}}},
                "strict": True,
            }, {"type": "web_search_preview"}],
            "parallel_tool_calls": True,
        }
        with patch.dict(os.environ, {
            "TONGBENCH_GATEWAY_MCP_ONLY_TOOLS": "1",
            "TONGBENCH_GATEWAY_REASONING_EFFORT": "high",
        }):
            messages = _response_input_to_messages(payload)
            request = _base_chat_request(payload, messages, "ZHIPU/GLM-5.3-Flash")
        self.assertEqual(request["model"], "ZHIPU/GLM-5.3-Flash")
        self.assertEqual(request["messages"][0]["content"][1]["image_url"]["detail"], "high")
        self.assertTrue(request["tools"][0]["function"]["strict"])
        self.assertFalse(request["parallel_tool_calls"])
        self.assertEqual(request["reasoning_effort"], "high")
        self.assertEqual(request["thinking"], {"type": "enabled", "clear_thinking": False})

    def test_opaque_responses_reasoning_fails(self):
        with self.assertRaisesRegex(ValueError, "opaque encrypted Responses reasoning"):
            _response_input_to_messages({
                "input": [{"type": "reasoning", "encrypted_content": "opaque"}],
            })

    def test_codex_single_call_gate_keeps_first_tongsim_call(self):
        response = {"choices": [{"message": {"tool_calls": [
            {"id": "call_1", "function": {"name": "mcp__tongsim__select_option", "arguments": '{"option_id":"A"}'}},
            {"id": "call_2", "function": {"name": "mcp__tongsim__select_option", "arguments": '{"option_id":"B"}'}},
        ]}}]}
        with patch.dict(os.environ, {"TONGBENCH_GATEWAY_SINGLE_TONGSIM_CALL": "1"}):
            output = _chat_to_responses(response, "glm")["output"]
        calls = [item for item in output if item["type"] == "function_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["call_id"], "call_1")
        self.assertEqual(calls[0]["name"], "select_option")
        self.assertEqual(calls[0]["namespace"], "mcp__tongsim__")

        messages = _response_input_to_messages({
            "input": [
                calls[0],
                {"type": "function_call_output", "call_id": "call_1", "output": "accepted"},
            ],
        })
        self.assertEqual(
            messages[0]["tool_calls"][0]["function"]["name"],
            "mcp__tongsim__select_option",
        )

    def test_codex_namespace_tool_is_flattened_without_builtin_tools(self):
        payload = {"tools": [
            {"type": "function", "name": "exec_command", "parameters": {"type": "object"}},
            {"type": "namespace", "name": "mcp__tongsim__", "description": "TongSIM", "tools": [
                {"type": "function", "name": "observe", "description": "observe", "parameters": {"type": "object"}, "strict": True},
                {"type": "function", "name": "select_option", "description": "select", "parameters": {"type": "object"}, "strict": True},
            ]},
        ]}
        with patch.dict(os.environ, {"TONGBENCH_GATEWAY_MCP_ONLY_TOOLS": "1"}):
            request = _base_chat_request(payload, [], "ZHIPU/GLM-5.3-Flash")
        self.assertEqual(
            [tool["function"]["name"] for tool in request["tools"]],
            ["mcp__tongsim__observe", "mcp__tongsim__select_option"],
        )
        self.assertTrue(all(tool["function"]["strict"] for tool in request["tools"]))


if __name__ == "__main__":
    unittest.main()
