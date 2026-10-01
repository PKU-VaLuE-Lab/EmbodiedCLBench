from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tongbench_eval.agents.responses_protocol_gateway import (
    _anthropic_request_to_responses,
    _attach_local_tool_output_images,
    _chat_messages_to_responses,
    _chat_stream_events,
    _input_image_count,
    _request_content_summary,
    _responses_request,
    _responses_to_chat,
)


PARAMETERS = {
    "type": "object",
    "properties": {"option_id": {"type": "string", "enum": ["A", "B", "C", "D"]}},
    "required": ["option_id"],
    "additionalProperties": False,
}


class ResponsesProtocolGatewayTests(unittest.TestCase):
    def test_local_tool_output_image_attachment_preserves_original_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir, "state.jpg")
            image_path.write_bytes(b"jpeg-bytes")
            original = "Wall time: 0.1 seconds\nOutput:\n" + json.dumps(
                {"result": {"image_paths": [str(image_path)]}}
            )
            request = {
                "input": [{
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": original,
                }]
            }
            with patch.dict(os.environ, {"TONGBENCH_GATEWAY_ATTACH_LOCAL_IMAGES": "1"}):
                attached = _attach_local_tool_output_images(request)
        self.assertEqual(attached, 1)
        output = request["input"][0]["output"]
        self.assertEqual(output[0], {"type": "input_text", "text": original})
        self.assertEqual(output[1]["type"], "input_image")
        self.assertTrue(output[1]["image_url"].startswith("data:image/jpeg;base64,"))

    def test_local_tool_output_image_attachment_is_opt_in(self):
        original = json.dumps({"result": {"image_paths": ["/tmp/state.jpg"]}})
        request = {"input": [{"type": "function_call_output", "output": original}]}
        with patch.dict(os.environ, {}, clear=True):
            attached = _attach_local_tool_output_images(request)
        self.assertEqual(attached, 0)
        self.assertEqual(request["input"][0]["output"], original)

    def test_request_content_summary_reports_shapes_without_payloads(self):
        summary = _request_content_summary([
            {
                "type": "function_call_output",
                "output": json.dumps({
                    "content": [
                        {"type": "input_image", "image_url": "data:image/jpeg;base64,aGVsbG8="},
                        {"type": "input_text", "text": "/tmp/observation.png"},
                    ]
                }),
            }
        ])
        self.assertEqual(summary["content_type_counts"]["function_call_output"], 1)
        self.assertEqual(summary["content_type_counts"]["input_image"], 1)
        self.assertEqual(summary["data_image_string_count"], 1)
        self.assertEqual(summary["image_path_string_count"], 1)

    def test_chat_history_preserves_roles_image_tools_and_results(self):
        messages = [
            {"role": "system", "content": "system unchanged"},
            {"role": "user", "content": [
                {"type": "text", "text": "task unchanged"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aGVsbG8=", "detail": "high"}},
            ]},
            {
                "role": "assistant",
                "content": "choice unchanged",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "select_option", "arguments": '{"option_id":"B"}'},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "feedback unchanged"},
        ]
        converted = _chat_messages_to_responses(messages)
        self.assertEqual(converted[0]["role"], "system")
        self.assertEqual(converted[0]["content"][0]["text"], "system unchanged")
        self.assertEqual(converted[1]["content"][0]["text"], "task unchanged")
        self.assertEqual(converted[1]["content"][1], {
            "type": "input_image",
            "image_url": "data:image/jpeg;base64,aGVsbG8=",
            "detail": "high",
        })
        self.assertEqual(converted[2]["role"], "assistant")
        self.assertEqual(converted[3]["call_id"], "call_1")
        self.assertEqual(converted[3]["arguments"], '{"option_id":"B"}')
        self.assertEqual(converted[4], {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "feedback unchanged",
        })

    def test_chat_request_preserves_schema_and_uses_medium_reasoning(self):
        source = {
            "max_tokens": 4096,
            "parallel_tool_calls": False,
            "tool_choice": "required",
            "tools": [{
                "type": "function",
                "function": {
                    "name": "select_option",
                    "description": "Choose one option",
                    "parameters": PARAMETERS,
                    "strict": True,
                },
            }],
        }
        with patch.dict(os.environ, {"TONGBENCH_GATEWAY_REASONING_EFFORT": "medium"}):
            request = _responses_request(source, [{"role": "user", "content": "task"}], "gpt-5.6-luna")
        self.assertEqual(request["model"], "gpt-5.6-luna")
        self.assertEqual(request["reasoning"], {"effort": "medium"})
        self.assertEqual(request["max_output_tokens"], 4096)
        self.assertFalse(request["parallel_tool_calls"])
        self.assertEqual(request["tool_choice"], "required")
        self.assertEqual(request["tools"][0], {
            "type": "function",
            "name": "select_option",
            "description": "Choose one option",
            "parameters": PARAMETERS,
            "strict": True,
        })

    def test_fastmcp_structured_tool_image_is_restored(self):
        content = json.dumps({
            "result": "Current image: attached.",
            "structuredContent": {"result": [
                {"type": "text", "text": "Current image: attached."},
                {"type": "image", "data": "aGVsbG8=", "mimeType": "image/jpeg"},
            ]},
        })
        converted = _chat_messages_to_responses([
            {"role": "tool", "tool_call_id": "call_1", "content": content},
        ])
        self.assertEqual(converted[0]["output"], [
            {"type": "input_text", "text": "Current image: attached."},
            {"type": "input_image", "image_url": "data:image/jpeg;base64,aGVsbG8="},
        ])

    def test_json_wrapped_fastmcp_tool_image_is_restored(self):
        blocks = [
            {"type": "text", "text": "Current image: attached."},
            {"type": "image", "data": "aGVsbG8=", "mimeType": "image/jpeg"},
        ]
        converted = _chat_messages_to_responses([{
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{
                "type": "text",
                "text": json.dumps({"structuredContent": {"result": blocks}}),
            }],
        }])
        self.assertEqual(converted[0]["output"][0]["type"], "input_text")
        self.assertEqual(converted[0]["output"][1]["type"], "input_image")
        self.assertEqual(_input_image_count(converted), 1)

    def test_anthropic_request_preserves_system_image_and_tool(self):
        source = {
            "system": "system unchanged",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "task unchanged"},
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": "aGVsbG8=",
                }},
            ]}],
            "tools": [{
                "name": "select_option",
                "description": "Choose one option",
                "input_schema": PARAMETERS,
            }],
            "tool_choice": {"type": "any"},
            "max_tokens": 4096,
        }
        with patch.dict(os.environ, {"TONGBENCH_GATEWAY_REASONING_EFFORT": "medium"}):
            request = _anthropic_request_to_responses(source, "gpt-5.6-luna")
        self.assertEqual(request["input"][0]["role"], "system")
        self.assertEqual(request["input"][0]["content"][0]["text"], "system unchanged")
        self.assertEqual(request["input"][1]["content"][1]["image_url"], "data:image/jpeg;base64,aGVsbG8=")
        self.assertEqual(request["tools"][0]["parameters"], PARAMETERS)
        self.assertEqual(request["tool_choice"], "required")

    def test_response_preserves_text_reasoning_tool_call_and_usage(self):
        source = {
            "id": "resp_1",
            "output": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "reason unchanged"}]},
                {"type": "message", "content": [{"type": "output_text", "text": "text unchanged"}]},
                {"type": "function_call", "call_id": "call_1", "name": "select_option", "arguments": '{"option_id":"C"}'},
            ],
            "usage": {
                "input_tokens": 123,
                "output_tokens": 45,
                "total_tokens": 168,
                "input_tokens_details": {"cached_tokens": 10},
                "output_tokens_details": {"reasoning_tokens": 7},
            },
        }
        converted = _responses_to_chat(source, "gpt-5.6-luna")
        message = converted["choices"][0]["message"]
        self.assertEqual(message["content"], "text unchanged")
        self.assertEqual(message["reasoning_content"], "reason unchanged")
        self.assertEqual(message["tool_calls"][0]["id"], "call_1")
        self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"]), {"option_id": "C"})
        self.assertEqual(converted["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(converted["usage"]["prompt_tokens"], 123)
        self.assertEqual(converted["usage"]["completion_tokens_details"]["reasoning_tokens"], 7)

    def test_chat_stream_has_role_tool_finish_usage_and_done_ready_events(self):
        response = _responses_to_chat({
            "id": "resp_1",
            "output": [{
                "type": "function_call",
                "call_id": "call_1",
                "name": "select_option",
                "arguments": '{"option_id":"A"}',
            }],
            "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
        }, "gpt-5.6-luna")
        events = _chat_stream_events(response)
        self.assertEqual(events[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(events[1]["choices"][0]["delta"]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(events[-2]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(events[-1]["usage"]["total_tokens"], 13)

    def test_opaque_reasoning_round_trips_without_being_dropped(self):
        reasoning = {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "encrypted_content": "opaque",
        }
        converted = _responses_to_chat({"output": [reasoning]}, "gpt-5.6-luna")
        history = [{"role": "assistant", **converted["choices"][0]["message"]}]
        restored = _chat_messages_to_responses(history)
        self.assertEqual(restored[0], reasoning)


if __name__ == "__main__":
    unittest.main()
