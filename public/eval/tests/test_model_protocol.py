from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tongbench_eval.cli.framework.protocol import mcp_image_env, protocol_tool_names
from tongbench_eval.graph_env.interface.framework_prompt import build_framework_agent_prompt
from tongbench_eval.graph_env.runtime.framework_runtime import _public_action_result_payload
from tongbench_eval.mcp.server import _model_visible_payload


class ModelProtocolTest(unittest.TestCase):
    def test_opt_in_uses_standard_uncompressed_mcp_image_content(self) -> None:
        with patch.dict(
            "os.environ", {"TONGBENCH_MCP_CANONICAL_IMAGE_CONTENT": "1"}, clear=False
        ):
            env = mcp_image_env(
                backend="claudecode",
                no_image_input=False,
                direct_image_input=False,
                compress_images=False,
            )
        self.assertEqual(env["TONGSIM_MCP_IMAGE_CONTENT_ONLY"], "1")
        self.assertEqual(env["TONGSIM_MCP_PI_IMAGE_CONTENT"], "1")
        self.assertNotIn("TONGSIM_MCP_ANTHROPIC_IMAGE_BLOCKS", env)
        self.assertEqual(env["TONGSIM_MCP_COMPRESS_IMAGES"], "0")

    def test_existing_provider_image_transport_is_unchanged_without_opt_in(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            hermes = mcp_image_env(
                backend="hermesagent",
                no_image_input=False,
                direct_image_input=False,
                compress_images=False,
            )
            claude = mcp_image_env(
                backend="claudecode",
                no_image_input=False,
                direct_image_input=False,
                compress_images=False,
            )
        self.assertNotIn("TONGSIM_MCP_IMAGE_CONTENT_ONLY", hermes)
        self.assertEqual(claude["TONGSIM_MCP_ANTHROPIC_IMAGE_BLOCKS"], "1")

    def test_finish_is_not_in_model_tool_surface(self) -> None:
        self.assertEqual(protocol_tool_names("safe_choice"), ("observe", "select_option", "status"))
        self.assertEqual(protocol_tool_names("legacy"), ("observe_current_state", "choose_action", "check_task_status"))

    def test_safe_choice_prompt_uses_done_as_terminal_signal(self) -> None:
        prompt = build_framework_agent_prompt(
            SimpleNamespace(task_id="task_L2_test", description="Arrange the room."),
            max_steps=12,
            action_interface="full_action",
            action_library_mode="atomic_only",
            interaction_mode="semantic_tool",
            protocol_surface="safe_choice",
        )
        self.assertIn("done=true", prompt)
        self.assertNotIn("finish", prompt.lower())
        self.assertNotIn("reached_goal", prompt)
        self.assertNotIn("confidence", prompt)

    def test_terminal_feedback_is_explicit_without_revealing_internal_flag(self) -> None:
        success = _public_action_result_payload({"valid": True, "done": True, "reached_goal": True})
        failure = _public_action_result_payload({"valid": False, "done": True, "reached_goal": False})
        self.assertEqual(set(success), {"accepted", "done", "action_feedback"})
        self.assertEqual(set(failure), {"accepted", "done", "action_feedback"})
        self.assertIn("complete", success["action_feedback"].lower())
        self.assertIn("without completing", failure["action_feedback"].lower())
        self.assertNotIn("reached_goal", success)
        self.assertNotIn("reached_goal", failure)

    def test_defense_in_depth_hides_internal_fields(self) -> None:
        visible = _model_visible_payload(
            {
                "done": True,
                "reached_goal": True,
                "confidence": 0.9,
                "nested": {"reached_goal": False, "confidence": 0.1},
            }
        )
        self.assertEqual(visible, {"done": True, "nested": {}})

    def test_generated_codex_sequence_server_has_no_finish_or_confidence(self) -> None:
        import tongbench_eval.cli.framework.dialogue_runtime.main as dialogue

        step = SimpleNamespace(
            index=1,
            task_id="task_L2_test",
            level="L2",
            workspace_name="task_L2_test",
        )
        with tempfile.TemporaryDirectory() as raw_dir:
            path = dialogue._write_codex_sequence_mcp_server(Path(raw_dir), [step])
            source = path.read_text(encoding="utf-8").lower()
        self.assertNotIn("finish_task", source)
        self.assertNotIn("confidence: float", source)
        self.assertNotIn('"confidence":', source)


if __name__ == "__main__":
    unittest.main()
