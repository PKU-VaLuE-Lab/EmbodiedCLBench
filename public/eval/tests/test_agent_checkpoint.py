from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tongbench_eval.agents.base import AgentCheckpoint
from tongbench_eval.agents.checkpoint_validation import (
    NEXT_TASK_MESSAGE,
    build_hermes_in_context_checkpoint,
    validate_checkpoint_branch,
)


class AgentCheckpointTest(unittest.TestCase):
    def test_manifest_detects_payload_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoint"
            payload = checkpoint_dir / "native" / "sessions" / "session_test.json"
            payload.parent.mkdir(parents=True)
            payload.write_text('{"messages": []}\n', encoding="utf-8")
            created = AgentCheckpoint.create(
                backend="hermesagent",
                path=checkpoint_dir,
                session_id="test",
                metadata={"fingerprint_sha256": "abc"},
            )
            loaded = AgentCheckpoint.load(checkpoint_dir, expected_backend="hermesagent")
            self.assertEqual(created.payload_sha256, loaded.payload_sha256)

            payload.write_text('{"messages": [{"role": "user"}]}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
                AgentCheckpoint.load(checkpoint_dir)

    def test_hermes_branch_prefix_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoint"
            checkpoint_session = checkpoint_dir / "native" / "sessions" / "session_learning.json"
            checkpoint_session.parent.mkdir(parents=True)
            prefix_messages = [
                {"role": "user", "content": "learning"},
                {"role": "assistant", "content": "done"},
            ]
            checkpoint_session.write_text(
                json.dumps(
                    {
                        "session_id": "learning",
                        "system_prompt": "system",
                        "messages": prefix_messages,
                    }
                ),
                encoding="utf-8",
            )
            checkpoint = AgentCheckpoint.create(
                backend="hermesagent",
                path=checkpoint_dir,
                session_id="learning",
            )
            branch_dir = root / "branch"
            branch_session = branch_dir / "hermes_session" / "session_branch.json"
            branch_session.parent.mkdir(parents=True)
            branch_session.write_text(
                json.dumps(
                    {
                        "session_id": "branch",
                        "system_prompt": "system",
                        "messages": [
                            *prefix_messages,
                            {"role": "user", "content": "summarize"},
                            {"role": "assistant", "content": "skill"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = validate_checkpoint_branch(
                checkpoint=checkpoint,
                branch_output_dir=branch_dir,
                branch_prompt="summarize",
            )
            self.assertTrue(report["equivalent"])
            self.assertEqual(
                report["expected_model_input_prefix_sha256"],
                report["actual_model_input_prefix_sha256"],
            )

    def test_hermes_branch_can_continue_without_synthetic_user_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoint"
            checkpoint_session = checkpoint_dir / "native" / "sessions" / "session_learning.json"
            checkpoint_session.parent.mkdir(parents=True)
            prefix_messages = [
                {"role": "assistant", "content": "", "tool_calls": [{"id": "observe"}]},
                {"role": "tool", "tool_call_id": "observe", "content": "next task"},
            ]
            checkpoint_session.write_text(
                json.dumps(
                    {
                        "session_id": "learning",
                        "system_prompt": "system",
                        "tools": [{"name": "observe"}],
                        "messages": prefix_messages,
                    }
                ),
                encoding="utf-8",
            )
            checkpoint = AgentCheckpoint.create(
                backend="hermesagent",
                path=checkpoint_dir,
                session_id="learning",
            )
            branch_dir = root / "branch"
            branch_session = branch_dir / "hermes_session" / "session_branch.json"
            branch_session.parent.mkdir(parents=True)
            branch_session.write_text(
                json.dumps(
                    {
                        "session_id": "branch",
                        "system_prompt": "system",
                        "tools": [{"name": "observe"}],
                        "messages": [
                            *prefix_messages,
                            {"role": "assistant", "content": "", "tool_calls": [{"id": "observe"}]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = validate_checkpoint_branch(
                checkpoint=checkpoint,
                branch_output_dir=branch_dir,
                branch_prompt=None,
            )
            self.assertTrue(report["equivalent"])
            self.assertTrue(report["continued_without_user_message"])

    def test_hermes_skill_branch_preserves_history_with_tools_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoint"
            checkpoint_session = checkpoint_dir / "native" / "sessions" / "session_learning.json"
            checkpoint_session.parent.mkdir(parents=True)
            prefix_messages = [{"role": "assistant", "content": "learning complete"}]
            checkpoint_session.write_text(
                json.dumps(
                    {
                        "session_id": "learning",
                        "system_prompt": "system",
                        "tools": [{"name": "select_option"}],
                        "messages": prefix_messages,
                    }
                ),
                encoding="utf-8",
            )
            checkpoint = AgentCheckpoint.create(
                backend="hermesagent",
                path=checkpoint_dir,
                session_id="learning",
            )
            branch_dir = root / "branch"
            branch_session = branch_dir / "hermes_session" / "session_branch.json"
            branch_session.parent.mkdir(parents=True)
            branch_session.write_text(
                json.dumps(
                    {
                        "session_id": "branch",
                        "system_prompt": "system",
                        "tools": [],
                        "messages": [
                            *prefix_messages,
                            {"role": "user", "content": "summarize"},
                            {"role": "assistant", "content": "skill"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = validate_checkpoint_branch(
                checkpoint=checkpoint,
                branch_output_dir=branch_dir,
                branch_prompt="summarize",
                tools_enabled=False,
            )
            self.assertTrue(report["equivalent"])
            self.assertTrue(report["history_prefix_exactly_equal"])
            self.assertTrue(report["tools_exactly_equal"])

    def test_hermes_icl_checkpoint_keeps_learning_and_removes_completion_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoint"
            checkpoint_session = checkpoint_dir / "native" / "sessions" / "session_learning.json"
            checkpoint_session.parent.mkdir(parents=True)
            completed = {
                "accepted": True,
                "done": True,
                "action_feedback": "Action accepted. The task is complete.",
                "task_complete": True,
                "all_tasks_complete": True,
                "message": "All tasks are complete. Stop using task tools.",
            }
            messages = [
                {"role": "user", "content": "learning"},
                {"role": "tool", "content": "final learning action accepted"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "select_option", "name": "select_option"}],
                },
                {
                    "role": "tool",
                    "tool_call_id": "select_option",
                    "content": json.dumps(
                        {
                            "result": json.dumps(completed, indent=2),
                            "structuredContent": completed,
                        }
                    ),
                },
                {"role": "assistant", "content": "Learning tasks are complete."},
            ]
            checkpoint_session.write_text(
                json.dumps(
                    {
                        "session_id": "learning",
                        "system_prompt": "system",
                        "tools": [{"name": "select_option"}, {"name": "observe"}],
                        "messages": messages,
                        "message_count": len(messages),
                    }
                ),
                encoding="utf-8",
            )
            source = AgentCheckpoint.create(
                backend="hermesagent",
                path=checkpoint_dir,
                session_id="learning",
            )
            derived = build_hermes_in_context_checkpoint(
                source=source,
                destination=root / "in_context_checkpoint",
            )
            derived_session = json.loads(
                (derived.path / "native" / "sessions" / "session_learning.json").read_text(
                    encoding="utf-8"
                )
            )
            source_session = json.loads(checkpoint_session.read_text(encoding="utf-8"))
            self.assertEqual(source_session["messages"], messages)
            self.assertEqual(derived_session["messages"][:-1], messages[:-2])
            self.assertEqual(derived_session["message_count"], len(messages) - 1)
            terminal_payload = json.loads(derived_session["messages"][-1]["content"])
            self.assertFalse(terminal_payload["structuredContent"]["all_tasks_complete"])
            self.assertEqual(
                terminal_payload["structuredContent"]["message"],
                NEXT_TASK_MESSAGE,
            )
            self.assertNotIn("reached_goal", terminal_payload["structuredContent"])
            self.assertEqual(
                derived.metadata["source_checkpoint_payload_sha256"],
                source.payload_sha256,
            )


if __name__ == "__main__":
    unittest.main()
