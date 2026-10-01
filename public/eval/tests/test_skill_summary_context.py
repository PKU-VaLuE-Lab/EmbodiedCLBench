from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tongbench_eval.agents.base import AgentCheckpoint
from tongbench_eval.cli.framework.dialogue_runtime.main import _audit_skill_summary_context


class SkillSummaryContextAuditTest(unittest.TestCase):
    def test_accepts_full_saved_learning_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoint"
            payload = checkpoint_dir / "native" / "sessions" / "session.json"
            payload.parent.mkdir(parents=True)
            payload.write_text('{"messages": []}\n', encoding="utf-8")
            checkpoint = AgentCheckpoint.create(
                backend="hermesagent",
                path=checkpoint_dir,
                session_id="session",
                metadata={"learning_task_ids": ["l2_a", "l2_b"]},
            )
            learning_dir = root / "skill_learning"
            learning_dir.mkdir()
            (learning_dir / "dialogue_plan.json").write_text(
                json.dumps(
                    {
                        "phase": "skill_summary",
                        "context_mode": "full",
                        "context_mode_override": "full",
                    }
                ),
                encoding="utf-8",
            )
            audit = _audit_skill_summary_context(
                learning_dir=learning_dir,
                checkpoint=checkpoint,
                summary_call={"error": None},
                learning_task_ids=["l2_a", "l2_b"],
            )
            self.assertTrue(audit["passed"])

    def test_rejects_compaction_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoint"
            payload = checkpoint_dir / "native" / "sessions" / "session.json"
            payload.parent.mkdir(parents=True)
            payload.write_text('{"messages": []}\n', encoding="utf-8")
            checkpoint = AgentCheckpoint.create(
                backend="hermesagent",
                path=checkpoint_dir,
                session_id="session",
                metadata={"learning_task_ids": ["l2_a"]},
            )
            learning_dir = root / "skill_learning"
            learning_dir.mkdir()
            (learning_dir / "dialogue_plan.json").write_text(
                json.dumps({"phase": "skill_summary", "context_mode": "full"}),
                encoding="utf-8",
            )
            (learning_dir / "chat.jsonl").write_text(
                "[CONTEXT COMPACTION — REFERENCE ONLY] Summary generation was unavailable.\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "context audit failed"):
                _audit_skill_summary_context(
                    learning_dir=learning_dir,
                    checkpoint=checkpoint,
                    summary_call={"error": None},
                    learning_task_ids=["l2_a"],
                )


if __name__ == "__main__":
    unittest.main()
