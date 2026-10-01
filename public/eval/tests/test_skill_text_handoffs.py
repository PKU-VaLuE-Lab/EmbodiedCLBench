import importlib.util
import json
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


def _load_extractor(backend: str):
    path = SRC_ROOT / "tongbench_eval" / "agents" / backend / "skill_text.py"
    spec = importlib.util.spec_from_file_location(f"{backend}_skill_text_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load skill extractor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_skill_text


extract_claude_skill_text = _load_extractor("claudecode")
extract_codex_skill_text = _load_extractor("codex")
extract_hermes_skill_text = _load_extractor("hermesagent")
extract_openclaw_skill_text = _load_extractor("openclaw")


class SkillTextHandoffTests(unittest.TestCase):
    def test_native_harness_extractors_preserve_markdown(self):
        raw = "\n# Skill\n\nKeep the target visible.\n"

        self.assertEqual(extract_hermes_skill_text(raw), raw.strip())
        self.assertEqual(extract_codex_skill_text(raw), raw.strip())
        self.assertEqual(extract_claude_skill_text(raw), raw.strip())

    def test_openclaw_extractor_removes_gateway_prefix_and_outer_fence(self):
        payload = {
            "payloads": [
                {"text": "```markdown\n# Skill\n\nUse the latest observation.\n```"}
            ]
        }
        raw = "gateway warning: reconnecting\n" + json.dumps(payload)

        self.assertEqual(
            extract_openclaw_skill_text(raw),
            "# Skill\n\nUse the latest observation.",
        )
        self.assertNotIn("gateway warning", extract_openclaw_skill_text(raw))

    def test_openclaw_plain_markdown_is_not_reinterpreted(self):
        raw = "# Skill\n\nA literal { brace is allowed."

        self.assertEqual(extract_openclaw_skill_text(raw), raw)


