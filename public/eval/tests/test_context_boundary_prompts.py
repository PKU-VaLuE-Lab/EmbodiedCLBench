import unittest

from tongbench_eval.cli.framework.dialogue_runtime.main import (
    _build_sequence_prompt,
    _safe_choice_checkpoint_application_prompt,
)


class ContextBoundaryPromptTests(unittest.TestCase):
    def test_in_context_application_prompt_marks_learning_as_reference(self):
        prompt = _safe_choice_checkpoint_application_prompt()

        self.assertIn("learning examples", prompt)
        self.assertIn("not the current task", prompt)
        self.assertIn("A new task starts now and is not complete yet", prompt)
        self.assertIn("environment result reports task_complete=true", prompt)
        self.assertNotIn("A new household task is now available", prompt)
        self.assertNotIn("Study them for useful visual reasoning", prompt)

    def test_skill_prompt_marks_summary_as_reference(self):
        prompt = _build_sequence_prompt(
            [],
            protocol_surface="safe_choice",
            skill_text="learned guidance",
        )

        self.assertIn("learned guidance to help you solve the new task", prompt)
        self.assertIn("reference guidance, not a task episode or a completion result", prompt)
        self.assertIn("The current task is new and has just started, so it is not complete yet", prompt)
        self.assertIn("environment result reports task_complete=true", prompt)
        self.assertNotIn("Study them for useful visual reasoning", prompt)


if __name__ == "__main__":
    unittest.main()
