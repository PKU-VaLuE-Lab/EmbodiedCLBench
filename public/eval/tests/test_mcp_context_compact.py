from tongbench_eval.mcp.context_compact import SharedMcpCompactor


def test_compaction_preserves_task_fields_and_images(tmp_path):
    payload = {
        "observation_text": (
            "Task: Put the cup on the table.\n"
            "State: the cup is ready.\n"
            "Choices:\nA. ...\nB. ..."
        ),
        "image_paths": ["state_0001.jpg"],
        "choices": [{"id": "A", "role": "success_path"}],
        "action_feedback": "Action accepted.",
        "state_id": "s1",
        "step_count": 1,
        "result_text": "Display-only duplicate.",
        "next_observation": {
            "image_paths": ["state_0002.jpg"],
            "done": False,
            "feedback": "Continue.",
        },
    }
    compactor = SharedMcpCompactor(tmp_path / "audit.json")
    compacted = compactor.transform(payload)

    assert compacted["image_paths"] == payload["image_paths"]
    assert compacted["next_observation"]["image_paths"] == ["state_0002.jpg"]
    assert compacted["choices"] == payload["choices"]
    assert compacted["action_feedback"] == payload["action_feedback"]
    assert compacted["state_id"] == payload["state_id"]
    assert compacted["step_count"] == payload["step_count"]
    assert "Choices:" not in compacted["observation_text"]
    assert "result_text" not in compacted
    assert compactor.summary["images_unchanged"] is True
    assert compactor.summary["protected_content_unchanged"] is True
    assert (tmp_path / "audit.json").is_file()


def test_duplicate_task_line_is_removed_only_after_first_result():
    payload = {
        "observation_text": "Task: Same task.\nState: ready.",
        "image_paths": [],
        "choices": [],
    }
    compactor = SharedMcpCompactor()
    first = compactor.transform(payload)
    second = compactor.transform(payload)

    assert "Task: Same task." in first["observation_text"]
    assert "Task: Same task." not in second["observation_text"]
    assert "State: ready." in second["observation_text"]
