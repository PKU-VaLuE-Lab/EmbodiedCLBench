from __future__ import annotations

from typing import Any

FRAMEWORK_WORKDIR = "/tmp_workspace"
FRAMEWORK_CLIENT_PATH = f"{FRAMEWORK_WORKDIR}/tongsim_client.py"


def build_framework_agent_prompt(
    task: Any,
    *,
    max_steps: int,
    action_interface: str,
    action_library_mode: str,
    public_interface_mode: str = "structured",
    interaction_mode: str = "command_line",
    decision_only: bool = False,
    protocol_surface: str = "legacy",
) -> str:
    if interaction_mode == "semantic_tool":
        if protocol_surface == "safe_choice":
            lines = [
                "You will solve one visual household task.",
                "For the current task, inspect the image and choose the best next option.",
                "Use only the current task goal, current image, previous feedback, and the listed choices.",
                "Start by calling observe. For each decision, call select_option with exactly one listed letter.",
                "Call at most one TongSIM communication tool in each assistant response. After calling observe, select_option, or status, stop immediately and wait for the returned tool result before calling another TongSIM tool.",
                "Do not call select_option more than once in one assistant response. Do not retry the same tool call or try alternate tool names before receiving feedback.",
                "When a tool result contains done=true, stop using environment tools and give one short final answer summarizing completion.",
                "",
                f"Maximum environment actions: {max_steps}",
            ]
            return "\n".join(lines) + "\n"

        observation_line = (
            "The observation is returned as natural language in observation_text, with the current image attached in the same observation result, plus done."
            if public_interface_mode == "natural_language"
            else "The observation contains task_instruction, action_candidates, object_candidates, optional previous_action_feedback, done, and the current image attached in the same observation result."
        )
        output_rule = (
            "- Do not output explanatory text before tool calls. The choose_action tool should contain only the selected action fields."
            if decision_only
            else "- Do not output explanatory text before tool calls. Use natural language only inside brief_reason."
        )
        lines = [
            "# TongSIM Interactive Agent Task",
            "",
            f"Task ID: {task.task_id}",
            f"Task instruction: {task.description}",
            "",
            "You are operating a hidden TongSIM environment through dedicated interaction tools.",
            "Do not run shell commands, do not write code, and do not look for repository entrypoints.",
            "Interact with the environment step by step: observe the current situation, choose one action, then continue until the task is complete.",
            "",
            "Environment contract:",
            f"- You may take at most {max_steps} environment actions.",
            f"- {observation_line}",
            "- Use the attached observation image and visual evidence as the primary source of truth.",
            "- Use previous_action_feedback to recover from invalid or no-progress actions.",
            "",
            "Action decision rules:",
            "- Choose exactly one exposed action template for each step.",
            "- Keep template_id exactly equal to the exposed candidate name such as look_at_{object} or place_{object1}_on_{object2}.",
            "- Never substitute object ids into template_id. For example, do not use place_BP_Cup_Kitchen_003_C_0_on_BP_Plate_325_036_C_1.",
            "- Bind each placeholder in the selected template, such as object, object1, or object2, to exactly one provided object_id.",
            "- Do not invent object aliases such as cup, fruit, or plate when object_id values are provided.",
            "- If the previous action made no progress, do not repeat it unchanged.",
            output_rule,
            "- When the environment reports done=true, stop using environment tools and finalize your response.",
        ]
        return "\n".join(lines) + "\n"

    observation_return_line = (
        "   - Returns the current observation JSON with only: task_instruction, image_paths, action_candidates, object_candidates, optional previous_action_feedback, and done."
        if public_interface_mode != "natural_language"
        else "   - Returns the current observation as a natural-language summary in observation_text, plus image_paths and done."
    )
    action_return_line = (
        ""
        if public_interface_mode != "natural_language"
        else "   - Action results also include a natural-language summary in result_text, and refreshed observations may appear as next_observation."
    )
    lines = [
        "# TongSIM Interactive Agent Task",
        "",
        f"Task ID: {task.task_id}",
        f"Task instruction: {task.description}",
        "",
        "You are operating a hidden TongSIM environment through a command-line tool.",
        "You must interact with the environment step by step instead of assuming the hidden symbolic state.",
        f"The agent runtime starts in {FRAMEWORK_WORKDIR}, which already contains tongsim_client.py.",
        "Do not search the repository for other TongSIM executables or alternative entrypoints.",
        "Use the provided client path directly for every environment interaction.",
        "",
        "Available commands:",
        f"1. python {FRAMEWORK_CLIENT_PATH} observe",
        observation_return_line,
        "   - Also refreshes the local current_observation/ folder with the latest image files, if any exist.",
        f"2. python {FRAMEWORK_CLIENT_PATH} act-simple --action-level atomic --template-id '...' --action-type '...' --role ROLE_NAME=EXACT_OBJECT_ID",
        "   - Preferred action command. Avoids shell quoting problems from inline JSON.",
        "   - Repeat --role for multiple bindings.",
        "   - One-role example:",
        f"     python {FRAMEWORK_CLIENT_PATH} act-simple --action-level atomic --template-id 'look_at_{{object}}' --action-type 'look_at' --role object=BP_Cup_Kitchen_003_C_0",
        "   - Two-role example:",
        f"     python {FRAMEWORK_CLIENT_PATH} act-simple --action-level atomic --template-id 'place_{{object1}}_on_{{object2}}' --action-type 'place' --role object1=BP_Cup_Kitchen_003_C_0 --role object2=BP_Plate_325_036_C_1",
        f"3. python {FRAMEWORK_CLIENT_PATH} act --decision-json /tmp_workspace/decision.json",
        "   - Fallback action command if you explicitly write a JSON file first.",
        "   - Only selected_action is needed for normal use.",
        action_return_line,
        f"4. python {FRAMEWORK_CLIENT_PATH} status",
        "   - Returns only done and step_count.",
        "",
        "Environment contract:",
        f"- action_interface: {action_interface}",
        f"- action_library_mode: {action_library_mode}",
        f"- max_steps: {max_steps}",
        "- The hidden symbolic state is not directly available to you.",
        "- Use the returned image_paths and current_observation/ images as the primary source of truth for the current situation.",
        "- Use previous_action_feedback to recover from invalid or no-progress actions.",
        "",
        "Action decision rules:",
        "- Prefer the act-simple command instead of inline JSON whenever possible.",
        "- Use exactly one template_id from action_candidates.",
        "- template_id must stay exactly equal to the exposed candidate name such as look_at_{object} or place_{object1}_on_{object2}.",
        "- Never substitute object ids into template_id. For example, do not write place_BP_Cup_Kitchen_003_C_0_on_BP_Plate_325_036_C_1.",
        "- Use exactly one provided object_id from object_candidates when binding each role.",
        "- Use the required role names exactly as exposed by the action candidate, such as object, object1, or object2.",
        "- Do not use a generic role name like role unless the candidate itself literally requires a role named role.",
        "- Do not invent object aliases such as cup, fruit, or plate when the environment gives object_id values.",
        "- If the previous action made no progress, do not repeat it unchanged.",
        "- When the environment reports done=true, stop using environment tools and finalize your response.",
    ]
    return "\n".join(lines) + "\n"
