# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import json
from typing import Any, Dict, List, Optional, Tuple


def merge_runtime_error_context(
    base_error: str,
    error_descriptions: Optional[List[str]],
) -> str:
    runtime_errors = [desc.strip() for desc in (error_descriptions or []) if desc and desc.strip()]
    if len(runtime_errors) == 0:
        return base_error

    bullet_list = "\n".join(f"- {desc}" for desc in runtime_errors)
    return (
        f"{base_error}\n\n"
        "Active environment errors for this trajectory:\n"
        f"{bullet_list}"
    )


def format_memory(memory: Any) -> str:
    if isinstance(memory, str):
        return memory
    return json.dumps(memory, ensure_ascii=True, default=str)


def format_runtime_errors(error_descriptions: Optional[List[str]]) -> str:
    runtime_errors = [desc.strip() for desc in (error_descriptions or []) if desc and desc.strip()]
    if len(runtime_errors) == 0:
        return "None"
    return "\n".join(f"- {desc}" for desc in runtime_errors)


def extract_status_error(status_return: Dict[str, Any]) -> str:
    if "error" in status_return:
        return status_return["error"]

    status_items = []
    for key, value in status_return.items():
        if key == "previous_tool_call":
            continue
        status_items.append(f"{key}: {value}")

    if len(status_items) == 0:
        return ""

    return "\n".join(status_items)


def format_tool_names(tools: List[Dict[str, Any]], include_description : bool = False) -> List[str]:
    if include_description:
        return [str(tool["name"]) + " : " + str(tool["description"]) for tool in tools]
    return [tool["name"] for tool in tools]


def format_stage_trajectory(
    trajectory: List[Dict],
) -> str:
    """
    Transform trajectory steps into a paragraph.

    Robot feedback is already represented by the execution results attached to
    the preceding decision, so only user instructions are displayed as inputs.
    """
    lines: List[str] = []
    step_index = 0
    for step_dict in trajectory:
        if step_dict['diagnosable']:
            step_index += 1
        else:
            step_index = 0
        if step_dict['input_type'] != "FEEDBACK":
            lines.append(
                f"{step_dict['input_type']}: {step_dict['input_content']}"
            )
        if "answer" in step_dict:
            lines.append(f"DECISION {step_index}: {step_dict['answer']}")
            execution_results = step_dict.get("execution_results")
            if isinstance(execution_results, list):
                for execution_result in execution_results:
                    robot = execution_result.get("robot", "unknown robot")
                    tool = execution_result.get("tool")
                    execution_name = f"{robot}.{tool}" if tool else robot
                    result = "success" if execution_result.get("result") else "failure"
                    error_flag = execution_result.get("error_flag", "none")
                    message = execution_result.get("message", "")
                    formatted_result = f"EXECUTION RESULT [{execution_name}]"
                    if error_flag != "none":
                        formatted_result += f" ERROR_FLAG={error_flag}"
                    if message:
                        formatted_result += f"; message={message}"
                    lines.append(formatted_result)
            elif step_dict.get("error_flag", "none") != "none":
                lines.append(
                    f"EXECUTION RESULT: {step_dict['error_flag']}"
                )
    return "\n".join(lines)


def format_action_trajectory(trajectory: List[Tuple[Any, bool]]) -> str:
    lines: List[str] = []
    step_index = 0
    for action, in_current_stage in trajectory:
        if in_current_stage:
            step_index += 1
            current_index = step_index
        else:
            current_index = 0
        lines.append(f"ACTION {current_index}: {action}")
    return "\n".join(lines)
