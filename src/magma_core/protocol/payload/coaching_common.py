# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

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


def format_memory(memory: List[Any]) -> str:
    return "".join(f"- {mem}\n" for mem in memory)


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


def format_stage_trajectory(trajectory: List[Tuple[str, str, bool]]) -> str:
    lines: List[str] = []
    step_index = 0
    for event_type, content, diagnosable in trajectory:
        if event_type in {"USER","SYSTEM","QUERY"}:
            if diagnosable:
                step_index += 1
            else:
                step_index = 0
        current_index = step_index if diagnosable else 0
        lines.append(f"{event_type} {current_index}: {content}")
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
