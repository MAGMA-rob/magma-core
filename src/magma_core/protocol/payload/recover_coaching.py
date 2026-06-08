# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
import json
from typing import Any, Dict, List, Optional, Union

from magma_core._prompts.coach.recovery.fix_injection_prompt import BAD_DECISION_PART, BAD_RECOVERY_PART
from ..registry import ExternalRequestType
from .base_payload import BasePayload
from .coaching_common import (
    extract_status_error,
    format_memory,
    format_stage_trajectory,
    merge_runtime_error_context,
)


class InformUserPayload(BasePayload):
    """
    Payload to generate an error answer to inform the user about repetitive failure.
    """

    def __init__(
        self,
        status_return: Dict[str, Any],
        is_dual_agent: bool,
        id: int,
        error_descriptions: Optional[List[str]] = None,
        max_tokens: int = 2500,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(ExternalRequestType.INFORM_USER, id, max_tokens, model)

        status = status_return.copy()
        self.tool_call = status.pop("previous_tool_call")
        self.error = merge_runtime_error_context(status["error"], error_descriptions)
        self.mem_text = ", using the sentence 'Looking at my memory,'" if is_dual_agent else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_call": self.tool_call,
            "error_message": self.error,
            "is_dual_agent": self.mem_text,
        }


class PlannerRecoveryPayload(BasePayload):
    """
    Payload to generate the immediate planner-recovery answer.
    """

    def __init__(
        self,
        stage_description: str,
        interaction_history: str,
        tools: List[Dict[str, Any]],
        status_return: Dict[str, Any],
        id: int,
        error_descriptions: Optional[List[str]] = None,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(ExternalRequestType.BUILD_PLANNER_RECOVERY, id, max_tokens, model)

        status = status_return.copy()
        self.tool_call = status.pop("previous_tool_call")
        self.error = merge_runtime_error_context(extract_status_error(status), error_descriptions)
        self.description = stage_description
        self.interaction = interaction_history
        self.tools = tools

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tools": self.tools,
            "interaction": self.interaction,
            "description": self.description,
            "tool_call": self.tool_call,
            "error_message": self.error,
        }


class RecoveryDiagnosticPayload(BasePayload):
    """
    Payload to diagnose the earliest causal step in a failed trajectory under
    execution uncertainty.
    """

    def __init__(
        self,
        task_description: str,
        instruction: str,
        memory: List[Any],
        trajectory: List[tuple[str, str, bool]],
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(ExternalRequestType.RECOVERY_DIAGNOSE, id, max_tokens, model)

        self.task_description = task_description
        self.instruction = instruction
        self.memory = format_memory(memory)
        self.trajectory = format_stage_trajectory(trajectory)
        self.keep_message = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_description": self.task_description,
            "instruction": self.instruction,
            "memory": self.memory,
            "trajectory": self.trajectory,
        }


class RecoveryFixPayload(BasePayload):
    """
    Payload to regenerate a corrected answer after a recovery-oriented
    diagnosis.
    """

    def __init__(
        self,
        old_messages: List[Dict[str, Any]],
        original_query: str,
        interaction_history: str,
        current_memory: Union[List[Any], str],
        model_answer: Union[Dict[str, Any], str],
        problem_type: str,
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(ExternalRequestType.RECOVERY_FIX, id, max_tokens, model)

        self.old_messages = old_messages
        self.original_query = original_query
        self.interaction = interaction_history
        if isinstance(current_memory, str):
            self.memory = current_memory
        else:
            self.memory = format_memory(current_memory)
        if isinstance(model_answer, str):
            self.model_answer = model_answer
        else:
            self.model_answer = json.dumps(model_answer, ensure_ascii=False, default=str)
        self.problem_type = problem_type
        self.specific_part = self._resolve_specific_part(problem_type)
        self.output_rule = (
            '{"think": short reasoning, "say": "One sentence to say what you are doing", '
            '"action": json_call}'
        )

    @staticmethod
    def _resolve_specific_part(problem_type: str) -> str:
        normalized_problem = problem_type.strip().upper()
        if normalized_problem in {"DECISION_ERROR", "PLANNING_ERROR"}:
            return BAD_DECISION_PART
        if normalized_problem == "RECOVERY_ERROR":
            return BAD_RECOVERY_PART
        raise ValueError(f"Unsupported recovery problem type '{problem_type}'")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "old_messages": self.old_messages,
            "original_query": self.original_query,
            "interaction": self.interaction,
            "memory": self.memory,
            "model_answer": self.model_answer,
            "specific_part": self.specific_part,
            "format": self.output_rule,
        }


class CleanMemoryPayload(BasePayload):
    """
    Payload to clean or regenerate memory at the end of a stage.
    """

    def __init__(
        self,
        memory: List[str],
        preserved_indices: List[int],
        list_constraints: Optional[List[str]],
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        if list_constraints is None:
            request_type = ExternalRequestType.CLEAN_MEMORY
            self.constraints = None
        else:
            request_type = ExternalRequestType.GEN_MEMORY
            self.constraints = "".join(constraint + "\n" for constraint in list_constraints)
        super().__init__(request_type, id, max_tokens, model)

        self.memory_str = ""
        for i, mem in enumerate(memory):
            idx = "X" if i in preserved_indices else i
            self.memory_str += f"{idx}. {mem}\n"

    def to_dict(self) -> Dict[str, Any]:
        if self.constraints is None:
            return {"memory": self.memory_str}

        return {
            "memory": self.memory_str,
            "constraints": self.constraints,
        }
