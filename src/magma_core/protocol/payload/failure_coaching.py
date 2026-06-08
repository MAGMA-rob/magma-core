# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import json
from typing import Any, Dict, List, Optional, Tuple

from ..registry import ExternalRequestType
from .base_payload import BasePayload
from .coaching_common import format_memory, format_stage_trajectory, format_tool_names


class FailureTextOnlyFixPayload(BasePayload):
    """
    Payload to rewrite a text-only failed answer.
    """

    def __init__(
        self,
        stage_description: str,
        model_answer: Dict[str, Any],
        user_instruction: str,
        memory: List[Any],
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(ExternalRequestType.FAILURE_TEXT_ONLY_FIX, id, max_tokens, model)

        self.description = stage_description
        self.answer = model_answer
        self.instruction = user_instruction
        self.memory = format_memory(memory)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "instruction": self.instruction,
            "memory": self.memory,
            "answer": self.answer,
        }


class FailureTextOnlyAnswerDiagnosticPayload(BasePayload):
    """
    Payload to identify the earliest causal step in a failed text-only
    trajectory that may contain tool calls before the final answer.
    """

    def __init__(
        self,
        stage_description: str,
        original_instruction: str,
        original_memory: List[Any],
        trajectory: List[Tuple[str, str, bool]],
        tools: List[Dict[str, Any]],
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(ExternalRequestType.FAILURE_TEXT_ONLY_ANSWER_DIAGNOSE, id, max_tokens, model)

        self.description = stage_description
        self.instruction = original_instruction
        self.memory_constraints = original_memory
        self.memory = format_memory(original_memory)
        self.trajectory = format_stage_trajectory(trajectory)
        self.tools = tools
        self.output_rule = (
            '{"error_step": the index of the first causal erroneous step, '
            '"error_description": "Concise explanation of why this step caused the failure."}'
        )
        self.keep_message = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "instruction": self.instruction,
            "memory": self.memory,
            "trajectory": self.trajectory,
            "tools": format_tool_names(self.tools, include_description=True),
            "format": self.output_rule,
        }


class FailureTextOnlyAnswerFixPayload(BasePayload):
    """
    Payload to regenerate one step in a failed text-only stage where tools may
    be used before the final answer.
    """

    def __init__(
        self,
        old_messages: List[Dict[str, Any]],
        stage_description: str,
        tools: List[Dict[str, Any]],
        original_query: str,
        current_query: str,
        current_memory: List[Any],
        model_answer: Dict[str, Any],
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(ExternalRequestType.FAILURE_TEXT_ONLY_ANSWER_FIX, id, max_tokens, model)

        self.old_messages = old_messages
        self.description = stage_description
        self.tools = tools
        self.original_query = original_query
        self.current_query = current_query
        self.current_memory = format_memory(current_memory)
        self.model_answer = json.dumps(model_answer, ensure_ascii=False, default=str)
        self.output_rule = (
            '{"think": short reasoning, "say": "One sentence to say what you are doing or the final answer", '
            '"action": json_call_or_empty_object}'
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "description": self.description,
            "tools": format_tool_names(self.tools, include_description=True),
            "original_query": self.original_query,
            "query": self.current_query,
            "memory": self.current_memory,
            "model_answer": self.model_answer,
            "format": self.output_rule,
            "additional": ""
        }
        if len(self.old_messages) > 0:
            d["old_messages"] = self.old_messages
        else:
            d["additional"] = "DIAGNOSIS\nThe model probably try to give an answer to the user without using any of the tool that can help get the answer."
        return d


class MissingActionDiagnosticPayload(BasePayload):
    """
    Payload to diagnose which action was missing on a no-action failure.
    """

    def __init__(
        self,
        stage_description: str,
        interaction_history: str,
        memory: List[Any],
        tools: List[Dict[str, Any]],
        model_answer: Dict[str, Any],
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(ExternalRequestType.MISSING_ACTION_DIAGNOSE, id, max_tokens, model)

        self.description = stage_description
        self.interaction_history = interaction_history
        self.memory = format_memory(memory)
        self.tools = tools
        self.answer = model_answer
        self.keep_message = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "interaction_history": self.interaction_history,
            "memory": self.memory,
            "tools": format_tool_names(self.tools, include_description=True),
            "answer": self.answer,
        }


class MissingActionFixPayload(BasePayload):
    """
    Payload to generate the final corrected answer for a missing-action failure.
    """

    def __init__(
        self,
        old_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        attributes: Dict[str, Any],
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(ExternalRequestType.MISSING_ACTION_FIX, id, max_tokens, model)

        self.old_messages = old_messages
        self.tools = tools
        self.attributes = attributes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "old_messages": self.old_messages,
            "tools": self.tools,
            "attributes": self.attributes,
        }


class FailureDiagnosticPayload(BasePayload):
    """
    Payload to identify the earliest causal step in a failed trajectory.
    """

    def __init__(
        self,
        task_description: str,
        original_instruction: str,
        original_memory: List[Any],
        trajectory: List[Tuple[str, str, bool]],
        dual_mode: bool,
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        request_type = (
            ExternalRequestType.FAILURE_DIAGNOSE_DUAL
            if dual_mode
            else ExternalRequestType.FAILURE_DIAGNOSE_SINGLE
        )
        super().__init__(request_type, id, max_tokens, model)

        self.task_description = task_description
        self.instruction = original_instruction
        self.memory = format_memory(original_memory)
        self.trajectory = format_stage_trajectory(trajectory)
        self.output_rule = (
            '{"error_step": the index of the first causal erroneous step, '
            '"error_description": "Concise explanation of why this step caused the failure."}'
        )
        self.keep_message = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_description": self.task_description,
            "instruction": self.instruction,
            "memory": self.memory,
            "trajectory": self.trajectory,
            "format": self.output_rule,
        }


class FailureFixPayload(BasePayload):
    """
    Payload to regenerate the selected failed step after diagnosis.
    """

    def __init__(
        self,
        old_messages: List[Dict[str, Any]],
        original_query: str,
        current_query: str,
        current_memory: List[Any],
        model_answer: Dict[str, Any],
        dual_mode: bool,
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        request_type = (
            ExternalRequestType.FAILURE_FIX_DUAL
            if dual_mode
            else ExternalRequestType.FAILURE_FIX_SINGLE
        )
        super().__init__(request_type, id, max_tokens, model)

        self.old_messages = old_messages
        self.original_query = original_query
        self.current_query = current_query
        self.current_memory = format_memory(current_memory)
        self.model_answer = json.dumps(model_answer, ensure_ascii=False, default=str)
        self.output_rule = (
            '{"think": short reasoning, "say": "One sentence to say what you are doing", '
            '"action": json_call}'
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "old_messages": self.old_messages,
            "original_query": self.original_query,
            "query": self.current_query,
            "memory": self.current_memory,
            "model_answer": self.model_answer,
            "format": self.output_rule,
        }
