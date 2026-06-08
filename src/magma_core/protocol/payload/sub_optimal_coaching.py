# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Any, Dict, List, Optional, Tuple

from ..registry import ExternalRequestType
from .base_payload import BasePayload
from .coaching_common import format_memory, format_stage_trajectory


class SubOptimalDiagnosticPayload(BasePayload):
    """
    Payload to detect the earliest sub-optimal step in a successful trajectory.
    """

    def __init__(
        self,
        task_description: str,
        stage_instruction: str,
        stage_memory: List[str],
        trajectory: List[Tuple[str, str, bool]],
        dual_mode: bool,
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        if dual_mode:
            request_type = ExternalRequestType.SUBOPTIMAL_DIAGNOSE_DUAL
            self.output_rule = (
                '{"error_type": "ACTION or MEMORY of the first non-optimal step", '
                '"error_step": the index of the step, '
                '"error_description": "Concise explanation of what went wrong."}'
            )
        else:
            request_type = ExternalRequestType.SUBOPTIMAL_DIAGNOSE_SINGLE
            self.output_rule = (
                '{"error_step": the index of the step, '
                '"error_description": "Concise explanation of what went wrong and what should be done."}'
            )

        super().__init__(request_type, id, max_tokens, model)

        self.task_description = task_description
        self.instruction = stage_instruction
        self.memory = format_memory(stage_memory)
        self.trajectory = format_stage_trajectory(trajectory)
        self.keep_message = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_description": self.task_description,
            "instruction": self.instruction,
            "memory": self.memory,
            "trajectory": self.trajectory,
            "format": self.output_rule,
        }


class SubOptimalFixPayload(BasePayload):
    """
    Payload to regenerate a locally optimal answer from a sub-optimal step.
    """

    def __init__(
        self,
        old_messages: List[Dict[str, Any]],
        original_query: str,
        current_query: str,
        current_memory: str,
        model_answer: str,
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(ExternalRequestType.SUBOPTIMAL_FIX, id, max_tokens, model)

        self.old_messages = old_messages
        self.original_query = original_query
        self.current_query = current_query
        self.current_memory = current_memory
        self.model_answer = model_answer
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
