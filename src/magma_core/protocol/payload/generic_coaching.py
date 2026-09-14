# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
import json
from typing import Any, ClassVar, Dict, List, Optional

from magma_core._prompts.coach.specific_failure.fix_say_prompt import FIX_TEXT_ONLY
from magma_core._prompts.coach.specific_failure.format_prompt import FIX_FORMAT

from .base_payload import BasePayload


class FormatFixPayload(BasePayload):
    """
    Payload to repair a malformed agent answer into an executable answer.
    """

    prompt_template: ClassVar[str] = FIX_FORMAT
    debug_log: ClassVar[bool] = True

    def __init__(
        self,
        stage_goal: str,
        rejected_answer: Any,
        rejection_reason: str,
        desired_output_format: Dict[str, Any],
        output_format_rules: str,
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
        trajectory: Optional[List[Dict[str, Any]]] = None,
        task_attributes: Optional[Dict[str, Any]] = None,
        task_coaching_hint: Optional[str] = None,
        format_component: Optional[str] = None,
        task_state_view: Optional[Dict[str, Any]] = None,
        rejected_answer_display: Optional[str] = None,
        response_instructions: Optional[str] = None,
    ) -> None:
        super().__init__(id, max_tokens, model)
        self.stage_goal = stage_goal
        self.rejected_answer = rejected_answer
        self.rejection_reason = rejection_reason
        self.desired_output_format = desired_output_format
        self.output_format_rules = output_format_rules
        self.trajectory_steps = trajectory or []
        self.task_attributes = task_attributes or {}
        self.task_coaching_hint = task_coaching_hint or ""
        self.format_component = format_component
        self.task_state_view = task_state_view or {}
        self.rejected_answer_display = rejected_answer_display
        self.response_instructions = response_instructions or (
            "Return one JSON object only, without markdown fences. "
            "The object must strictly match the desired output format."
        )

    def to_dict(self) -> Dict[str, Any]:
        if self.task_state_view:
            input_context = json.dumps(
                self.task_state_view,
                ensure_ascii=False,
                indent=2,
            )
        else:
            input_context = (
                "No additional structured input context was provided."
            )
        rejected_answer = (
            self.rejected_answer_display
            if self.rejected_answer_display is not None
            else json.dumps(
                self.rejected_answer,
                ensure_ascii=False,
                default=str,
            )
        )
        return {
            "stage_goal": self.stage_goal,
            "input_context": input_context,
            "rejected_answer": rejected_answer,
            "rejection_reason": self.rejection_reason,
            "desired_output_format": json.dumps(
                self.desired_output_format,
                ensure_ascii=False,
                default=str,
            ),
            "output_format_rules": self.output_format_rules,
            "response_instructions": self.response_instructions,
        }

    def to_human_dict(self) -> Dict[str, Any]:
        return {
            "stage_goal": self.stage_goal,
            "rejected_answer": self.rejected_answer,
            "rejection_reason": self.rejection_reason,
            "desired_output_format": self.desired_output_format,
            "output_format_rules": self.output_format_rules,
            "trajectory_steps": self.trajectory_steps,
            "task_attributes": self.task_attributes,
            "task_coaching_hint": self.task_coaching_hint,
            "format_component": self.format_component,
            "task_state_view": self.task_state_view,
            "rejected_answer_display": self.rejected_answer_display,
            "response_instructions": self.response_instructions,
        }


class FailureTextOnlyFixPayload(BasePayload):
    """
    Payload to rewrite a text-only failed answer.
    """

    prompt_template: ClassVar[str] = FIX_TEXT_ONLY
    debug_log: ClassVar[bool] = True

    def __init__(
        self,
        stage_goal: str,
        rejected_answer: str,
        rejection_reason: str,
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
        trajectory: Optional[List[Dict[str, Any]]] = None,
        task_attributes: Optional[Dict[str, Any]] = None,
        task_coaching_hint: Optional[str] = None,
    ) -> None:
        super().__init__(id, max_tokens, model)

        self.stage_goal = stage_goal
        self.rejected_answer = rejected_answer
        self.rejection_reason = rejection_reason
        self.trajectory_steps = trajectory or []
        self.task_attributes = task_attributes or {}
        self.task_coaching_hint = task_coaching_hint or ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_goal": self.stage_goal,
            "rejected_answer": self.rejected_answer,
            "rejection_reason": self.rejection_reason,
        }

    def to_human_dict(self) -> Dict[str, Any]:
        return {
            **self.to_dict(),
            "trajectory_steps": self.trajectory_steps,
            "task_attributes": self.task_attributes,
            "task_coaching_hint": self.task_coaching_hint,
        }
