# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import json
from typing import Any, ClassVar, Dict, List, Optional

from magma_core._prompts.coach.diagnostic.coach_diagnose import COACH_DIAGNOSIS, FAILURE_PARAGRAPHE
from magma_core._prompts.coach.diagnostic.diag_non_optimal import DIAGNOSIS_SUB_OPTI
from magma_core._prompts.coach.diagnostic.select_similar_cases import (
    SELECT_SIMILAR_CASES,
)

from .base_payload import BasePayload
from .coaching_common import format_stage_trajectory


class CoachDiagnosisPayload(BasePayload):
    """
    Payload to identify the earliest causal decision in a failed trajectory.
    """

    prompt_template: ClassVar[str] = COACH_DIAGNOSIS
    debug_log: ClassVar[bool] = True

    def __init__(
        self,
        stage_goal: str,
        trajectory: List[Dict],
        failure_type: str,
        id: int,
        task_coaching_hint: Optional[str] = None,
        terminal_feedback: Optional[str] = None,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(id, max_tokens, model)
        self.stage_goal = stage_goal
        self.failure_type = failure_type
        self.terminal_feedback = terminal_feedback
        self.trajectory_steps = trajectory if isinstance(trajectory, list) else []
        if isinstance(trajectory, str):
            self.trajectory = trajectory
        else:
            self.trajectory = format_stage_trajectory(trajectory)
        self.specific_failure_paragraph = FAILURE_PARAGRAPHE[failure_type]
        self.task_coaching_hint = task_coaching_hint or ""
        self.validated_similar_cases = ""

    def set_coaching_examples(
        self,
        examples: List[Dict[str, Any]],
    ) -> None:
        formatted_examples = []
        for index, example in enumerate(examples, start=1):
            stage_goal = example.get("stage_goal")
            trajectory = example.get("trajectory")
            diagnosis = example.get("diagnosis")
            if (
                not isinstance(stage_goal, str)
                or not stage_goal.strip()
                or not isinstance(trajectory, list)
                or not trajectory
                or any(not isinstance(step, dict) for step in trajectory)
                or not isinstance(diagnosis, dict)
            ):
                continue
            formatted_examples.append(
                f"CASE {index}\n"
                f"STAGE GOAL\n{stage_goal}\n\n"
                "STAGE TRAJECTORY\n"
                f"{format_stage_trajectory(trajectory)}\n\n"
                "VALIDATED DIAGNOSIS\n"
                f"{json.dumps(diagnosis, ensure_ascii=False, indent=2)}"
            )

        if not formatted_examples:
            self.validated_similar_cases = ""
            return
        joined_examples = "\n\n".join(formatted_examples)
        self.validated_similar_cases = f"""
--------------------------------------------------
VALIDATED SIMILAR CASES
--------------------------------------------------

The following successful coaching cases are references only.
Do not copy their decision indices or follow their trajectories mechanically.
Analyze the current trajectory independently.

{joined_examples}

END OF VALIDATED SIMILAR CASES
"""

    def extract_coaching_example_phase(
        self,
        output: str,
    ) -> Optional[tuple[str, Dict[str, Any]]]:
        try:
            diagnosis = json.loads(output)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(diagnosis, dict) or set(diagnosis) != {
            "decision_index",
            "reason",
            "expected_decision",
        }:
            return None
        decision_index = diagnosis["decision_index"]
        reason = diagnosis["reason"]
        expected_decision = diagnosis["expected_decision"]
        if (
            not isinstance(decision_index, int)
            or isinstance(decision_index, bool)
            or decision_index <= 0
            or not isinstance(reason, str)
            or not reason.strip()
            or not isinstance(expected_decision, str)
            or not expected_decision.strip()
        ):
            return None
        return "diagnosis", {
            "decision_index": decision_index,
            "reason": reason.strip(),
            "expected_decision": expected_decision.strip(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_goal": self.stage_goal,
            "trajectory": self.trajectory,
            "specific_failure_paragraph": self.specific_failure_paragraph,
            "task_coaching_hint": self.task_coaching_hint,
            "validated_similar_cases": self.validated_similar_cases,
        }

    def to_human_dict(self) -> Dict[str, Any]:
        data = self.to_dict()
        data.update(
            {
                "failure_type": self.failure_type,
                "terminal_feedback": self.terminal_feedback,
                "trajectory_steps": self.trajectory_steps,
            }
        )
        return data


class SimilarCaseSelectionPayload(BasePayload):
    """Select relevant cached coaching cases before causal diagnosis."""

    prompt_template: ClassVar[str] = SELECT_SIMILAR_CASES
    debug_log: ClassVar[bool] = True

    def __init__(
        self,
        stage_goal: str,
        trajectory: List[Dict[str, Any]],
        cases: List[Dict[str, Any]],
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(id, max_tokens, model)
        self.stage_goal = stage_goal
        self.trajectory_steps = list(trajectory)
        self.trajectory = format_stage_trajectory(trajectory)
        self.case_values = list(cases)
        formatted_cases = []
        for index, case in enumerate(cases, start=1):
            formatted_cases.append(
                f"CASE {index}\n"
                f"STAGE GOAL\n{case['stage_goal']}\n\n"
                "STAGE TRAJECTORY\n"
                f"{format_stage_trajectory(case['trajectory'])}\n\n"
                "VALIDATED DIAGNOSIS\n"
                f"{json.dumps(case['diagnosis'], ensure_ascii=False)}"
            )
        self.cases = "\n\n".join(formatted_cases)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_goal": self.stage_goal,
            "trajectory": self.trajectory,
            "cases": self.cases,
        }

    def to_human_dict(self) -> Dict[str, Any]:
        return {
            "stage_goal": self.stage_goal,
            "trajectory": self.trajectory,
            "trajectory_steps": self.trajectory_steps,
            "cases": self.case_values,
        }


class SuboptimalDiagnosisPayload(BasePayload):
    """Payload used to identify a defensible local inefficiency."""

    prompt_template: ClassVar[str] = DIAGNOSIS_SUB_OPTI
    debug_log: ClassVar[bool] = True

    def __init__(
        self,
        task_description: str,
        stage_goal: str,
        permanent_rules: List[str],
        trajectory: List[Dict[str, Any]],
        example_trajectory: Optional[List[Dict[str, Any]]],
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(id, max_tokens, model)
        self.task_description = task_description
        self.stage_goal = stage_goal
        self.permanent_rules_list = list(permanent_rules)
        self.trajectory_steps = list(trajectory)
        self.example_trajectory_steps = (
            None if example_trajectory is None else list(example_trajectory)
        )
        self.permanent_rules = (
            "None"
            if len(permanent_rules) == 0
            else "\n".join(f"- {rule}" for rule in permanent_rules)
        )
        self.trajectory = format_stage_trajectory(trajectory)
        self.example_trajectory = (
            "No validated efficient trajectory is available."
            if example_trajectory is None
            else format_stage_trajectory(example_trajectory)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_description": self.task_description,
            "stage_goal": self.stage_goal,
            "permanent_rules": self.permanent_rules,
            "example_trajectory": self.example_trajectory,
            "trajectory": self.trajectory,
        }

    def to_human_dict(self) -> Dict[str, Any]:
        data = self.to_dict()
        data.update({
            "permanent_rules_list": self.permanent_rules_list,
            "trajectory_steps": self.trajectory_steps,
            "example_trajectory_steps": self.example_trajectory_steps,
        })
        return data
