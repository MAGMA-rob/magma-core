# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.simulation.stage.base_stage import (
    BaseTaskStage,
    StageGlobalParameters,
    TextOnlyValidationMode,
)
from magma_core.simulation.data_structures import (
    EmptyInstruction,
    StageInput,
    UserInstruction,
    Instruction,
    Log,
)

from typing import List, Dict, Literal, Optional

# Some Stage template you can use as base for common Stage as Asking a question or giving a constraint to the model that it must memorize.

class CompletionAnswerStage(BaseTaskStage):
    """Text-only stage accepting a direct user-facing completion message."""

    def __init__(
            self,
            allowed_tools: Optional[List[str]] = None,
            reset_at_end: bool = False,
        ) -> None:
        detector_names = sorted(set(allowed_tools or []))
        self.target_tool_calls = 1 if detector_names else 0
        self.max_tool_calls = 3 if detector_names else 0
        super().__init__(
            goals=[],
            stage_goal_description=(
                "Clearly inform the user that the requested mission is complete."
            ),
            stage_input=StageInput(
                instruction=EmptyInstruction(),
                flag_answer_to_user=False,
                linked_to_prev=True,
            ),
            global_parameters=StageGlobalParameters(
                reset_at_end=reset_at_end,
                verification_prompt=(
                    "The answer must be a non-empty user-facing completion message."
                ),
                text_only_validation=TextOnlyValidationMode.SAY_ONLY,
                allow_tools_before_answer=bool(detector_names),
                allowed_tools=detector_names,
            ),
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "allowed_tools": list(self.get_allowed_tools()),
            "reset_at_end": self.should_reset_at_end(),
        }

class AskingBaseStage(BaseTaskStage):
    """
    [TEXT ONLY] (can allow tools)
    One-step text-only stage that checks whether the agent answers a question.
    """

    target_tool_calls = 1
    max_tool_calls = 1

    def __init__(
            self,
            question: str,
            answer : str,
            linked_to_prev: bool = False,
            allow_tools_before_answer: bool = False,
            allowed_tools: Optional[List[str]] = None,
        ) -> None:
        """
        You can pass a question and the associated answer that the model must output.
        Most of the time, linked_to_prev must be false. The only reason to turn it to true is
        if the question is in fact an action request and that the answer of the model is to ask a question.
        """
        self.question = question
        self.answer = answer
        stage_input = StageInput(
            instruction=UserInstruction(question),
            flag_answer_to_user=False, # Always False because Asking Stage is already an answer stage
            linked_to_prev=linked_to_prev,
        )
        super().__init__(
            goals=[],
            stage_goal_description=f"The stage of the goal is to ensure that the model correctly answer to the question by answering : {answer}",
            stage_input=stage_input,
            global_parameters=StageGlobalParameters(
                reset_at_end=True,
                verification_prompt="The model correctly answer : " + answer,
                allow_tools_before_answer=allow_tools_before_answer,
                allowed_tools=[] if allowed_tools is None else list(allowed_tools),
            ),
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "linked_to_prev": self.get_stage_input().linked_to_prev,
            "allow_tools_before_answer": self.allows_tools_before_answer(),
            "allowed_tools": list(self.get_allowed_tools()),
        }

class ConstraintBaseStage(BaseTaskStage):
    """
    [TEXT ONLY]
    One-step text-only stage that checks rule understanding.
    """

    target_tool_calls = 1
    max_tool_calls = 1

    def __init__(
            self,
            constraint: str,
            linked_to_prev: bool = False,
            reset_at_end: bool = True,
        ) -> None:
        stage_input = StageInput(
            instruction=UserInstruction(constraint, has_constraint=True),
            flag_answer_to_user=False,
            linked_to_prev=linked_to_prev,
        )
        super().__init__(
            goals=[],
            stage_goal_description=f"The stage of the goal is to ensure that the model understand what the user said : {constraint}",
            stage_input=stage_input,
            global_parameters=StageGlobalParameters(
                reset_at_end=reset_at_end,
                verification_prompt=f"The model understand what the user asked : {constraint}",
            ),
        )
        self.constraint = constraint

    def _to_spec_arguments(self) -> Dict:
        return {
            "constraint": self.constraint,
            "linked_to_prev": self.get_stage_input().linked_to_prev,
            "reset_at_end": self.should_reset_at_end(),
        }

class ModifAttributesBaseStage(BaseTaskStage):
    """
    One-step stage that verifies an attribute add/remove action in the logs.
    Any stage that requires attributes modification must inherits from it.
    """

    target_tool_calls = 1
    max_tool_calls = 1

    def __init__(
            self,
            mode : Literal["ADD","REMOVE"],
            stage_input : StageInput | Dict,
            val_name : str,
            att_name : str,
            target_tool_calls: int = 1,
            max_tool_calls: int = 1,
        ) -> None:
        if isinstance(stage_input, dict):
            stage_input = StageInput(**stage_input)
        if mode == "ADD":
            action = "add"
            preposition = "to"
        elif mode == "REMOVE":
            action = "remove"
            preposition = "from"
        else:
            raise ValueError(f"Unsupported attribute modification mode: {mode}")

        goal = f"The goal is that the robot call the correct function to {action} {val_name} {preposition} its {att_name}."
        super().__init__(
            goals=[],
            stage_goal_description=goal,
            stage_input=stage_input,
            global_parameters=StageGlobalParameters(additive_stage=True),
        )
        self.val_name = val_name
        self.att_name = att_name
        self.mode = mode
        self.target_tool_calls = target_tool_calls
        self.max_tool_calls = max_tool_calls

    def _to_spec_arguments(self) -> Dict:
        return {
            "mode": self.mode,
            "stage_input": {
                "instruction": self.get_stage_input().instruction,
                "flag_answer_to_user": (
                    self.get_stage_input().flag_answer_to_user
                ),
                "linked_to_prev": self.get_stage_input().linked_to_prev,
            },
            "val_name": self.val_name,
            "att_name": self.att_name,
            "target_tool_calls": self.target_tool_calls,
            "max_tool_calls": self.max_tool_calls,
        }

    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        if len(stage_log) == 0: return 0
        if stage_log[-1].action == self.mode:
            if stage_log[-1].content == (self.att_name, self.val_name): return 1
        return -1
