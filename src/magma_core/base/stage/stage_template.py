# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from .base_stage import BaseTaskStage
from ..data_structures import Situation, UserInstruction, Instruction, Log

from typing import List, Dict, Literal, Optional

# Some Stage template you can use as base for common Stage as Asking a question or giving a constraint to the model that it must memorize.

class AskingBaseStage(BaseTaskStage):
    """One-step text-only stage that checks whether the agent answers a question."""

    target_steps = 1
    acceptance_steps = 0

    def __init__(
            self,
            question: str,
            answer : str,
            memory : List[str],
            attributes : Dict,
            linked_to_prev: bool = False,
            allow_tools_before_answer: bool = False,
            allowed_tools: Optional[List[str]] = None,
        ) -> None:
        super().__init__(
            [],
            True,
            f"The stage of the goal is to ensure that the model correctly answer to the question by answering : {answer}",
            linked_to_prev=linked_to_prev,
        )
        self.allow_tools_before_answer = allow_tools_before_answer
        self.allowed_tools = [] if allowed_tools is None else list(allowed_tools)
        self.situation = Situation(
            memory=memory,
            preserved_memory_indices=[e for e in range(len(memory))],
            instruction=UserInstruction(question),
            attributes=attributes,
            flag_answer_to_user=False
        )
        self.verification_prompt = "The model correctly answer : " + answer

class ConstraintBaseStage(BaseTaskStage):
    """One-step text-only stage that checks rule understanding."""

    target_steps = 1
    acceptance_steps = 0

    def __init__(
            self,
            constraint: str,
            memory : List[str],
            attributes : Dict,
            linked_to_prev: bool = False,
        ) -> None:
        super().__init__(
            [],
            True,
            f"The stage of the goal is to ensure that the model understand what the user said : {constraint}",
            linked_to_prev=linked_to_prev,
        )

        self.verification_prompt = f"The model understand what the user asked : {constraint}"
        self.situation = Situation(
            memory=memory,
            instruction=UserInstruction(constraint,has_constraint=True),
            flag_answer_to_user=False,
            attributes=attributes,
            preserved_memory_indices=[e for e in range(len(memory))]
        )

class ModifAttributesBaseStage(BaseTaskStage):
    """One-step stage that verifies an attribute add/remove action in the logs."""

    target_steps = 1
    acceptance_steps = 0

    additive_stage = True

    def __init__(
            self,
            mode : Literal["ADD","REMOVE"],
            instruction : Instruction,
            val_name : str,
            att_name : str,
            memory: List[str],
            preserved_memory_indices : List[int],
            attributes : Dict,
            flag_answer_to_user : bool,
            linked_to_prev: bool = False,
        ) -> None:
        if mode == "ADD":
            action = "add"
            preposition = "to"
        elif mode == "REMOVE":
            action = "remove"
            preposition = "from"
        else:
            raise ValueError(f"Unsupported attribute modification mode: {mode}")

        goal = f"The goal is that the robot call the correct function to {action} {val_name} {preposition} its {att_name}."
        super().__init__([],True, goal, linked_to_prev=linked_to_prev)
        self.situation = Situation(
            memory,preserved_memory_indices,attributes, instruction, flag_answer_to_user=flag_answer_to_user
        )
        self.val_name = val_name
        self.att_name = att_name
        self.mode = mode

    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        if len(stage_log) == 0: return 0
        if stage_log[-1].action == self.mode:
            if stage_log[-1].content == (self.att_name, self.val_name): return 1
        return -1
