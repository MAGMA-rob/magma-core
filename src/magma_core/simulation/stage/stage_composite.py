# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.simulation.data_structures import StageState, StageSuccess
from abc import ABC, abstractmethod
from torch._tensor import Tensor
import torch

from magma_core.simulation.stage.base_stage import BaseTaskStage, StageGlobalParameters
from magma_core.simulation.data_structures import StageInput, EmptyInstruction, Instruction, Log

from typing import Any, List, Dict, Literal, Optional
from dataclasses import dataclass

@dataclass
class StageData:
    """
    Helper for representing stage inside the composite
    """
    stage : BaseTaskStage
    desired_nb_of_completion : int


class BaseStageComposite(BaseTaskStage, ABC):
    """
    Experimental base class for composite stages.

    A composite stage groups several regular stages and tracks how many times
    each one must be completed. Subclasses must implement
    ``is_fully_completed`` and ``get_stage_state`` to define the composite
    success criteria and stage progression.

    This API is still experimental and is not ready to be used yet. Only the
    composite situation is taken into account, while per-stage situations are
    ignored. Text-only stages, additive stages, and full environment
    reinitialization through ``build_init_state`` are not supported.
    """
    stages : List[StageData]
    must_reset : bool 

    #runtime storage of envs. As verification is always env (batched) then logs (per env), we keep the batch of env
    #and do the complete verif in log
    all_envs_result : List[Tensor]
    env_cpt : int

    def __init__(self, stages : List[BaseTaskStage], number_per_stage : List[int], stage_input : StageInput) -> None:
        if len(stages) != len(number_per_stage):
            raise RuntimeError(f"Lenght mismatch between stages ({len(stages)}) and number_per_stage ({len(number_per_stage)})")
        self.stages = [StageData(s,n) for s,n in zip(stages,number_per_stage)]
        super().__init__(
            goals=[],
            stage_goal_description="The goal of this composite is to realize each stage a desired number of time",
            stage_input=stage_input,
            global_parameters=StageGlobalParameters(additive_stage=False),
        )
        raise NotImplementedError("This is experimental for now and not supported in the rest of the pipeline")

    def validate(self, env_agents: List[str]):
        for s in self.stages:
            s.stage.validate(env_agents)
            if s.stage.is_text_only():
                raise RuntimeError(f"We can not have text-only stage ({s.__class__.__name__}) in StageComposite")
            if s.stage.get_stage_input().flag_answer_to_user:
                raise RuntimeError(f"The stage {s.__class__.__name__} has flag_answer_to_user set to True. \
                                   In StageComposite, this is handled by the stage_input of the composite.")
            if s.stage.is_additive_stage():
                raise RuntimeError("Additive composite are not yet supported")
            


    def get_description(self) -> Dict:
        stage_input = self.get_stage_input()
        ei = isinstance(stage_input.instruction, EmptyInstruction)
        return {
            "goal_description" : "The goal of this composite is to realize each stage a desired number of time",
            "request_type": self.get_request_type(),
            "composite" : [s.stage.get_description for s in self.stages],
            "text_only" : False,
            "empty_instruction" : ei,
            "linked_to_prev": self.is_linked_to_prev(),
            "instruction": "" if ei else stage_input.instruction.get_content()
        }

    def _verif_env_completion(self, obs: Dict) -> Tensor:
        self.all_envs_result = []
        self.env_cpt = 0
        for s in self.stages:
            self.all_envs_result.append(s.stage._verif_env_completion(obs))
        
        return torch.zeros_like(self.all_envs_result[0])   
    
    def verif_log_completion(
            self,
            stage_log: List[Log],
            full_log: List[Log],
            composite_progress: Optional[Dict[str, Any]] = None,
        ) -> int:
        """
        Override version which compute for a StageComposite.

        This value is the definitive one (the combine will return it).
        """
        # It's not very robust, we rely on increment at the class level. But it works for now.
        # todo: Implement a cleaner code
        full_error = True
        self.must_reset = False
        valid_idx = -1
        for i, s in enumerate(self.stages):
            log_res = s.stage.verif_log_completion(stage_log, full_log)
            total_res = s.stage.combine_stage_completion(
                self.all_envs_result[i][self.env_cpt].item(), #type: ignore
                log_res
            )
            if total_res == 1:
                full_error = False
                self.must_reset = s.stage.should_reset_at_end()
                valid_idx = i
                break
            elif total_res == 0:
                full_error = False

        self.env_cpt += 1

        if full_error: return -1
        # cropping logs to contains only logs related to composite
        n = 0
        if len(full_log) > 0:
            last_id_log = full_log[-1].stage_id
            for i, l in enumerate(full_log):
                n=i
                if l.stage_id == last_id_log:
                    break
        
        
        if valid_idx:
            return self.is_fully_completed(full_log[n:])
        return 0

    @abstractmethod
    def is_fully_completed(self, full_log: List[Log]) -> int:
        raise NotImplementedError
        
    def combine_stage_completion(self, task_completion: int, log_completion: int) -> int:
        return log_completion
    
    def is_composite_and_reset(self) -> bool:
        """
        Allows to know if we need to reset or no the env.
        It's managed throughout the self.must_reset bool. Which is set to false automatically at the beginning of each
        log loop (set to true inside if one env suceed and that env requires reset_at_end).
        """
        return self.must_reset
    
    @abstractmethod
    def get_stage_state(
            self,
            effective_tool_calls: int,
            stage_success: StageSuccess,
            log: List[Log],
        ) -> StageState:
        raise NotImplementedError
