# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Any, Dict, List, Optional, Tuple, Union
import copy
import inspect
from abc import ABC, abstractmethod

from magma_core.simulation.data_structures import Situation, Log, Instruction

from magma_core.simulation.tasks import BaseTask
from magma_core.simulation.envs import DefaultEnv
from magma_core.simulation.randomizer import Randomizer

from magma_core.simulation.executor.executor import ToolsBaseExecutor

from magma_core.workers import LMWorker, LMWorkerPool


class SingleTaskExecutor(ToolsBaseExecutor, ABC):
    """
    Tools executor is the main class responsible for transforming LLM calls into actions using specific defined class.
    """

    # References
    task_ref : BaseTask # The underlying task data.

    # Randomization
    randomized : bool
    randomizer : Randomizer

    def __init__(
            self,
            nb_env : int,
            ollama_worker : Union[LMWorker, LMWorkerPool],
            planner_endpoint : str,
            gui : bool = False,
            nb_randomization : int = 0,
            random_seed: Optional[int] = None,
        ):
        super().__init__(
            nb_env, ollama_worker, planner_endpoint, gui
        )
        
        self.randomized = nb_randomization > 0
        if self.randomized:
            self.randomizer = Randomizer(nb_randomization, seed=random_seed)

    ####### getters
    
    def get_init_situation(self) -> Situation:
        """Allows to get the initial situation of the task to initialize the manager.
        Query, Attributes, Memory, Preserved Memory indices"""
        original_situation = self.task_ref.get_init_situation()

        if self.randomized:
            return self.randomizer.get_randomized_situation(original_situation)
        
        return original_situation
    
    def get_instruction(self, stage_id : int) -> Instruction:
        """Allows to get the initial situation of the task to initialize the manager.
        Query, Attributes, Memory, Preserved Memory indices"""
        instruction = self.task_ref.get_stage_input(stage_id).instruction

        if self.randomized:
            return self.randomizer.get_randomized_instruction(instruction)
        
        return instruction
    
    def get_tools(self) -> List[Dict]:
        """
        Get the tools available for this executor.
        """
        if self.randomized:
            return self.randomizer.get_tools()
        return self.task_ref.get_tools()

    def get_tools_with_real_names(self) -> List[Tuple[Dict, str]]:
        tools = self.get_tools()
        if self.randomized:
            return [
                (tool, self.randomizer.get_real_tool_name(tool["name"]))
                for tool in tools
            ]
        return [(tool, tool["name"]) for tool in tools]

    def get_skill_vocabulary_translator(self) -> Optional[Randomizer]:
        return self.randomizer if self.randomized else None

    def get_loaded_task_descriptions(self) -> Dict:
        """
        Return the overall task description and stages description
        """
        task_description = inspect.getdoc(self.task_ref)
        stages_descriptions = []
        for stage in self.task_ref.stages:
            stage_description = stage.get_description()
            stage_description["stage_name"] = type(stage).__name__
            stages_descriptions.append(stage_description)
        coaching_hint = self.task_ref.task_metadata.coaching_hint
        if self.randomized:
            task_description = self.randomizer.traduce_attributes_to_llm(task_description if task_description else '')
            if coaching_hint is not None:
                coaching_hint = self.randomizer.traduce_attributes_to_llm(coaching_hint)
            for s in stages_descriptions:
                s['goal_description'] = self.randomizer.traduce_attributes_to_llm(s['goal_description'])
                s['instruction'] = self.randomizer.traduce_attributes_to_llm(s['instruction'])

        return {
            "task_description" : task_description,
            "stages_description" : stages_descriptions,
            "coaching_hint" : coaching_hint,
        }

    def get_env_options(self) -> Dict:
        """
        Return the env options dict for reset / initialization parameters
        """
        return self.task_ref.initialization_parameters.env_options
    
    def get_task_info(self) -> Dict:
        return self.task_ref.get_task_info()

    def get_try_randomization_info(self) -> Dict[str, Any]:
        """
        Return the exact attribute/tool naming exposed to the LLM for the
        current variation so benchmark try logs can persist it.
        """
        if self.randomized:
            visible_attributes = self.randomizer.get_randomized_attributes(
                self.task_ref.get_complete_attributes()
            )
            variation_index: Optional[int] = self.randomizer.variation_idx
        else:
            visible_attributes = self.task_ref.get_complete_attributes()
            variation_index = None

        visible_tools = []
        for tool in self.get_tools():
            visible_tools.append(
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameter_names": list(tool.get("parameters", {}).keys()),
                }
            )

        return {
            "enabled": self.randomized,
            "variation_index": variation_index,
            "attributes": copy.deepcopy(visible_attributes),
            "tools": visible_tools,
        }

    ################ public function

    def initialize(
            self,
            task_ref: BaseTask,
            build_first_stage: bool = True,
            obs_mode: str = "state_dict",
            sim_backend: str = "auto",
        ) -> DefaultEnv:
        """
        Initialize the Executor with a task name class and some arguments for its __init__.
        Tasks must be present in the scenarios.scenarios package.
        """
        self.task_ref = task_ref
        self.env = self._create_envs(
            task_ref.maniskill_env_id,
            self.get_env_options(),
            task_ref.initialization_parameters.planner_options,
            obs_mode,
            sim_backend
        )
        
        env_state = self.env.unwrapped.get_state_dict()

        # task init
        init_env_state = self.task_ref.initialize_task(
            self.trajectory_converter.agents,
            self.trajectory_converter.agents_name,
            env_state=env_state, #type: ignore
            nb_env=self.nb_env,
            build_first_stage=build_first_stage
        )

        if self.randomized:
            self.randomizer.initialize_randomizer(self.task_ref)

        self.env.unwrapped.set_state_dict(init_env_state)

        return self.env

    ################ private functionmultiple_

    @abstractmethod
    def _get_logs(self, env_id: int) -> Tuple[List[Log], List[Log]]:
        """
        Return the log corresponding to the given env_id.
        First list for the full logs and the second is stage only logs
        """
        raise NotImplementedError("This function must be implemented in child class")

    @abstractmethod
    def _get_node_infos(self, node_id):
        """Allows to retrieve the env information (env_state and logs) from a specific nodes"""
        raise NotImplementedError("This function must be implemented in child class")
