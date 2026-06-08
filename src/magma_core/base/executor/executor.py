# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Any, Dict, List, Optional, Tuple, Union, Type
from collections import OrderedDict
import torch, inspect, logging, copy
import gymnasium as gym
from abc import ABC, abstractmethod

from ..trajectory import TrajectoryConverter
from ..data_structures import (
    ActiveStageErrorState, ToolExecution, ToolInfos, Situation, ToolRobot,
    EnvCreationInfos, CommanderAnswer, Log, ToolErrorFlag
)

from ..tasks import BaseTask
from ..envs import DefaultEnv
from ..randomizer import Randomizer

from magma_core.workers import LMWorker, LMWorkerPool


class ToolsBaseExecutor(ABC):
    """
    Tools executor is the main class responsible for transforming LLM calls into actions using specific defined class.
    """

    nb_env : int
    logger : logging.Logger

    # References
    task_ref : BaseTask # The underlying task data.
    env : DefaultEnv
    trajectory_converter : TrajectoryConverter

    # Params
    _gui : bool
    worker : Union[LMWorker, LMWorkerPool]
    planner_endpoint : str

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
        self.worker = ollama_worker
        self.nb_env = nb_env
        self._gui = gui
        
        self.randomized = nb_randomization > 0
        if self.randomized:
            self.randomizer = Randomizer(nb_randomization, seed=random_seed)

        self.logger = logging.getLogger("EXECUTION")
        self.planner_endpoint = planner_endpoint

    ####### getters
    
    def get_task_attributes(self, stage_id : int) -> Dict:
        """Return task attributes as dict"""
        t = self.task_ref.get_stage_attributes(stage_id)
        if self.randomized:
            return self.randomizer.get_randomized_attributes(t)
        return t
    
    def get_init_situation(self, stage_id : int) -> Situation:
        """Allows to get different elements to initialize the manager.
        Query, Attributes, Memory, Preserved Memory indices"""
        original_situation = self.task_ref.get_init_situation(stage_id)

        if self.randomized:
            return self.randomizer.get_randomized_situation(original_situation)
        
        return original_situation
    
    def get_tools(self) -> List[Dict]:
        """
        Get the tools available for this executor.
        """
        if self.randomized:
            return self.randomizer.get_tools()
        return self.task_ref.get_tools()
    
    def get_env_id(self) -> str:
        """
        Return the gym env id (name).
        """
        return self.task_ref.env_id

    def get_loaded_task_descriptions(self) -> Dict:
        """
        Return the overall task description and stages description
        """
        task_description = inspect.getdoc(self.task_ref)
        stages_descriptions = [stage.get_description() for stage in self.task_ref.stages]
        if self.randomized:
            task_description = self.randomizer.traduce_attributes_to_llm(task_description if task_description else '')
            for s in stages_descriptions:
                s['goal_description'] = self.randomizer.traduce_attributes_to_llm(s['goal_description'])
                s['instruction'] = self.randomizer.traduce_attributes_to_llm(s['instruction'])

        return {
            "task_description" : task_description,
            "stages_description" : stages_descriptions
        }

    def get_env_options(self, stage_id: Optional[int] = None) -> Dict:
        """
        Return the env options dict for reset / initialization parameters
        """
        return self.task_ref.env_options
    
    def get_task_info(self) -> Dict:
        return {
            "styles" : self.task_ref.styles,
            "approximal_difficulty" : self.task_ref.approximal_difficulty,
            "env_id" : self.task_ref.env_id
        }

    def get_try_randomization_info(self) -> Dict[str, Any]:
        """
        Return the exact attribute/tool naming exposed to the LLM for the
        current variation so benchmark try logs can persist it.
        """
        if self.randomized:
            visible_attributes = self.randomizer.get_randomized_attributes(
                self.task_ref.all_task_attributes
            )
            variation_index: Optional[int] = self.randomizer.variation_idx
        else:
            visible_attributes = self.task_ref.all_task_attributes
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
        if sim_backend == "auto" and not torch.cuda.is_available():
            sim_backend = "cpu"

        # task creation
        current_env_id = None
        if hasattr(self,"task_ref"):
            current_env_id = self.task_ref.env_id

        self.task_ref = task_ref
        
        if self.task_ref.env_id != current_env_id: # We create a new env only if the task required a different env_id then precedently.
            if hasattr(self, "env"):
                self.env.close()
            
            # env init
            if self._gui:
                self.env = gym.make( #type: ignore
                    self.task_ref.env_id,
                    num_envs = self.nb_env,
                    obs_mode=obs_mode,
                    control_mode="pd_joint_pos",
                    render_mode="human",
                    sim_backend=sim_backend,
                    parallel_in_single_scene=True,
                    )
            else:
                self.env = gym.make( #type: ignore
                    self.task_ref.env_id,
                    num_envs = self.nb_env,
                    obs_mode=obs_mode,
                    control_mode="pd_joint_pos",
                    render_mode="rgb_array",
                    sim_backend=sim_backend,
                )

        reset_options = copy.deepcopy(self.get_env_options())
        reset_options["reconfigure"] = True
        self.env.reset(seed=0, options=reset_options)

        # trajectory converter init
        self.trajectory_converter = TrajectoryConverter(
            self.env.unwrapped, #type: ignore
            self.nb_env,
            self.planner_endpoint,
            planner_init_options = self.task_ref.planner_options
        )
        a = self.step()
        self.env.step(a) # Steps just for make sure the env state is updated
        env_state = self.env.get_state_dict()

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

        self.env.set_state_dict(init_env_state)

        return self.env
    
    @abstractmethod
    def compute_actions(self, tools_call : Dict[int,CommanderAnswer]):
        """
        Take a batch of tools_calls. It's a dict where each key is a node_id associate with a dict.
        In case of GENERATION mode, the value dict contains 'tool' (the action dict) and 'src_id' (the parent node_id).
        In case of EVALUATION mode, the value dict contains directly the action dict.
        Transforms this into a batched sequence of steps per environment.
        """
        raise NotImplementedError("This function must be implemented in child class")

    @abstractmethod
    def step(self) -> Union[torch.Tensor, OrderedDict]:
        """
        Execute the actions in the environment.

        Returns:
            Dict: If SingleAGent env, return a batched tensor of action. For MultiAGent, return an OrderedDict
        """

        raise NotImplementedError("This function must be implemented in child class")

    @abstractmethod
    def verif_ended_tool(self, obs : Dict) -> Dict:
        """
        Verify if some envs have finished executing their ToolsExecution elements from the observation.

        If in GENERATION mode, it will also check if the overall task is finished.
        
        :param obs: Observation of the environment
        :type obs: Dict
        :return: A dict where each finished node_id have its assoiated return status. COuld be empty if no environments have finished.
        :rtype: Dict[int, Dict]
        """
        raise NotImplementedError("This function must be implemented in child class")


    ################ private function

    def _update_env_states(self, envs_infos: List[EnvCreationInfos]):
        """
        Override specific envs in the current state with provided env_state dicts.
        To apply change, you need to execute one env.step() after.
        """

        st = self.env.get_state_dict().copy() #type: ignore

        def _recursive_update_state(main_dict: Dict, env_id: int, update_dict: Dict):
            for key, value in update_dict.items():
                if isinstance(value, dict):
                    _recursive_update_state(main_dict[key], env_id, value)
                else:
                    main_dict[key][env_id] = value.clone()

        for env_info in envs_infos:
            env_id = env_info.env_id
            env_state = env_info.env_state
            _recursive_update_state(st, env_id, env_state)

        self.env.set_state_dict(st)

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

    def _initialize_stage_error_state(
            self,
            stage_id: int,
            error_state: ActiveStageErrorState,
            obs: Dict,
            env_id: int,
            agent_id: int = 0,
        ) -> ActiveStageErrorState:
        """
        Lazily select the active error(s) for a stage from the validated
        observation seen by the tools.
        """
        if error_state != {}:
            return error_state
        if len(self.task_ref.stages[stage_id].possible_errors) == 0:
            return {}

        return self.task_ref.initialize_stage_errors(stage_id, obs, env_id, agent_id)


    def _compute_single_tool(
            self,
            func_name : str,
            params : Dict,
            env_id : int,
            obs : Dict,
            error_state : ActiveStageErrorState,
            stage_id : int,
            current_node_step : int,
            original_log_length : int,
            source_node_id : int,
            node_id : int = -1,
            logs : Optional[List] = None,
            composite_progress : Optional[Dict[str, Any]] = None
        ) -> ToolInfos:
        # We initialize the stage error(s) from the post-step observation so the
        # same logic is shared by generation, evaluation and testing executors.
        error_state = self._initialize_stage_error_state(
            stage_id=stage_id,
            error_state=error_state,
            obs=obs,
            env_id=env_id,
        )

        if self.randomized:
            out = self.randomizer.map_tool_call(func_name, params)
            self.randomizer.set_tmp_translation(node_id, func_name)
            if isinstance(out, ToolExecution): # That's means that we encounter an error
                robot_tool = [ToolRobot("default",out)]
            else:
                robot_tool = [ToolRobot("default",self.task_ref.execute_tools(
                    obs,
                    env_id,
                    out[0],
                    out[1],
                    stage_id,
                    error_state,
                ))]
        else:
            robot_tool = [ToolRobot("default",self.task_ref.execute_tools(
                obs,
                env_id,
                func_name,
                params,
                stage_id,
                error_state,
            ))]

        return ToolInfos( # normal non-randomized execution
                    node_id=node_id,
                    tool_robots=robot_tool,
                    error_state=error_state,
                    logs=logs,
                    task_stage=stage_id,
                    stage_log_start_idx=original_log_length,
                    source_id=source_node_id,
                    agent_step=current_node_step,
                    composite_progress=composite_progress
                )
    
    def _compute_multiple_tool(
            self,
            actions : Dict,
            env_id : int,
            obs : Dict,
            error_state : ActiveStageErrorState,
            stage_id : int,
            current_node_step : int,
            original_log_length : int,
            source_node_id : int,
            node_id : int = -1,
            logs : Optional[List] = None,
            composite_progress : Optional[Dict[str, Any]] = None
        ) -> ToolInfos:
        # Stage errors are initialized once per child rollout before resolving
        # any robot-specific tool execution.
        error_state = self._initialize_stage_error_state(
            stage_id=stage_id,
            error_state=error_state,
            obs=obs,
            env_id=env_id,
        )

        names = self.task_ref.get_agent_names()
        unknow_robots = [r for r in actions.keys() if r not in names]
        if len(unknow_robots) > 0:
            s = ",".join(unknow_robots)
            if self.randomized:
                self.randomizer.set_tmp_translation(node_id, None)
            return ToolInfos(
                    node_id=node_id,
                    tool_robots=[ToolRobot(
                        "default",
                        ToolExecution(
                            [],
                            verifier=None,
                            reason=f"Unknow robot names {s}. Please use only knowns robots names",
                        ),
                    )],
                    error_state=error_state,
                    logs=logs,
                    stage_log_start_idx=original_log_length,
                    task_stage=stage_id,
                    source_id=source_node_id,
                    agent_step=current_node_step,
                    composite_progress=composite_progress
                )
        
        robot_tool = []
        fn_names = []
        for robot, tool in actions.items():
            func_name = tool.get("name",None)
            fn_names.append(func_name)
            params = tool.get("arguments",{})
            agent_id = names.index(robot)
            if func_name:
                if self.randomized:
                    out = self.randomizer.map_tool_call(func_name, params)
                    if isinstance(out, ToolExecution): # That's means that we encounter an error
                        robot_tool.append(ToolRobot(robot, out))
                    else:
                        robot_tool.append(ToolRobot(robot, self.task_ref.execute_tools(
                            obs,
                            env_id,
                            out[0],
                            out[1],
                            stage_id,
                            error_state,
                            agent_id
                        )))      
                else:
                    robot_tool.append(ToolRobot(robot, self.task_ref.execute_tools(
                        obs,
                        env_id,
                        func_name,
                        params,
                        stage_id,
                        error_state,
                        agent_id
                    )))
        

        if len(robot_tool) == 0:
            raise RuntimeError("A fully empty multiple tool call have reached the compute multiple tool.")
        
        if self.randomized:
            self.randomizer.set_tmp_translation(node_id, fn_names)
            
        return ToolInfos( # normal non-randomized execution
                    node_id=node_id,
                    tool_robots=robot_tool,
                    error_state=error_state,
                    logs=logs,
                    source_id=source_node_id,
                    stage_log_start_idx=original_log_length,
                    task_stage=stage_id,
                    agent_step=current_node_step,
                    composite_progress=composite_progress
                )

    def _pass_to_the_next_stage(
            self, 
            current_stage_id : int,
            env_ids : List[int],
            env_state : Dict,
        ) -> Dict:
        """
        Allows to call the post-effects of current stage and the init of the new ones.
        Building the new env_state for the next stage associated with the env_id.
        
        :param current_stage_id: The current stage id
        :type current_stage_id: int
        :param env_ids: The list of env which have completed the current stage
        :type env_ids: List[int]
        :param env_state: The current env state of all envs
        :type env_state: Dict
        :return: The updates env state of all envs (modifying only env_id index)
        :rtype: Dict[Any, Any]
        """
        self.task_ref.call_stage_change_effects(current_stage_id, env_state, env_ids)
        return env_state
