from typing import Dict, List, Any, Tuple, Union, Type
from collections import OrderedDict
import torch, inspect, logging
import gymnasium as gym
from abc import ABC, abstractmethod

from ..trajectory import TrajectoryConverter
from ..data_structures import (
    ToolExecution, ToolInfos, Situation, ToolRobot,
    EnvCreationInfos, CommanderAnswer, Log
)
from .tool_random_wrapper import ToolRandomizerWrapper
from ..tasks import BaseTask
from ..envs import DefaultEnv

from magma_core.utils.global_utils import load_module_from_name
import magma_scenarios as tool_pkg
from magma_core.workers import LMWorker

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
    worker : LMWorker

    # Randomization
    randomized : bool
    randomizer : ToolRandomizerWrapper

    def __init__(
            self,
            nb_env : int,
            ollama_worker : LMWorker,
            gui : bool = False,
            randomized : bool = False,
        ):
        self.worker = ollama_worker
        self.nb_env = nb_env
        self._gui = gui
        self.randomized = randomized
        self.logger = logging.getLogger("EXECUTION")

        if randomized:
            self.randomizer = ToolRandomizerWrapper(0)

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

    def get_env_options(self, stage_id : int) -> Dict:
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

    ################ public function

    def compute_partial_reward(self, node_id : int) -> float:
        """
        Compute a partial reward for a given node_id. Must be implemented in child class.
        """
        return 0

    def initialize(self, task_name : str, task_arguments : Dict, build_first_stage : bool = True, obs_mode : str = "state_dict") -> DefaultEnv:
        """
        Initialize the Executor with a task name class and some arguments for its __init__.
        Tasks must be present in the scenarios.scenarios package.
        """
        # task creation
        current_env_id = None
        if hasattr(self,"task_ref"):
            current_env_id = self.task_ref.env_id

        TaskCls : Type[BaseTask] = load_module_from_name(tool_pkg, task_name)
        self.task_ref = TaskCls(**task_arguments)
        
        if self.task_ref.env_id != current_env_id: # We create a new env only if the task required a different env_id then precedently.
            if hasattr(self, "env"):
                self.env.close()
            
            # env init
            if self._gui:
                self.env = gym.make(
                    self.task_ref.env_id,
                    num_envs = self.nb_env,
                    obs_mode=obs_mode,
                    control_mode="pd_joint_pos",
                    render_mode="human",
                    parallel_in_single_scene=True
                    )
            else:
                self.env = gym.make(
                    self.task_ref.env_id,
                    num_envs = self.nb_env,
                    obs_mode=obs_mode,
                    control_mode="pd_joint_pos",
                    render_mode="rgb_array",
                )

        self.env.reset(seed=0, options=self.get_env_options(0))

        # trajectory converter init
        self.trajectory_converter = TrajectoryConverter(
            self.env.unwrapped,
            self.nb_env,
            planner_init_options = self.task_ref.planner_options
        )
        a = self.step()
        self.env.step(a) # Steps just for make sure the env state is updated
        env_state = self.env.get_state_dict()

        # task init
        init_env_state = self.task_ref.initialize_task(
            self.trajectory_converter.agents,
            self.trajectory_converter.agents_name,
            env_state=env_state,
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

        st = self.env.get_state_dict().copy()

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


    def _compute_single_tool(
            self,
            func_name : str,
            params : Dict,
            env_id : int,
            obs : Dict,
            stage_id : int,
            current_node_step : int,
            original_log_lenght : int,
            source_node_id : int,
            node_id : int = -1,
            logs : List = [],
        ) -> ToolInfos:

        if self.randomized:
            out = self.randomizer.map_tool_call(func_name, params)
            self.randomizer.tmp_translation[node_id] = func_name
            if isinstance(out, ToolExecution): # That's means that we encounter an error
                robot_tool = [ToolRobot("default",out)]
            else:
                robot_tool = [ToolRobot("default",self.task_ref.execute_tools(obs, env_id, out[0], out[1], stage_id))]
        else:
            robot_tool = [ToolRobot("default",self.task_ref.execute_tools(obs, env_id, func_name, params, stage_id))]

        return ToolInfos( # normal non-randomized execution
                    node_id=node_id,
                    tool_robots=robot_tool,
                    logs=logs,
                    task_stage=stage_id,
                    stage_log_start_idx=original_log_lenght,
                    source_id=source_node_id,
                    agent_step=current_node_step
                )
    
    def _compute_multiple_tool(
            self,
            actions : Dict,
            env_id : int,
            obs : Dict,
            stage_id : int,
            current_node_step : int,
            original_log_lenght : int,
            source_node_id : int,
            node_id : int = -1,
            logs : List = [],
        ) -> ToolInfos:
        names = self.task_ref.get_agent_names()
        unknow_robots = [r for r in actions.keys() if r not in names]
        if len(unknow_robots) > 0:
            s = ",".join(unknow_robots)
            if self.randomized:
                self.randomizer.tmp_translation[node_id] = None
            return ToolInfos(
                    node_id=node_id,
                    tool_robots=[ToolRobot(
                        "default",
                        ToolExecution(
                            [],
                            verifier=None,
                            reason=f"Unknow robot names {s}. Please use only knowns robots names"
                        ),
                    )],
                    logs=logs,
                    stage_log_start_idx=original_log_lenght,
                    task_stage=stage_id,
                    source_id=source_node_id,
                    agent_step=current_node_step
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
                        robot_tool.append(ToolRobot(robot, self.task_ref.execute_tools(obs, env_id, out[0], out[1], stage_id, agent_id)))      
                else:
                    robot_tool.append(ToolRobot(robot, self.task_ref.execute_tools(obs, env_id, func_name, params, stage_id, agent_id)))
        

        if len(robot_tool) == 0:
            raise RuntimeError("A fully empty multiple tool call have reached the compute multiple tool.")
        
        if self.randomized:
            self.randomizer.tmp_translation[node_id] = fn_names
            
        return ToolInfos( # normal non-randomized execution
                    node_id=node_id,
                    tool_robots=robot_tool,
                    logs=logs,
                    source_id=source_node_id,
                    stage_log_start_idx=original_log_lenght,
                    task_stage=stage_id,
                    agent_step=current_node_step
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