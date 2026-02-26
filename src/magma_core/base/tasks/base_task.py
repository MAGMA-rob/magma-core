# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Dict, List, Any, Tuple, Type, Optional
from abc import ABC, abstractmethod
import torch, copy

from ..tools import BaseToolsAPI
from ..data_structures import Situation, StageState, ToolExecution, EmptyInstruction, Log
from ..tasks_style import TaskStyle
from magma_core.utils.global_utils import slice_obs, batch_set_value, extract_env_state_val

required_task_attrs = [
    "env_id",
    "name",
    "stages",
    "all_task_attributes",
    "Tools_cls",
    "styles",
    "approximal_difficulty",
    "env_options",
    "planner_options"
]

required_stage_attrs = [
    "target_steps",
    "acceptance_steps",
    "situation",
    "stage_goal_description"
]

class BaseTaskStage(ABC):
    """
    Base class to define a task stage which allows to define the situation, the number of steps and the completion goal for the step.

    You can also define if you want to reset the env at the end of this stage and you must give a short contextual description of the stage. Used for coaching.
    """

    reset_at_end : bool
    target_steps : int # The number of target steps required to complete the task.
    acceptance_steps : int # The number of steps (+ target_steps) to let the model try to complete the task.

    # The initial situation of the step (memory, attributes, instructions...)
    # Could be overriden by precedent step final step if the model complete it.
    situation : Situation
    stage_goal_description : str

    additive_stage : bool # Mark if this stage can modify the attributes (Default to False)
    verification_prompt : Optional[str] # If this verification prompt is set, this stage is considered as TextOnly stage (analyzing model answer).

    def __init__(self, reset_at_end : bool, stage_goal_description : str) -> None:
        self.reset_at_end = reset_at_end
        self.stage_goal_description = stage_goal_description
        if not hasattr(self, "additive_stage"): self.additive_stage = False

    def get_description(self) -> Dict:
        ei = isinstance(self.situation.instruction, EmptyInstruction)
        return {
            "goal_description" : self.stage_goal_description,
            "empty_instruction" : ei,
            "text_only" : hasattr(self,"verification_prompt") and self.verification_prompt != None,
            "instruction" : "" if ei else self.situation.instruction.get_content()
            }

    def validate(self, env_agents : List[str]):
        """Verify that all required attributes are correctly defined. Raise TypeError if not."""
        for attr in required_stage_attrs:
            if not hasattr(self, attr):
                raise TypeError(
                    f"Stage '{self.__class__.__name__}' must define attribute '{attr}' "
                    f"(either in __init__ or in class)."
                )
        overrides_f1 = self.__class__.verif_env_completion is not BaseTaskStage.verif_env_completion
        overrides_f2 = self.__class__.verif_log_completion is not BaseTaskStage.verif_log_completion
        has_mode_str = hasattr(self, "verification_prompt")

        if has_mode_str:
            if overrides_f1 or overrides_f2:
                raise TypeError(
                    f"{self.__class__.__name__} defines a verification_prompt, "
                    "so it must NOT override verif_env_completion or verif_log_completion."
                )
        else:
            if not (overrides_f1 or overrides_f2):
                raise TypeError(
                    f"{self.__class__.__name__} must override verif_env_completion or verif_log_completion, "
                    f"or define verification_prompt to mark this stage as TextOnly."
                )
            
        self.situation.verify_robot_name(env_agents)

    def build_init_state(self, env_state : Dict, env_ids : torch.Tensor) -> Dict:
        """
        This function allows to define a specific custom initialization state for all objects present in the scene.
        Ensure that each field: tensor keep the exact same size, name as the original one.
        """
        return env_state
 
    def verif_env_completion(self, obs : Dict) -> torch.Tensor:
        """
        Check if the stage is completed according to a sliced dict of observation from the env. [BATCHED]
        
        :param obs: The env observation
        :type obs: Dict
        :return: A tensor of int of nb_env length. 1 for success, -1 for catastrophic failure, 0 otherwise. 1 default
        :rtype: torch.Tensor
        """
        nb_env = obs["agent"]["qpos"].shape[0]
        return torch.tensor([1]*nb_env)
    
    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        """
        Check if the stage is completed according to logs from one instance only.
        
        :param log: The list of logs
        :type log: List
        :return: 1 for success, -1 for catastrophic failure, 0 otherwise. 1 default
        :rtype: int
        """
        return 1
    
    def combine_stage_completion(self, task_completion : int, log_completion : int) -> int:
        """
        Combine logic between the stage env completion value, stage log completion value and stage model answer completion.
        By default it returns -1 if one of them equal -1. 1 if all are equal to 1. 0 otherwise.

        Could be overridde to define a custom combine logic (In case of log only verif for exemple).
        """
        if task_completion == -1 or log_completion == -1:
            return -1
        if task_completion == 1 and log_completion == 1:
            return 1
        return 0


class BaseTask(ABC):
    """
    Base class to define custom tools API and a sequence of TaskStage. This class will be load by the tools_executor class.
    """
    
    env_id : str # The name of the maniskill environment to use.
    env_options : Dict # The env options to send to the _initialize_episode of the env.
    planner_options : Dict # Planner options
    name : str # The name of the task.
    styles : List[TaskStyle]
    approximal_difficulty : str # Store the difficulty estimated following the initialisation parameters.
    tools_constant : Dict # Allows to define some constant that are passed to the tools through the observation by adding a "add_constants" field (same level as "extra")
    agent_names : List[str] # Used in Multi-Agent Task. If you want to override real maniskill agents name.
    all_task_attributes : Dict[str, Any] # Store the full task attributes. Used by the randomizer to compute initial correspondance.

    # The path to the randomized config
    randomized_config_path : str = ""

    # References to class
    stages : List[BaseTaskStage]
    Tools_cls : Type[BaseToolsAPI]

    # Do not touch these parameters.
    _tools : BaseToolsAPI # direct instance of the tools
    _default_env_state : Dict # direct copy of the default initial env state (used for reset)
    _agents_name_equivalence : Dict

    def __init__(self) -> None:
        """
        Default empty __init__. No parameters needed.
        """
        super().__init__()
        self.env_options = {}
        self.planner_options = {
            "robot": 'panda_v2',
            "joint_vel_limit": 0.9,
            "joint_acc_limit": 0.9,
            "control_timesteps": 0
        }
        if not hasattr(self, "tools_constant"): self.tools_constant = {}
        if not hasattr(self, "agent_names"): self.agent_names = ['default']

    # getters

    def get_nb_total_stage(self):
        return self.NB_STAGES
    
    def get_init_situation(self, stage_id : int):
        """
        Return the situation of the selected stage id.
        """
        self._verif_stage(stage_id)
        return self.stages[stage_id].situation
    
    def get_tools(self) -> List[Dict]:
        """
        Return the availaible tools API.
        """
        return self._tools.get_api_description()
    
    def get_stage_attributes(self, stage_id : int):
        self._verif_stage(stage_id)
        return self.stages[stage_id].situation.attributes
    
    def get_stage_rule(self, stage_id : int) -> Optional[str]:
        self._verif_stage(stage_id)
        return self.stages[stage_id].verification_prompt

    # common function  
    #      
    def validate(self):
        """Verify that all required attributes are correctly defined and that each StageTask is valid. Raise TypeError if not."""
        for attr in required_task_attrs:
            if not hasattr(self, attr):
                raise TypeError(
                    f"Task '{self.__class__.__name__}' must define attribute '{attr}' "
                    f"(either in __init__ or in class)."
                )
        
        if not self.stages:
            raise TypeError("Stages are empty. You must define at least one stage.")
        
        self.NB_STAGES = len(self.stages)
        precedent_flag = False
        for i, stage in enumerate(self.stages):
            stage.validate(self.agent_names)
            # Check litle particularity

            if precedent_flag and isinstance(stage.situation.instruction, EmptyInstruction):
                raise TypeError(f"The stage {stage.__class__.__name__} ({i}) has an EmptyInstruction. Meaning that you are planning to override it by the last return status of the precedent stage."
                                f"However, stage {self.stages[i-1].__class__.__name__} ({i-1}) has set the flag_user_answer to True. Meaning that the system will ask the model to generate an additional steps."
                                f"This is not compatible. Define a custom user instruction at stage {i} or set the flag to false at stage {i-1}")

            precedent_flag = stage.situation.flag_answer_to_user


    def execute_tools(
            self,
            obs: Dict,
            env_id : int,
            function_name: str,
            params: Dict,
            stage_id : int,
            agent_id : int = 0,
        ) -> ToolExecution:
        if self.tools_constant:
            obs["add_constants"] = self.tools_constant
        obs["selected_robot_name"] = self.agent_names[agent_id]
        obs['task_attributes'] = self.get_stage_attributes(stage_id)
        exec = self._tools.execute_tools(obs, env_id, function_name, params)
        exec.robot_idx = agent_id
        exec._set_function_name(function_name)
        return exec
    
    def get_agent_names(self) -> List[str]:
        return self.agent_names

    def initialize_task(self, agents : Dict, agents_name : List[str], env_state : Dict, nb_env : int, build_first_stage : bool = True) -> Dict:
        """
        Initialize the task and tools by passing reference to agents
        """
        env_agents = [agents[a] for a in agents_name]
        if self.agent_names == ['default']:
            print("[TASK-BUILDER] Automatically assign real robot names to the Task")
            self.agent_names = agents_name

        if len(env_agents) != len(self.agent_names):
            raise TypeError(f"agent_names ({len(self.agent_names)}) must have the exact same length that the number of agents ({len(env_agents)}) in the env. Names and Agent will be associated by index")
                
        
        equivalence = {self.agent_names[i] : i for i in range(len(agents_name))}
        self._tools = self.Tools_cls(env_agents, equivalence)

        if build_first_stage:
            new_env_state = self.stages[0].build_init_state(env_state, torch.tensor(range(nb_env)))
        else:
            new_env_state = env_state
        self._default_env_state = extract_env_state_val(env_state, 0)

        self.validate()

        return new_env_state

    def verif_stage_env_completion(self, stage_id : int, obs : Dict, env_ids : List[int]) -> torch.Tensor:
        """
        Verify if some envs have completed their stage.
        It returns a torch tensor with -1 for catastrophic failure, 1 for success, 0 otherwise.

        :param stage_id: The id of the stage to verify
        :type stage_id: int
        :param obs: The observation dict from the environment
        :type obs: Dict
        :param env_ids: The list of ID of envs to verif
        :type env_ids: List[int]
        :return: torch tensor of size [len(env_ids)]
        """
        if self.is_stage_text_only(stage_id):
            raise RuntimeError("A text only stage has reached a verif env completion")
        
        if not env_ids:
            return torch.tensor([])
        
        _env_ids = torch.tensor(env_ids)
        _obs = slice_obs(obs,_env_ids)
        
        out = self.stages[stage_id].verif_env_completion(_obs)
        
        if not torch.is_tensor(out):
            out = torch.tensor([out])

        return out
    
    def verif_stage_log_completion(self, stage_id : int, full_log : List[Log], stage_log : List[Log]):
        """
        Check if the stage at id stage_id is completed according to logs from one instance only.
        
        :param stage_id: The stage_id
        :type stage_id: int
        :param log: The list of logs
        :type log: List
        :return: 1 for success, -1 for catastrophic failure, 0 otherwise. None if not implemented.
        :rtype: int
        """
        self._verif_stage(stage_id)
        if not self.stages[stage_id].additive_stage:
            for l in stage_log:
                if l.action != None:
                    # That's means that the model modify attributes in a stage which do not require any attributes modification.
                    return -1 
        return self.stages[stage_id].verif_log_completion(stage_log, full_log)
    
    def combine_stage_verif_scores(self, stage_id : int, env_score : int, log_score : int):
        """
        Check if the stage at id stage_id is completed according to score from env completion and log completion.
        
        :param stage_id: The stage_id
        :type stage_id: int
        :param env_score: env completion score
        :type env_score: int
        :param log_score: log completion score
        :type log_score: int
        :return: 1 for success, -1 for catastrophic failure, 0 otherwise. None if not implemented.
        :rtype: int
        """
        self._verif_stage(stage_id)
        return self.stages[stage_id].combine_stage_completion(env_score,log_score)

    def _verif_stage(self, stage_id : int):
        if stage_id < 0 or stage_id >= self.NB_STAGES:
            raise ValueError(f"stage_id ({stage_id}) is superior to the total length ({self.NB_STAGES}) of steps for this task")
    

    def get_stage_state(self, stage_id : int, agent_step : int) -> StageState:
        """
        Return the state of the stage based on the agent_step.
        
        :param stage_id: the id of the task stage
        :type stage_id: int
        :param agent_step: the number of step of the agent to solve the step
        :type agent_step: int
        :return: True if exceed, False otherwise
        :rtype: StageState
        """
        self._verif_stage(stage_id)
        
        if agent_step < self.stages[stage_id].target_steps:
            return StageState.OPTIMAL
        # If acceptance is set to 0, this means that the last stage is EXCEEDED to stop the dual agent
        
        if agent_step == self.stages[stage_id].target_steps:
            if self.stages[stage_id].acceptance_steps == 0:
                return StageState.EXCEEDED_OPTIMAL
            return StageState.OPTIMAL
        
        if agent_step < self.stages[stage_id].target_steps + self.stages[stage_id].acceptance_steps:
            return StageState.ACCEPTABLE 
        return StageState.EXCEEDED

    def is_stage_text_only(self, stage_id : int) -> bool:
        """
        Return True if the stage is considered as textonly
        """
        self._verif_stage(stage_id)
        return hasattr(self.stages[stage_id],"verification_prompt") and self.stages[stage_id].verification_prompt != None

    
    def call_stage_change_effects(self, stage_id : int, env_state : Dict, env_ids : List[int]):
        """
        Allows to apply the post effects of the stage_id to all env present in the env_state. 
        And applies the initial effect of the next stage.
        It will modify inplace.

        :param stage_id: the stage id
        :type stage_id: int
        :param env_state: The full env_state.
        :type env_state: Dict
        """

        self._verif_stage(stage_id)
        _env_ids = torch.tensor(env_ids)

        # End
        if self.stages[stage_id].reset_at_end:
            batch_set_value(state_env=env_state, env_ids=_env_ids, template=self._default_env_state)

        # Init
        self._verif_stage(stage_id+1)
        env_state = self.stages[stage_id+1].build_init_state(env_state, _env_ids)

    # CAN BE OVERRIDDE
    
    def compare_log(self, ref_log : List[Dict], current_log : List[Log]) -> Tuple[bool, str]:
        """
        Allows to compare two logs and verify that current_log is valid regarding ref_log.
        current_log is a list of Log while ref_log is a list of dict.
        These dict can have either 'function_name', 'action' or 'content' as key.

        This function it's only used witch Benchmark.
        """
        if len(current_log) == len(ref_log):
            for i in range(len(current_log)):
                fc = ref_log[i].get("function_name",None)
                c = ref_log[i].get('content',None)
                a = ref_log[i].get('action', None)
                r = f"At indice {i}, ref_logs = {ref_log[i]} but log = {current_log[i]}"
                if fc and fc != current_log[i].function: return False, r
                if a and a != current_log[i].action: return False, r
                if c:
                    if isinstance(current_log[i].content, Tuple) and isinstance(c, List):
                        if len(current_log[i].content) != len(c): return False, f"At {i}, log have different content length : ref {len(c)} != cur ({len(current_log[i].content)})"
                        for j in range(len(c)):
                            if current_log[i].content[j] != c[j]: return False,  f"At {i}-{j}, log have different content : ref {c[j]} != cur {current_log[i].content[j]}"
                        return True, ""
                    elif type(current_log[i].content) != type(c): 
                        return False, f"At {i}, log have different types : ref {type(c)} != cur ({type(current_log[i].content)})"
                    if current_log[i].content != c: return False, r
        else:
            return False, "logs and ref_logs do not have the same length"

        return True, ""
    
    def reset_stage(self) -> bool:
        raise RuntimeError("Calling a reset stage on a non-benchmark task class")
    
class BaseBenchmarkTask(BaseTask):

    stages = []
    styles = []
    approximal_difficulty = "easy"

    saved_task_attributes : Dict

    def validate(self):
        for attr in required_task_attrs:
            if not hasattr(self, attr):
                raise TypeError(
                    f"Task '{self.__class__.__name__}' must define attribute '{attr}' "
                    f"(either in __init__ or in class)."
                )
        if len(self.stages) > 0:
            raise TypeError("A Bench Task has some stages defined")

        for att_values in self.all_task_attributes.values():
            if att_values == self.agent_names:
                break
        else:
            if self.all_task_attributes.get("known_robots",None) is not None:
                raise TypeError(f"Fail to verify that attributes {self.all_task_attributes} contains correct robots name {self.agent_names}")
            
            self.all_task_attributes["known_robots"] = self.agent_names

        self.saved_task_attributes = copy.deepcopy(self.all_task_attributes)

    def get_stage_attributes(self, stage_id: int):
        return self.all_task_attributes
    
    def reset_stage(self) -> bool:
        self.all_task_attributes = copy.deepcopy(self.saved_task_attributes)
        return True