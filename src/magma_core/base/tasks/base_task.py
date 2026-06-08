# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Dict, List, Any, Tuple, Type, Optional
from abc import ABC
import torch, copy

from ..tools import BaseToolsAPI
from ..stage import BaseTaskStage, BaseStageComposite
from ..data_structures import (
    ActiveStageErrorState,
    StageState,
    ToolExecution,
    EmptyInstruction,
    Log,
    Observation,
)
from ..errors import BaseError
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
        self.all_task_attributes = copy.deepcopy(
            getattr(self.__class__, "all_task_attributes", {})
        )
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
            error_state : ActiveStageErrorState,
            agent_id : int = 0
        ) -> ToolExecution:

        magma_obs = self.build_tool_observation(obs, stage_id, agent_id)

        if self.is_stage_text_only(stage_id):
            stage = self.stages[stage_id]
            if not stage.allow_tools_before_answer or (len(stage.allowed_tools) > 0 and function_name not in stage.allowed_tools):
                exec = ToolExecution(
                    poses=[],
                    verifier=None,
                    reason=(
                        f"{function_name} is not allowed in this stage. "
                        f"Allowed tools are: {sorted(stage.allowed_tools)}"
                    ),
                )
                exec.must_fail_stage = True
                exec.robot_idx = agent_id
                exec._set_execution_info(
                    function_name,
                    stage_id,
                    self.get_active_stage_error(stage_id, error_state),
                )
                return exec

        exec = self._tools.execute_tools(magma_obs, env_id, function_name, params)
        exec.robot_idx = agent_id
        exec._set_execution_info(
            function_name,
            stage_id,
            self.get_active_stage_error(stage_id, error_state),
        )
        return exec

    def build_tool_observation(self, obs: Dict, stage_id: int, agent_id: int = 0) -> Observation:
        """
        Build the observation object exposed to tools and runtime errors.
        """
        magma_obs = Observation(
            selected_robot_name=self.agent_names[agent_id],
            task_attributes=self.get_stage_attributes(stage_id),
            maniskill_obs=obs,
        )

        if self.tools_constant:
            magma_obs.add_constants = self.tools_constant

        return magma_obs

    def initialize_stage_errors(
            self,
            stage_id : int,
            obs : Dict,
            env_id : int,
            agent_id : int = 0,
        ) -> ActiveStageErrorState:
        """
        Ask the stage to choose the active error(s) for this trajectory.
        """
        self._verif_stage(stage_id)

        stage = self.stages[stage_id]
        if len(stage.possible_errors) == 0:
            return {}

        # We initialize stage errors from the public observation so the runtime
        # binding uses the same object naming convention as the tools.
        error_state = stage.initialize_errors(
            self.build_tool_observation(obs, stage_id, agent_id),
            env_id,
        )
        if error_state is None:
            raise RuntimeError(
                f"{stage.__class__.__name__}.initialize_errors must return a dict, got None."
            )

        allowed_names = {error.get_name() for error in stage.possible_errors}
        unknown_names = [name for name in error_state.keys() if name not in allowed_names]
        if unknown_names:
            raise RuntimeError(
                f"{stage.__class__.__name__}.initialize_errors returned unknown errors {unknown_names}. "
                f"Allowed errors are {sorted(allowed_names)}."
            )

        return error_state

    def get_available_stage_errors(self, stage_id: int) -> List[BaseError]:
        """
        Return the pool of error instances that are allowed to be activated for
        the given runtime context.

        Regular MAGMA tasks resolve errors from the underlying stage
        declaration. Benchmark tasks can override this hook to expose a
        dedicated error registry without relying on scenario stages.
        """
        self._verif_stage(stage_id)
        return self.stages[stage_id].possible_errors

    def get_active_stage_error(
            self,
            stage_id : int,
            error_state : ActiveStageErrorState
        ) -> List[BaseError]:
        """
        Resolve the currently active runtime errors from their persisted state.
        """
        if len(error_state) == 0:
            return []

        available_errors = self.get_available_stage_errors(stage_id)
        resolved_errors: List[BaseError] = []
        unknown_names: List[str] = []

        for error_name in error_state.keys():
            for error in available_errors:
                if error.get_name() == error_name:
                    resolved_errors.append(error)
                    break
            else:
                unknown_names.append(error_name)

        if unknown_names:
            raise RuntimeError(
                f"Unable to resolve active error(s) {unknown_names} for stage {stage_id}."
            )

        return resolved_errors
    
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
    
    def should_reset_same_stage(self, stage_id : int) ->bool:
        """
        Return True if the env should be resetted within the same stage.

        It's false by default. Only StageComposite use this function.
        """
        self._verif_stage(stage_id)
        return self.stages[stage_id].is_composite_and_reset()

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
        
        out = self.stages[stage_id]._verif_env_completion(_obs)
        
        if not torch.is_tensor(out):
            out = torch.tensor([out])

        return out
    
    def verif_stage_log_completion(
            self,
            stage_id : int,
            full_log : List[Log],
            stage_log : List[Log],
            composite_progress : Optional[Dict[str, Any]] = None,
        ):
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
        if isinstance(self.stages[stage_id], BaseStageComposite):
            return self.stages[stage_id].verif_log_completion(stage_log, full_log, composite_progress) #type: ignore --> pass to a stage composite
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
    
    def get_stage_recovery_extra_steps(
            self,
            stage_id: int,
            error_state: Optional[ActiveStageErrorState] = None,
        ) -> int:
        """
        Resolve the currently active error for this trajectory and return how
        many extra steps should be forgiven when computing the stage state.
        """
        if not error_state:
            return 0

        active_errors = self.get_active_stage_error(stage_id, error_state)
        if len(active_errors) == 0:
            return 0
        return sum(
            error.get_recovery_extra_steps(error_state.get(error.get_name()))
            for error in active_errors
        )

    def get_stage_state(
            self,
            stage_id : int,
            agent_step : int,
            optional_log : List[Log] = [],
            error_state: Optional[ActiveStageErrorState] = None,
        ) -> StageState:
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

        return self.stages[stage_id].get_stage_state(
            agent_step,
            optional_log,
            recovery_extra_steps=self.get_stage_recovery_extra_steps(stage_id, error_state),
        )

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

    # must be specified
    benchmark_possible_errors: List[BaseError]

    saved_task_attributes : Dict
    saved_tools_constant : Dict
    saved_env_options : Dict

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

        # Benchmark tasks have no internal MAGMA stages, but the rest of the
        # runtime still passes stage_id=0 for tool execution helpers.
        self.NB_STAGES = 1

        if not hasattr(self, "benchmark_possible_errors"):
            self.benchmark_possible_errors = []

        normalized_errors: List[BaseError] = []
        for error in self.benchmark_possible_errors:
            if isinstance(error, BaseError):
                normalized_errors.append(error)
            elif isinstance(error, type) and issubclass(error, BaseError):
                normalized_errors.append(error())
            else:
                raise TypeError(
                    "benchmark_possible_errors must contain BaseError instances "
                    f"or BaseError classes. Got {error!r}."
                )
        # Ensuring different names per error
        error_names = [error.get_name() for error in normalized_errors]
        duplicate_names = sorted({name for name in error_names if error_names.count(name) > 1})
        if duplicate_names:
            raise TypeError(
                "benchmark_possible_errors contains duplicate error names: "
                f"{duplicate_names}. Please keep unique error identifiers."
            )
        self.benchmark_possible_errors = normalized_errors

        self.saved_task_attributes = copy.deepcopy(self.all_task_attributes)
        self.saved_tools_constant = copy.deepcopy(self.tools_constant)
        self.saved_env_options = copy.deepcopy(self.env_options)

    def get_stage_attributes(self, stage_id: int):
        return self.all_task_attributes

    def get_available_stage_errors(self, stage_id: int) -> List[BaseError]:
        # Benchmark runtime errors are declared directly on the benchmark task
        # rather than on internal MAGMA stages.
        return self.benchmark_possible_errors
    
    def reset_stage(self) -> bool:
        self.all_task_attributes = copy.deepcopy(self.saved_task_attributes)
        self.tools_constant = copy.deepcopy(self.saved_tools_constant)
        self.env_options = copy.deepcopy(self.saved_env_options)
        return True

    def initialize_benchmark_variant(self, attributes : Dict, tools_constant : Dict, env_options : Dict):
        self.saved_task_attributes = copy.deepcopy(attributes)
        self.saved_tools_constant = copy.deepcopy(tools_constant)
        self.saved_env_options = copy.deepcopy(env_options)
        self.reset_stage()

    def rewrite_benchmark_task_payload(self, task_data: Dict) -> Dict:
        """
        Rewrite one benchmark task payload according to the currently active
        benchmark variant.

        Benchmark scenarios can override this hook when their JSON tasks are
        canonically authored with placeholder or scenario-specific names that
        must be resolved after `initialize_benchmark_variant(...)`.
        """
        return copy.deepcopy(task_data)

    def execute_tools(
            self,
            obs: Dict,
            env_id : int,
            function_name: str,
            params: Dict,
            stage_id : int,
            error_state : ActiveStageErrorState,
            agent_id : int = 0
        ) -> ToolExecution:
        """
        In this version we do not check any 
        """

        magma_obs = self.build_tool_observation(obs, stage_id, agent_id)
        exec = self._tools.execute_tools(magma_obs, env_id, function_name, params)
        exec.robot_idx = agent_id
        exec._set_execution_info(
            function_name,
            stage_id,
            self.get_active_stage_error(stage_id, error_state),
        )
        return exec
