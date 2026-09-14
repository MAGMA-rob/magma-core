# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Dict, List, Any, Tuple, Type, Optional, Sequence
from abc import ABC
from copy import deepcopy
import torch
from dataclasses import dataclass, field

from magma_core.simulation.tools import BaseToolsAPI
from magma_core.simulation.stage import (
    BaseTaskStage,
    BaseStageComposite,
    TextOnlyValidationMode,
)
from magma_core.simulation.data_structures import (
    ActiveStageErrorState,
    StageState,
    StageSuccess,
    ToolExecution,
    EmptyInstruction,
    Log,
    Observation,
    ToolBatchContext,
    Situation,
    SituationInit,
    StageInput,
)
from magma_core.simulation.errors import BaseError
from magma_core.simulation.tasks_style import TaskStyle
from magma_core.utils.global_utils import slice_obs, batch_set_value, extract_env_state_val
from magma_core.simulation.serialization import decode_value, encode_value

required_task_attrs = [
    "maniskill_env_id",
    "stages",
    "Tools_cls",
    "situation_init",
    "initialization_parameters",
    "task_metadata"
]

@dataclass
class TaskMetadata():
    styles : List[TaskStyle] = field(default_factory=list)
    approximal_difficulty : str = "None" # Store the difficulty estimated following the initialisation parameters.
    coaching_hint : Optional[str] = None

@dataclass
class InitializationParameters():
    env_options : Dict = field(default_factory=dict) # The env options to send to the _initialize_episode of the env.
    planner_options : Dict = field(default_factory=lambda: {
        "robot": "panda_v2",
        "joint_vel_limit": 0.9,
        "joint_acc_limit": 0.9,
        "control_timesteps": 0,
    }) # Planner options
    agent_names : List[str] = field(default_factory=lambda: ["default"]) # Used in Multi-Agent Task. If you want to override real maniskill agents name.

    def to_spec(self) -> Dict[str, Any]:
        return encode_value({
            "env_options": deepcopy(self.env_options),
            "planner_options": deepcopy(self.planner_options),
            "agent_names": deepcopy(self.agent_names),
        })

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "InitializationParameters":
        values = decode_value(spec)
        if not isinstance(values, dict):
            raise TypeError("InitializationParameters spec must be a dictionary")
        expected = {"env_options", "planner_options", "agent_names"}
        if set(values) != expected:
            raise ValueError(
                "InitializationParameters spec keys mismatch: "
                f"expected={sorted(expected)}, got={sorted(values)}"
            )
        if not isinstance(values["env_options"], dict):
            raise TypeError("Initialization env_options must be a dictionary")
        if not isinstance(values["planner_options"], dict):
            raise TypeError("Initialization planner_options must be a dictionary")
        if not isinstance(values["agent_names"], list) or not all(
            isinstance(name, str) for name in values["agent_names"]
        ):
            raise TypeError("Initialization agent_names must be a list of strings")
        return cls(**values)


class BaseTask(ABC):
    """
    Base class to define custom tools API and a sequence of TaskStage. This class will be load by the tools_executor class.
    """
    
    maniskill_env_id : str # The name of the maniskill environment to use.
    name : str # The name of the task.
    task_metadata : TaskMetadata

    # The path to the randomized config
    randomized_config_path : str = ""
    # Allows to define some constant that are passed to the tools through the observation by adding a "add_constants" field (same level as "extra")
    tools_constant : Dict

    # References to class
    stages : List[BaseTaskStage]
    Tools_cls : Type[BaseToolsAPI]
    situation_init : SituationInit

    initialization_parameters : InitializationParameters    

    # Do not touch these parameters.
    _tools : BaseToolsAPI # direct instance of the tools
    _default_env_state : Dict # direct copy of the default initial env state (used for reset)
    _agents_name_equivalence : Dict

    def __init__(self) -> None:
        """
        Default empty __init__. No parameters needed.
        """
        super().__init__()
        if not hasattr(self, "tools_constant"): self.tools_constant = {}
        if not hasattr(self, "initialization_parameters"): self.initialization_parameters = InitializationParameters()
        if not hasattr(self, "task_metadata"): self.task_metadata = TaskMetadata()
        if not hasattr(self, "name"): self.name = self.__class__.__name__

    # getters

    def get_task_info(self) -> Dict:
        return {
            "styles" : self.task_metadata.styles,
            "approximal_difficulty" : self.task_metadata.approximal_difficulty,
            "env_id" : self.maniskill_env_id
        }

    def get_nb_total_stage(self) -> int:
        return self.NB_STAGES
    
    def get_complete_attributes(self) -> Dict:
        return self.situation_init.all_task_attributes
    
    def get_stage_input(self, stage_id : int) -> StageInput:
        self._verif_stage(stage_id)
        return self.stages[stage_id].get_stage_input()

    def get_init_attributes(self) -> Dict[str, Any]:
        return self.situation_init.attributes

    def get_init_situation(self) -> Situation:
        """
        Return the initial runtime situation for the first stage.
        """
        return self.situation_init.to_situation(self.get_stage_input(0))

    def get_initialization_spec(self) -> Dict[str, Any]:
        return {
            "parameters": self.initialization_parameters.to_spec(),
            "situation": self.situation_init.to_spec(),
            "tools_constant": encode_value(deepcopy(self.tools_constant)),
            "randomized_config_path": self.randomized_config_path,
        }

    def apply_initialization_spec(self, spec: Dict[str, Any]) -> None:
        if not isinstance(spec, dict):
            raise TypeError("Task initialization spec must be a dictionary")
        expected = {
            "parameters",
            "situation",
            "tools_constant",
            "randomized_config_path",
        }
        if set(spec) != expected:
            raise ValueError(
                "Task initialization spec keys mismatch: "
                f"expected={sorted(expected)}, got={sorted(spec)}"
            )
        randomized_config_path = spec["randomized_config_path"]
        if not isinstance(randomized_config_path, str):
            raise TypeError("randomized_config_path must be a string")
        self.initialization_parameters = InitializationParameters.from_spec(
            spec["parameters"]
        )
        self.situation_init = SituationInit.from_spec(spec["situation"])
        tools_constant = decode_value(spec["tools_constant"])
        if not isinstance(tools_constant, dict):
            raise TypeError("Task tools_constant must be a dictionary")
        self.tools_constant = tools_constant
        self.randomized_config_path = randomized_config_path
    
    def get_tools(self) -> List[Dict]:
        """
        Return the availaible tools API.
        """
        return self._tools.get_api_description()
    
    def get_stage_rule(self, stage_id : int) -> Optional[str]:
        self._verif_stage(stage_id)
        return self.stages[stage_id].get_verification_prompt()

    def get_stage_text_only_validation(
            self,
            stage_id: int,
        ) -> TextOnlyValidationMode:
        self._verif_stage(stage_id)
        return self.stages[stage_id].get_text_only_validation()
    
    def get_available_stage_errors(self, stage_id: int) -> List[BaseError]:
        """
        Return the pool of error instances that are allowed to be activated for
        the given runtime context.

        Regular MAGMA tasks resolve errors from the underlying stage
        declaration. Benchmark tasks can override this hook to expose a
        dedicated error registry without relying on scenario stages.
        """
        self._verif_stage(stage_id)
        return self.stages[stage_id].get_possible_errors()

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
        return self.initialization_parameters.agent_names

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
        self._verify_initial_robot_names()
        precedent_flag = False
        for i, stage in enumerate(self.stages):
            stage.validate(self.get_agent_names())
            # Check litle particularity

            stage_input = stage.get_stage_input()
            if precedent_flag and isinstance(stage_input.instruction, EmptyInstruction):
                raise TypeError(f"The stage {stage.__class__.__name__} ({i}) has an EmptyInstruction. Meaning that you are planning to override it by the last return status of the precedent stage."
                                f"However, stage {self.stages[i-1].__class__.__name__} ({i-1}) has set the flag_user_answer to True. Meaning that the system will ask the model to generate an additional steps."
                                f"This is not compatible. Define a custom user instruction at stage {i} or set the flag to false at stage {i-1}")

            precedent_flag = stage_input.flag_answer_to_user

    def _verify_initial_robot_names(self) -> None:
        attributes = self.get_init_attributes()
        agents_name = self.get_agent_names()

        if attributes.get("known_robots", None) is None:
            r = None
            for att_key, att_values in attributes.items():
                if att_values == agents_name:
                    r = att_key
                    break
            if r is not None:
                print(f"[WARNING] Changing key {r} to 'known_robots'")
                attributes.pop(r)
        else:
            if attributes["known_robots"] != agents_name:
                raise TypeError(
                    f"Attributes {attributes} has a key 'known_robots' that do not contains "
                    f"correct robots name {agents_name}"
                )

        attributes["known_robots"] = self.get_agent_names()


    def execute_tools(
            self,
            obs: Dict,
            env_id : int,
            attributes : Dict[str, Any],
            function_name: str,
            params: Dict,
            stage_id : int,
            error_state : ActiveStageErrorState,
            agent_id : int = 0,
            tool_batch_context: Optional[ToolBatchContext] = None,
        ) -> ToolExecution:

        magma_obs = self.build_tool_observation(
            obs,
            attributes,
            agent_id,
            tool_batch_context,
        )

        if self.is_stage_text_only(stage_id):
            stage = self.stages[stage_id]
            allowed_tools = stage.get_allowed_tools()
            if not stage.allows_tools_before_answer() or (len(allowed_tools) > 0 and function_name not in allowed_tools):
                exec = ToolExecution(
                    poses=[],
                    verifier=None,
                    reason=(
                        f"{function_name} is not allowed in this stage. "
                        f"Allowed tools are: {sorted(allowed_tools)}"
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

    def build_tool_observation(
            self,
            obs: Dict,
            attributes: Dict[str, Any],
            agent_id: int = 0,
            tool_batch_context: Optional[ToolBatchContext] = None,
        ) -> Observation:
        """
        Build the observation object exposed to tools and runtime errors.
        """
        return Observation(
            selected_robot_name=self.get_agent_names()[agent_id],
            task_attributes=attributes,
            maniskill_obs=obs,
            add_constants=self.tools_constant,
            tool_batch_context=(
                ToolBatchContext()
                if tool_batch_context is None
                else tool_batch_context
            ),
        )

    def initialize_stage_errors(
            self,
            stage_id : int,
            obs : Dict,
            attributes: Dict[str, Any],
            env_id : int,
            agent_id : int = 0
        ) -> ActiveStageErrorState:
        """
        Ask the stage to choose the active error(s) for this trajectory.
        """
        self._verif_stage(stage_id)

        stage = self.stages[stage_id]
        possible_errors = stage.get_possible_errors()
        if len(possible_errors) == 0:
            return {}

        # We initialize stage errors from the public observation so the runtime
        # binding uses the same object naming convention as the tools.
        error_state = stage.initialize_errors(
            self.build_tool_observation(obs, attributes, agent_id),
            env_id,
        )
        if error_state is None:
            raise RuntimeError(
                f"{stage.__class__.__name__}.initialize_errors must return a dict, got None."
            )

        allowed_names = {error.get_name() for error in possible_errors}
        unknown_names = [name for name in error_state.keys() if name not in allowed_names]
        if unknown_names:
            raise RuntimeError(
                f"{stage.__class__.__name__}.initialize_errors returned unknown errors {unknown_names}. "
                f"Allowed errors are {sorted(allowed_names)}."
            )

        return error_state


    def initialize_task(
            self,
            agents: Dict,
            agents_name: List[str],
            env_state: Dict,
            nb_env: int,
            build_first_stage: bool = True,
            env_ids: Optional[Sequence[int]] = None,
        ) -> Dict:
        """
        Initialize the task and tools by passing references to the agents.

        ``env_ids`` limits stage initialization to selected slots of a vectorized
        environment. Omitting it preserves the historical behavior and
        initializes every slot.
        """
        selected_env_ids = list(range(nb_env)) if env_ids is None else list(env_ids)
        if not selected_env_ids:
            raise ValueError("env_ids must contain at least one environment index")
        if len(set(selected_env_ids)) != len(selected_env_ids):
            raise ValueError(f"env_ids contains duplicated indices: {selected_env_ids}")
        invalid_env_ids = [
            env_id
            for env_id in selected_env_ids
            if not isinstance(env_id, int) or isinstance(env_id, bool)
            or env_id < 0 or env_id >= nb_env
        ]
        if invalid_env_ids:
            raise ValueError(
                f"env_ids contains indices outside [0, {nb_env}): {invalid_env_ids}"
            )

        env_agents = [agents[a] for a in agents_name]
        if self.get_agent_names() == ['default']:
            print("[TASK-BUILDER] Automatically assign real robot names to the Task")
            resolved_agent_names = list(agents_name)
            self.initialization_parameters.agent_names = resolved_agent_names
            self.situation_init.attributes["known_robots"] = list(
                resolved_agent_names
            )
            self.situation_init.all_task_attributes["known_robots"] = list(
                resolved_agent_names
            )

        if len(env_agents) != len(self.get_agent_names()):
            raise TypeError(f"agent_names ({len(self.get_agent_names())}) must have the exact same length that the number of agents ({len(env_agents)}) in the env. Names and Agent will be associated by index")
                
        
        equivalence = {self.get_agent_names()[i] : i for i in range(len(agents_name))}
        self._tools = self.Tools_cls(env_agents, equivalence)
        self._default_env_state = extract_env_state_val(
            env_state,
            selected_env_ids[0],
        )

        if build_first_stage:
            new_env_state = self.stages[0].build_init_state(
                env_state,
                torch.tensor(selected_env_ids),
            )
        else:
            new_env_state = env_state

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
        if not self.stages[stage_id].is_additive_stage():
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
    
    def get_stage_state(
            self,
            stage_id: int,
            tool_calls: int,
            forgiven_tool_calls: int,
            stage_success: StageSuccess,
            optional_log: Optional[List[Log]] = None,
        ) -> StageState:
        """
        Return the stage efficiency state based on effective tool calls.
        """
        self._verif_stage(stage_id)

        return self.stages[stage_id].get_stage_state(
            tool_calls - forgiven_tool_calls,
            stage_success,
            [] if optional_log is None else optional_log,
        )

    def is_stage_text_only(self, stage_id : int) -> bool:
        """
        Return True if the stage is considered as textonly
        """
        self._verif_stage(stage_id)
        return self.stages[stage_id].is_text_only()

    
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
        if self.stages[stage_id].should_reset_at_end():
            batch_set_value(state_env=env_state, env_ids=_env_ids, template=self._default_env_state)

        # Init
        self._verif_stage(stage_id+1)
        env_state = self.stages[stage_id+1].build_init_state(env_state, _env_ids)


    ##### CAN BE OVERRIDDE
    
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
    
