from abc import ABC
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional
import random
import torch
from dataclasses import dataclass, field

from magma_core.simulation.data_structures import (
    StageInput,
    EmptyInstruction,
    Log,
    StageState,
    StageSuccess,
    ActiveStageErrorState,
    Observation,
)
from magma_core.simulation.goals import BaseGoal
from magma_core.simulation.errors import BaseError
from magma_core.simulation.serialization import build_spec, construct_from_spec
from magma_core.simulation.stage.environment_transition import BaseStageEnvironmentTransition

class TextOnlyValidationMode(Enum):
    JUDGE = "judge"
    SAY_ONLY = "say_only"


@dataclass
class StageGlobalParameters():
    reset_at_end : bool = False # Indicates if the stage must reset the env after completion
    additive_stage : bool = False # Mark if this stage can modify the attributes (Default to False)
    verification_prompt : Optional[str] = None # If this verification prompt is set, this stage is considered as TextOnly stage (analyzing model answer).
    allow_tools_before_answer : bool = False # Mark if the text-only stage can allows the execution of some tools
    allowed_tools : List[str] = field(default_factory=list) # The list of allowed tool, if empty, all tools are allowed. Otherwise, only tool in the list are allowed.
    text_only_validation: TextOnlyValidationMode = TextOnlyValidationMode.JUDGE

@dataclass
class StageErrorParameters():
    possible_errors : List[BaseError] = field(default_factory=list)
    # Bounds for the number of stage errors sampled per trajectory. The sampled
    # count is clamped to the number of possible errors declared on the stage.
    min_active_errors : int = 1
    max_active_errors : int = 1

required_stage_attrs = [
    "stage_input",
    "target_tool_calls",
    "max_tool_calls",
    "goals",
    "stage_goal_description"
]


def _find_observation_reference(value) -> Optional[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return value
    if not isinstance(value, dict):
        return None

    pose = value.get("pose")
    if isinstance(pose, torch.Tensor):
        return pose

    for nested_value in value.values():
        reference = _find_observation_reference(nested_value)
        if reference is not None:
            return reference
    return None

class BaseTaskStage(ABC):
    """
    Base class to define a task stage which allows to define the situation, the number of steps and the completion goal for the step.

    You can also define if you want to reset the env at the end of this stage and you must give a short contextual description of the stage. Used for coaching.
    """

    # The list of Goals and Constraints related to the stage
    goals : List[BaseGoal]

    target_tool_calls: Optional[int]
    max_tool_calls: Optional[int]

    # Lightweight user/model input declared by this stage.
    stage_input : StageInput
    stage_goal_description : str
    request_type: Optional[str]
    
    # Parameters
    error_parameters : StageErrorParameters
    global_parameters : StageGlobalParameters

    def __init__(
            self,
            goals : List[BaseGoal],
            stage_goal_description : str,
            stage_input : StageInput,
            global_parameters : Optional[StageGlobalParameters] = None,
            error_parameters : Optional[StageErrorParameters] = None,
            entry_transition: Optional[BaseStageEnvironmentTransition] = None,
        ) -> None:
        self.goals = goals
        self.stage_input = stage_input
        self.stage_goal_description = stage_goal_description
        self.entry_transition = entry_transition
        self.request_type = None
        # Artifact-backed stages can provide already-materialized error state.
        # ``None`` keeps the regular scenario behavior and samples at runtime.
        self.active_error_arguments_override: Optional[ActiveStageErrorState] = None

        target_tool_calls = self.target_tool_calls
        max_tool_calls = self.max_tool_calls

        if target_tool_calls is None and max_tool_calls is None:
            raise ValueError(
                "target_tool_calls and max_tool_calls cannot both be None"
            )
        if target_tool_calls is not None and target_tool_calls < 0:
            raise ValueError("target_tool_calls must be greater than or equal to 0")
        if max_tool_calls is None:
            if target_tool_calls is None:
                raise RuntimeError("Missing target_tool_calls for default max computation")
            max_tool_calls = target_tool_calls * 2
            self.max_tool_calls = max_tool_calls
        if max_tool_calls < 0:
            raise ValueError("max_tool_calls must be greater than or equal to 0")
        if (
            target_tool_calls is not None
            and target_tool_calls > max_tool_calls
        ):
            raise ValueError("target_tool_calls cannot exceed max_tool_calls")

        if global_parameters:
            self.global_parameters = global_parameters
        elif not hasattr(self, "global_parameters"):
            self.global_parameters = StageGlobalParameters()

        if error_parameters:
            self.error_parameters = error_parameters
        elif not hasattr(self, "error_parameters"):
            self.error_parameters = StageErrorParameters()

    def get_stage_input(self) -> StageInput:
        return self.stage_input

    def set_request_type(self, request_type: str) -> None:
        if not isinstance(request_type, str) or not request_type.strip():
            raise ValueError("request_type must be a non-empty string")
        normalized_request_type = request_type.strip()
        if (
            self.request_type is not None
            and self.request_type != normalized_request_type
        ):
            raise RuntimeError(
                f"Stage {type(self).__name__} is already assigned to request "
                f"{self.request_type!r}"
            )
        self.request_type = normalized_request_type

    def get_request_type(self) -> str:
        return self.request_type or "unknown"

    def is_linked_to_prev(self) -> bool:
        return self.stage_input.linked_to_prev

    def get_stage_goal_description(self) -> str:
        return self.stage_goal_description

    def get_target_tool_calls(self) -> Optional[int]:
        return self.target_tool_calls

    def get_max_tool_calls(self) -> Optional[int]:
        return self.max_tool_calls

    def get_goals(self) -> List[BaseGoal]:
        return self.goals

    def should_reset_at_end(self) -> bool:
        return self.global_parameters.reset_at_end

    def is_additive_stage(self) -> bool:
        return self.global_parameters.additive_stage

    def get_verification_prompt(self) -> Optional[str]:
        return self.global_parameters.verification_prompt

    def get_text_only_validation(self) -> TextOnlyValidationMode:
        return self.global_parameters.text_only_validation

    def is_text_only(self) -> bool:
        return self.get_verification_prompt() is not None

    def allows_tools_before_answer(self) -> bool:
        return self.global_parameters.allow_tools_before_answer

    def get_allowed_tools(self) -> List[str]:
        return self.global_parameters.allowed_tools

    def get_possible_errors(self) -> List[BaseError]:
        return self.error_parameters.possible_errors

    def to_spec(self) -> Dict[str, Any]:
        """Serialize a stage using arguments declared by the concrete class."""

        if "_to_spec_arguments" not in type(self).__dict__:
            raise NotImplementedError(
                f"{type(self).__name__} must implement _to_spec_arguments() "
                "before it can be serialized"
            )
        arguments = self._to_spec_arguments()
        if not isinstance(arguments, dict):
            raise TypeError("_to_spec_arguments() must return a dict")
        return build_spec(self, arguments)

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "BaseTaskStage":
        """Load a stage by validating and calling its concrete constructor."""

        stage = construct_from_spec(spec, cls)
        if not isinstance(stage, cls):
            raise TypeError(f"Stage spec is not compatible with {cls.__name__}")
        return stage

    def _to_spec_arguments(self) -> Dict[str, Any]:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _to_spec_arguments()"
        )

    def get_min_active_errors(self) -> int:
        return self.error_parameters.min_active_errors

    def get_max_active_errors(self) -> int:
        return self.error_parameters.max_active_errors

    def get_description(self) -> Dict:
        stage_input = self.get_stage_input()
        ei = isinstance(stage_input.instruction, EmptyInstruction)
        return {
            "goal_description" : self.get_stage_goal_description(),
            "target_tool_calls": self.target_tool_calls,
            "max_tool_calls": self.max_tool_calls,
            "request_type": self.get_request_type(),
            "empty_instruction" : ei,
            "linked_to_prev": self.is_linked_to_prev(),
            "text_only" : self.is_text_only(),
            "text_only_validation": self.get_text_only_validation().value,
            "allow_tools_before_answer": self.allows_tools_before_answer(),
            "instruction" : "" if ei else stage_input.instruction.get_content(),
            "has_constraints": stage_input.instruction.has_constraint,
            "composite": []
            }

    def validate(self, env_agents : List[str]):
        """Verify that all required attributes are correctly defined. Raise TypeError if not."""
        for attr in required_stage_attrs:
            if not hasattr(self, attr):
                raise TypeError(
                    f"Stage '{self.__class__.__name__}' must define attribute '{attr}' "
                    f"(either in __init__ or in class)."
                )
        self.get_stage_input()

        overrides_f2 = self.__class__.verif_log_completion is not BaseTaskStage.verif_log_completion
        has_verification_prompt = self.is_text_only()

        if not isinstance(
            self.get_text_only_validation(),
            TextOnlyValidationMode,
        ):
            raise TypeError(
                f"{self.__class__.__name__}.text_only_validation must be a "
                "TextOnlyValidationMode"
            )
        if (
            self.get_text_only_validation() is not TextOnlyValidationMode.JUDGE
            and not has_verification_prompt
        ):
            raise TypeError(
                f"{self.__class__.__name__} defines a non-judge text-only "
                "validation mode without a verification_prompt."
            )

        if self.allows_tools_before_answer() and not has_verification_prompt:
            raise TypeError(
                f"{self.__class__.__name__} enables allow_tools_before_answer, "
                "so it must define a verification_prompt."
            )
        if len(self.get_allowed_tools()) > 0 and not self.allows_tools_before_answer():
            raise TypeError(
                f"{self.__class__.__name__} defines allowed_tools, "
                "so it must enable allow_tools_before_answer."
            )

        if has_verification_prompt:
            if len(self.get_goals()) > 0:
                raise TypeError(
                    f"{self.__class__.__name__} defines a verification_prompt, "
                    "so it must NOT defines any goals."
                )
            if overrides_f2:
                raise TypeError(
                    f"{self.__class__.__name__} defines a verification_prompt, "
                    "so it must NOT override verif_log_completion."
                )
        else:
            if not overrides_f2 and len(self.get_goals()) == 0:
                raise TypeError(
                    f"{self.__class__.__name__} must override verif_log_completion OR/AND define a list of goals, "
                    f"or define verification_prompt to mark this stage as TextOnly."
                )

        # Error states are keyed by error name, so duplicate names inside one
        # stage would make the persisted runtime state ambiguous.
        error_names = [error.get_name() for error in self.get_possible_errors()]
        duplicates = {name for name in error_names if error_names.count(name) > 1}
        if duplicates:
            raise TypeError(
                f"{self.__class__.__name__} defines duplicated error names: {sorted(duplicates)}. "
                "Each stage error must expose a unique get_name()."
            )

        self._validate_error_sampling_policy()

    def _validate_error_sampling_policy(self) -> None:
        for attr in ("min_active_errors", "max_active_errors"):
            value = self.get_min_active_errors() if attr == "min_active_errors" else self.get_max_active_errors()
            if type(value) is not int:
                raise TypeError(
                    f"{self.__class__.__name__}.{attr} must be an int, "
                    f"got {type(value).__name__}."
                )
            if value < 0:
                raise TypeError(
                    f"{self.__class__.__name__}.{attr} must be >= 0, got {value}."
                )

        if self.get_min_active_errors() > self.get_max_active_errors():
            raise TypeError(
                f"{self.__class__.__name__}.min_active_errors "
                f"({self.get_min_active_errors()}) must be <= max_active_errors "
                f"({self.get_max_active_errors()})."
            )

    def _sample_active_error_count(self) -> int:
        nb_possible_errors = len(self.get_possible_errors())
        if nb_possible_errors == 0:
            return 0

        self._validate_error_sampling_policy()

        min_count = min(self.get_min_active_errors(), nb_possible_errors)
        max_count = min(self.get_max_active_errors(), nb_possible_errors)
        if max_count == 0:
            return 0

        return random.randint(min_count, max_count)

    def build_init_state(self, env_state : Dict, env_ids : torch.Tensor) -> Dict:
        """
        This function allows to define a specific custom initialization state for all objects present in the scene.
        Ensure that each field: tensor keep the exact same size, name as the original one.
        """
        if self.entry_transition is None:
            return env_state
        return self.entry_transition.apply(env_state, env_ids)
 
    def _verif_env_completion(self, obs : Dict) -> torch.Tensor:
        """
        Check if the stage is completed according to a sliced dict of observation from the env. [BATCHED]
        Automatically call all goals associated with the Stage.
        
        :param obs: The env observation
        :type obs: Dict
        :return: A tensor of int of nb_env length. 1 for success, -1 for catastrophic failure, 0 otherwise. 1 default
        :rtype: torch.Tensor
        """
        goals = self.get_goals()
        if goals:
            out = goals[0].verify(obs)
            for goal in goals[1:]:
                out = torch.minimum(out, goal.verify(obs))
            return out

        reference = _find_observation_reference(obs)
        if reference is None:
            raise ValueError(
                "Cannot verify a stage without goals: the observation does "
                "not contain a reference tensor."
            )
        nb_envs = 1 if reference.ndim <= 1 else reference.shape[0]
        return torch.ones(
            nb_envs,
            dtype=torch.int32,
            device=reference.device,
        )
    
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
    
    def is_composite_and_reset(self) -> bool:
        """
        This function is used only inside StageComposite
        """
        return False
    
    def get_stage_state(
            self,
            effective_tool_calls: int,
            stage_success: StageSuccess,
            log: List[Log],
        ) -> StageState:
        """
        Classify stage efficiency from executed tool calls and completion.
        """
        target_tool_calls = self.get_target_tool_calls()

        if stage_success == StageSuccess.FINISH:
            if target_tool_calls is None:
                return StageState.UNASSESSED
            if effective_tool_calls <= target_tool_calls:
                return StageState.OPTIMAL
            return StageState.ACCEPTABLE

        max_tool_calls = self.get_max_tool_calls()
        if max_tool_calls is not None and effective_tool_calls >= max_tool_calls:
            return StageState.EXCEEDED
        if target_tool_calls is None:
            return StageState.UNASSESSED
        if effective_tool_calls <= target_tool_calls:
            return StageState.OPTIMAL
        return StageState.ACCEPTABLE
    
    def initialize_errors(self, obs : Observation, env_id : int) -> ActiveStageErrorState:
        """
        Allows to initialize the different errors according to the current observation.
        """
        if self.active_error_arguments_override is not None:
            expected_error_names = {
                error.get_name() for error in self.get_possible_errors()
            }
            provided_error_names = set(self.active_error_arguments_override)
            if provided_error_names != expected_error_names:
                raise RuntimeError(
                    "Materialized error arguments do not match the stage errors: "
                    f"expected {sorted(expected_error_names)}, got "
                    f"{sorted(provided_error_names)}."
                )
            return deepcopy(self.active_error_arguments_override)

        nb_errors = self._sample_active_error_count()
        if nb_errors == 0:
            return {}

        if nb_errors == 1:
            chosen_errors = [random.choice(self.get_possible_errors())]
        else:
            chosen_errors = random.sample(self.get_possible_errors(), k=nb_errors)

        return {
            error.get_name(): error.initialize(obs, env_id)
            for error in chosen_errors
        }
