from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import random
import torch

from ..data_structures import Situation, EmptyInstruction, Log, StageState, ActiveStageErrorState, Observation
from ..goals import BaseGoal
from ..errors import BaseError

required_stage_attrs = [
    "target_steps",
    "goals",
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

    # The list of Goals and Constraints related to the stage
    goals : List[BaseGoal]
    possible_errors : List[BaseError]
    # Bounds for the number of stage errors sampled per trajectory. The sampled
    # count is clamped to the number of possible errors declared on the stage.
    min_active_errors : int = 1
    max_active_errors : int = 1

    target_steps : int # The number of target steps required to complete the task.
    acceptance_steps : int # The number of steps (+ target_steps) to let the model try to complete the task.

    # The initial situation of the step (memory, attributes, instructions...)
    # Could be overriden by precedent step final step if the model complete it.
    situation : Situation
    stage_goal_description : str

    additive_stage : bool # Mark if this stage can modify the attributes (Default to False)
    verification_prompt : Optional[str] # If this verification prompt is set, this stage is considered as TextOnly stage (analyzing model answer).
    linked_to_prev : bool # Indicates whether this stage belongs to the same logical interaction block as the previous stage.
    
    allow_tools_before_answer : bool # Mark if the text-only stage can allows the execution of some tools
    allowed_tools : List[str] # The list of allowed tool, if empty, all tools are allowed. Otherwise, only tool in the list are allowed.

    def __init__(
            self,
            goals : List[BaseGoal],
            reset_at_end : bool,
            stage_goal_description : str,
            possible_errors : Optional[List[BaseError]] = None,
            linked_to_prev: bool = False,
            min_active_errors: Optional[int] = None,
            max_active_errors: Optional[int] = None,
        ) -> None:
        self.goals = goals
        if not hasattr(self, "possible_errors"):
            self.possible_errors = [] if possible_errors is None else possible_errors
        if min_active_errors is not None:
            self.min_active_errors = min_active_errors
        if max_active_errors is not None:
            self.max_active_errors = max_active_errors
        self.reset_at_end = reset_at_end
        self.stage_goal_description = stage_goal_description
        if not hasattr(self, "additive_stage"): self.additive_stage = False
        if not hasattr(self, "allow_tools_before_answer"): self.allow_tools_before_answer = False
        if not hasattr(self, "allowed_tools"): self.allowed_tools = []
        self.linked_to_prev = linked_to_prev

    def _sync_linked_to_prev_with_instruction(self) -> None:
        if not isinstance(self.linked_to_prev, bool):
            raise TypeError(
                f"{self.__class__.__name__}.linked_to_prev must be a bool, got {type(self.linked_to_prev).__name__}"
            )
        if isinstance(self.situation.instruction, EmptyInstruction):
            self.linked_to_prev = True

    def get_description(self) -> Dict:
        self._sync_linked_to_prev_with_instruction()
        ei = isinstance(self.situation.instruction, EmptyInstruction)
        has_verification_prompt = hasattr(self, "verification_prompt") and self.verification_prompt is not None
        return {
            "goal_description" : self.stage_goal_description,
            "empty_instruction" : ei,
            "linked_to_prev": self.linked_to_prev,
            "text_only" : has_verification_prompt,
            "allow_tools_before_answer": self.allow_tools_before_answer,
            "instruction" : "" if ei else self.situation.instruction.get_content(),
            "has_constraints": self.situation.instruction.has_constraint,
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

        self._sync_linked_to_prev_with_instruction()

        overrides_f2 = self.__class__.verif_log_completion is not BaseTaskStage.verif_log_completion
        has_verification_prompt = hasattr(self, "verification_prompt") and self.verification_prompt is not None

        if self.allow_tools_before_answer and not has_verification_prompt:
            raise TypeError(
                f"{self.__class__.__name__} enables allow_tools_before_answer, "
                "so it must define a verification_prompt."
            )
        if len(self.allowed_tools) > 0 and not self.allow_tools_before_answer:
            raise TypeError(
                f"{self.__class__.__name__} defines allowed_tools, "
                "so it must enable allow_tools_before_answer."
            )

        if has_verification_prompt:
            if len(self.goals) > 0:
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
            if not overrides_f2 and len(self.goals) == 0:
                raise TypeError(
                    f"{self.__class__.__name__} must override verif_log_completion OR/AND define a list of goals, "
                    f"or define verification_prompt to mark this stage as TextOnly."
                )

        self.situation.verify_robot_name(env_agents)

        # Error states are keyed by error name, so duplicate names inside one
        # stage would make the persisted runtime state ambiguous.
        error_names = [error.get_name() for error in self.possible_errors]
        duplicates = {name for name in error_names if error_names.count(name) > 1}
        if duplicates:
            raise TypeError(
                f"{self.__class__.__name__} defines duplicated error names: {sorted(duplicates)}. "
                "Each stage error must expose a unique get_name()."
            )

        self._validate_error_sampling_policy()

    def _validate_error_sampling_policy(self) -> None:
        for attr in ("min_active_errors", "max_active_errors"):
            value = getattr(self, attr)
            if type(value) is not int:
                raise TypeError(
                    f"{self.__class__.__name__}.{attr} must be an int, "
                    f"got {type(value).__name__}."
                )
            if value < 0:
                raise TypeError(
                    f"{self.__class__.__name__}.{attr} must be >= 0, got {value}."
                )

        if self.min_active_errors > self.max_active_errors:
            raise TypeError(
                f"{self.__class__.__name__}.min_active_errors "
                f"({self.min_active_errors}) must be <= max_active_errors "
                f"({self.max_active_errors})."
            )

    def _sample_active_error_count(self) -> int:
        nb_possible_errors = len(self.possible_errors)
        if nb_possible_errors == 0:
            return 0

        self._validate_error_sampling_policy()

        min_count = min(self.min_active_errors, nb_possible_errors)
        max_count = min(self.max_active_errors, nb_possible_errors)
        if max_count == 0:
            return 0

        return random.randint(min_count, max_count)

    def build_init_state(self, env_state : Dict, env_ids : torch.Tensor) -> Dict:
        """
        This function allows to define a specific custom initialization state for all objects present in the scene.
        Ensure that each field: tensor keep the exact same size, name as the original one.
        """
        return env_state
 
    def _verif_env_completion(self, obs : Dict) -> torch.Tensor:
        """
        Check if the stage is completed according to a sliced dict of observation from the env. [BATCHED]
        Automatically call all goals associated with the Stage.
        
        :param obs: The env observation
        :type obs: Dict
        :return: A tensor of int of nb_env length. 1 for success, -1 for catastrophic failure, 0 otherwise. 1 default
        :rtype: torch.Tensor
        """
        nb_env = obs["agent"]["qpos"].shape[0]
        out = torch.ones(size=(nb_env,), dtype=torch.int32, device=obs["agent"]["qpos"].device)

        for goal in self.goals:
            goal_res = goal.verify(obs)
            out = torch.minimum(out,goal_res)

        return out
    
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
            agent_step : int,
            log : List[Log],
            recovery_extra_steps : int = 0,
        ) -> StageState:
        """
        This function can be overriden if needed to define a custom stage state depending on the log.
        By default it relies solely on target steps and acceptance steps.
        """
        effective_target = self.target_steps + recovery_extra_steps

        if agent_step < effective_target:
            return StageState.OPTIMAL
        # If acceptance is set to 0, this means that the last stage is EXCEEDED to stop the dual agent
        
        if agent_step == effective_target:
            if self.acceptance_steps == 0:
                return StageState.EXCEEDED_OPTIMAL
            return StageState.OPTIMAL
        
        if agent_step < effective_target + self.acceptance_steps:
            return StageState.ACCEPTABLE 
        return StageState.EXCEEDED
    
    def initialize_errors(self, obs : Observation, env_id : int) -> ActiveStageErrorState:
        """
        Allows to initialize the different errors according to the current observation.
        """
        nb_errors = self._sample_active_error_count()
        if nb_errors == 0:
            return {}

        if nb_errors == 1:
            chosen_errors = [random.choice(self.possible_errors)]
        else:
            chosen_errors = random.sample(self.possible_errors, k=nb_errors)

        return {
            error.get_name(): error.initialize(obs, env_id)
            for error in chosen_errors
        }
