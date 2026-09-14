# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
from typing import Callable, Any, Dict, Hashable, List, Optional, Tuple, Literal, Union
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import TYPE_CHECKING

from magma_core.simulation.data_structures.errors import ActiveStageErrorState

import sapien
import numpy as np
import torch


if TYPE_CHECKING:
    from magma_core.simulation.errors.base_error import BaseError
else:
    BaseError = Any

# Trajectory of the gripper, it is a list of crossing points or actions.
Trajectory = list[Union[sapien.Pose , Literal["OPEN", "CLOSE", "OK"]]]

Point = np.ndarray[Literal[3], np.dtype[np.float32]]


@dataclass
class ToolBatchContext:
    """Ephemeral resources shared by tool calls from one execution request."""

    _reservations: Dict[str, Dict[Hashable, str]] = field(default_factory=dict)

    def get_reservations(self, namespace: str) -> Dict[Hashable, str]:
        """Return a copy of the resources reserved in ``namespace``."""
        return dict(self._reservations.get(namespace, {}))

    def try_reserve(
        self,
        namespace: str,
        resource: Hashable,
        owner: str,
    ) -> bool:
        """Reserve a resource, returning False when another owner has it."""
        reservations = self._reservations.setdefault(namespace, {})
        current_owner = reservations.get(resource)
        if current_owner is not None:
            return current_owner == owner
        reservations[resource] = owner
        return True

@dataclass(frozen=True)
class ToolErrorSupport:
    """
    Describe how a tool supports a given error for pre-exec and post-verif.
    """
    error_type: type[BaseError]
    pre: bool = True
    post: bool = True


class ToolErrorFlag(str, Enum):
    NONE = "none"
    PLANNER_ERROR = "planner_error"
    INJECTION_ERROR = "injection_error"
    BAD_CALL = "bad_call"


class StageSuccess(IntEnum):
    FINISH = 1
    ONGOING = 0
    FAILED = -1

class StageState(IntEnum):
    """
    Represent the efficiency state of a task stage.

    UNASSESSED means that the stage has no target tool-call count, so its
    completion can be observed but its optimality cannot be classified.
    """
    OPTIMAL = 1
    ACCEPTABLE = 0
    UNASSESSED = 2
    EXCEEDED = -1


#########

class Log:
    function : str # Is set in the execute tool of the BaseTask class.
    stage_id : int # Is set in the execute tool of the BaseTask class.
    content : Any
    action : Optional[Literal["ADD","REMOVE"]] = None

    def __init__(
            self,
            content : Any,
            action : Optional[Literal["ADD","REMOVE"]] = None
        ) -> None:
        self.content = content
        self.action = action

    def set_fn_and_stage(self, func_name : str, stage_id : int):
        self.function = func_name
        self.stage_id = stage_id

    def to_string(self) -> str:
        return f"({self.stage_id}) function_name={self.function}, content={self.content}, action={self.action}"

@dataclass
class EnvStateUpdate:
    path : Tuple[str, ...]
    value : Any


@dataclass
class ToolResult:
    ok : bool
    reason : str = ""
    logs : Optional[Log] = None
    context : Dict[str, Any] = field(default_factory=dict)
    state_updates : List[EnvStateUpdate] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.ok, bool):
            return
        if isinstance(self.ok, torch.Tensor):
            try:
                tensor_value = self.ok.item()
            except RuntimeError as error:
                raise TypeError(
                    "ToolResult.ok tensors must contain exactly one value, "
                    f"got shape {tuple(self.ok.shape)}"
                ) from error
            self.ok = bool(tensor_value)
            return
        raise TypeError(
            "ToolResult.ok must be a bool or a single-value torch.Tensor, "
            f"got {type(self.ok).__name__}"
        )


class ToolExecution:

    function_name : str # store the function_name (for log purpose)
    associated_stage_id : int # store the stage_id (for log purpose)
    compatible_error_supports : List[ToolErrorSupport]
    execution_group : Optional[str] # Scheduling group selected for this execution.
    context : Dict[str, Any]
    failure_flag : ToolErrorFlag
    must_fail_stage : bool
    injection_applied : bool
    reason_is_public: bool
    allowed_moving_actors: Optional[List[str]]

    active_error_instances : List[BaseError]
    active_error_supports : List[ToolErrorSupport]

    def __init__(
        self,
        poses: Trajectory,
        verifier: Optional[Callable[[Dict], ToolResult]],
        reason: Optional[str] = "",
        robot_idx : int = 0,
        context: Optional[Dict[str, Any]] = None,
        compatible_error_supports: Optional[List[ToolErrorSupport]] = None, # set via tool API
        execution_group: Optional[str] = None,
        reason_is_public: bool = False,
        allowed_moving_actors: Optional[List[str]] = None,
    ):
        self.poses = poses # sequence of poses to complete the movement
        self.verifier = verifier # function to call to verify the tool success
        self.reason = reason # status message in case of errors
        self.robot_idx = robot_idx # the index of the robot in the env. In case of multi-agent env.
        # Copy the context so each tool call owns its own mutable runtime data.
        self.context = {} if context is None else dict(context)
        self.compatible_error_supports = [] if compatible_error_supports is None else list(compatible_error_supports)
        self.execution_group = execution_group
        self.reason_is_public = reason_is_public
        self.allowed_moving_actors = (
            None
            if allowed_moving_actors is None
            else list(allowed_moving_actors)
        )
        
        # Compute possible error flag
        self.failure_flag = ToolErrorFlag.NONE if len(poses) > 0 else ToolErrorFlag.BAD_CALL
        self.must_fail_stage = False
        
        self.injection_applied = False
        self.active_error_instances = []
        self.active_error_supports = []

    def _set_execution_info(self, fn_name : str, stage_id : int, active_error_instances : Optional[List[BaseError]]):
        """
        Is called internally by the BaseTask class to assign the function name,
        stage id and the active stage errors selected for this trajectory.
        """
        self.function_name = fn_name
        self.associated_stage_id = stage_id

        self.active_error_instances = []
        self.active_error_supports = []
        if not active_error_instances:
            return

        # We keep the exact support declarations so pre/post hooks only run for
        # errors that this tool explicitly opted into.
        for active_error_instance in active_error_instances:
            for error_support in self.compatible_error_supports:
                if isinstance(active_error_instance, error_support.error_type):
                    self.active_error_instances.append(active_error_instance)
                    self.active_error_supports.append(error_support)
                    break

    def fail(
            self,
            reason: str,
            failure_flag: ToolErrorFlag = ToolErrorFlag.BAD_CALL,
            must_fail_stage: bool = False,
        ):
        self.poses = []
        self.verifier = None
        self.reason = reason
        self.failure_flag = failure_flag
        self.must_fail_stage = must_fail_stage

    def apply_pre_exec_errors(self, error_state : ActiveStageErrorState):
        # Pre-exec hooks run sequentially for each compatible active error.
        if len(self.active_error_instances) == 0:
            return
        for active_error_instance, active_error_support in zip(self.active_error_instances, self.active_error_supports):
            if self.has_error(): break
            
            if not active_error_support.pre:
                continue

            error_name = active_error_instance.get_name()
            if error_name not in error_state:
                continue

            param = error_state.get(error_name)
            if param is None:
                # ``None`` means the error was selected for the stage but still
                # needs to lazily bind its runtime arguments on first use.
                param = {}
                error_state[error_name] = param

            injection_applied = active_error_instance.apply_pre_exec(self, param)
            if injection_applied is True or (
                injection_applied is None and self.has_error()
            ):
                self.injection_applied = True

    def apply_post_verif_errors(self, res : ToolResult, error_state : ActiveStageErrorState):
        if len(self.active_error_instances) == 0:
            return
        for active_error_instance, active_error_support in zip(self.active_error_instances, self.active_error_supports):
            if not active_error_support.post:
                continue

            error_name = active_error_instance.get_name()
            if error_name not in error_state:
                continue

            param = error_state.get(error_name)
            if param is None:
                # Post-verification errors can also complete the lazy binding when
                # the needed information only appears in the tool result context.
                param = {}
                error_state[error_name] = param

            self.injection_applied = True
            active_error_instance.apply_post_verif(res, param)

    def has_error(self):
        return self.failure_flag != ToolErrorFlag.NONE

    def call_verif(self, env_state : Dict, error_state : ActiveStageErrorState) -> ToolResult:
        if self.verifier is not None:
            res = self.verifier(env_state)
            if res.logs:
                res.logs.set_fn_and_stage(self.function_name,self.associated_stage_id)
            #apply the post verif errors
            self.apply_post_verif_errors(res,error_state)
            return res
        raise ValueError(f"Try to call verif on a toolExecution with a None verifier : {self.function_name}, {self.reason}")

    def get_error_description(self, error_state : ActiveStageErrorState) -> str:
        """
        Return the active stage-error description(s) for this tool execution, or
        an empty string when no compatible stage error is attached to this tool
        call.
        """
        if len(self.active_error_instances) == 0:
            return ""

        descriptions = []
        for active_error_instance in self.active_error_instances:
            error_name = active_error_instance.get_name()
            if error_name not in error_state:
                continue
            description = active_error_instance.get_description(error_state.get(error_name))
            if description:
                descriptions.append(description)

        return " | ".join(descriptions)

class RobotToolContext:
    """
    Represent a tool for a specific robot inside a complete tool infos (all tools for all robots for an env)
    """
    robot_name : str
    tool_execution : ToolExecution # stock the reference to the tool execution

    # Will be set by the trajectory converter as soon as we compute the tool
    latent_actions : List #stock for the corresponding robot, the remaining action step to do.
    real_robot_name : str

    tool_result : Optional[ToolResult] # If its set, we consider this RobotToolContext as already computed

    def __init__(self, robot_name : str, tool_execution : ToolExecution) -> None:
        self.robot_name = robot_name
        self.tool_execution = tool_execution
        self.latent_actions = []
        self.real_robot_name = ""
        self.tool_result = None
     
    def is_finish(self) -> bool:
        """Return True if there is no latent action left and that the tool_result was not already computed"""
        return len(self.latent_actions) == 0 and self.tool_result is None
    
    def is_error(self) -> bool:
        return self.tool_execution.has_error()
    
    def get_result(self) -> Optional[ToolResult]:
        return self.tool_result
