# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from __future__ import annotations

from typing import Callable, Any, Dict, List, Optional, Tuple, Literal, Union
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import json
from typing import TYPE_CHECKING

from .situation import Situation
from .answer import CommanderAnswer
from .env import ActiveStageErrorState

import sapien
import numpy as np

if TYPE_CHECKING:
    from ..errors.base_error import BaseError

# Trajectory of the gripper, it is a list of crossing points or actions.
Trajectory = list[Union[sapien.Pose , Literal["OPEN", "CLOSE", "OK"]]]

Point = np.ndarray[Literal[3], np.dtype[np.float32]]

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
    Represent the state of a task stage. 

    A state is optimal if the agent step is inferior or equal to the number of target_steps.
    A state is acceptable if the agent step is higher than target_steps but below target_steps+acceptance_steps.
    A state is exceeded if the agent step is higher or equal target_steps+acceptance_steps.
    """
    OPTIMAL = 1
    ACCEPTABLE = 0
    EXCEEDED = -1
    EXCEEDED_OPTIMAL = -2 # Edge-case if acceptance_step is set to 0. Allows to make difference between last optimal step and last acceptable step.

@dataclass
class ToolStatus:
    """
    Encapsulate a tool status return by the system
    """
    result : List[bool] # List of success, one per robot tool called.
    error_flags : List[ToolErrorFlag] # Error qualification aligned with each tool.
    stage_id : int
    error_descriptions : List[str] = field(default_factory=list) # Stage-error description aligned with each tool index.
    reward : Optional[float] = None # 0-1 reward. If not set, which suppose that the eward has not changed with the execution of the tool. (Empty tool)
    mess : Optional[str] = None

    # Success about the overall step
    stage_success : StageSuccess = StageSuccess.ONGOING
    stage_state : StageState = StageState.OPTIMAL
    
    # Next Situation Information (None if final step)
    next_situation : Optional[Situation] = None
    attributes_modif : List[Tuple] = field(default_factory=list)

    def has_single_error_flag(self, flag: ToolErrorFlag) -> bool:
        return len(self.error_flags) == 1 and self.error_flags[0] == flag

    def has_single_planner_error(self) -> bool:
        return self.has_single_error_flag(ToolErrorFlag.PLANNER_ERROR)

    def all_error_flags_are(self, flag: ToolErrorFlag) -> bool:
        return len(self.error_flags) > 0 and all(error_flag == flag for error_flag in self.error_flags)

    def get_mixed_next_situation(self, precedent_situation : Situation) -> Optional[Situation]:
        if self.next_situation is None:
            return None
        return self.next_situation.copy_with(attributes=precedent_situation.attributes)

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
class ToolResult:
    ok : bool
    reason : str = ""
    logs : Optional[Log] = None
    context : Dict[str, Any] = field(default_factory=dict)


class ToolExecution:

    function_name : str # store the function_name (for log purpose)
    associated_stage_id : int # store the stage_id (for log purpose)
    compatible_error_supports : List[ToolErrorSupport]
    context : Dict[str, Any]
    failure_flag : ToolErrorFlag
    must_fail_stage : bool
    injection_applied : bool

    active_error_instances : List[BaseError]
    active_error_supports : List[ToolErrorSupport]

    def __init__(
        self,
        poses: Trajectory,
        verifier: Optional[Callable[[Dict], ToolResult]],
        redo: Optional[Callable[[Dict], Trajectory]] = None,
        reason: Optional[str] = "",
        robot_idx : int = 0,
        context: Optional[Dict[str, Any]] = None,
        compatible_error_supports: Optional[List[ToolErrorSupport]] = None # set via tool API
    ):
        self.poses = poses # sequence of poses to complete the movement
        self.verifier = verifier # function to call to verify the tool success
        self.redo = redo # function to call when a new call is needed
        self.reason = reason # status message in case of errors
        self.robot_idx = robot_idx # the index of the robot in the env. In case of multi-agent env.
        # Copy the context so each tool call owns its own mutable runtime data.
        self.context = {} if context is None else dict(context)
        self.compatible_error_supports = [] if compatible_error_supports is None else list(compatible_error_supports)
        
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
        self.redo = None
        self.reason = reason
        self.failure_flag = failure_flag
        self.must_fail_stage = must_fail_stage

    def replace_poses(self, poses: Trajectory):
        self.poses = poses
        self.failure_flag = ToolErrorFlag.NONE if len(poses) > 0 else ToolErrorFlag.BAD_CALL
        if len(poses) > 0:
            self.must_fail_stage = False

    def clear_redo(self):
        self.redo = None

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

            self.injection_applied = True
            active_error_instance.apply_pre_exec(self, param)

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

class ToolRobot:
    """
    Represent a tool for a specific robot inside a complete tool infos (all tools for all robots for an env)
    """
    robot_name : str
    tool_execution : ToolExecution #stock the reference to the execution tool class element (verifier, redo...)

    # Will be set by the trajectory converter as soon as we compute the tool
    latent_actions : List #stock for the corresponding robot, the remaining action step to do.
    real_robot_name : str

    tool_result : Optional[ToolResult] # If its set, we consider this ToolRobot as already computed

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


class ToolInfos:
    """
    Encapsulate all tools for all robots of a specific environments
    """
    node_id : int
    tool_robots : List[ToolRobot] # List of tool per robot
    
    # env information
    logs : List[Log] #stock optional logs for the env
    current_agent_step: int
    current_task_stage: int
    source_id : int
    stage_log_start_idx : int
    error_state : ActiveStageErrorState

    # unused for now
    composite_progress : Dict[str, Any]

    def __init__(
            self,
            node_id : int,
            task_stage : int,
            agent_step : int,
            stage_log_start_idx : int,
            source_id : int,
            tool_robots : List[ToolRobot],
            error_state : ActiveStageErrorState,
            logs : Optional[List[Log]] = None,
            composite_progress : Optional[Dict[str, Any]] = None,
        ) -> None:
        self.node_id = node_id
        self.current_agent_step = agent_step
        self.current_task_stage = task_stage
        self.stage_log_start_idx = stage_log_start_idx
        self.logs = [] if logs is None else logs
        self.source_id = source_id
        self.tool_robots = tool_robots

        # The stage/source node already decided which errors are active.
        # ToolInfos only forwards that state to compatible tool executions.
        self.error_state = error_state
        self._apply_pre_exec_errors()

        self.composite_progress = {} if composite_progress is None else composite_progress
    
    def all_finished(self) -> bool:
        """
        Return True if all tool per robots has been done
        """
        if len(self.tool_robots) == 0:
            return False
        for tool in self.tool_robots:
            if tool.tool_result is None and not tool.is_error():
                return False
        return True
    
    def is_full_error(self):
        """
        Return True if all tool execution are in error
        """
        return all([e.is_error() for e in self.tool_robots])

    def has_terminal_tool_failure(self) -> bool:
        return any(tool.tool_execution.must_fail_stage for tool in self.tool_robots)
    
    def _apply_pre_exec_errors(self):
        """
        Apply the pre-execution effect of active errors to the tools.
        """
        for tool in self.tool_robots:
            tool.tool_execution.apply_pre_exec_errors(self.error_state)
        
    def build_return(self) -> Tuple[List[bool],str,List]:
        """
        Return a list of bool corresponding to the list of tool success and a string for the system status
        """
        mess_list = []
        return_status = []
        attributes_modif = []
        for tool in self.tool_robots:
            if tool.is_error():
                return_status.append(False)
                mess_list.append(tool.tool_execution.reason)
                continue
            else:
                result = tool.get_result()
                if result is None:
                    raise RuntimeError("A tool infos which has not fully ended tries to compute all return")
                return_status.append(result.ok)
                mess_list.append(result.reason)

            if result.logs is not None and result.logs.action is not None:
                attributes_modif.append((result.logs.action, result.logs.content))
                
                
        if len(mess_list) == 1:
            return return_status, mess_list[0], attributes_modif
    
        return return_status, json.dumps({tool.robot_name : mess_list[i] for i, tool in enumerate(self.tool_robots)}), attributes_modif

    def get_error_descriptions(self) -> List[str]:
        """
        Return the stage-error descriptions aligned with each tool robot.
        """
        return [
            tool.tool_execution.get_error_description(self.error_state)
            for tool in self.tool_robots
        ]

    def get_error_flags(self) -> List[ToolErrorFlag]:
        """
        Return the final error qualification aligned with each tool robot.
        """
        error_flags = []
        for tool in self.tool_robots:
            # tool in error means either bad call or injection
            if tool.is_error():
                if tool.tool_execution.injection_applied:
                    error_flags.append(ToolErrorFlag.INJECTION_ERROR)
                else:
                    error_flags.append(ToolErrorFlag.BAD_CALL)
                continue

            result = tool.get_result()
            if result is None:
                raise RuntimeError("A tool infos which has not fully ended tries to compute error flags")

            # Normal case, no errors
            if result.ok:
                error_flags.append(ToolErrorFlag.NONE)
            
            else: #result not ok 
                if tool.tool_execution.injection_applied:
                    error_flags.append(ToolErrorFlag.INJECTION_ERROR)
                else:
                    error_flags.append(ToolErrorFlag.PLANNER_ERROR)

        return error_flags

    def get_result_contexts(self) -> List[Dict[str, Any]]:
        """
        Return the final verifier contexts aligned with each tool robot.

        Contexts are only available after a tool has reached a final verifier
        result. Intermediate failed verifier attempts that still produce a
        redo trajectory are intentionally not exposed here.
        """
        contexts = []
        for tool in self.tool_robots:
            result = tool.get_result()
            contexts.append({} if result is None else dict(result.context))
        return contexts
    
    def compute_tool_results(self, obs : Dict) -> bool:
        """
        Compute result for ended tool. Return true if some trajectory needs to be computed again.
        """
        traj_to_compute = False
        for tool in self.tool_robots:
            if not tool.is_error() and tool.is_finish():
                # call of the verifier
                tool_result : ToolResult = tool.tool_execution.call_verif(obs,self.error_state)

                if tool_result.logs != None:
                    self.logs.append(tool_result.logs)

                if not tool_result.ok and tool.tool_execution.redo:
                    # If there is a redo in the tool
                    p = tool.tool_execution.redo(obs)
                    if p: # If there is some trajectory to execute, we add
                        tool.tool_execution.replace_poses(p)
                        traj_to_compute = True
                        continue

                tool.tool_result = tool_result
        
        return traj_to_compute
    
    def _get_logs(self) -> Tuple[List[Log], List[Log]]:
        """
        Return a tuple with first list, the full log, and second list the specific stage log
        """
        return self.logs, self.logs[self.stage_log_start_idx:]


@dataclass
class EmptyToolCall:
    node_id : int
    answer : CommanderAnswer
    current_agent_step: int
    current_task_stage: int
