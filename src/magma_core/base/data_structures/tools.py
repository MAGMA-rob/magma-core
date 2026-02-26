# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Callable, Any, Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, field
from enum import IntEnum
import json

from magma_core.base.data_structures import Trajectory
from .situation import Situation
from .answer import CommanderAnswer
from ..data_structures import EmptyInstruction

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
    planning_errors : List[bool] # True in case of planning errors.
    stage_id : int
    reward : Optional[float] = None # 0-1 reward. If not set, which suppose that the eward has not changed with the execution of the tool. (Empty tool)
    mess : Optional[str] = None

    # Success about the overall step
    stage_success : StageSuccess = StageSuccess.ONGOING
    stage_state : StageState = StageState.OPTIMAL
    
    # Next Situation Information (None if final step)
    next_situation : Optional[Situation] = None
    attributes_modif : List[Tuple] = field(default_factory=list)

    def has_single_planning_errors(self) -> bool:
        return len(self.planning_errors) == 1 and self.planning_errors[0]

    def get_mixed_next_situation(self, precedent_situation : Situation) -> Optional[Situation]:
        if self.next_situation is None:
            return None
        return self.next_situation.copy_with(attributes=precedent_situation.attributes)

#########

class Log:
    function : str # Is set in the execute tool of the BaseTask class.
    content : Any
    action : Optional[Literal["ADD","REMOVE"]] = None

    def __init__(
            self,
            content : Any,
            action : Optional[Literal["ADD","REMOVE"]] = None
        ) -> None:
        self.content = content
        self.action = action

    def set_function_name(self, func_name : str):
        self.function = func_name

    def to_string(self) -> str:
        return f"function_name={self.function}, content={self.content}, action={self.action}"

@dataclass
class ToolResult:
    ok : bool
    reason : str = ""
    logs : Optional[Log] = None


class ToolExecution:
    def __init__(
        self,
        poses: Trajectory,
        verifier: Optional[Callable[[Dict], ToolResult]],
        redo: Optional[Callable[[Dict], Trajectory]] = None,
        reason: Optional[str] = "",
        robot_idx : int = 0,
    ):
        self.poses = poses # sequence of poses to complete the movement
        self.verifier = verifier # function to call to verify the tool success
        self.redo = redo # function to call when a new call is needed
        self.reason = reason # status message in case of errors
        self.robot_idx = robot_idx # the index of the robot in the env. In case of multi-agent env.

    def _set_function_name(self, fn_name : str):
        """
        Is called internally by the Base Task class to assign the function name directly.
        """
        self.function_name = fn_name

    def has_error(self):
        return len(self.poses) == 0

    def call_verif(self, env_state : Dict) -> ToolResult:
        if self.verifier is not None:
            res = self.verifier(env_state)
            if res.logs:
                res.logs.set_function_name(self.function_name)
            return res
        raise ValueError(f"Try to call verif on a toolExecution with a None verifier : {self.function_name}, {self.reason}")

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
    tool_robots : List[ToolRobot]
    logs : List[Log] #stock optional logs for the env
    current_agent_step: int
    current_task_stage: int
    source_id : int

    stage_log_start_idx : int 

    def __init__(
            self,
            node_id : int,
            task_stage : int,
            agent_step : int,
            stage_log_start_idx : int,
            source_id : int,
            tool_robots : List[ToolRobot],
            logs : List[Log] = []
        ) -> None:
        self.node_id = node_id
        self.current_agent_step = agent_step
        self.current_task_stage = task_stage
        self.stage_log_start_idx = stage_log_start_idx
        self.logs = logs
        self.source_id = source_id
        self.tool_robots = tool_robots
    
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
    
    def compute_tool_results(self, obs : Dict) -> bool:
        """
        Compute result for ended tool. Return true if some trajectory needs to be computed again.
        """
        traj_to_compute = False
        for tool in self.tool_robots:
            if not tool.is_error() and tool.is_finish():
                # call of the verifier
                tool_result : ToolResult = tool.tool_execution.call_verif(obs)

                if tool_result.logs != None:
                    self.logs.append(tool_result.logs)

                if not tool_result.ok and tool.tool_execution.redo:
                    # If there is a redo in the tool
                    p = tool.tool_execution.redo(obs)
                    if p: # If there is some trajectory to execute, we add
                        tool.tool_execution.poses = p
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