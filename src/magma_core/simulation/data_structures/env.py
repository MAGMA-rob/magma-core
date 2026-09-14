# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy

from magma_core.domain.agent_call import ValidExecutionReq
from magma_core.simulation.data_structures.tools import (
    RobotToolContext, Log, ToolResult,
    ToolErrorFlag, EnvStateUpdate,
    StageSuccess
)
from magma_core.simulation.data_structures.errors import ActiveStageErrorState
from magma_core.simulation.errors import BaseError
from magma_core.simulation.data_structures.situation import StageInput
from magma_core.simulation.data_structures.executor_agent_link import (
    RobotToolStatus,
    ToolStatus,
    build_text_only_action_failure_status,
)

from magma_core.utils.data_utils import apply_att_modif

MAX_FORGIVEN_TOOL_CALLS_PER_STAGE = 3


@dataclass
class SavedEnvData:
    """Saved per-node runtime data used by generation executors."""

    env_state: Dict
    logs: List
    stage_id: int
    stage_log_length: int
    attributes: Dict[str, Any] = field(default_factory=dict)
    tool_calls: int = 0
    forgiven_tool_calls: int = 0

    composite_progress: Dict[str, Any] = field(default_factory=dict)
    active_stage_error_state: ActiveStageErrorState = field(default_factory=dict)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

@dataclass
class EnvCreationInfos:
    """
    Data structure to transport information required to initialize an env to execute the execution request
    """

    env_id : int
    node_id : int

    saved_data : SavedEnvData
    request : ValidExecutionReq

    def get_source_node_id(self):
        return self.request.source_node_id
    
    def get_env_state(self):
        return self.saved_data.env_state

class EnvToolContext:
    """
    Encapsulate all tools for all robots of a specific environments
    """
    node_id : int
    tool_robots : List[RobotToolContext] # List of tool per robot
    
    # env information
    logs : List[Log] #stock optional logs for the env
    current_agent_step: int
    current_task_stage: int
    source_id : int
    stage_log_start_idx : int
    error_state : ActiveStageErrorState
    attributes : Dict[str, Any]
    tool_calls: int
    forgiven_tool_calls: int
    source_forgiven_tool_calls: int

    # unused for now
    composite_progress : Dict[str, Any]

    def __init__(
            self,
            node_id : int,
            task_stage : int,
            agent_step : int,
            stage_log_start_idx : int,
            source_id : int,
            tool_robots : List[RobotToolContext],
            error_state : ActiveStageErrorState,
            attributes : Dict[str, Any],
            tool_calls: int,
            forgiven_tool_calls: int,
            logs : Optional[List[Log]] = None,
            composite_progress : Optional[Dict[str, Any]] = None,
            active_stage_errors: Optional[List[BaseError]] = None,
        ) -> None:
        self.node_id = node_id
        self.current_agent_step = agent_step
        self.current_task_stage = task_stage
        self.stage_log_start_idx = stage_log_start_idx
        self.logs = [] if logs is None else logs
        self.source_id = source_id
        self.tool_robots = tool_robots
        self.attributes = deepcopy(attributes)
        self.tool_calls = tool_calls
        self.forgiven_tool_calls = forgiven_tool_calls
        self.source_forgiven_tool_calls = forgiven_tool_calls

        # The stage/source node already decided which errors are active.
        # EnvToolContext only forwards that state to compatible tool executions.
        self.error_state = error_state
        self.active_stage_errors = list(active_stage_errors or [])
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

    def get_allowed_moving_actors(self) -> Optional[set[str]]:
        """Return the batch-level actor allowlist, or None when unprotected."""
        allowed_actors: set[str] = set()
        for tool in self.tool_robots:
            tool_allowed_actors = tool.tool_execution.allowed_moving_actors
            if tool_allowed_actors is None:
                return None
            allowed_actors.update(tool_allowed_actors)
        return allowed_actors

    def has_terminal_tool_failure(self) -> bool:
        return any(tool.tool_execution.must_fail_stage for tool in self.tool_robots)
    
    def _apply_pre_exec_errors(self):
        """
        Apply the pre-execution effect of active errors to the tools.
        """
        for tool in self.tool_robots:
            tool.tool_execution.apply_pre_exec_errors(self.error_state)
        
    def build_tool_status(self) -> ToolStatus:
        """
        Return The Tool Status associated with this env tool context
        """

        attribute_updates = []
        robot_status : List[RobotToolStatus] = []
        for tool_context in self.tool_robots:
            if tool_context.is_error():
                if tool_context.tool_execution.injection_applied:
                    flag = ToolErrorFlag.INJECTION_ERROR
                else:
                    flag = ToolErrorFlag.BAD_CALL
                robot_status.append(
                    RobotToolStatus(
                        robot_name=tool_context.robot_name,
                        result=False,
                        mess=tool_context.tool_execution.reason or "",
                        error_flag=flag,
                        mess_is_public=tool_context.tool_execution.reason_is_public,
                    )
                )
                continue
            
            result = tool_context.get_result()
            if result is None:
                raise RuntimeError("A tool infos which has not fully ended tries to compute all return")
            
            flag = ToolErrorFlag.NONE
            if not result.ok:
                if tool_context.tool_execution.injection_applied:
                    flag = ToolErrorFlag.INJECTION_ERROR
                else:
                    flag = ToolErrorFlag.PLANNER_ERROR

            robot_status.append(
                RobotToolStatus(
                    robot_name=tool_context.robot_name,
                    result=result.ok,
                    mess=result.reason,
                    error_flag=flag
                )
            )

            if result.logs is not None and result.logs.action is not None:
                attribute_updates.append((result.logs.action, result.logs.content))

        apply_att_modif(self.attributes, attribute_updates)

        self.forgiven_tool_calls = min(
            MAX_FORGIVEN_TOOL_CALLS_PER_STAGE,
            self.source_forgiven_tool_calls + sum(
                status.error_flag in {
                    ToolErrorFlag.PLANNER_ERROR,
                    ToolErrorFlag.INJECTION_ERROR,
                }
                for status in robot_status
            ),
        )

        return ToolStatus(
            robots_status=robot_status,
            stage_id=self.current_task_stage,
            attributes=self.attributes,
            error_descriptions=self.get_error_descriptions(),
            tool_calls=self.tool_calls,
            forgiven_tool_calls=self.forgiven_tool_calls,
        )
    
    def build_catastrophic_failure_status(self, next_input : Optional[StageInput]) -> ToolStatus:
        return build_text_only_action_failure_status(
            robot_names=[
                tool_context.robot_name
                for tool_context in self.tool_robots
            ],
            stage_id=self.current_task_stage,
            attributes=self.attributes,
            tool_calls=self.tool_calls,
            forgiven_tool_calls=self.forgiven_tool_calls,
            next_input=next_input,
        )

    def get_error_descriptions(self) -> List[str]:
        """
        Describe all active stage errors, independently of tool compatibility.
        """
        return [
            error.get_description(self.error_state[error.get_name()])
            for error in self.active_stage_errors
            if error.get_name() in self.error_state
        ]

    def get_result_contexts(self) -> List[Dict[str, Any]]:
        """
        Return the final verifier contexts aligned with each tool robot.

        Contexts are only available after a tool has reached a final verifier
        result.
        """
        contexts = []
        for tool in self.tool_robots:
            result = tool.get_result()
            contexts.append({} if result is None else dict(result.context))
        return contexts

    def get_state_updates(self) -> List[EnvStateUpdate]:
        """
        Allows to manage move_to, state changement of object etc...
        """
        updates = []
        for tool in self.tool_robots:
            result = tool.get_result()
            if result is not None:
                updates.extend(result.state_updates)
        return updates
    
    def compute_tool_results(self, obs : Dict) -> None:
        """
        Compute results for tools whose execution has ended.
        """
        for tool in self.tool_robots:
            if not tool.is_error() and tool.is_finish():
                # call of the verifier
                tool_result : ToolResult = tool.tool_execution.call_verif(obs,self.error_state)

                if tool_result.logs != None:
                    self.logs.append(tool_result.logs)

                tool.tool_result = tool_result
    
    def _get_logs(self) -> Tuple[List[Log], List[Log]]:
        """
        Return a tuple with first list, the full log, and second list the specific stage log
        """
        return self.logs, self.logs[self.stage_log_start_idx:]
