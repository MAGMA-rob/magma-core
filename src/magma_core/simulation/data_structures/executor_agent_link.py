from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from magma_core.simulation.data_structures.tools import ToolErrorFlag, StageSuccess, StageState
from magma_core.simulation.data_structures.situation import StageInput
from magma_core.simulation.agents import AgentAnswer

@dataclass()
class RobotToolStatus:

    robot_name : str
    mess : str
    result : bool
    error_flag : ToolErrorFlag
    mess_is_public: bool = False

    def to_string(self, tool_name : str) -> str:
        if self.mess == "":
            return ""
        state = "succeed" if self.result else "failed"
        return f"{self.robot_name}: {tool_name} {state} with message: {self.mess}"


@dataclass
class ToolStatus:
    """
    Encapsulate a tool status return by the system
    """
    
    robots_status : List[RobotToolStatus]
    stage_id : int
    attributes : Dict[str, Any]
    error_descriptions : List[str] # Descriptions of all active stage errors.


    # Failure is set only in general failure like text-only, calling action in text-only
    failure_reason : Optional[str] = None
    reward : Optional[float] = None # 0-1 reward. If not set, which suppose that the eward has not changed with the execution of the tool. (Empty tool)
    failure_diagnostics: Optional[Dict[str, Any]] = None

    # Success about the overall step
    stage_success : StageSuccess = StageSuccess.ONGOING
    stage_state : StageState = StageState.OPTIMAL
    tool_calls: int = 0
    forgiven_tool_calls: int = 0
    
    # Next Situation Information (None if final step)
    next_input : Optional[StageInput] = None

    @property
    def effective_tool_calls(self) -> int:
        return self.tool_calls - self.forgiven_tool_calls

    def should_restore_source_env_state(self) -> bool:
        return any(
            robot_status.error_flag == ToolErrorFlag.PLANNER_ERROR
            for robot_status in self.robots_status
        )
    
    def get_robot_status(self, robot_name : str) -> RobotToolStatus:
        for status in self.robots_status:
            if status.robot_name == robot_name:
                return status
        
        raise RuntimeError(f"Asking for a robot {robot_name} which is not in the Tool status {self.robots_status}")

    def build_status_return(self, agent_answer : AgentAnswer) -> Optional[Dict[str, Any]]:
        calls = agent_answer.get_action()
        message_lines = []
        for robot_status in self.robots_status:
            matching_call = None
            for c in calls:
                if robot_status.robot_name == c.target_robot_name:
                    matching_call = c
                    break

            tool_name = matching_call.name if matching_call is not None else "action"
            line = robot_status.to_string(tool_name)
            if line:
                message_lines.append(line)

        has_robot_feedback = len(message_lines) > 0
        if not has_robot_feedback and self.failure_reason is None:
            return None

        status_return: Dict[str, Any] = {}
        if has_robot_feedback:
            status_return["infos"] = "\n".join(message_lines)
            status_return["previous_tool_call"] = agent_answer.to_dict()["action"]

        if self.failure_reason:
            status_return["failure_reason"] = self.failure_reason

        return status_return

    def build_dict_return(self, agent_answer : AgentAnswer) -> Optional[Dict[str, Any]]:
        return self.build_status_return(agent_answer)

    def is_full_of_this_flag(self, flag : ToolErrorFlag):
        return len(self.robots_status) > 0 and all(
            st.error_flag == flag
            for st in self.robots_status
        )


def build_text_only_action_failure_status(
        robot_names: List[str],
        stage_id: int,
        attributes: Dict[str, Any],
        tool_calls: int,
        forgiven_tool_calls: int,
        next_input: Optional[StageInput] = None,
    ) -> ToolStatus:
    """Build the canonical failure returned when a text-only stage receives an action."""
    return ToolStatus(
        robots_status=[
            RobotToolStatus(robot_name, "", False, ToolErrorFlag.NONE)
            for robot_name in robot_names
        ],
        error_descriptions=[""] * len(robot_names),
        failure_reason=(
            "The answer called a tool, but this text-only stage "
            "requires a user-facing response and does not allow tool calls."
        ),
        next_input=next_input,
        stage_id=stage_id,
        stage_success=StageSuccess.FAILED,
        attributes=attributes,
        tool_calls=tool_calls,
        forgiven_tool_calls=forgiven_tool_calls,
    )
