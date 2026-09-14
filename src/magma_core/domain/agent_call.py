from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

@dataclass
class Call:

    name : str
    arguments : Dict
    target_robot_name : str = "default"

    def to_function_string(self) -> str:
        arguments = ", ".join(
            f"{name}={value}" for name, value in self.arguments.items()
        )
        return f"{self.name}({arguments})"

    def to_string(self) -> str:
        return f"{self.target_robot_name}.{self.to_function_string()}\n"

class ExecutionRequest(ABC):

    source_node_id : int
    agent_step_id : int

    def __init__(self, source_node_id : int, agent_step_id : int) -> None:
        self.source_node_id = source_node_id
        self.agent_step_id = agent_step_id

    def get_tool_calls(self) -> List[Call]:
        raise RuntimeError("Requesting a bad ExecutionRequest 'tool_call")
    
    def contains_call(self) -> bool:
        return False
    
    def get_say(self) -> str:
        raise RuntimeError("Requesting a bad ExecutionRequest the 'say'")


class ValidExecutionReq(ExecutionRequest):

    say : str
    calls : List[Call]

    def __init__(
            self,
            source_node_id: int,
            agent_step_id: int,
            calls : List[Call],
            say : str,
        ) -> None:
        super().__init__(source_node_id, agent_step_id)
        self.calls = calls
        self.say = say
        

    def get_tool_calls(self) -> List[Call]:
        return self.calls
    
    def contains_call(self) -> bool:
        return len(self.calls) > 0
    
    def get_say(self) -> str:
        return self.say

class InvalidExecutionReq(ExecutionRequest):

    bad_answer : str

    def __init__(self, source_node_id: int, agent_step_id : int, output : str) -> None:
        super().__init__(source_node_id, agent_step_id)
        self.bad_answer = output

@dataclass
class EmptyToolCall:
    """
    Encapsulate an Execution Request that is destinaed to be evaluated by the judge (no tool call).
    """
    node_id : int
    execution_request : ExecutionRequest
    current_task_stage: int
    attributes: Dict[str, Any]
    tool_calls: int
    forgiven_tool_calls: int

    def get_source_node_id(self) -> int:
        return self.execution_request.source_node_id
    
    def get_id(self) -> int:
        return self.node_id
    
    def get_step_id(self) -> int:
        return self.execution_request.agent_step_id
    
    def get_answer(self) -> str:
        return self.execution_request.get_say()
