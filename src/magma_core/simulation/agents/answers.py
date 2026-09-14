# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from magma_core.domain.agent_call import Call


@dataclass(frozen=True)
class AgentAnswer(ABC):
    """
    Normalized answer produced by an agent.
    """

    source_node_id: int
    agent_step_id: int

    @abstractmethod
    def to_dict(self) -> Dict:
        raise NotImplementedError()

    def get_say(self) -> str:
        raise RuntimeError("Asking failed answer a say")

    def get_action(self) -> List["Call"]:
        raise RuntimeError("Asking failed answer an action")

    @abstractmethod
    def to_string(self) -> str:
        pass

    def is_valid(self) -> bool:
        return False


@dataclass(frozen=True)
class BadAgentAnswer(AgentAnswer):
    say: str
    raw_action: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "say": self.say,
            "raw_action": self.raw_action,
            "reason": self.reason,
        }

    def to_string(self) -> str:
        return f"BAD_ANSWER: {self.reason} | MESSAGE: {self.say} | RAW_ACTION: {self.raw_action}"


@dataclass(frozen=True)
class ValidAgentAnswer(AgentAnswer):
    """
    Normalized answer with parsed tool calls.

    Parsing is intentionally not done here because each agent can expose its own
    model output schema.
    """

    say: str
    calls: List["Call"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "say": self.say,
            "action": {
                call.target_robot_name: {
                    "name": call.name,
                    "arguments": call.arguments,
                }
                for call in self.calls
            },
        }

    def get_say(self) -> str:
        return self.say

    def get_action(self) -> List["Call"]:
        return self.calls

    def to_string(self) -> str:
        if self.calls:
            calls = " | ".join(call.to_string() for call in self.calls)
            return f"CALLS: {calls}"
        if self.say != "":
            return f"MESSAGE: {self.say}"
        return ""

    def is_valid(self) -> bool:
        return True

    def is_empty(self) -> bool:
        return self.say == "" and not self.calls
