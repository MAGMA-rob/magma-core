from typing import Dict, Union, ClassVar, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from magma_core.simulation.data_structures import RobotToolStatus, ToolErrorFlag

@dataclass(frozen=True)
class SkillSpec:
    """
    Specify a skill name, description and arguments.
    Give name of tools that are used inside.
    """
    name: str
    description: str
    required_tools: list[str]
    argument_schema: dict

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.argument_schema
        }

class SkillStatus(Enum):
    WAITING_START = "waiting_start" #waiting for start
    RUNNING = "running"
    WAITING_TICK = "waiting_tick" #waiting for tick
    FINISH = "finish"

@dataclass(frozen=True)
class CallTick:
    tool_name : str
    arguments : Dict

@dataclass(frozen=True)
class MessTick:
    message : str
    result : bool
    error_flag: ToolErrorFlag

SkillTick = Union[CallTick,MessTick]

class BaseSkill(ABC):
    """
    Base class to define a Skill.

    A skill is a sort of meta tool. It orchestrates multiple tools together,
    handling chaining between some tools.

    It could be coded by hand or generated directly by agents.

    An instance of a skill class is created at each new call, managing state variables.
    """
    spec: ClassVar[SkillSpec]

    arguments : Dict

    def __init_subclass__(
        cls,
        *,
        require_spec: bool = True,
        **kwargs,
    ) -> None:
        super().__init_subclass__(**kwargs)

        if not require_spec:
            return

        if "spec" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must define a class attribute 'spec: SkillSpec'."
            )

        if not isinstance(cls.spec, SkillSpec):
            raise TypeError(
                f"{cls.__name__}.spec must be a SkillSpec, "
                f"got {type(cls.spec).__name__}."
            )
        
    def __init__(self, arguments : Dict) -> None:
        super().__init__()
        self.arguments = arguments

    @abstractmethod
    def start(
        self,
    ) -> SkillTick:
        ...

    @abstractmethod
    def tick(
        self,
        status: RobotToolStatus,
    ) -> SkillTick:
        ...


class CancelCurrentActionSkill(BaseSkill):
    """Control skill handled directly by SkillManager."""

    spec = SkillSpec(
        name="cancel_current_action",
        description="Cancel the current action running on the targeted robot.",
        required_tools=[],
        argument_schema={},
    )

    def start(self) -> SkillTick:
        raise RuntimeError("CancelCurrentActionSkill must be handled by SkillManager")

    def tick(self, status: RobotToolStatus) -> SkillTick:
        raise RuntimeError("CancelCurrentActionSkill must be handled by SkillManager")



class MonoToolSkill(BaseSkill, require_spec=False):
    """
    Specific Skill Class that allows to encapsulate a tool within a skill to be managed
    by the skill manager properly.

    Used when a tool must be exposed directly to the agent (without being in a complexe skill).
    """

    def start(self) -> SkillTick:
        return CallTick(
            tool_name = self.spec.name,
            arguments=self.arguments,
        )

    def tick(self, status: RobotToolStatus) -> SkillTick:
        return MessTick(status.mess, status.result, status.error_flag)

def build_mono_tool_skill_class(tool: dict) -> Type[MonoToolSkill]:
    tool_name = tool["name"]

    return type(
        f"{tool_name}MonoToolSkill",
        (MonoToolSkill,),
        {
            "spec": SkillSpec(
                name=tool_name,
                description=tool.get("description", ""),
                required_tools=[tool_name],
                argument_schema=tool.get("parameters", {}),
            )
        },
    )
