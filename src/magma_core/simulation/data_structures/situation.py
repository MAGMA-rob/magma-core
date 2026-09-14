# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from __future__ import annotations
from typing import List, Dict, Any, Literal, Optional, Tuple
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, field
import json

from magma_core.utils.data_utils import apply_att_modif
from magma_core.simulation.serialization import (
    build_spec,
    construct_from_spec,
    decode_value,
    encode_value,
    extract_constructor_arguments,
)


class Instruction(ABC):
    """
    Base class to represent Instruction.
    It has a content, a role and a flag has_constraint.

    The flag must be set to True if the instruction contains constraints injection.
    Possible only with UserInstruction.
    """

    content : str
    role : str
    timestamp : int
    has_constraint : bool

    def __init__(
            self,
            content : str,
            role : Literal["USER","MODEL","SYSTEM"],
            has_constraint : bool,
            timestamp : int = 0,
        ) -> None:
        self.content = content
        self.role = role
        self.timestamp = timestamp
        self.has_constraint = has_constraint

    def get_content(self):
        return self.content
    
    def get_role(self):
        if self.role not in {"USER","MODEL","SYSTEM"}:
            raise RuntimeError()
        return self.role
    
    def get_timestamp(self):
        return self.timestamp
    
    def to_string(self) -> str:
        return self.content

    def to_spec(self) -> Dict[str, Any]:
        """Serialize an instruction from its constructor arguments."""

        return build_spec(self, extract_constructor_arguments(self))

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "Instruction":
        """Reconstruct an instruction from a specification."""

        instruction = construct_from_spec(spec, cls)
        if not isinstance(instruction, cls):
            raise TypeError(f"Instruction spec is not compatible with {cls.__name__}")
        return instruction

class UserInstruction(Instruction):

    def __init__(self, content: str, timestamp: int = 0, has_constraint : bool = False) -> None:
        super().__init__(content, "USER", has_constraint, timestamp)
    
class StatusReturn(Instruction):

    str_simplified : str

    def __init__(self, status: Dict, timestamp: int = 0) -> None:
        self.status = status
        mess = status.get("infos", None)
        if mess is None:
            mess = status.get("error", None)
            if mess is None:
                raise RuntimeError("Should not happen")
        self.str_simplified = mess
        super().__init__(json.dumps(status), "SYSTEM", False, timestamp)

    def to_string(self) -> str:
        return self.str_simplified

class EmptyInstruction(Instruction):

    def __init__(self) -> None:
        self.content = ""
        self.role =  ""
        self.timestamp = 0
        self.has_constraint = False

class TemplateInstruction(Instruction):
    """
    Template Instruction is an instruction that need to be augmented by the UserSim components.
    You can pass any information through the template that will be pass to the UserSim to generate a plausible instruction.
    """

    template : Dict
    context : str

    def __init__(self, template: Dict, context : str, timestamp: int = 0) -> None:
        self.template = template
        self.context = context
        super().__init__(str(template), "USER", False, timestamp)

    def set_content(self, text : str):
        self.content = text

    def get_content(self):
        if self.content is None:
            return str(self.template)
        return self.content


@dataclass
class StageInput:
    """
    Lightweight declarative input for a stage.
    """

    instruction: Instruction
    flag_answer_to_user: bool = False
    # True when this stage should be evaluated as part of the previous user instruction.
    linked_to_prev: bool = False

    def __init__(self, instruction : Instruction, flag_answer_to_user : bool, linked_to_prev : Optional[bool] = None) -> None:
        """
        Build the StageInput from an instruction and a flag to mark this stage as the end of a 'sequence' requiring an answer of the model to the user.
        linked_to_prev indicates if the stage is linked to a previous one. It's automatically set to True if instruction is Empty.
        You can force to be True.
        """
        self.instruction = instruction
        self.flag_answer_to_user = flag_answer_to_user
        if isinstance(self.instruction, EmptyInstruction):
            self.linked_to_prev = True
        elif linked_to_prev is None:
            self.linked_to_prev = False
        else:
            self.linked_to_prev = linked_to_prev


class SituationInit:
    """
    Initial runtime state for a task/preset.

    all_task_attributes store the complete attributes possible (in case of updating for randomization sake).
    By default, if attributes does not evolve during execution, you can let it to None, it wil ltake the attributes value.
    Otherwise, define it with all possible values.
    """

    attributes: Dict[str, Any]
    memory: Dict[str, Any] 
    all_task_attributes : Dict[str, Any] # Store the full task attributes. Used by the randomizer to compute initial correspondance.
    history: List

    def __init__(
            self,
            attributes : Dict,
            all_task_attributes : Optional[Dict[str, Any]] = None,
            memory : Optional[Dict] = None,
            history : Optional[List] = None,
        ) -> None:
        self.attributes = attributes
        self.memory = {} if memory is None else memory
        self.history = [] if history is None else history
        if all_task_attributes is None:
            self.all_task_attributes = self.attributes
        else:
            self.all_task_attributes = all_task_attributes

    def to_situation(self, stage_input : StageInput) -> Situation:
        return Situation(
            memory=deepcopy(self.memory),
            attributes=deepcopy(self.attributes),
            instruction=stage_input.instruction,
            flag_answer_to_user=stage_input.flag_answer_to_user,
            history=deepcopy(self.history),
        )

    def to_spec(self) -> Dict[str, Any]:
        return encode_value({
            "attributes": deepcopy(self.attributes),
            "all_task_attributes": deepcopy(self.all_task_attributes),
            "memory": deepcopy(self.memory),
            "history": deepcopy(self.history),
        })

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "SituationInit":
        values = decode_value(spec)
        if not isinstance(values, dict):
            raise TypeError("SituationInit spec must be a dictionary")
        expected = {"attributes", "all_task_attributes", "memory", "history"}
        if set(values) != expected:
            raise ValueError(
                "SituationInit spec keys mismatch: "
                f"expected={sorted(expected)}, got={sorted(values)}"
            )
        if not isinstance(values["attributes"], dict):
            raise TypeError("SituationInit attributes must be a dictionary")
        if not isinstance(values["all_task_attributes"], dict):
            raise TypeError("SituationInit all_task_attributes must be a dictionary")
        if not isinstance(values["memory"], dict):
            raise TypeError("SituationInit memory must be a dictionary")
        if not isinstance(values["history"], list):
            raise TypeError("SituationInit history must be a list")
        return cls(**values)

class Situation(ABC):
    """
    Base class to define the initial situation of a task : memory, instruction, attributes...
    """

    memory : Dict[str, Any] # runtime memory state
    attributes : Dict[str, Any] # task attributes
    instruction : Instruction # the instruction
    history : List # the history of messages

    #Flag to indicate to the model if it needs to compute an additional agent steps to have a proper model answer to the user. 
    flag_answer_to_user : bool #(False only if you want to simulate an interuption)

    def __init__(
            self,
            memory : Dict[str, Any],
            attributes : Optional[Dict[str, Any]] = None,
            instruction : Optional[Instruction] = None,
            flag_answer_to_user : bool = False,
            history : List = []
        ) -> None:
        super().__init__()
        if not isinstance(memory, dict):
            raise TypeError(f"Situation.memory must be a dict, got {type(memory).__name__}")
        self.memory = memory
        self.attributes = {} if attributes is None else attributes
        self.instruction = EmptyInstruction() if instruction is None else instruction
        self.history = history        
        self.flag_answer_to_user = flag_answer_to_user

    def copy_with(
        self,
        *,
        memory: Optional[Dict[str, Any]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        instruction: Optional[Instruction] = None,
        flag_answer_to_user : Optional[bool] = None,
        history: Optional[List] = None
    ) -> Situation:
        """
        Create a copy of the situation and override some specific elements
        """
        return Situation(
            memory = deepcopy(memory) if memory is not None else deepcopy(self.memory),
            attributes = deepcopy(attributes) if attributes is not None else deepcopy(self.attributes),
            instruction = instruction if instruction is not None else self.instruction,
            history = deepcopy(history) if history is not None else deepcopy(self.history),
            flag_answer_to_user=flag_answer_to_user if flag_answer_to_user is not None else self.flag_answer_to_user
        )
    
    def mix_with(
            self,
            *,
            new_memory : Optional[Dict[str, Any]]=None,
            new_history :  Optional[List]=None,
            new_query : Optional[Instruction]=None,
            new_attributes: Optional[Dict[str, Any]] = None,
    ) -> Situation:
        """
        Define a custom implementation to mix a situation with modified field.
        By default, it creates a copy of the current situation with overriden fields.
        """
        if isinstance(self.instruction, EmptyInstruction) and new_query is None:
            raise RuntimeError("Trying to create a new situation from a situation with an empty Instruction, without overriding it.\nIt's probably due because the n-1 steps has flag_answer_to_user to true and the current step start with an empty Instruction.")
        
        return self.copy_with(
            memory=new_memory,
            history=new_history,
            instruction=new_query,
            attributes=new_attributes,
        )
    
    def apply_attributes_modif(self, modif_list : List[Tuple[Literal["ADD","REMOVE"],Tuple[str,str]]]):
        """
        Apply to the current situation some attributes modification according to modif_list.
        WHich each element must be a tuple like ("ADD"|"REMOVE",(attributes_field_name,attributes_val))
        Supporting only list modification for now.
        """
        apply_att_modif(self.attributes, modif_list)
    
    def verify_robot_name(self, env_agents : List[str]) -> bool:
        for att_values in self.attributes.values():
            if att_values == env_agents:
                return True
        
        if self.attributes.get("known_robots",None) is not None:
            raise TypeError(f"Fail to verify that attributes {self.attributes} contains correct robots name {env_agents}")
        
        self.attributes["known_robots"] = env_agents
        return True

    def to_dict(self):
        return {
            "instruction" : {
                "content" : self.instruction.get_content(),
                "author" : self.instruction.get_role()
            },
            "memory" : self.memory,
            "attributes" : self.attributes,
            "history": self.history
        }
