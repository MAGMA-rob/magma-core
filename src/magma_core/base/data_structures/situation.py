# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from __future__ import annotations
from typing import List, Dict, Any, Literal, Optional, Tuple
from abc import ABC, abstractmethod
from copy import deepcopy
import json

from magma_core.utils.data_utils import apply_att_modif

class Instruction(ABC):

    content : str
    role : str
    timestamp : int

    def __init__(
            self,
            content : str,
            role : Literal["user","model","status"],
            timestamp : int = 0,
        ) -> None:
        self.content = content
        self.role = role
        self.timestamp = timestamp

    def get_content(self):
        return self.content
    
    def get_role(self):
        return self.role
    
    def get_timestamp(self):
        return self.timestamp

class UserInstruction(Instruction):

    def __init__(self, content: str, timestamp: int = 0) -> None:
        super().__init__(content, "user", timestamp)
    
class StatusReturn(Instruction):

    def __init__(self, status: Dict, timestamp: int = 0) -> None:
        super().__init__(json.dumps(status), "status", timestamp)

class EmptyInstruction(Instruction):

    def __init__(self) -> None:
        pass

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
        super().__init__(str(template), "user", timestamp)

    def set_content(self, text : str):
        self.content = text

    def get_content(self):
        if self.content is None:
            return str(self.template)
        return self.content

class Situation(ABC):
    """
    Base class to define the initial situation of a task : memory, instruction, attributes...
    """

    memory : List[str] # initial memory
    preserved_memory_indices : List[int] # initial static memory indices
    attributes : Dict[str, Any] # task attributes
    instruction : Instruction # the instruction
    history : Optional[List] # the history of messages

    user_scenario : str # the user scenario describing what the simulated user must answer to the model to guide it. Optional
    
    #Flag to indicate to the model if it needs to compute an additional agent steps to have a proper model answer to the user. 
    flag_answer_to_user : bool #(False only if you want to simulate an interuption)

    def __init__(
            self,
            memory : List[str],
            preserved_memory_indices : List[int],
            attributes : Dict[str, Any],
            instruction : Instruction,
            flag_answer_to_user : bool,
            user_scenario : str = "none",
            history : Optional[List] = None
        ) -> None:
        super().__init__()
        self.memory = memory
        self.preserved_memory_indices = preserved_memory_indices
        self.attributes = attributes
        self.instruction = instruction
        self.user_scenario = user_scenario
        self.history = history        
        self.flag_answer_to_user = flag_answer_to_user

    def copy_with(
        self,
        *,
        memory: Optional[List[str]] = None,
        preserved_memory_indices: Optional[List[int]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        instruction: Optional[Instruction] = None,
        flag_answer_to_user : Optional[bool] = None,
        user_scenario: Optional[str] = None,
        history: Optional[List] = None
    ) -> Situation:
        """
        Create a copy of the situation and override some specific elements
        """
        
        return Situation(
            memory = deepcopy(memory) if memory is not None else deepcopy(self.memory),
            preserved_memory_indices = (
                deepcopy(preserved_memory_indices)
                if preserved_memory_indices is not None
                else deepcopy(self.preserved_memory_indices)
            ),
            attributes = deepcopy(attributes) if attributes is not None else deepcopy(self.attributes),
            instruction = instruction if instruction is not None else self.instruction,
            user_scenario = (
                user_scenario
                if user_scenario is not None
                else self.user_scenario
            ),
            history = deepcopy(history) if history is not None else deepcopy(self.history),
            flag_answer_to_user=flag_answer_to_user if flag_answer_to_user is not None else self.flag_answer_to_user
        )
    
    def mix_with(
            self,
            *,
            new_memory : Optional[List[str]]=None,
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
            "preserved_memory_indices" : self.preserved_memory_indices,
            "attributes" : self.attributes,
            "history": self.history
        }
    
    def memory_to_memorizer_format(self) -> str:
        memory = "Memory:\n"
        for j, mem in enumerate(self.memory):
            if j in self.preserved_memory_indices:
                id = "X"
            else:
                id = str(j)
            memory += f"[{id}] {mem}\n"
        return memory

    

class InterruptSituation(Situation):
    """
    An interupt situation is a Situation you can use to specify that you want to keep the query of the situation and not override it.
    Like this, you can use it inside a stage and avoid the automatic query override.
    """

    def mix_with(
            self,
            *,
            new_memory : Optional[List[str]]=None,
            new_history :  Optional[List]=None,
            new_query : Optional[Instruction]=None,
            new_attributes: Optional[Dict[str, Any]] = None,
    ) -> Situation:
        return super().mix_with(new_memory=new_memory, new_history=new_history, new_attributes=new_attributes)