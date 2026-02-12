from typing import Union, Dict, Optional, List
from dataclasses import dataclass

class ToolCall:
    """
    Encapsulate a tool call by the agent to the system
    """

    name : Union[str, None]
    arguments : Dict
    agent_step_id : int

    source_node_id : int

    def __init__(
            self,
            agent_step_id : int,
            source_node_id : int,
            name : Optional[str] = None,
            arguments : Dict = {},
        ) -> None:
        self.agent_step_id = agent_step_id
        self.source_node_id = source_node_id

        self.name = name
        self.arguments = arguments

    def to_dict(self) -> Dict:
        if self.name:
            return {'name':self.name, 'arguments':self.arguments}
        else:
            return {}
        
    def get_names(self) -> List:
        if self.name: return [self.name]
        return [None]
    
@dataclass    
class BadToolCall:

    action : Dict

    def to_dict(self) -> Dict:
        return self.action

class MultipleToolCall:

    actions : Dict[str,Dict]

    agent_step_id : int
    source_node_id : int

    def __init__(
            self,
            agent_step_id : int,
            source_node_id : int,
            actions_dict : Dict
        ) -> None:
        self.actions = {}
        self.agent_step_id = agent_step_id
        self.source_node_id = source_node_id
        for k,v in actions_dict.items():
            if isinstance(v,Dict):
                n = v.get("name",None)
                if n is not None:
                    self.actions[k] = v
                    continue
            self.actions[k] = {}
        
    def to_dict(self):
        return self.actions

    def get_names(self) -> List:
        return [v.get("name",None) for v in self.actions.values()]

class CommanderAnswer:

    think : str
    say : str
    actions : Union[ToolCall,BadToolCall,MultipleToolCall]

    def __init__(self, think : str, say : str, action_dict : Dict, id : int, level : int) -> None:
        self.think = think
        self.say = say

        if action_dict == {} or action_dict is None:
            self.actions=ToolCall(
                name=None,
                arguments={},
                source_node_id=id,
                agent_step_id=level # no +1 here because its the situation, when created which get the +1
            )
        else:
            name = action_dict.get("name",None)
            arguments = action_dict.get("arguments",{})

            if name == None:
                # Try multiple robot call
                self.actions=MultipleToolCall(
                    agent_step_id=level,
                    source_node_id=id,
                    actions_dict=action_dict
                )
            else:
                self.actions=ToolCall(
                    name=name,
                    arguments=arguments,
                    source_node_id=id,
                    agent_step_id=level # no +1 here because its the situation, when created which get the +1
                )

    def to_dict(self):
        return {
            "think" : self.think,
            "say" : self.say,
            "action" : self.actions.to_dict()
        }
    
    def get_tool_call(self) -> Union[ToolCall,None,MultipleToolCall]:
        """
        Return tool call only if it's a valid tool call
        """
        if not isinstance(self.actions, BadToolCall): return self.actions
        return None


class BadCommanderAnswer:

    answer : Union[str, Dict]

    def __init__(self, think : str, say : str = "", action : str = "") -> None:
        if action == "X":
            self.answer = think
        else:
            self.answer = {"think":think, "say":say, "action":action}

    def to_dict(self):
        return {"bad_answer" : self.answer}