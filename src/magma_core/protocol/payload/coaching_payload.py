from typing import Dict, List, Any, Tuple, Optional, Type
from ..registry import ExternalRequestType
from .base_payload import BasePayload

class BuildRepeatPayload(BasePayload):
    """
    Payload to retry the same action after a failure
    """

    def __init__(self,
            model_answer : Dict,
            status_return : Dict,
            id: int, max_tokens=2500, model = None) -> None:
        super().__init__(ExternalRequestType.REPEAT, id, max_tokens, model)

        if model_answer:
            self.template = model_answer.copy()
            self.template.pop("action")
        else:
            self.template = "Not Available"

        s = status_return.copy()
        self.tool_call = s.pop("previous_tool_call")
        self.error = s['error']

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template" : self.template,
            "tool_call" : self.tool_call,
            "error_message" : self.error
        }

class InformUserPayload(BasePayload):
    """
    Payload to generate an error answer to inform the user about repetitive failure.
    """

    def __init__(self,
            status_return : Dict,
            is_dual_agent : bool,
            id: int, max_tokens=2500, model = None) -> None:
        super().__init__(ExternalRequestType.INFORM_USER, id, max_tokens, model)

        s = status_return.copy()
        self.tool_call = s.pop("previous_tool_call")
        self.error = s['error']
        self.mem_text = ", using the sentence 'Looking at my memory,'," if is_dual_agent else ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_call" : self.tool_call,
            "error_message" : self.error,
            "is_dual_agent" : self.mem_text
        }
    
class DiagnosticSubOptimalPayload(BasePayload):
    """
    Payload to ask the coach to detect the sub-optimal step inside a trajectory
    """

    def __init__(
            self,
            task_description : str,
            stage_instruction : str,
            stage_memory : List[str],
            trajectory : List[Tuple[str,str,bool]],
            dual_mode : bool,
            id: int, max_tokens=5000, model: Optional[str] = None
        ) -> None:
        if dual_mode:
            t = ExternalRequestType.DIAGNOSTIC_DUAL_SUBOPTIMAL
            self.output_rule = "{{\"error_type\" :  ACTION or MEMORY of the first step which is non-optimal., \"error_step\" : the index of the step, \"error_description\" : Concise explanation of what went wrong.}}"
        else:
            t = ExternalRequestType.DIAGNOSTIC_SINGLE_SUBOPTIMAL
            self.output_rule = "{{\"error_step\" : the index of the step, \"error_description\" : Concise explanation of what went wrong.}}"

        super().__init__(t, id, max_tokens, model)

        self.task_description = task_description
        self.instruction = stage_instruction
        self.memory = stage_memory
        s = ""
        cpt = 0
        for type, content, new in trajectory:            
            if new and type == "QUERY":
                cpt+=1
            s+= f"{type} {cpt}: {content}\n"
  
        self.trajectory = s
        self.keep_message = True # Allows to mark that we want the full message as return

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_description" : self.task_description,
            "instruction" : self.instruction,
            "memory" : self.memory,
            "trajectory" : self.trajectory,
            "format" : self.output_rule
        }
    
class FixTextOnlyStage(BasePayload):
    """
    Paylod to fix Stage where the model must memorize a constraint or answer to an user
    Currently supporting only 1-lenght stage
    """

    def __init__(self, 
                stage_description : str, #Describe the desired answer or the constraint passed
                model_answer : Dict,
                user_instruction : str,
                memory : List,
                id: int, max_tokens=5000, model = None) -> None:
        super().__init__(ExternalRequestType.FIX_TEXT_ONLY_STAGE, id, max_tokens, model)

        self.description = stage_description
        self.answer = model_answer
        self.instruction = user_instruction
        self.memory = ""
        for mem in memory:
            self.memory += f"- {mem}\n"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "description" : self.description,
            "instruction" : self.instruction,
            "memory": self.memory,
            "answer" : self.answer
        }
    
class DiagnosticFailure(BasePayload):
    """
    Paylod to diagnostic a failed action. An action that conducts to a catastrophic end
    """

    def __init__(self, 
                stage_description : str, #Describe the desired answer or the constraint passed
                model_answer : Dict,
                user_instruction : str,
                memory : List,
                tools : List,
                attributes : Dict,
                dual_mode : bool,
                id: int, max_tokens=5000, model = None) -> None:
        if dual_mode:
            raise NotImplementedError()
            d = ExternalRequestType.DIAGNOSTIC_DUAL_FAILURE
        else:
            d = ExternalRequestType.DIAGNOSTIC_SINGLE_FAILURE
        super().__init__(d, id, max_tokens, model)

        self.tools = [t['name'] for t in tools]
        self.attributes = attributes
        self.description = stage_description
        self.answer = model_answer
        self.instruction = user_instruction
        self.memory = ""
        for mem in memory:
            self.memory += f"- {mem}\n"

        self.keep_message = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "description" : self.description,
            "instruction" : self.instruction,
            "memory": self.memory,
            "answer" : self.answer,
            "attributes" : self.attributes,
            "tools" : self.tools
        }

class FixActionStage(BasePayload):
    """
    Paylod to fix Stage after DiagnosticFailure
    """

    def __init__(self, 
                old_messages : List[Dict],
                tools : List,
                attributes : Dict,
                id: int, max_tokens=5000, model = None) -> None:
        super().__init__(ExternalRequestType.FIX_SINGLE_FAILURE, id, max_tokens, model)

        self.old_messages = old_messages
        self.tools = tools
        self.attributes = attributes
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "attributes" : self.attributes,
            "old_messages" : self.old_messages,
            "tools": self.tools
        }
    
class CleanMemoryPayload(BasePayload):
    """
    ayload to ask the coach to clean a memory at the end of stage. Ensuring efficient memory representation.
    """

    def __init__(self, 
                memory : List[str],
                preserved_indices : List[int],
                list_constraints : Optional[List[str]],
                id: int, max_tokens=5000, model = None
            ) -> None:
        if list_constraints is None:
            t = ExternalRequestType.CLEAN_MEMORY
            self.constraints = None
        else:
            t = ExternalRequestType.GEN_MEMORY
            self.constraints = ""
            for c in list_constraints:
                self.constraints += c + "\n"
        super().__init__(t, id, max_tokens, model)

        self.memory_str = ""
        for i, mem in enumerate(memory):
            if i in preserved_indices:
                idx = "X"
            else:
                idx = i
            self.memory_str += f"{idx}. {mem}\n"

    def to_dict(self) -> Dict[str, Any]:
        if self.constraints is None:
            return {
                "memory" : self.memory_str
            }
        
        return {
            "memory" : self.memory_str,
            "constraints" : self.constraints
        }
        
    
class FixSubOptimalPayload(BasePayload):
    """
    Payload to ask the coach to propose a new answer from a sub-optimal answer by the model.
    """

    old_messages : List[Dict]

    def __init__(
            self,
            old_messages : List[Dict],
            ori_query : str, cur_query : str, cur_mem : str, model_answer : str,
            id: int, max_tokens=5000, model = None) -> None:
        super().__init__(ExternalRequestType.FIX_SUBOPTIMAL, id, max_tokens, model)
        self.old_messages = old_messages
        self.original_query = ori_query
        self.current_query = cur_query
        self.current_memory = cur_mem
        self.model_answer = model_answer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "old_messages" : self.old_messages,
            "original_query" : self.original_query,
            "query" : self.current_query,
            "memory" : self.current_memory,
            "model_answer" : self.model_answer
        }