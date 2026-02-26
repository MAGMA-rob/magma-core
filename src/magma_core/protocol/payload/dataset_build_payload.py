# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Dict, List, Any, Tuple, Optional, Type
from ..registry import ExternalRequestType
from .base_payload import BasePayload
from magma_core.utils.text_utils import transform_json_to_memorizer_output

class EvaluateLeafPayload(BasePayload):
    """
    Payload to evaluate the quality of a leaf, select or reject.
    """

    def __init__(
            self,
            user_instruction : str,
            task_description : str,
            stage_description : str,
            answer : Dict,
            situations : List[Dict],
            success : bool,
            id: int, max_tokens=5000, model = None) -> None:
        super().__init__(ExternalRequestType.EVALUATE_LEAF, id, max_tokens, model)

        self.user_instruction = user_instruction
        self.task_description = task_description
        self.data = {
            "situations" : situations,
            "answers" : answer
        }
        self.reasoning = answer["think"]
        self.say = answer['say']
        self.stage_description = stage_description
        if success:
            self.failtext = ""
        else:
            self.failtext = "This data has failed. Therefore the answer must inform the user of the failure (Potentially, trying again but it must inform the user about the failure)."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reasoning" : self.reasoning,
            "stage_description" : self.stage_description,
            "task_description" : self.task_description,
            "user_instruction" : self.user_instruction,
            "say" : self.say,
            "failtext" : self.failtext
        }
    
    def get_data(self) -> Dict:
        return self.data
    
class MemoryCoherenceLeafPayload(BasePayload):
    """
    Payload to evaluate the memory coherence of multiple answer and select the best or reject all
    """

    def __init__(
            self,
            answers : List[Dict],
            situation : Dict,
            id: int, max_tokens=5000, model = None) -> None:
        super().__init__(ExternalRequestType.EVALUATE_MEM_LEAF, id, max_tokens, model)
        self.memory_str = ""
        self.think = situation['think']
        for i, mem in enumerate(situation['memory']):
            if i in situation['preserved_memory_indices']:
                idx = "X"
            else:
                idx = i
            self.memory_str += f"{idx}. {mem}\n"
        self.answers_str = ""
        for i, ans in enumerate(answers):
            answer_str = transform_json_to_memorizer_output(ans)
            self.answers_str += f"{i}.\n{answer_str}\n\n"
        
        self.answers = answers
        self.situation = situation
    
    def get_data(self) -> Dict:
        return {
            "answers" : self.answers,
            "situation" : self.situation
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answers" : self.answers_str,
            "memory" : self.memory_str,
            "think": self.think
        }
    
class CompareLeafPayload(BasePayload):
    """
    Payload to compare multiple leaf and select one or reject all.
    """

    def __init__(
            self,
            user_instruction : str,
            task_description : str,
            stage_description : str,
            answers : List[Dict],
            success : bool,
            situations : List[Dict],
            id: int, max_tokens=5000, model = None) -> None:
        super().__init__(ExternalRequestType.COMPARE_LEAF, id, max_tokens, model)

        self.user_instruction = user_instruction
        self.task_description = task_description
        self.stage_description = stage_description
        self.answers = ""
        for i,answer in enumerate(answers):
            self.answers += f"[ANSWER {i}]\nReasoning\n{answer['think']}\nUser-visible Answer\n{answer['say']}\n\n"
        self.data = {
            "situations" : situations,
            "answers" : answers
        }
        if success:
            self.failtext = ""
        else:
            self.failtext = "This data has failed. Therefore the answer must inform the user of the failure (Potentially, trying again but it must inform the user about the failure)."


    def to_dict(self) -> Dict[str, Any]:
        return {
            "answers" : self.answers,
            "stage_description" : self.stage_description,
            "task_description" : self.task_description,
            "user_instruction" : self.user_instruction,
        }
    
    def get_data(self) -> Dict:
        return self.data