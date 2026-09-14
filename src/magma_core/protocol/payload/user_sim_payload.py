# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Any, ClassVar, Dict, List, Optional

from magma_core._prompts.userSim.judge_prompt import BENCHMARK_JUDGE_PROMPT
from magma_core._prompts.userSim.paraphrasing_prompt import FROM_TEMPLATE, PARAPHRASING

from .base_payload import BasePayload

class JudgePayload(BasePayload):
    """
    Payload to ask an external model to judge a model answer based on a rule.
    """  

    prompt_template: ClassVar[str] = BENCHMARK_JUDGE_PROMPT
    debug_log: ClassVar[bool] = True
    
    def __init__(
        self,
        rule : str,
        model_answer : str,
        id : int,
        question: str = "",
        max_tokens=2000,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(id, max_tokens, model)
        self.question = question
        self.rule = rule
        self.model_answer = model_answer
        self.exp = "{{\"explanation\": <very short reasoning>, \"verdict\": true or false}}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exp" : self.exp,
            "question" : self.question,
            "rule" : self.rule,
            "model_answer" : self.model_answer
        }
    
class ParaphrasingPayload(BasePayload):
    """
    Payload to generate variation for user instruction, allowing more robust data
    """

    prompt_template: ClassVar[str] = PARAPHRASING

    def __init__(self, 
            queries : List[str],
            real_conversation: List[Dict],
            previous_variant : List[str],
            id: int, max_tokens=5000, model = None) -> None:
        super().__init__(id, max_tokens, model)

        self.template_answer = queries[-1]
        self.previous_variants = previous_variant[:5]
        self.template_queries = ""
        for q in queries:
            self.template_queries += f"user: {q}\n"
        
        self.conversation = ""
        for conv in real_conversation:
            if conv['author'] == "SYSTEM":
                continue
            self.conversation += f"{conv['author']}: {conv['content']}\n"        

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_answer": self.template_answer,
            "user_sequence": self.template_queries,
            "real_conversation": self.conversation,
            "previous_variants": self.previous_variants
        }

class GenInstructionPayload(BasePayload):
    """
    Payload to generate an instruction from a template
    """

    prompt_template: ClassVar[str] = FROM_TEMPLATE

    def __init__(self, 
            template : Dict,
            context : str,
            last_model_answer: str,
            previous_variant : List[str],
            id: int, max_tokens=5000, model = None) -> None:
        super().__init__(id, max_tokens, model)

        self.previous_variants = previous_variant[:5]
        self.template = template
        self.context = context
        self.last_model_answer = last_model_answer    

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template": self.template,
            "context": self.context,
            "last_model_answer": self.last_model_answer,
            "previous_variants": self.previous_variants
        }
