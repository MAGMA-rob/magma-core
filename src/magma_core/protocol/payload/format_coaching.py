# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
from typing import Any, Dict, List, Optional

from ..registry import ExternalRequestType
from .base_payload import BasePayload
from .coaching_common import format_memory, format_tool_names


class FormatFixPayload(BasePayload):
    """
    Payload to repair a malformed agent answer into an executable answer.
    """

    def __init__(
        self,
        stage_description: str,
        user_instruction: str,
        memory: List[Any],
        tools: List[Dict[str, Any]],
        attributes: Dict[str, Any],
        invalid_answer: Any,
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(ExternalRequestType.FORMAT_FIX, id, max_tokens, model)
        self.description = stage_description
        self.instruction = user_instruction
        self.memory = format_memory(memory)
        self.tools = format_tool_names(tools)
        self.attributes = attributes
        self.invalid_answer = invalid_answer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "instruction": self.instruction,
            "memory": self.memory,
            "tools": self.tools,
            "attributes": self.attributes,
            "invalid_answer": self.invalid_answer,
        }
