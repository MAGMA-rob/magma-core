# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..registry import ExternalRequestType
from .base_payload import BasePayload
from .coaching_common import format_memory, format_tool_names


class BadCallDiagnosticPayload(BasePayload):
    """
    Payload to diagnose an invalid or inconsistent tool call.
    """

    def __init__(
        self,
        stage_description: str,
        model_answer: Dict[str, Any],
        interaction_history: str,
        memory: List[Any],
        tools: List[Dict[str, Any]],
        attributes: Dict[str, Any],
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(ExternalRequestType.BAD_CALL_DIAGNOSE, id, max_tokens, model)

        self.description = stage_description
        self.answer = model_answer
        self.interaction_history = interaction_history
        self.memory = format_memory(memory)
        self.tools = tools
        self.attributes = attributes
        self.keep_message = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "interaction_history": self.interaction_history,
            "memory": self.memory,
            "answer": self.answer,
            "attributes": self.attributes,
            "tools": format_tool_names(self.tools),
        }


class BadCallFixPayload(BasePayload):
    """
    Payload to generate the corrected answer after a bad-call diagnosis.
    """

    def __init__(
        self,
        query : str,
        old_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        attributes: Dict[str, Any],
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(ExternalRequestType.BAD_CALL_FIX, id, max_tokens, model)

        self.old_messages = old_messages
        self.tools = tools
        self.attributes = attributes
        self.query = query

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attributes": self.attributes,
            "old_messages": self.old_messages,
            "tools": self.tools,
            "query" : self.query,
        }


class BadCallRecoveryPayload(BasePayload):
    """
    Payload to generate the immediate recovery answer after a bad tool call.
    """

    def __init__(
        self,
        old_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        attributes: Dict[str, Any],
        failed_call: Dict[str, Any],
        failure_message: str,
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(ExternalRequestType.BAD_CALL_RECOVERY, id, max_tokens, model)

        self.old_messages = old_messages
        self.tools = tools
        self.attributes = attributes
        self.failed_call = failed_call
        self.failure_message = failure_message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attributes": self.attributes,
            "old_messages": self.old_messages,
            "tools": self.tools,
            "failed_call": self.failed_call,
            "failure_message": self.failure_message,
        }
