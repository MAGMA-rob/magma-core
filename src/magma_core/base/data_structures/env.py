# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .answer import ToolCall, MultipleToolCall

ActiveStageErrorState = Dict[str, Optional[Dict[str, Any]]]

@dataclass
class SavedEnvData:
    """Saved per-node runtime data used by generation executors."""

    env_state: Dict
    logs: List
    stage_id: int
    stage_log_length: int
    composite_progress: Dict[str, Any] = field(default_factory=dict)
    active_stage_error_state: ActiveStageErrorState = field(default_factory=dict)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

@dataclass
class EnvCreationInfos:

    env_id : int 
    env_state : Dict
    logs : List
    stage_log_length : int
    stage_id : int

    tool_call : Union[ToolCall, MultipleToolCall]
    node_id :  int
    composite_progress : Dict[str, Any] = field(default_factory=dict)
    active_stage_error_state: ActiveStageErrorState = field(default_factory=dict)

@dataclass
class WaitingTool:

    id : int
    tool_call : Union[ToolCall, MultipleToolCall]
