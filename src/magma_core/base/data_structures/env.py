# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from dataclasses import dataclass
from typing import Dict, List, Union

from .answer import ToolCall, MultipleToolCall

@dataclass
class EnvCreationInfos:

    env_id : int 
    env_state : Dict
    logs : List
    stage_log_length : int
    stage_id : int

    tool_call : Union[ToolCall, MultipleToolCall]
    node_id :  int

@dataclass
class WaitingTool:

    id : int
    tool_call : Union[ToolCall, MultipleToolCall]