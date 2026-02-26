# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Union, Literal
import numpy as np
import sapien

Trajectory = list[Union[sapien.Pose , Literal["OPEN", "CLOSE", "OK"]]]
""" Trajectory of the gripper.

The trajectory is a list of crossing points or actions.
"""

Point = np.ndarray[Literal[3], np.dtype[np.float32]]

from .answer import ToolCall, CommanderAnswer, BadCommanderAnswer, BadToolCall, MultipleToolCall
from .situation import (
    Situation, InterruptSituation,
    Instruction, UserInstruction, EmptyInstruction, StatusReturn, TemplateInstruction    
)
from .tools import (
    ToolExecution, ToolInfos, ToolResult,  ToolStatus, 
    StageSuccess, StageState, EmptyToolCall, ToolRobot,
    Log
)
from .env import EnvCreationInfos, WaitingTool

__all__ = [
    # Generic
    "Trajectory",
    "Point",

    # answer
    "ToolCall",
    "CommanderAnswer",
    "BadCommanderAnswer",
    "BadToolCall",
    "MultipleToolCall",

    # situation
    "Situation",
    "InterruptSituation",
    "Instruction",
    "UserInstruction",
    "EmptyInstruction",
    "StatusReturn",
    "TemplateInstruction",

    # tools
    "ToolExecution",
    "ToolInfos",
    "ToolResult",
    "ToolStatus",
    "StageSuccess",
    "StageState",
    "EmptyToolCall",
    "ToolRobot",
    "Log",

    # env
    "EnvCreationInfos",
    "WaitingTool",
]
