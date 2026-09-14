# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.simulation.data_structures.situation import (
    Situation, SituationInit, StageInput,
    Instruction,
    UserInstruction, EmptyInstruction, StatusReturn, TemplateInstruction
)
from magma_core.simulation.data_structures.tools import (
    ToolExecution, ToolResult,
    StageSuccess, StageState, RobotToolContext,
    Log, Trajectory, Point, ToolErrorSupport, ToolErrorFlag, EnvStateUpdate,
    ToolBatchContext,
)
from magma_core.simulation.data_structures.env import (
    EnvCreationInfos,
    SavedEnvData,
    ActiveStageErrorState,
    EnvToolContext,
)

from magma_core.simulation.data_structures.observation import Observation

from magma_core.domain.agent_call import (
    Call, ValidExecutionReq, InvalidExecutionReq, 
    EmptyToolCall, ExecutionRequest
)

from magma_core.simulation.data_structures.executor_agent_link import (
    ToolStatus, RobotToolStatus
)

__all__ = [
    # Generic
    "Trajectory",
    "Point",

    # answer
    "Call",
    "ValidExecutionReq",
    "ExecutionRequest",
    "InvalidExecutionReq",

    # situation
    "Situation",
    "SituationInit",
    "StageInput",
    "Instruction",
    "UserInstruction",
    "EmptyInstruction",
    "StatusReturn",
    "TemplateInstruction",

    # tools
    "ToolExecution",
    "EnvToolContext",
    "ToolResult",
    "ToolStatus",
    "StageSuccess",
    "StageState",
    "EmptyToolCall",
    "RobotToolContext",
    "Log",
    "ToolErrorSupport",
    "ToolErrorFlag",
    "EnvStateUpdate",
    "ToolBatchContext",

    # env
    "EnvCreationInfos",
    "SavedEnvData",
    "ActiveStageErrorState",

    # observation
    "Observation",
]
