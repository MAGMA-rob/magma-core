# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from enum import IntEnum
from typing import Dict

import magma_core._prompts as prompts

class ExternalRequestType(IntEnum):
    JUDGE = 0
    SIMULATE_USER = 1
    SUBOPTIMAL_DIAGNOSE_DUAL = 2
    SUBOPTIMAL_DIAGNOSE_SINGLE = 3
    SUBOPTIMAL_FIX = 4
    EVALUATE_LEAF = 5
    COMPARE_LEAF = 6
    FAILURE_TEXT_ONLY_FIX = 7
    PARAPHRASING = 8
    BUILD_PLANNER_RECOVERY = 9
    INFORM_USER = 10
    CLEAN_MEMORY = 11
    GEN_MEMORY = 12
    EVALUATE_MEM_LEAF = 13
    GEN_INSTRUCTION = 14
    FAILURE_DIAGNOSE_SINGLE = 15
    FAILURE_DIAGNOSE_DUAL = 16
    FAILURE_FIX_SINGLE = 17
    FAILURE_FIX_DUAL = 18
    RECOVERY_DIAGNOSE = 19
    RECOVERY_FIX = 20
    FORMAT_FIX = 21
    BAD_CALL_DIAGNOSE = 22
    BAD_CALL_FIX = 23
    BAD_CALL_RECOVERY = 24
    MISSING_ACTION_DIAGNOSE = 25
    MISSING_ACTION_FIX = 26
    FAILURE_TEXT_ONLY_ANSWER_DIAGNOSE = 27
    FAILURE_TEXT_ONLY_ANSWER_FIX = 28


PROMPT_REGISTRY: Dict[ExternalRequestType, str] = {
    ExternalRequestType.JUDGE: prompts.BENCHMARK_JUDGE_PROMPT,
    ExternalRequestType.SIMULATE_USER: prompts.SIMULATE_USER_PROMPT,
    ExternalRequestType.SUBOPTIMAL_DIAGNOSE_SINGLE: prompts.DIAGNOSIS_SINGLE_PROMPT,
    ExternalRequestType.SUBOPTIMAL_DIAGNOSE_DUAL: prompts.DIAGNOSIS_DUAL_PROMPT,
    ExternalRequestType.SUBOPTIMAL_FIX: prompts.FIX_SUBOPTI_PROMPT,
    ExternalRequestType.EVALUATE_LEAF: prompts.EVALUATE_LEAF,
    ExternalRequestType.COMPARE_LEAF: prompts.COMPARE_LEAF,
    ExternalRequestType.FAILURE_TEXT_ONLY_FIX: prompts.FIX_TEXT_ONLY,
    ExternalRequestType.PARAPHRASING: prompts.PARAPHRASING,
    ExternalRequestType.BUILD_PLANNER_RECOVERY: prompts.BUILD_PLANNER_RECOVERY,
    ExternalRequestType.INFORM_USER: prompts.INFORM_USER,
    ExternalRequestType.CLEAN_MEMORY: prompts.CLEAN_MEMORY,
    ExternalRequestType.GEN_MEMORY: prompts.CORRECT_MEMORY,
    ExternalRequestType.EVALUATE_MEM_LEAF: prompts.MEMORY_LEAF,
    ExternalRequestType.GEN_INSTRUCTION: prompts.FROM_TEMPLATE,
    ExternalRequestType.FAILURE_DIAGNOSE_SINGLE: prompts.DIAGNOSIS_FAILURE_PROMPT,
    ExternalRequestType.FAILURE_DIAGNOSE_DUAL: prompts.DIAGNOSIS_FAILURE_PROMPT,
    ExternalRequestType.FAILURE_FIX_SINGLE: prompts.FIX_FAILURE_PROMPT,
    ExternalRequestType.FAILURE_FIX_DUAL: prompts.FIX_FAILURE_PROMPT,
    ExternalRequestType.RECOVERY_DIAGNOSE: prompts.DIAG_UNDER_UNCERTAINTY,
    ExternalRequestType.RECOVERY_FIX: prompts.FIX_RECOVERY,
    ExternalRequestType.FORMAT_FIX: prompts.FIX_FORMAT,
    ExternalRequestType.BAD_CALL_DIAGNOSE: prompts.DIAG_BAD_CALL,
    ExternalRequestType.BAD_CALL_FIX: prompts.FIX_ACTION,
    ExternalRequestType.BAD_CALL_RECOVERY: prompts.RECOVER_ACTION,
    ExternalRequestType.MISSING_ACTION_DIAGNOSE: prompts.DIAG_MISSING_ACTION,
    ExternalRequestType.MISSING_ACTION_FIX: prompts.FIX_MISSING_ACTION,
    ExternalRequestType.FAILURE_TEXT_ONLY_ANSWER_DIAGNOSE: prompts.DIAG_TEXT_ONLY_ANSWER,
    ExternalRequestType.FAILURE_TEXT_ONLY_ANSWER_FIX: prompts.FIX_TEXT_ONLY_ANSWER,
}
