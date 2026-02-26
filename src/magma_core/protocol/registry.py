# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from enum import IntEnum
from typing import Dict
import magma_core._prompts as prompts

class ExternalRequestType(IntEnum):
    JUDGE = 0
    SIMULATE_USER = 1
    DIAGNOSTIC_DUAL_SUBOPTIMAL = 2
    DIAGNOSTIC_SINGLE_SUBOPTIMAL = 3
    FIX_SUBOPTIMAL = 4
    EVALUATE_LEAF = 5
    COMPARE_LEAF = 6
    FIX_TEXT_ONLY_STAGE = 7
    PARAPHRASING = 8
    REPEAT = 9
    INFORM_USER = 10
    CLEAN_MEMORY = 11
    GEN_MEMORY = 12
    EVALUATE_MEM_LEAF = 13
    GEN_INSTRUCTION = 14
    DIAGNOSTIC_SINGLE_FAILURE = 15
    DIAGNOSTIC_DUAL_FAILURE = 16
    FIX_SINGLE_FAILURE = 17
    FIX_DUAL_FAILURE = 18

PROMPT_REGISTRY : Dict[ExternalRequestType,str] = {
    ExternalRequestType.JUDGE : prompts.BENCHMARK_JUDGE_PROMPT,
    ExternalRequestType.SIMULATE_USER : prompts.SIMULATE_USER_PROMPT,
    ExternalRequestType.DIAGNOSTIC_SINGLE_SUBOPTIMAL : prompts.DIAGNOSIS_SINGLE_PROMPT,
    ExternalRequestType.DIAGNOSTIC_DUAL_SUBOPTIMAL : prompts.DIAGNOSIS_DUAL_PROMPT,
    ExternalRequestType.FIX_SUBOPTIMAL : prompts.CORRECTION_PROMPT,
    ExternalRequestType.EVALUATE_LEAF : prompts.EVALUATE_LEAF,
    ExternalRequestType.COMPARE_LEAF : prompts.COMPARE_LEAF,
    ExternalRequestType.FIX_TEXT_ONLY_STAGE : prompts.FIX_TEXT_ONLY,
    ExternalRequestType.PARAPHRASING : prompts.PARAPHRASING,
    ExternalRequestType.REPEAT : prompts.BUILD_REPEAT,
    ExternalRequestType.INFORM_USER: prompts.INFORM_USER,
    ExternalRequestType.CLEAN_MEMORY : prompts.CLEAN_MEMORY,
    ExternalRequestType.GEN_MEMORY: prompts.CORRECT_MEMORY,
    ExternalRequestType.EVALUATE_MEM_LEAF: prompts.MEMORY_LEAF,
    ExternalRequestType.GEN_INSTRUCTION : prompts.FROM_TEMPLATE,
    ExternalRequestType.DIAGNOSTIC_SINGLE_FAILURE : prompts.REASON_ABOUT_FAILED_ACTION_SINGLE,
    ExternalRequestType.FIX_SINGLE_FAILURE : prompts.FIX_ACTION
}