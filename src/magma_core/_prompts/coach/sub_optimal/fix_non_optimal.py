# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

FIX_SUBOPTI_PROMPT = """
You need now to propose a correction of the bad step you identified.

IMPORTANT CONSTRAINTS:
- Reuse your previous diagnostic to build a better answer.
- Propose the MINIMAL correction necessary.
- Do NOT introduce new assumptions.
- Do not mentions "stage objective" or "task objective". The model must only reason about precedent instruction and memory to complete user goal.

ORIGINAL_QUERY : {original_query}
CURRENT_QUERY: {query}
CURRENT_MEMORY: {memory}

ORIGINAL MODEL ANSWER:
{model_answer}

REPAIR INSTRUCTIONS:
Using ONLY the diagnosed error above
Propose a fixed model answer keeping the exact same format as the original.
DO NOT mentions the stage objective. They serve just to help you determine what is the problem in this trajectory.
DO NOT say "the stage requires...", "the stage is...". Keep a reasoning close from the original.
If you need to select a tool call, you can only select one per step. Not multiple tool call.

The corrected step must:
- Preserve task success
- Reduce total steps
- Avoid unnecessary future commitments

You MUST do FORWARD CORRECTION:
- The original trajectory successfully completes the task, although suboptimally.
- Your correction will be RE-EXECUTED from this step onward. The future steps will NOT be reused verbatim.
- Therefore:
  - Do NOT try to preserve or edit the remaining trajectory.
  - Do NOT reason about redundancy with future steps.
  - Focus only on producing the best next action from the CURRENT state.
- If the current step is redundant or incorrect:
  - Replace it with the next meaningful action that should be executed NOW.
  - This action should naturally lead to a shorter trajectory when re-executed.
- The corrected step must be locally optimal and sufficient to continue the task correctly.

REQUIRED OUTPUT FORMAT (STRICT):

A Json with the format: {format}.
"""
