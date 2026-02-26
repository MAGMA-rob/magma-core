# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

CORRECTION_PROMPT = """\
You need now to propose a correction of this bad step by the model.

IMPORTANT CONSTRAINTS:
- Reuse your previous diagnostic to build a better answer.
- Propose the MINIMAL correction necessary.
- Do NOT introduce new tools or assumptions.
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

REQUIRED OUTPUT FORMAT (STRICT):

A Json with the exact same format as the model answer.
"""
