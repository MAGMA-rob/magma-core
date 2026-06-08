# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

DIAG_TEXT_ONLY_ANSWER = """\
You are analyzing a FAILED text-only stage trajectory.

In this stage, the agent may use tools before giving the final answer.
The trajectory does NOT satisfy the user instruction.

Your task:
Identify the EARLIEST model step that CAUSED the bad final answer or the
terminal failure.

IMPORTANT:
- The failure may be visible only in the final spoken answer.
- The first causal error may still be an earlier tool call.
- Select ONE step to modify.

You must NOT:
- Select a later symptom if an earlier tool choice caused it.
- Redesign the whole plan.
- Propose several fixes.
- Use hidden assumptions beyond the visible instruction, memory, tools, and trajectory.

TASK CONTEXT:
{description}

AVAILABLE TOOLS:
{tools}

INPUTS

ORIGINAL USER INSTRUCTION:
{instruction}

ORIGINAL MEMORY:
{memory}

FULL TRAJECTORY (ordered):
Old states have index 0 and cannot be selected as the error step.
{trajectory}

Each step may contain:
- USER or SYSTEM query visible at that step
- SAY content
- ACTION tool call
- ANSWER final spoken answer when no tool is called

ANALYSIS GUIDELINES

Step with index 0 must be considered valid. You cannot select it.

Select a step if:
- It uses the wrong tool or wrong tool arguments.
- It skips a necessary tool before answering.
- It commits to a wrong fact, object, or constraint.
- It gives an incorrect final answer despite having enough information.

Do NOT select steps that:
- Only reveal the failure.
- Are reasonable given an earlier wrong decision.

CAUSALITY RULE:
Assume all previous selected-index steps are fixed.
Choose the earliest step such that correcting only this step could put the
stage back on a valid path.

Include in your answer:
- The selected step.
- A short explanation of why it is the first causal error.
- What should have been done instead.

OUTPUT FORMAT (STRICT)

A simple json: {format}
"""
