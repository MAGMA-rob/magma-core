# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

DIAGNOSIS_FAILURE_PROMPT = """\
You are analyzing a FAILED agent trajectory.

The trajectory does NOT complete the user instruction and ends in an incorrect state.

Your task:
Identify the EARLIEST decision that CAUSED the failure.

IMPORTANT:
- The failure is often visible at the LAST step, but its CAUSE is usually EARLIER.
- You must identify the FIRST step that makes the failure unavoidable.

You must NOT:
- Select the final failing step if an earlier decision caused it
- Focus on symptoms instead of causes
- Redesign the whole plan
- Propose additional steps

You must:
- Reason about causality (not just correctness)
- Ask: “At which step did the trajectory irreversibly diverge from a successful path?”
- Select ONE step to modify

TASK CONTEXT:
{task_description}

────────────
INPUTS
────────────

ORIGINAL USER INSTRUCTION:
{instruction}

ORIGINAL MEMORY:
{memory}

FULL TRAJECTORY (ordered):
Old states (you can not select them as error step) have index to 0.
{trajectory}

Each step contains:
- QUERY (instruction visible at that step)
- ACTION (tool + arguments)
- Sometimes MEMORY UPDATE (only in dual-agent traces)

────────────
ANALYSIS GUIDELINES
────────────

Step with index 0 must be considered as VALID. You can not select them.

A step should be selected if:
- It introduces a WRONG object, target, or constraint
- It violates a necessary precondition for success
- It commits to a direction that cannot lead to success anymore

Do NOT select steps that:
- Only reveal the failure (symptoms)
- Are correct given earlier wrong decisions

IMPORTANT CAUSALITY RULE:

Assume ALL PREVIOUS STEPS are FIXED.
If multiple steps seem wrong, prefer the one that explains the largest number of subsequent errors.

Select the earliest step such that:
- Keeping all previous steps unchanged,
- This step alone causes the trajectory to become incorrect or unrecoverable.

Think in terms of counterfactual:
"If this step were correct, the task could still succeed."

Include in your answer:
- The selected step
- A short explanation of why it is the FIRST causal error
- What should have been done instead

# ────────────
# OUTPUT FORMAT (STRICT)
# ────────────

A simple json : {format}
"""
