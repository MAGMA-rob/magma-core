DIAGNOSIS_DUAL_PROMPT = """\
You are analyzing a completed agent trajectory.

The trajectory SUCCESSFULLY completed the user instruction, but required MORE steps than the target.
This means that at least one earlier step introduced an unnecessary error, detour, or premature commitment, which the agent later compensated for.

Your task:
Identify the EARLIEST step that SHOULD BE MODIFIED so that the later corrective behavior would not be necessary.

You must NOT:
- Identify repair or detour steps
- Propose additional steps
- Redesign the whole plan
- Use stage assumptions unless explicitly present

You must:
- Work backward from the final successful outcome
- Use later steps as evidence of what went wrong earlier
- Select ONE step to modify (action or memory)

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
- ACTION (tool + arguments)
- MEMORY UPDATE (add/remove)

────────────
ANALYSIS GUIDELINES
────────────

A step should be selected if:
- Its arguments were later implicitly corrected (e.g bad tool call wich was corrected after)
- It committed to a plan that was later contradicted (e.g taking an object without having depose the precedent)
- It skipped a prerequisite that was later enforced (e.g forgetting to open a drawer before putting inside)
- It forced the agent to "fix" something afterward

Do NOT penalize:
- Correct but partial progress
- Necessary exploration
- Valid memory summaries

Be aware that the model can select only one tool call at each step. ALso, without explicit mentionsin the task description, the robot can only take ONE object at the time. Taking another object while holding something is suboptimal because the first object will be dropped.

IMPORTANT CAUSALITY RULE:

When selecting the step to modify, assume that ALL PREVIOUS STEPS
(before the selected step) are FIXED and cannot be changed.

Select the earliest step such that:
- Keeping all previous steps unchanged,
- Modifying THIS step alone would remove the need for later corrective behavior.

Consider all steps that have index 0 valid. You can only select positive index.

# ────────────
# OUTPUT FORMAT (STRICT)
# ────────────

A simple json : {format}
"""

DIAGNOSIS_SINGLE_PROMPT = """\
You are analyzing a completed agent trajectory.

The trajectory SUCCESSFULLY completed the user instruction, but required MORE steps than the target.
This means that at least one earlier step introduced an unnecessary error, detour, or premature commitment, which the agent later compensated for.

Your task:
Identify the EARLIEST step that SHOULD BE MODIFIED so that the later corrective behavior would not be necessary.

You must NOT:
- Identify repair or detour steps
- Propose additional steps
- Redesign the whole plan
- Use stage assumptions unless explicitly present

You must:
- Work backward from the final successful outcome
- Use later steps as evidence of what went wrong earlier
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
- ACTION (tool + arguments)

────────────
ANALYSIS GUIDELINES
────────────

A step should be selected if:
- Its arguments were later implicitly corrected (e.g bad tool call wich was corrected after)
- It committed to a plan that was later contradicted (e.g taking an object without having depose the precedent)
- It skipped a prerequisite that was later enforced (e.g forgetting to open a drawer before putting inside)
- It forced the agent to "fix" something afterward

Do NOT penalize:
- Correct but partial progress
- Necessary exploration

Be aware that the model can select only one tool call at each step. ALso, without explicit mentionsin the task description, the robot can only take ONE object at the time. Taking another object while holding something is suboptimal because the first object will be dropped.

IMPORTANT CAUSALITY RULE:

When selecting the step to modify, assume that ALL PREVIOUS STEPS
(before the selected step) are FIXED and cannot be changed.

Select the earliest step such that:
- Keeping all previous steps unchanged,
- Modifying THIS step alone would remove the need for later corrective behavior.

Consider all steps that have index 0 valid. You can only select positive index.

# ────────────
# OUTPUT FORMAT (STRICT)
# ────────────

A simple json : {format}
"""
