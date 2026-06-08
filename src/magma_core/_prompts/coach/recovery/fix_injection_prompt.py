FIX_RECOVERY="""
You now need to propose a correction for the faulty step you identified.

IMPORTANT CONSTRAINTS:
- Reuse your previous diagnosis to build a better answer. 
- Do NOT introduce new assumptions.
- Do not mention "stage objective" or "task objective". The model must only reason about the instruction and memory.

ORIGINAL_QUERY :
{original_query}

INTERACTION:
{interaction}

CURRENT_MEMORY:
{memory}

ORIGINAL MODEL ANSWER:
{model_answer}

REPAIR INSTRUCTIONS:
Using ONLY the diagnosed error above,
propose a corrected model answer keeping the EXACT same format as the original.

DO NOT:
- Redesign the full plan
- Add extra steps
- Justify using hidden objectives
- Modify previous steps

If you need to select a tool call, you can only select ONE.

────────────
CORRECTION PRINCIPLE
────────────

{specific_part}

You MUST assume:
- The corrected step will be RE-EXECUTED from this point onward
- The future trajectory will be recomputed (NOT reused)

Therefore:
- Do NOT reason about future steps
- Do NOT try to align with the original continuation
- Focus only on the BEST next action from the CURRENT state

The corrected step must:
- Restore a valid path toward task completion
- Avoid irreversible mistakes
- Be locally optimal given the current state

Think of this as re-planning from the current step after removing the causal error.

────────────
REQUIRED OUTPUT FORMAT (STRICT)
────────────

A Json with the format: {format}.
"""

BAD_DECISION_PART = """
The original trajectory FAILS because of a wrong decision at this step.

Your correction must:
- Fix this decision so that the task becomes achievable again
- Keep all previous steps unchanged
- Be consistent with the current state (memory + observations)

If the step introduced a wrong object / target / constraint:
- Replace it with the correct one

If a precondition was missing:
- Replace the current action with the action that satisfies this precondition
- Do NOT attempt the original action yet
"""

BAD_RECOVERY_PART = """
The original trajectory FAILS because of a wrong recovery from an inevitable execution errors.

Your correction must:
- Fix the answer so that the agent still try to complete the task correctly
- Keep all previous steps unchanged
- Be consistent with the current state (observation + memory)

Do not forget to acknowledge in the 'say' that the execution failed ("X failed due to Y, I am doing Z now").
If the previous action failed, do NOT repeat the same action with identical parameters.
"""