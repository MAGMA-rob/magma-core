FIX_TEXT_ONLY = """
You are rewriting a rejected text-only robot answer.

The corrected answer must be a plain natural-language message to the user.
No tool call is possible for this correction.

Stage goal:
{stage_goal}

Rejected answer:
{rejected_answer}

Rejection reason:
{rejection_reason}

Rules:
- Return only the corrected message string.
- Do not return JSON.
- Do not add markdown fences or labels.
- Do not mention the rejection reason.
- Do not mention validation, formatting, or internal checks.
- Keep any useful content from the rejected answer when it is compatible with the stage goal.
- Rewrite the answer so it satisfies the stage goal and fixes the rejection reason.
"""
