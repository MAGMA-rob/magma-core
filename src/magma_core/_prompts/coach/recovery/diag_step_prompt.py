DIAG_UNDER_UNCERTAINTY="""
You are analyzing an agent trajectory under execution uncertainty.

The trajectory failed or timed out.

Your goal:
Identify the EARLIEST CAUSAL STEP that must be fixed for the task to succeed.

────────────
CORE PRINCIPLE
────────────

Reason in terms of CONSTRAINT VIOLATIONS, not just wrong actions.

Each step must be evaluated along 3 dimensions:
1. GOAL ALIGNMENT — does the action move toward the task goal?
2. REQUIRED CONDITIONS — were necessary conditions satisfied BEFORE acting?
3. EXECUTION — did the action fail due to environment uncertainty?

────────────
IMPORTANT DISTINCTIONS
────────────

- REQUIRED CONDITIONS include only what can be known BEFORE acting:
  (e.g., known object location, visible state, known dependencies)

- If a failure (e.g., unreachable) is revealed ONLY AFTER attempting the action (e.g not KNOWN by the agent),
  it must be treated as EXECUTION uncertainty, NOT a planning error.

- The TASK CONTEXT contain privileged information about active errors not available to the agent.
  Do NOT assume the agent knew this when evaluating earlier steps.

────────────
ERROR TYPES
────────────

- DECISION_ERROR:
  Wrong target/object/action for the goal OR A required condition (knowable before acting) was not satisfied.

- RECOVERY_ERROR:
  Failure to adapt AFTER an execution failure (e.g., repeating the same failing action).

────────────
RULES
────────────

- Do NOT classify a step as error if it only fails due to execution uncertainty.
- Only classify an error if:
  - the action is wrong under constraints and task objective (DECISION_ERROR), or
  - one action is missing (DECISION_ERROR), or
  - the agent fails to adapt after failure (RECOVERY_ERROR).

Hint:
Do not stop at the first failure.
If a failure is due to execution uncertainty, the root cause is often the next step,
where the agent either adapts correctly or fails to recover.

────────────
CAUSALITY RULE
────────────

Find the FIRST step such that:
- All previous steps are correct
- This step introduces a violation making success impossible or unlikely

Use counterfactual reasoning:
"If this step were correct, the task could succeed."

────────────
INPUTS
────────────

TASK CONTEXT:
{task_description}

INSTRUCTION:
{instruction}

MEMORY:
{memory}

TRAJECTORY:
{trajectory}

Steps with index 0 are valid and cannot be selected.

────────────
OUTPUT FORMAT (STRICT JSON)
────────────

{{
  "error_type": "DECISION_ERROR | RECOVERY_ERROR",
  "root_cause_step": <int>,
  "constraint_violated": "<short description>",
  "explanation": "<why this is the earliest causal error>",
  "minimal_fix": "<what should change at this step ONLY>"
}}
"""