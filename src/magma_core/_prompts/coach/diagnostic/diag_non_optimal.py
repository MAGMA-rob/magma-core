# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

DIAGNOSIS_SUB_OPTI = """\
You are a privileged trajectory coach analyzing a SUCCESSFUL agent trajectory.

The trajectory completed the Stage Goal but required more steps than the expected target.
This does NOT necessarily mean that it is suboptimal: additional steps may be justified by stochastic execution, partial observations, ambiguous feedback, or necessary recovery.

Your task is to determine whether there is a clear local opportunity to complete the SAME Stage Goal with fewer decisions.

If yes, identify the EARLIEST decision that should be modified.

--------------------------------------------------
CORE RULE
--------------------------------------------------

A decision at index k is a valid candidate if, while keeping ALL previous
decisions fixed, replacing that decision could plausibly:

1. still complete the Stage Goal;
2. avoid one or more later decisions;
3. reduce the total number of decisions.

Select the EARLIEST such decision.

Do NOT propose the replacement action.

--------------------------------------------------
CAUSALITY AND INFORMATION
--------------------------------------------------

Each step contains:

INPUT:
information available before the decision.

TOOL:
decision made by the agent.

RESULT:
Flag of the error types

Judge whether a decision was reasonable using only information available
in its INPUT and previous trajectory.

You may use later steps to understand the consequences of an earlier
decision, but do NOT use information that was unavailable to the agent
to claim that an earlier decision was suboptimal.

EXECUTION STOCHASTICITY

A correct decision may lead to additional steps because execution failed
or feedback was uncertain. IN this case it will be marked by injection_error.

Do NOT mark as suboptimal:
- justified retries after stochastic failures;
- observations needed to resolve uncertain or incomplete feedback;
- necessary exploration;
- appropriate recovery after execution failure;
- correct partial progress.

A decision MAY be suboptimal if it:
- performs an unnecessary observation or action;
- makes a choice that must later be unnecessarily reversed or corrected;
- ignores already available information;
- is a bad call;
- violates a known prerequisite and forces later recovery;
- creates avoidable additional work.

Do not assume that a suboptimal decision exists.
If no specific local decision can be defensibly identified, answer false.

All decisions with index 0 belong to a validated prefix.
They are correct and CANNOT be selected.
Only positive indices can be selected.

--------------------------------------------------
ROBOT CONSTRAINTS
--------------------------------------------------

- Only one tool call per robot can be selected at each step.
- Unless explicitly specified otherwise, a robot can hold only one object.
- Do not invent task constraints or robot capabilities.

TASK:
{task_description}

STAGE GOAL:
{stage_goal}

PERMANENT RULES:
{permanent_rules}


--------------------------------------------------
REFERENCE EFFICIENT TRAJECTORY:
--------------------------------------------------

{example_trajectory}

This trajectory is only a reference. Execution, objects, positions and
objectives may differ. Do not transfer assumptions from it.

--------------------------------------------------
TRAJECTORY TO ANALYZE:
--------------------------------------------------

{trajectory}

OUTPUT:
Return only:

{{
  "suboptimal": true/false,
  "decision_idx": <positive integer or null>,
  "reason": "<concise explanation>"
}}
"""
