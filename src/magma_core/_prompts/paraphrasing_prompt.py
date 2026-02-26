# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

PARAPHRASING = """
SYSTEM:
You are adapting user instructions in a human–robot interaction context.
The user gives ordered instructions and constraints to a robot to complete tasks.

Your role is to adapt the next user instruction so it fits naturally after the real conversation,
while staying as close as possible to the original planned instruction.

ALLOWED VARIATION AXES:
- Opening or acknowledgment (e.g., “Okay”, “Alright”, “Hey”)
- Imperative vs polite request phrasing
- Spoken-language style suitable for voice interaction
- Minor reordering of phrases that does not change meaning

FORBIDDEN VARIATIONS:
- Changing task scope or requirements
- Adding or removing actions
- Adding environment details
- Adding reasoning or explanations

Previously generated variants for this same instruction:
{previous_variants}

RULE:
- Generate a new variant that is clearly distinct in wording or tone
  from the variants above, while preserving the same meaning.
- The instruction is intended to be spoken aloud by a human to a robot.


This is the original sequence of user instructions imagined:
{user_sequence}

Here is the real conversation between the user and the robot:
{real_conversation}

The template of the next user instruction was:
{template_answer}

Generate a realistic, coherent instruction a user would say to the robot,
consistent with the conversation so far.
Ensure that no information is added or removed from the original instruction.
Directly output the adapted instruction.
"""

FROM_TEMPLATE = """
SYSTEM:
You are generating a realistic user instruction in a human–robot interaction context.
The user gives ordered instructions and constraints to a robot to complete tasks.

Your role is to generate the next user instruction so it responds naturally
to the LAST answer produced by the robot, using ONLY the information provided.

You will have access to a dict of parameters called Information.
You MUST use this dict to generate a plausible spoken instruction while preserving ALL information EXACTLY.

ALLOWED VARIATIONS (surface-level only):
- Opening or acknowledgment (e.g., "Okay", "Alright", "Yes")
- Imperative vs polite phrasing
- Spoken-language style suitable for voice interaction
- Minor reordering of phrases that does NOT change meaning

FORBIDDEN VARIATIONS (critical):
- Changing task scope or requirements
- Adding or removing actions
- Adding environment details
- Modifying, inferring, canceling, or simplifying Information

SEMANTIC PRESERVATION CONSTRAINT (CRITICAL):
- Every key in the Information dict MUST be explicitly reflected in the instruction.
- Every value associated with each key MUST be explicitly mentioned.
- If the same item appears in multiple fields (e.g., add and remove),
  ALL operations must be stated explicitly.
- Do NOT resolve redundancies or contradictions; state them as given.

Previously generated variants for this same instruction:
{previous_variants}

RULES:
- Generate a new variant that is clearly distinct in wording or tone
  from the variants above.
- If any required information is missing in the generated instruction,
  the instruction is invalid and must be regenerated.
- The instruction is intended to be spoken aloud by a human to a robot.

Context to understand the Information:
{context}

Information to integrate in the instruction:
{template}

Last answer produced by the robot:
{last_model_answer}

Before producing the final instruction, internally verify:
- All keys in Information are covered
- All items for each key are mentioned
- No extra actions or details are introduced

OUTPUT FORMAT:
- Output exactly one spoken instruction (one sentence or a short utterance).
- The instruction must naturally respond to the last robot answer.
- Do NOT explain, justify, summarize, or add meta-comments.
- Output only the adapted instruction text.
"""
