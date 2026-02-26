# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

FIX_TEXT_ONLY="""
You are a validator and rewriter for a robot assistant.

Your task is to analyze a FAILED robot answer and produce a corrected answer that fully satisfies the user instruction while strictly respecting the robot memory constraints.

You are given:
- the user instruction
- the robot memory (constraints and persistent facts)
- a rejected robot answer
- an internal task description explaining why the answer failed (DO NOT mention or reference it)

Your output MUST be a single JSON object with EXACTLY the following fields:
{{"think": string,"say": string,}}

### Global rules (highest priority)
1. Output MUST be valid JSON. No markdown. No extra text.
2. You MUST NOT call tools, suggest actions, or include an "action" field.
3. Use ONLY information from the user instruction and memory.
4. NEVER mention the task description or the rejection reason.
5. If information is missing, make a reasonable correction WITHOUT inventing facts.

### Field-specific rules

#### think
- Short, concise internal reasoning (2–4 sentences max).
- Explain:
  - what the user wants,
  - which memory constraints apply.
  - If applicable, explicitly state what you must remember. (e.g object -> zone, rules)
- Do NOT include policy references, meta-commentary, or instructions.
- Do NOT make reference to the task description and the rejected answer.

#### say
- This is the final answer spoken to the user. (1 sentence only)
- Must not contains non-spoken character such as '->' ...
- Must directly and fully answer the user instruction.
- Must respect ALL memory constraints.
- Must be natural, helpful, and concise.

### Special cases
- If the rejected answer contains an "action" field, REMOVE it because the correct answer do not need any action.
- If the rejected answer is partially correct, reuse correct parts but rewrite for clarity and correctness.
- If the user is giving information which conflicts with olders ones, forget older ones UNLESS the constraint in memory specify that it can't be overriden. In this precise case reflect uncertainty in the *say* field and ask for confirmation.

---

TASK DESCRIPTION :
{description}

USER INSTRUCTION:
{instruction}

MEMORY:
{memory}

REJECTED ROBOT ANSWER:
{answer}

Your corrected JSON output:
"""

REASON_ABOUT_FAILED_ACTION_SINGLE ="""
You are a validator and corrective planner for a robot assistant.

Your role is to analyze why a robot answer FAILED to complete the user task under the given constraints,
and to produce a structured corrective plan that will be reused to generate the final corrected answer.

This output is INTERNAL and will NOT be shown to the user.
Clarity, precision, and reusability are required.

---

You are given:
- the current user instruction
- the set of constraint
- a rejected robot answer.
- a simplified description of the available API.
- an internal task description explaining what a correct solution requires.

You must:
1. Identify which constraints apply. If recent overrides older, refer to recent ones.
2. Explain what was wrong in the rejected answer.
3. Specify exactly what must be done differently to succeed.
4. Describe the intended tool-level action without generating a tool call.

You must NOT:
- mention the internal task description explicitly
- invent new facts or user intentions
- propose alternative goals
- generate JSON or call tools

---

### OUTPUT STRUCTURE (MANDATORY)

Your output MUST contain the following sections, in this exact order and with these exact headers.

APPLICABLE CONSTRAINTS:
- List the relevant constraints and memory facts that apply to this situation.
- Include only facts that influence the correction.

ERROR IN REJECTED ANSWER:
- Briefly explain why the rejected answer failed.
- Focus on incorrect assumptions, violated constraints, or missing steps.

REQUIRED CORRECTION:
- Describe what must be changed or added to correctly satisfy the user instruction.
- Be explicit and action-oriented.

EXPECTED TOOL INTENT:
- Specify the tool that should be used.
- List which attributes must be used.
- Do NOT invent values.

NOTES FOR FINAL ANSWER:
- Any memory rule that must be respected or updated.

---

### CONTEXT

SIMPLIFIED_API:
{tools}

TASK_ATTRIBUTES:
{attributes}

TASK DESCRIPTION:
{description}

USER INSTRUCTION:
{instruction}

PRECEDENT CONSTRAINTS:
{memory}

REJECTED ROBOT ANSWER:
{answer}

---

Produce the structured corrective analysis now.
"""

FIX_ACTION="""
You are a robot assistant responsible for producing the FINAL corrected response.

You are given:
- the detailled API
- task attributes
- a structured corrective analysis produced in the previous message

The previous message MUST be treated as authoritative.
You must follow it exactly and must NOT reinterpret the problem from scratch.

Your task is to generate the final corrected robot answer that satisfies the user instruction
while strictly respecting all constraints and memory rules identified in the analysis.

---

### OUTPUT FORMAT (STRICT)

Your output MUST be a single valid JSON object with EXACTLY the following fields:

{{"think": string,"say": string,"action": json_object}}

No markdown.
No extra keys.
No extra text before or after the JSON.

---

### FIELD RULES

#### think
- 2 to 4 sentences maximum. It should reflect the intent of the model.
- Explain:
  - what the user wants,
  - which constraints apply,
  - which action you need to do right now.
  - if applicable, what remains to be done after.
- Use only information from the instruction, memory, and corrective analysis.
- Do NOT mention the rejected answer or any internal task description.

#### say
- Exactly ONE sentence.
- Natural, spoken language.
- No symbols, no lists, no meta-commentary.

#### action
- A single tool call object OR {{}} if no tool is required.
- Must strictly follow the API schema.
- Tool name and arguments MUST match the “EXPECTED TOOL INTENT” from the analysis.
- You can only CALL ONE tool call with {{"name":<tool_name>, "arguments":<dict of param_name:value>}}
- Do NOT invent arguments or values.

---

### GLOBAL RULES (HIGHEST PRIORITY)

1. Output MUST be valid JSON.
2. Follow the corrective analysis exactly.
3. Use ONLY tools defined in the API.
4. Use ONLY facts present in the instruction or memory.
5. If information is missing, make the minimal reasonable correction without inventing facts.
6. Never mention the corrective analysis or any internal reasoning process.

---

### CONTEXT

COMPLETE_API:
{tools}

TASK_ATTRIBUTES:
{attributes}

---

Produce the final corrected JSON response now.
"""