# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

FIX_TEXT_ONLY_ANSWER = """\
You need to correct one step from a failed text-only stage.

In this stage, tools may be used before producing the final answer.
Use the previous diagnostic to propose a simple and effective correction.

IMPORTANT CONSTRAINTS:
- Propose the MINIMAL correction necessary.
- Do NOT modify previous steps.
- Do NOT invent facts.
- Do NOT mention the task context, diagnostic, rejection, or failure to the user.
- The corrected answer must keep the normal model answer format.

The objective of this stage is to answer to a question or fetch an information. It requires to use tools and to give an answer to the user.

{additional}

TASK CONTEXT:
{description}

AVAILABLE TOOLS:
{tools}

ORIGINAL USER INSTRUCTION:
{original_query}

CURRENT QUERY:
{query}

CURRENT MEMORY:
{memory}

REJECTED MODEL ANSWER:
{model_answer}

REPAIR INSTRUCTIONS:

If the current step should use a tool:
- Put exactly one tool call in the "action" field.
- The "say" field should briefly state what you are doing.

If the current step should answer the user:
- Put an empty object in the "action" field: {{}}
- The "say" field must be the final spoken answer.

The corrected step must:
- Respect the current query and memory.
- Use only available tools when a tool is needed.
- Restore a valid path toward satisfying the user instruction.
- Be locally correct from the current state.

REQUIRED OUTPUT FORMAT (STRICT)

A JSON object with the format: {format}.
"""
