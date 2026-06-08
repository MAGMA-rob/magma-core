# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

FIX_FORMAT = """
You are correcting a malformed agent answer so it can be executed by the system.

Task goal:
{description}

User instruction:
{instruction}

Current memory:
{memory}

Available tools:
{tools}

Current environment attributes:
{attributes}

Malformed answer:
{invalid_answer}

Rewrite the malformed answer into one strict JSON object with exactly these top-level keys:
- "think": short reasoning string
- "say": message to the user string
- "action": either {{}} or a valid tool-call object {{"name": ..., "arguments": ...}}

Rules:
- Return JSON only.
- Do not add markdown fences.
- Keep the intent of the original malformed answer when possible.
- If the malformed answer clearly intends no tool call, use "action": {{}}.
- Do not invent unavailable tools.
"""
