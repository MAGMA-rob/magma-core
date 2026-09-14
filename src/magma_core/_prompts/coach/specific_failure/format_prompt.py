# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

FIX_FORMAT = """
You are correcting an agent answer that was rejected because it does not match
the required output format.

Stage goal:
{stage_goal}

Information available before the rejected answer:
{input_context}

Rejected answer:
{rejected_answer}

Rejection reason:
{rejection_reason}

Desired output format:
{desired_output_format}

Format rules:
{output_format_rules}

Rewrite the rejected answer so it strictly matches the desired output format.

Rules:
- Keep the useful intent of the rejected answer when possible.
- Fix only what is needed to satisfy the desired format and the rejection reason.
- Respect every field and rule from the desired output format.
- Do not add fields that are not allowed by the desired output format.

Response instructions:
{response_instructions}
"""
