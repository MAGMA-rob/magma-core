# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

SELECT_SIMILAR_CASES = """
Select the stored cases that are genuinely useful for diagnosing the current
failed trajectory.

CURRENT STAGE GOAL
{stage_goal}

CURRENT TRAJECTORY
{trajectory}

STORED CASES
{cases}

Return only a JSON array containing the useful case numbers, for example
[1, 3]. Return [] when none is useful. Do not return Markdown or explanations.
"""
