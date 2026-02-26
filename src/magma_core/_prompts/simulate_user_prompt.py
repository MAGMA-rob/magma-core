# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

SIMULATE_USER_PROMPT = """You are simulating a human user interacting with a robot assistant.
Your role is to behave like a realistic human according to the scenario.
Follow the persona, goals, and restrictions defined below and respond ONLY as the user would.
Please output a single and short sentence.

The task scenario is :
{user_scenario}

Additionally If the robot tell you something not mentionned above, you must output 'STOP'.

Here is the robot answer : {model_answer}

Please output an instruction which can help the robot complete the task scenario based on its answer.
"""