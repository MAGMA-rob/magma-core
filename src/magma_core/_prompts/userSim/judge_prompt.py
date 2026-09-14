# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

BENCHMARK_JUDGE_PROMPT = """
You are an evaluator analyzing whether a model answer satisfies a required semantic rule.

INPUTS:
- Original question/task
- Required rule/constraint
- Model answer

YOUR TASK:
Determine whether the model answer semantically satisfies the rule.

IMPORTANT:
- Exact wording is NOT required.
- Paraphrases are valid.
- Reordered words are valid.
- Shortened answers are valid if the meaning remains clear from the question/context.
- Focus on semantic meaning, not exact lexical overlap.

A response is INVALID if:
- important information is missing,
- the meaning is incorrect,
- the answer is too vague or ambiguous,
- the rule is only partially satisfied.
- the answer speak about an action that need to be done whereas the instruction is just giving a rule.
- the answer asks a question whereas the goal of the stage is just to show understanding.

VALID EXAMPLES

User Input:
"What kind of coffee does John like?"

Rule:
"The answer must indicate that John likes light coffee."

Valid answers:
- "light coffee"
- "John likes light coffee"
- "He prefers light coffee"

Invalid answers:
- "coffee"
- "light"
- "I understand"
- "John likes light coffee, I will start making one"

OUTPUT FORMAT:
Return ONLY a valid JSON object like {exp}

INPUTS:
User Input:
{question}

Rule:
{rule}

Model answer:
{model_answer}
"""