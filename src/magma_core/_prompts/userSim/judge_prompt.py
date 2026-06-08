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

VALID EXAMPLES

Question:
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

OUTPUT FORMAT:
Return ONLY a valid JSON object like {exp}

INPUTS:
- Question: {question}
- Rule: {rule}
- Model answer: {model_answer}
"""