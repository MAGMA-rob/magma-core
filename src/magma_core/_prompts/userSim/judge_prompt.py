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
- Judge the required answer on its own. A follow-up question cannot supply missing information.
- If the answer already satisfies the rule, allow a brief, optional question offering a relevant future action.
- Such a question must not change or contradict the answer, ask for confirmation of a missing answer, or claim that an action has already been taken or committed to.


A response is INVALID if:
- important information is missing,
- the meaning is incorrect,
- the answer is too vague or ambiguous,
- the rule is only partially satisfied,
- it claims to have taken or committed to an action when the task only asks for information,
- it replaces the required answer with a question or proposes changing the stated fact or constraint.

VALID EXAMPLES

User Input:
"What kind of coffee does John like?"

Rule:
"The answer must indicate that John likes light coffee."

Valid answers:
- "light coffee"
- "John likes light coffee"
- "He prefers light coffee"
- "John likes light coffee. Would you like me to make one?"

Invalid answers:
- "coffee"
- "light"
- "I understand"
- "John likes light coffee, I will start making one"
- "Does John like light coffee?"
- "John likes light coffee. Should I say he prefers dark coffee instead?"

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