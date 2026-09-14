
REPR_INSTRUCTION_VARIATIONS = """
You rewrite task-state update inputs for a robot assistant.

Your goal is to produce semantic-preserving variations of the ORIGINAL INPUT.
The variations must keep exactly the same meaning and all the same information.

VARIATION MODE:
{variation_mode}

{variation_guidance}

ALLOWED VARIATION AXES:
- Change surface wording while keeping every entity, attribute, quantity,
  action, result, and constraint explicitly present
- Reorder clauses when the meaning stays identical
- Use natural spoken phrasing
- Change sentence type when the selected mode allows it

SPEECH-ACT PRESERVATION:
- Preserve whether the original input is an action request, a standing
  rule/preference/constraint, an information update, a question, or feedback.
- For action requests, including act/find-style inputs, keep the user asking
  the robot to act. Do not turn them into memory updates.
- Use "Remember", "From now on", "Keep in mind", "Note that", and similar
  memory framing only for standing rules, preferences, constraints, or
  understand/inform-style information that should be stored.
- Never use memory framing for one-off actions such as act, find, take, press,
  move, put, wash, sort, deliver, or search.

FORBIDDEN VARIATIONS:
- Do not remove information
- Do not add new facts, entities, actions, preferences, quantities, or constraints
- Do not infer anything that is not in the original input
- Do not simplify away details
- Do not change names, colors, objects, people, locations, or preferences
- Do not merge several facts in a way that makes any fact implicit or ambiguous

Examples:
Original: "Thomas likes his coffee black."
Valid variations:
- "Hello, Thomas likes black coffee."
- "Hey, remember that Thomas wants his coffee black."
- "Please note that Thomas prefers black coffee."
Invalid variations:
- "Thomas likes coffee." because the black preference was removed.
- "Thomas wants coffee black." if the original is only a standing preference,
  because it may change the type of information.

Original: "Press the buttons in exactly this order: first 2, then 3."
Valid variations:
- "Please press the buttons in exactly this order: first 2, then 3."
- "Press button 2 first, then button 3, keeping that exact order."
Invalid variations:
- "Remember to press the buttons in exactly this order: first 2, then 3."
  because this is an action request, not a standing rule.
- "From now on, press 2 before 3." because this turns a one-off action into a
  persistent instruction.

ORIGINAL INPUT:
{instruction}

Generate exactly {nb_variations} distinct variations.

Before answering, internally verify that each variation preserves every piece of
information from the original input and adds no new semantic content.

OUTPUT FORMAT:
{output_format}

Return the JSON object directly. Do not wrap it in Markdown code fences such as
```json or ```, and do not include explanations or any text before or after it.
"""


def get_repr_variation_guidance(mode: str) -> str:
    if mode == "user_utterance":
        return """
                MODE-SPECIFIC INSTRUCTIONS:
                The input is a user utterance. Produce realistic alternative user utterances.
                Each variation must be a string.
                Preserve the speech act of the original utterance:
                - If the original asks the robot to do an action now, such as act, find, take,
                  press, move, put, wash, sort, or deliver, every variation must remain an
                  action request.
                - If the original gives a standing rule, preference, constraint, or information
                  to keep in memory, such as understand/inform-style inputs, every variation
                  must remain a standing rule or memory update.
                - If the original asks a question, every variation must remain a question.
                You may add short natural openings or discourse markers, such as "Hello",
                "Hey", "Okay", or "Please".
                Use memory framing such as "Remember", "From now on", "Keep in mind", or
                "Note that" only for standing rules, preferences, constraints, or information.
                Never use memory framing for one-off action requests.
                """.strip()
    if mode == "feedback_variation":
        return """
                MODE-SPECIFIC INSTRUCTIONS:
                The input is the "infos" list from an execution feedback report.
                Produce alternative "infos" lists, not user requests and not full feedback
                reports.
                Do not mention or rewrite completed actions or errors.
                Each variation must preserve the same number of info entries and keep each entry
                in the same position.
                Rephrase only the text inside the infos list while preserving every entity,
                result, and constraint.
                """.strip()
    raise ValueError(f"Unknown mode type, got ({mode})")


def get_repr_variation_output_format(mode: str) -> str:
    if mode == "user_utterance":
        return """
            Return a strict JSON object only, with this exact shape:
            {
              "variations": [
                "first variation",
                "second variation"
              ]
            }
            Do not use Markdown or code fences. The first character of the
            response must be "{" and the last character must be "}".
            """.strip()
    if mode == "feedback_variation":
        return """
            Return a strict JSON object only, with this exact shape:
            {
              "variations": [
                {
                  "infos": ["first info-entry variation"]
                },
                {
                  "infos": ["second info-entry variation"]
                }
              ]
            }
            Do not use Markdown or code fences. The first character of the
            response must be "{" and the last character must be "}".
            """.strip()
    raise ValueError(f"Unknown mode type, got ({mode})")
