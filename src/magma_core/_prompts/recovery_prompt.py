BUILD_REPEAT="""
You are in charge of generating a recovery response after a robot executed
a valid tool call but encountered an execution-time error.
  
This is NOT a function calling error.
This is an execution failure after a correct tool invocation.
  
Your task is to:
1. Produce a short reasoning paragraph ("think") explaining what happened and what will be done.
2. Produce a short spoken message ("say") addressed to the user.
  
You must follow the reasoning structure exactly as specified below.
  
INPUTS (some may be absent):
- base_template: a previous recovery response proposed by the model, if available
- original_call: the tool call and execution error message
  
MANDATORY REASONING STRUCTURE FOR "think":
The paragraph MUST:
- Acknowledge that an execution error occurred
- State that the tool call itself was correct
- State that the system will retry the same tool call once.
- State also that if it fails again, you will notify the user (use 'the user')
  
CONSTRAINTS ON "think":
- 3 to 4 sentences maximum
- Clear, factual, and concise
- Mention the action that failed ("while doing ...")
- No alternative strategies
- No mention of internal system details
  
CONSTRAINTS ON "say":
- Short spoken-language message for a user
- Inform the user that an error occurred by saying the precise failed action
- Inform the user that you are retrying (use first person 'I')
- Do NOT mention internal mechanics or retries beyond “retrying once”
  
RULES:
- Do NOT change the tool or its arguments
- Do NOT analyze the task or environment
- Do NOT introduce new actions
- Do NOT mention future handling beyond this retry
  
If a base_template is provided:
- You may reuse or lightly edit it
- You must correct it if it violates the rules above
- Generate a new response following the same constraints
  
Error message
{error_message}

Original Tool Call
{tool_call}

Template:
{template}

OUTPUT FORMAT (JSON ONLY):
{{"think": "<3–4 sentence concise reasoning paragraph>","say": "<short spoken message to the user>"}}
"""

INFORM_USER="""
You are in charge of generating a recovery response after a robot executed
a valid tool call but encountered an execution-time error.

This is NOT a function calling error.
This is an execution failure after a correct tool invocation.
In this case, you are NOT retrying. You must inform the user that the task stops here.

Your task is to:
1. Produce a short reasoning paragraph ("think") explaining what happened and what will be done.
2. Produce a short spoken message ("say") addressed to the user.

You must follow the reasoning structure exactly as specified below.

INPUTS:
- tool_call: the original tool call
- error_message: the execution error message

MANDATORY REASONING STRUCTURE FOR "think":
The paragraph MUST:
- Acknowledge that an execution error occurred while performing the action
- State that the tool call itself was correct
- Explicitly state {is_dual_agent} that this is the second time the error occurred
- Conclude that the system will inform the user and stop the task

CONSTRAINTS ON "think":
- 3 to 4 sentences maximum
- Clear, factual, and concise
- Explicitly mention the failed action (e.g., "while doing take_object")
- No alternative strategies
- No speculation about causes

CONSTRAINTS ON "say":
- Short spoken-language message addressed to the user
- Clearly state which action failed
- Inform the user that the task cannot continue
- Do NOT mention retries, internal mechanics, or system memory

RULES:
- Do NOT analyze the task or environment
- Do NOT introduce new actions
- Do NOT propose next steps or future handling

Error message:
{error_message}

Original tool call:
{tool_call}

OUTPUT FORMAT (JSON ONLY):
{{"think": "<3–4 sentence concise reasoning paragraph>","say": "<short spoken message to the user>"}}
"""

UPDATE_MEMORY=""""""