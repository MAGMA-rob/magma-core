"""Version 2 HTTP contract, independent of runtimes and simulators.

POST /v1/responses: AgentRequest -> AgentResponse.
GET /health: AgentHealth. GET /v1/info: AgentInfo.
TLS termination belongs to deployment; Python inheritance is not required.
"""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, JsonValue, RootModel, model_validator

PROTOCOL_VERSION = "2.0"
JsonObject = dict[str, JsonValue]
Identifier = Annotated[int, Field(strict=True)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class AgentInstruction(ContractModel):
    type: Literal["user", "env"]
    content: str


class AgentInput(ContractModel):
    id: Identifier
    instruction: AgentInstruction
    tools: list[JsonObject] = Field(default_factory=list)
    attributes: JsonObject = Field(default_factory=dict)
    memory: JsonObject = Field(default_factory=dict)
    num_outputs: Annotated[int, Field(strict=True, gt=0)] = 1
    extra_keys: JsonObject = Field(default_factory=dict)


class AgentRequest(ContractModel):
    request_id: str = Field(min_length=1)
    inputs: list[AgentInput]

    @model_validator(mode="after")
    def unique_input_ids(self) -> "AgentRequest":
        ids = [entry.id for entry in self.inputs]
        if len(ids) != len(set(ids)):
            raise ValueError("Input ids must be unique within a request")
        return self


class ToolCall(ContractModel):
    name: str = Field(min_length=1)
    arguments: JsonObject = Field(default_factory=dict)
    target_robot_name: str = Field(min_length=1)


class AgentDecision(ContractModel):
    say: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @model_validator(mode="after")
    def exclusive_action(self) -> "AgentDecision":
        if self.say and self.tool_calls:
            raise ValueError("say and tool_calls are mutually exclusive")
        return self


class AgentError(ContractModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    component: str | None = None


class AgentOutput(ContractModel):
    request_id: str = Field(min_length=1)
    source_id: Identifier
    candidate_index: Annotated[int, Field(strict=True, ge=0)]
    status: Literal["completed", "error"]
    memory: JsonObject
    internal_steps: list[JsonObject] = Field(default_factory=list)
    output: AgentDecision | None = None
    error: AgentError | None = None

    @model_validator(mode="after")
    def consistent_status(self) -> "AgentOutput":
        if self.status == "completed" and (self.output is None or self.error is not None):
            raise ValueError("Completed candidates require a decision and no error")
        if self.status == "error" and (self.output is not None or self.error is None):
            raise ValueError("Failed candidates require an error and no decision")
        return self


class AgentResponse(RootModel[list[AgentOutput]]):
    def validate_request(self, request: AgentRequest) -> None:
        expected = [
            (request.request_id, entry.id, index)
            for entry in request.inputs
            for index in range(entry.num_outputs)
        ]
        actual = [(item.request_id, item.source_id, item.candidate_index) for item in self.root]
        if actual != expected:
            raise ValueError("Response candidates must match request counts and order")


class AgentHealth(ContractModel):
    status: Literal["ready"] = "ready"


class AgentInfo(ContractModel):
    protocol_version: str = PROTOCOL_VERSION
    agent_id: str
    agent_version: str
    specialized_coaching: list[str] = Field(default_factory=list)
    coaching_resume: bool = False
    coaching_session_version: str | None = None
    coaching_unavailable_reason: str | None = None
    capabilities: dict[str, bool]
