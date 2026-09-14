"""Simulator-independent specialized coaching exchanges."""
from typing import Literal
from pydantic import Field, model_validator
from .agent import AgentInput, AgentDecision, AgentError, ContractModel, JsonObject


class InternalStep(ContractModel):
    id: str
    component: str
    origin: Literal['model', 'coaching'] = 'model'
    full_prompt: str
    input_elements: JsonObject = Field(default_factory=dict)
    output_raw: str


class CoachingStage(ContractModel):
    id: int
    goal: str
    hint: str | None = None


class CoachingError(ContractModel):
    code: str
    description: str
    details: JsonObject = Field(default_factory=dict)


class CoachingDiagnosis(ContractModel):
    explanation: str
    expected_decision: str | None = None


class CoachingStep(ContractModel):
    id: int
    input: AgentInput
    output: AgentDecision | None = None
    error: AgentError | None = None
    internal_steps: list[JsonObject] = Field(default_factory=list)
    execution: JsonObject = Field(default_factory=dict)


class ReinjectionPoint(ContractModel):
    id: str
    input_step_id: int | None = None
    input: AgentInput | None = None

    @model_validator(mode='after')
    def exclusive_input(self) -> "ReinjectionPoint":
        if (self.input_step_id is None) == (self.input is None):
            raise ValueError('Provide exactly one input or input_step_id')
        return self


class CoachingBackendConfig(ContractModel):
    type: str
    endpoint: str
    default_model: str
    headers: dict[str, str] = Field(default_factory=dict, repr=False)
    timeout: float = Field(default=30, gt=0)
    max_retry: int = Field(default=3, ge=0)


class CoachingSessionConfig(ContractModel):
    backends: dict[str, CoachingBackendConfig] = Field(default_factory=dict)
    provider: Literal["llm", "human"] = "llm"
    endpoint: str | None = None
    connect_timeout: float = Field(default=10, gt=0)
    capture_logs: bool = True

    @model_validator(mode="after")
    def validate_provider(self) -> "CoachingSessionConfig":
        if self.provider == "llm" and not self.backends:
            raise ValueError("LLM coaching requires at least one backend")
        if self.provider == "human" and not self.endpoint:
            raise ValueError("Human coaching requires an endpoint")
        return self


class CoachingLog(ContractModel):
    coaching_type: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    content: str


class SpecializedCoachingRequest(ContractModel):
    run_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    kind: Literal['failure', 'suboptimal', 'format']
    stage: CoachingStage
    active_errors: list[CoachingError] = Field(default_factory=list)
    diagnosis: CoachingDiagnosis | None = None
    target_step_id: int
    trajectory: list[CoachingStep]
    reinjection_points: list[ReinjectionPoint]
    additional_info: JsonObject = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_references(self) -> "SpecializedCoachingRequest":
        ids = [step.id for step in self.trajectory]
        points = [point.id for point in self.reinjection_points]
        if len(ids) != len(set(ids)) or len(points) != len(set(points)):
            raise ValueError('Step and reinjection point ids must be unique')
        if self.target_step_id not in ids:
            raise ValueError('Target step is absent from trajectory')
        if any(point.input_step_id is not None and point.input_step_id not in ids
               for point in self.reinjection_points):
            raise ValueError('Unknown reinjection input reference')
        return self


class CoachingProposal(ContractModel):
    reinjection_point_id: str
    extra_keys: JsonObject


class SpecializedCoachingResponse(ContractModel):
    logs: list[CoachingLog] = Field(default_factory=list)
    request_id: str = Field(min_length=1)
    status: Literal['corrected', 'abandoned', 'error']
    proposals: list[CoachingProposal] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode='after')
    def validate_status(self) -> "SpecializedCoachingResponse":
        if self.status == 'corrected':
            if not self.proposals:
                raise ValueError('A correction requires proposals')
        elif self.proposals or not self.reason:
            raise ValueError('Abandon/error requires a reason and no proposals')
        return self

    def validate_request(self, request: SpecializedCoachingRequest) -> None:
        if self.request_id != request.request_id:
            raise ValueError('Coaching request id mismatch')
        allowed = {point.id for point in request.reinjection_points}
        if any(proposal.reinjection_point_id not in allowed for proposal in self.proposals):
            raise ValueError('Unknown reinjection point')


class ReplaceSay(ContractModel):
    kind: Literal['replace_say'] = 'replace_say'
    text: str = Field(min_length=1)
