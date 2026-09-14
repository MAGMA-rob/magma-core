"""Portable generation records shared by disk, observation and dataset export."""
from typing import Literal
from pydantic import Field, model_validator
from .agent import AgentInput, AgentDecision, AgentError, ContractModel, JsonObject

SAVE_VERSION = '2.0'


class SituationLink(ContractModel):
    situation_id: int
    kind: Literal['next', 'alternative']


class CandidateRecord(ContractModel):
    id: int
    parent_id: int | None
    stage_id: int
    kind: Literal['agent', 'simulation_resume']
    response_status: Literal['completed', 'error']
    output: AgentDecision | None = None
    error: AgentError | None = None
    internal_steps: list[JsonObject] = Field(default_factory=list)
    execution: JsonObject = Field(default_factory=dict)
    status: str
    execution_finished: bool
    score: JsonObject = Field(default_factory=dict)
    coaching: JsonObject = Field(default_factory=dict)
    coaching_source: JsonObject = Field(default_factory=dict)
    coaching_attempt_ids: list[int] = Field(default_factory=list)
    origin: Literal['model', 'coaching', 'simulation']
    links: list[SituationLink] = Field(default_factory=list)

    @model_validator(mode='after')
    def decision_status(self) -> 'CandidateRecord':
        if self.response_status == 'error':
            if self.output is not None or self.error is None:
                raise ValueError('An invalid response requires error and no output')
        elif self.output is None or self.error is not None:
            raise ValueError('A completed response requires output and no error')
        return self


class SituationRecord(ContractModel):
    schema_version: Literal['2.0'] = SAVE_VERSION
    id: int
    stage_id: int
    parent_id: int | None
    input: AgentInput
    candidates: list[CandidateRecord]
