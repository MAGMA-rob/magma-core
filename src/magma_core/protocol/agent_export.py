"""Local dataset export contracts, independent of inference and transport."""
from abc import ABC, abstractmethod
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Literal, TypeVar

from pydantic import Field, model_validator

from .agent import AgentInput, ContractModel, JsonObject
from .graph import CandidateRecord

EXPORT_VERSION = '2.0'
ProviderT = TypeVar('ProviderT')


def discover_exporters(group: str) -> dict[str, EntryPoint]:
    """Inspect installed providers without importing their modules."""
    providers: dict[str, EntryPoint] = {}
    for entry in sorted(entry_points(group=group), key=lambda item: (item.name, item.value)):
        if entry.name in providers:
            raise ValueError(f'Duplicate exporter {entry.name!r} in {group!r}')
        providers[entry.name] = entry
    return providers


def load_exporter(group: str, name: str, contract: type[ProviderT]) -> ProviderT:
    providers = discover_exporters(group)
    if name not in providers:
        raise ValueError(f'Unknown exporter {name!r} in {group!r}; installed: {sorted(providers)}')
    entry = providers[name]
    try:
        provider = entry.load()()
    except ImportError as error:
        raise ImportError(f'Cannot load exporter {name!r} ({entry.value}); install its required extras') from error
    if not isinstance(provider, contract):
        raise TypeError(f'Exporter {name!r} must implement {contract.__name__}')
    return provider


class ExportExample(ContractModel):
    example_id: str
    input: AgentInput
    candidate: CandidateRecord


class ExportRecord(ContractModel):
    record_id: str = Field(min_length=1)
    dataset: str = Field(pattern=r'^[a-zA-Z0-9][a-zA-Z0-9_-]*$')
    data: JsonObject


class ExportResult(ContractModel):
    example_id: str
    status: Literal['exported', 'skipped', 'error']
    records: list[ExportRecord] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode='after')
    def validate_status(self) -> 'ExportResult':
        if self.status == 'exported':
            if not self.records:
                raise ValueError('Exported examples require records')
        elif self.records or not self.reason:
            raise ValueError('Skipped/error examples require a reason and no records')
        return self


class DatasetRenderer(ABC):
    """Agent-owned schema and rendering, shared by all input adapters."""
    filenames: dict[str, str]

    @abstractmethod
    def render(self, channel: str, example: dict[str, Any]) -> JsonObject:
        raise NotImplementedError


class GenExporter(ABC):
    agent_id: str
    agent_version: str
    renderer: DatasetRenderer

    @abstractmethod
    def export(self, examples: list[ExportExample], *, options: JsonObject | None = None) -> list[ExportResult]:
        raise NotImplementedError
