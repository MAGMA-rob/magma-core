from abc import ABC
from typing import Generic, TypeVar

from magma_core.simulation.state import TaskState
from magma_core.simulation.stage import BaseTaskStage


ParametersT = TypeVar("ParametersT")


class BaseRequest(ABC, Generic[ParametersT]):
    """Build a runtime request from one immutable sampled parameter object."""

    def sampling_weight(self, state: TaskState) -> float:
        return 1.0

    def sample_parameters(self, state: TaskState) -> ParametersT:
        raise NotImplementedError(
            "sample_parameters must be defined by each request."
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: ParametersT,
    ) -> list[BaseTaskStage]:
        raise NotImplementedError(
            "create_stages must be defined by each request."
        )

    def apply_request(
        self,
        state: TaskState,
        parameters: ParametersT,
    ) -> TaskState:
        return state

    def force_state_recompute(self) -> bool:
        return False
