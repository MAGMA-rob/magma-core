from abc import ABC, abstractmethod
from typing import Any, Dict

import torch

from magma_core.simulation.serialization import build_spec, construct_from_spec


class BaseStageEnvironmentTransition(ABC):
    """Apply an automatic environment change when a stage starts."""

    def to_spec(self) -> Dict[str, Any]:
        serializer = type(self).__dict__.get("_to_spec_arguments")
        if serializer is None:
            raise NotImplementedError(
                f"{type(self).__name__} must implement _to_spec_arguments() "
                "before it can be serialized"
            )
        return build_spec(self, self._to_spec_arguments())

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "BaseStageEnvironmentTransition":
        transition = construct_from_spec(spec, cls)
        if not isinstance(transition, cls):
            raise TypeError(f"Transition spec is not compatible with {cls.__name__}")
        return transition

    @abstractmethod
    def apply(
        self,
        env_state: Dict,
        env_ids: torch.Tensor,
    ) -> Dict:
        """Return the environment state after applying the transition."""
        raise NotImplementedError
