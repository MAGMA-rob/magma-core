from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any, Optional, ClassVar, List

from magma_core.simulation.serialization import (
    build_spec,
    construct_from_spec,
    extract_constructor_arguments,
)

if TYPE_CHECKING:
    from magma_core.simulation.data_structures.observation import Observation
    from magma_core.simulation.data_structures.tools import ToolExecution, ToolResult

class BaseError(ABC):
    """
    Base class for managing error injection (perception, movement failure).

    This class must ne inherited and register inside some tool registration decorator.
    """
    required_key: ClassVar[List[str]] = []
    optional_key: ClassVar[List[str]] = []

    def get_name(self) -> str:
        return self.__class__.__name__

    def to_spec(self) -> Dict[str, Any]:
        serializer = type(self).__dict__.get("_to_spec_arguments")
        arguments = (
            extract_constructor_arguments(self)
            if serializer is None
            else self._to_spec_arguments()
        )
        return build_spec(self, arguments)

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "BaseError":
        error = construct_from_spec(spec, cls)
        if not isinstance(error, cls):
            raise TypeError(f"Error spec is not compatible with {cls.__name__}")
        return error

    def _to_spec_arguments(self) -> Dict[str, Any]:
        return extract_constructor_arguments(self)

    def validate_arguments(self, arguments: Optional[Dict[str, Any]]) -> None:
        """
        Validate benchmark-provided runtime error arguments against the error
        schema declared by the concrete class.
        """
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise RuntimeError(
                f"{self.get_name()} arguments must be a dict or None. "
                f"Got {type(arguments)}."
            )

        argument_keys = set(arguments)
        required_keys = set(self.required_key)
        allowed_keys = required_keys | set(self.optional_key)
        missing = required_keys - argument_keys
        unexpected = argument_keys - allowed_keys
        if missing or unexpected:
            raise RuntimeError(
                f"{self.get_name()} invalid arguments: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
    
    def apply_pre_exec(
        self,
        tool_execution: "ToolExecution",
        arguments: Dict[str, Any],
    ) -> bool:
        """Apply a pre-execution error and report whether it took effect."""
        return False

    @abstractmethod
    def initialize(self, obs: "Observation", env_id: int) -> Optional[Dict[str, Any]]:
        """
        Build the initial per-trajectory runtime state for this error.

        Returning ``None`` keeps the error selected for the stage but defers
        the actual binding until a compatible tool provides enough context.
        """
        raise NotImplementedError()

    def apply_post_verif(self, tool_result: "ToolResult", arguments: Dict[str, Any]):
        pass

    @abstractmethod
    def get_description(self, arguments : Optional[Dict[str, Any]]) -> str:
        raise NotImplementedError()
