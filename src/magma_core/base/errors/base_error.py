from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any, Optional, ClassVar, List

from ...utils.data_utils import verify_parameters_dict

if TYPE_CHECKING:
    from ..data_structures.observation import Observation
    from ..data_structures.tools import ToolExecution, ToolResult

class BaseError(ABC):
    """
    Base class for managing error injection (perception, movement failure).

    This class must ne inherited and register inside some tool registration decorator.
    """
    recovery_extra_steps: int = 0
    required_key: ClassVar[List[str]] = []
    optional_key: ClassVar[List[str]] = []

    def get_name(self) -> str:
        return self.__class__.__name__

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

        required = {key: object for key in self.required_key}
        optional = {key: object for key in self.optional_key}
        valid, reason = verify_parameters_dict(arguments, required, optional)
        if not valid:
            raise RuntimeError(f"{self.get_name()} invalid arguments. {reason}")
    
    def apply_pre_exec(self, tool_execution: "ToolExecution", arguments: Dict[str, Any]):
        pass

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

    def get_recovery_extra_steps(self, arguments: Optional[Dict[str, Any]]) -> int:
        """
        Return the number of additional steps that should be forgiven because of
        this active error on the current trajectory.
        """
        return self.recovery_extra_steps
