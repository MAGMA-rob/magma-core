from abc import ABC
from typing import List

from ..stage import BaseTaskStage
from ..state import TaskState

class BaseRequest(ABC):
    """
    Base interface for user requests.

    A request can create one or more stages from the current ``TaskState`` and
    optionally update that state afterward so future stages remember the change.
    """

    def __init__(
            self
        ) -> None:
        super().__init__()

    def sampling_weight(self, state: TaskState) -> float:
        """
        Returns the sampling weight for this request given the current state.

        A weight of 0 means the request cannot be sampled.
        Any positive value represents the relative probability of sampling
        compared to other requests.

        The final sampling probability is computed as:

            weight_i / sum(weights)

        Implementations may return any non-negative float.
        """
        return 1.0
    
    def force_state_recompute(self) -> bool:
        """
        Override this function to define a logic to return True or False depending on the request process.
        This function must be True if you remove constraints or modify attributes.
        """
        return False
    
    def apply_request(self, state : TaskState) -> TaskState:
        """
        Apply modifications to the current state after the execution of the stage.

        This function is called after the stage creation by the task generator to ensure that
        the state is currently updated.
        Use this function to reflect the stage effect on the latent task state.

        Return the updated TaskState.
        """
        return state
    
    def create_stages(self, state : TaskState) -> List[BaseTaskStage]:
        """
        This function create stages directly from the current state.
        It is called from the Task Generator
        """
        raise NotImplementedError("The function create_stage must be defined in child class")
