from abc import ABC
from typing import Dict

from ..state import TaskState

class BaseConstraint(ABC):
    """
    Base class for latent task constraints.

    Constraints are applied to a ``TaskState`` during task generation to
    register new rules or mutate derived relations. Override ``apply`` to
    update the state and ``outdated`` to tell the generator when a stored
    constraint is no longer valid.
    """

    def __init__(self) -> None:
        super().__init__()

    def apply(self, state : TaskState):
        """
        Register the constraint to the state constraint_history list.
        You can override this function and keep the call to the parent to keep the register and define your custom apply logic.
        """
        state.constraints_history.append(self)

    def outdated(self, state : TaskState) -> bool:
        """
        Return True if the state does not allow to apply the constraint anymore (e.g constraint on attributes that was removed / override)
        """
        return False
    
class AttributesModifConstraint(BaseConstraint):
    """
    Constraint that replaces the current task attributes snapshot.

    This is the default building block used by attribute-edit requests after
    they compute a new set of available task attributes.
    """

    def __init__(self, attributes : Dict) -> None:
        super().__init__()
        self.attributes = attributes

    def apply(self, state: TaskState):
        super().apply(state)
        state.attributes = self.attributes
