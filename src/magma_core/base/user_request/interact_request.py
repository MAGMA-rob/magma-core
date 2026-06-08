from typing import Optional, List, Dict, Any
import copy

from magma_core.base.stage import BaseTaskStage

from ..stage import ConstraintBaseStage
from ..constraints import BaseConstraint, AttributesModifConstraint
from..state.task_state import TaskState
from ..user_request import BaseRequest


class BaseConstraintRequest(BaseRequest):
    """Base request helper for permanent-rule updates.

    Subclasses build one or more constraints plus a short message describing
    the rule, then this class turns them into a single ``ConstraintBaseStage``.
    """

    # Stock All initialized constraints
    constraints : List[BaseConstraint]
    # Stock the computed message for the underlying stage
    constraint_msg : Optional[str]

    def __init__(self):
        self.constraints = []
        self.constraint_msg = None

    def initialize_constraints(self, state: TaskState):
        raise NotImplementedError(f"The function initialize must be defined per constraint request and define self.constraints / consraint_msg")

    def apply_request(self, state: TaskState) -> TaskState:
        for c in self.constraints:
            c.apply(state)
        return state
    
    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        self.initialize_constraints(state)
        if self.constraint_msg is None:
            raise ValueError("The constraint must define a constraint_msg that will be gave to the agent.")
        return [ConstraintBaseStage(
            self.constraint_msg,
            [str(m) for m in state.memory],
            copy.deepcopy(state.attributes)
        )]
    
   
class BaseAttributesModifRequest(BaseRequest):
    """Base request helper for attribute edits.

    Subclasses compute a new attribute snapshot in ``self.att_state`` and this
    class applies it through ``AttributesModifConstraint`` after the stage is
    generated.
    """

    att_state : Dict

    def __init__(self, modifiable_task_attributes : Dict[str,Any]) -> None:
        """
        Pass the dict of attributes {name:list of values} that you want to modify
        """
        super().__init__()
        self.attributes = modifiable_task_attributes
        self.att_state = {}

    def sampling_weight(self, state: TaskState) -> float:
        for entity_name, entity_val in self.attributes.items():
            if type(entity_val) == type(state.attributes[entity_name]):
                return 1
            else:
                raise ValueError(f"Incompatible type between task_attributes {entity_val} and entities {state.attributes[entity_name]} from state")
        return 0
    
    def apply_request(self, state: TaskState) -> TaskState:
        c = AttributesModifConstraint(copy.deepcopy(self.att_state))
        c.apply(state)
        return super().apply_request(state)
    
    def force_state_recompute(self) -> bool:
        return True
