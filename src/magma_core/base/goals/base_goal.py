from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List
import torch

class BaseGoal(ABC):
    """
    Base class for goal predicates.

    A goal represents a condition that must become true in the environment.
    It is evaluated from environment observations.

    Goals are intended to be composed inside a GoalStage.
    
    We recommend to make the difference between pure positive predicates and failure predicates.
    But you are free to design a predicates which evaluate both success and failure.

    Return convention for verification:
        1  -> goal satisfied
        0  -> not yet satisfied
        -1 -> failure / violation
    """

    def __init__(
        self,
        name: Optional[str] = None,
        metadata : Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        name:
            Optional human-readable identifier for debugging or logging.

        metadata:
            Optional dictionary storing predicate-specific data.
            Useful for debugging, logging, or visualization.
        """

        self.name = name or self.__class__.__name__
        self.metadata = metadata if metadata else "no info"

    # ---------------------------------------------------------------------
    # Main API
    # ---------------------------------------------------------------------

    @abstractmethod
    def verify(self, obs: Dict) -> torch.Tensor:
        """
        Evaluate goal completion from environment observations.

        Parameters
        ----------
        obs : Dict
            Observation dictionary coming from the environment.

        Returns
        -------
        torch.Tensor
            Tensor of shape (num_envs,) with values in {-1, 0, 1}.
        """
        raise NotImplementedError("Must be defined in child class")

    # ---------------------------------------------------------------------
    # Debug utilities
    # ---------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, metadata={self.metadata})"


class Or(BaseGoal):

    goals : List[BaseGoal]

    def __init__(self, goal_list : List[BaseGoal]) -> None:
        if not goal_list:
            raise ValueError("Or goal requires at least one sub-goal")
        super().__init__("OR", str([g.metadata for g in goal_list]))
        self.goals = goal_list

    def verify(self, obs: Dict) -> torch.Tensor:
        local_results = [goal.verify(obs) for goal in self.goals]
        stacked = torch.stack(local_results)

        has_success = (stacked == 1).any(dim=0)
        has_pending = (stacked == 0).any(dim=0)

        out = torch.full_like(local_results[0], fill_value=-1)
        out[has_pending] = 0
        out[has_success] = 1

        return out


class And(BaseGoal):
    """
    And is mostly use for benchmark purpose as it's corresponding to the 
    default behavior to verify all goals from the goal_list.
    """

    goals : List[BaseGoal]

    def __init__(self, goal_list : List[BaseGoal]) -> None:
        if not goal_list:
            raise ValueError("And goal requires at least one sub-goal")
        super().__init__("AND", str([g.metadata for g in goal_list]))
        self.goals = goal_list

    def verify(self, obs: Dict) -> torch.Tensor:
        out = self.goals[0].verify(obs)
        for goal in self.goals[1:]:
            out = torch.minimum(out, goal.verify(obs))
        return out
