from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import torch

from magma_core.simulation.serialization import (
    build_spec,
    extract_constructor_arguments,
    load_spec,
    validate_constructor_arguments,
)

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

    # ------------------------------------------------------------------
    # Serialization API
    # ------------------------------------------------------------------

    def to_spec(self) -> Dict[str, Any]:
        """Return a reconstructible, JSON-oriented goal specification.

        The default implementation inspects the concrete constructor and uses
        instance attributes with identical names. Goals that rename, derive or
        discard constructor arguments must override ``_to_spec_arguments``.
        """

        # A subclass does not implicitly reuse a parent's custom argument
        # mapping: its constructor may expose a completely different API.
        serializer = type(self).__dict__.get("_to_spec_arguments")
        arguments = (
            BaseGoal._to_spec_arguments(self)
            if serializer is None
            else self._to_spec_arguments()
        )
        return build_spec(self, arguments)

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "BaseGoal":
        """Reconstruct a goal produced by :meth:`to_spec`."""

        goal_class, arguments = load_spec(spec, BaseGoal)
        if cls is not BaseGoal and not issubclass(goal_class, cls):
            raise TypeError(
                f"Goal type {spec['type']!r} is not compatible with {cls.__name__}"
            )
        try:
            if "_from_spec_arguments" in goal_class.__dict__:
                return goal_class._from_spec_arguments(arguments)
            validate_constructor_arguments(goal_class, arguments)
            return goal_class(**arguments)
        except (TypeError, ValueError):
            raise
        except Exception as error:
            raise ValueError(
                f"Cannot reconstruct goal {spec['type']!r}: {error}"
            ) from error

    def _to_spec_arguments(self) -> Dict[str, Any]:
        return extract_constructor_arguments(self)

    @classmethod
    def _from_spec_arguments(cls, arguments: Dict[str, Any]) -> "BaseGoal":
        return cls(**arguments)

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

    def _to_spec_arguments(self) -> Dict[str, Any]:
        return {"goal_list": [goal.to_spec() for goal in self.goals]}

    @classmethod
    def _from_spec_arguments(cls, arguments: Dict[str, Any]) -> "Or":
        if set(arguments) != {"goal_list"}:
            raise ValueError("Or expects only the 'goal_list' argument")
        return cls([BaseGoal.from_spec(spec) for spec in arguments["goal_list"]])


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

    def _to_spec_arguments(self) -> Dict[str, Any]:
        return {"goal_list": [goal.to_spec() for goal in self.goals]}

    @classmethod
    def _from_spec_arguments(cls, arguments: Dict[str, Any]) -> "And":
        if set(arguments) != {"goal_list"}:
            raise ValueError("And expects only the 'goal_list' argument")
        return cls([BaseGoal.from_spec(spec) for spec in arguments["goal_list"]])
