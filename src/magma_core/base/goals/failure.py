from typing import Dict, Optional, Any, List
import torch

from magma_core.utils.env_utils import is_object_inside_target
from .base_goal import BaseGoal


def _get_pose(obs_entry):
    if isinstance(obs_entry, dict):
        return obs_entry["pose"]
    return obs_entry


class NotAt(BaseGoal):
    """
    Failure-oriented goal that rejects forbidden placements.

    It returns success while the object stays outside every forbidden area and
    switches to ``-1`` or ``0`` as soon as the object enters one, depending on
    the ``strict`` flag.
    """

    def __init__(self, obj_name: str, locations: List[str], strict : bool, thresh: float = 0.1):
        super().__init__(name="NotAt", metadata=f"{obj_name} not in any {locations}")
        self.obj = obj_name
        self.locations = locations
        self.strict = strict
        self.thresh = thresh

    def verify(self, obs: Dict) -> torch.Tensor:

        pos = _get_pose(obs["extra"][self.obj])
        device = pos.device
        nb_envs = pos.shape[0]

        inside_any = torch.zeros(nb_envs, dtype=torch.bool, device=device)

        for location in self.locations:
            inside_any |= is_object_inside_target(
                pos,
                _get_pose(obs["extra"][location]),
                thresh=self.thresh,
                keep_tensor=True
            )

        out = torch.ones(nb_envs, dtype=torch.int8, device=device)

        out[inside_any] = -1 if self.strict else 0

        return out


class MaxAt(BaseGoal):
    """Failure goal that forbids having more than ``maximum`` objects in one target."""

    def __init__(self, objects: List[str], location: str, maximum: int, strict: bool = True):
        if maximum < 0:
            raise ValueError("maximum must be non-negative")
        super().__init__(name="MaxAt", metadata=f"max {maximum} of {objects} at {location}")
        self.objects = objects
        self.location = location
        self.maximum = maximum
        self.strict = strict

    def verify(self, obs: Dict) -> torch.Tensor:
        location_pose = _get_pose(obs["extra"][self.location])
        device = location_pose.device
        nb_envs = location_pose.shape[0]

        count = torch.zeros(nb_envs, device=device, dtype=torch.int32)

        for obj in self.objects:
            count += is_object_inside_target(
                _get_pose(obs["extra"][obj]),
                location_pose,
                keep_tensor=True,
            ).int()

        out = torch.ones(nb_envs, dtype=torch.int8, device=device)
        out[count > self.maximum] = -1 if self.strict else 0

        return out
