from typing import Dict, List
import torch

from magma_core.utils.env_utils import is_object_inside_target
from .base_goal import BaseGoal


def _get_pose(obs_entry):
    if isinstance(obs_entry, dict):
        return obs_entry["pose"]
    return obs_entry


def _count_objects_at_location(
        obs: Dict,
        objects: List[str],
        location: str,
        thresh: float = 0.1,
    ) -> torch.Tensor:
    location_pose = _get_pose(obs["extra"][location])
    nb_envs = 1 if location_pose.ndim == 1 else location_pose.shape[0]

    device = location_pose.device
    count = torch.zeros(
        nb_envs,
        device=device,
        dtype=torch.int32
    )

    for obj in objects:
        count += is_object_inside_target(
            _get_pose(obs["extra"][obj]),
            location_pose,
            thresh=thresh,
            keep_tensor=True,
        ).int()

    return count


class At(BaseGoal):
    """Success goal that checks whether one object is inside one target area."""

    def __init__(self, obj_name: str, location: str, thresh : float = 0.1):
        super().__init__(name="At", metadata=f"{obj_name} to {location}")
        self.obj = obj_name
        self.location = location
        self.thresh = thresh

    def verify(self, obs: Dict) -> torch.Tensor:
        pos = _get_pose(obs["extra"][self.obj])
        target = _get_pose(obs["extra"][self.location])

        inside = is_object_inside_target(pos, target, thresh=self.thresh, keep_tensor=True)

        return inside.int()

class On(BaseGoal):
    """Success goal that checks whether one object is above another one."""

    def __init__(self, top_object: str, bottom_object: str, thresh: float = 0.05):
        super().__init__(name="On", metadata=f"{top_object} on {bottom_object}")
        self.top_object = top_object
        self.bottom_object = bottom_object
        self.thresh = thresh

    def verify(self, obs: Dict) -> torch.Tensor:
        top_pose = _get_pose(obs["extra"][self.top_object])
        bottom_pose = _get_pose(obs["extra"][self.bottom_object])

        if top_pose.ndim == 1:
            top_pose = top_pose.unsqueeze(0)
        if bottom_pose.ndim == 1:
            bottom_pose = bottom_pose.unsqueeze(0)

        aligned_xy = is_object_inside_target(
            top_pose,
            bottom_pose,
            thresh=self.thresh,
            keep_tensor=True,
        )
        return (aligned_xy & (top_pose[:, 2] > bottom_pose[:, 2])).int()
    
class AtLeastCountAt(BaseGoal):
    """Success goal that requires at least ``minimum`` objects in one target."""

    def __init__(self, objects : List, location : str, minimum: int, thresh: float = 0.1):
        super().__init__(name="AtLeastCountAt")
        self.objects = objects
        self.location = location
        self.minimum = minimum
        self.thresh = thresh

    def verify(self, obs: Dict) -> torch.Tensor:
        count = _count_objects_at_location(obs, self.objects, self.location, thresh=self.thresh)
        return (count >= self.minimum).int()


class ExactCountAt(BaseGoal):
    """Success goal that requires exactly ``expected`` objects in one target."""

    def __init__(self, objects: List[str], location: str, expected: int, thresh: float = 0.1):
        super().__init__(name="ExactCountAt")
        self.objects = objects
        self.location = location
        self.expected = expected
        self.thresh = thresh

    def verify(self, obs: Dict) -> torch.Tensor:
        count = _count_objects_at_location(obs, self.objects, self.location, thresh=self.thresh)
        return (count == self.expected).int()
