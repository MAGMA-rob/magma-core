# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import torch
import random, string

def is_object_inside_target(obj_pose: torch.Tensor, target_pose: torch.Tensor, thresh: float = 0.1, keep_tensor : bool = True) -> torch.Tensor:
    """
    Check if the object is within the target position.

    Works with both batched tensors (N × D) and single environment tensors (D,).
    Returns a boolean tensor of shape (N,) or a single boolean if only one environment.

    If keep_tensor is set to True, it will return a torch tensor even if there is only one environment.
    """
    # Ensure both have a batch dimension
    if obj_pose.ndim == 1:
        obj_pose = obj_pose.unsqueeze(0)
    if target_pose.ndim == 1:
        target_pose = target_pose.unsqueeze(0)
    
    # Compute Euclidean distance in the XY plane
    dist = torch.norm(obj_pose[:, :2] - target_pose[:, :2], dim=1)
    result = dist <= thresh
    return result if result.numel() > 1 or keep_tensor else result.item()

def craft_random_manu_order(nb_character: int) -> str:
    """
    Generate a random manufacturing order code.

    Mix of character and digits
    """
    if nb_character < 2:
        raise ValueError("Manufacturing order must have at least 2 characters.")

    charset = string.ascii_uppercase + string.digits
    return ''.join(random.choices(charset, k=nb_character))