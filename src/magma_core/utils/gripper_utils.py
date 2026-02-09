import torch
from typing import Dict, Optional

def find_object_in_gripper(
    tcp_pose: torch.Tensor,
    object_poses: Dict[str, torch.Tensor],
    threshold: float = 0.02,
    z_dist_threshold: float = 0.05
) -> Optional[str]:
    """
    Find the first object in the gripper based on the TCP pose and a dict of object poses.

    Args:
        tcp_pose (torch.Tensor): The TCP pose of the gripper (shape: (3,)).
        object_poses (Dict[str, torch.Tensor]): A dictionary where keys are object names and values are their positions (shape: (3,)).
        threshold (float): Distance threshold to consider an object as being in the gripper.
        z_dist_threshold (float): Vertical distance threshold to pre-filter objects by height.

    Returns:
        str: The name of the first object found within the threshold, or None if no object is found.
    """
    names = list(object_poses.keys())
    positions = torch.stack([object_poses[name] for name in names], dim=0)  # shape (N, 3)

    # quick z-axis pre-filter
    dz = torch.abs(positions[:, 2] - tcp_pose[2])
    close_mask = dz < z_dist_threshold
    if not close_mask.any():
        return None

    candidates = positions[close_mask]
    candidate_names = [name for i, name in enumerate(names) if close_mask[i]]

    # compute full distance for candidates
    tcp_pos = tcp_pose.view(1, 3)
    distances = torch.norm(candidates - tcp_pos, dim=1)
    mask = distances < threshold

    if mask.any():
        idx = mask.nonzero(as_tuple=True)[0][0]
        return candidate_names[idx]

    return None

def is_object_in_gripper(tcp_pose : torch.Tensor, object_pose : torch.Tensor, threshold=0.005):
    """
    Check if objects are in the gripper for a batch of TCP poses and object poses.
    Works with single poses ([3] or [7]) and batched poses ([B, 3] or [B, 7]).

    Args:
        tcp_poses (torch.Tensor): Tensor of shape [3] or [B, N] (at least first 3 are xyz).
        object_poses (torch.Tensor): Tensor of shape [3] or [B, N] (at least first 3 are xyz).
        threshold (float): Distance threshold to consider the object as being in the gripper.

    Returns:
        torch.Tensor: Boolean scalar if input was 1D, or boolean tensor of shape [B] if input was batched.
    """
    # Ensure both inputs are at least 2D
    if tcp_pose.ndim == 1:
        tcp_pose = tcp_pose.unsqueeze(0)
    if object_pose.ndim == 1:
        object_pose = object_pose.unsqueeze(0)

    # Compute pairwise distances
    dists = torch.norm(tcp_pose[:, :3] - object_pose[:, :3], dim=1)
    result = dists < threshold

    # If the original input was 1D, return a scalar boolean
    return result if result.shape[0] > 1 else result.item()