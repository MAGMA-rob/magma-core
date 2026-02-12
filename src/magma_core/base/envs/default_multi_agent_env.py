import sapien
import torch
from typing import List


from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from .default_env import DefaultEnv

class DefaultMultiAgentEnv(DefaultEnv):
    
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**19,
                max_rigid_contact_count=2**21,
            )
        )