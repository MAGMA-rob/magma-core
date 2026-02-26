# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import sapien
import torch
import numpy as np

from typing import Union, Any, List, Tuple, Optional

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs import Pose as MSPose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

class DefaultEnv(BaseEnv):

    agents_name : List[str]

    def __init__(
        self,
        *args,
        robot_uids,
        robot_init_qpos_noise,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.agents_name = []
        if isinstance(robot_uids,str):
            self.agents_name.append(robot_uids)
        elif isinstance(robot_uids,Tuple):
            for i, uid in enumerate(robot_uids):
                self.agents_name.append(f"{uid}-{i}")
        else:
            raise TypeError(f"robot_uids must be 'str' (singleAgent) or Tuple (multiAGent). Got {type(robot_uids)}")
        
        super().__init__(*args, robot_uids=robot_uids, **kwargs)


    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    def _load_agent(
        self,
        options: dict,
        initial_agent_poses: Union[sapien.Pose , MSPose , None] = sapien.Pose(
            p=[-0.615, 0, 0]
        )
    ):
        # this code loads the agent into the current scene. You should use it to specify the initial pose(s) of the agent(s)
        # such that they don't collide with other objects initially
        super()._load_agent(
            options,
            initial_agent_poses=initial_agent_poses
        )

    @property
    def _default_sensor_configs(self):
        # the camera used by the policy
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return [
             CameraConfig(
            "base_camera", pose=pose, width=256, height=256, fov=1, near=0.01, far=100
        )
        ]

    @property
    def _default_human_render_camera_configs(self):
        # this is just like _sensor_configs, but for adding cameras used for rendering when you call env.render()
        # when render_mode="rgb_array" or env.render_rgb_array()
        # Another feature here is that if there is a camera called render_camera, this is the default view shown initially when a GUI is opened
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _setup_sensors(self, options: dict):
        # default code here will setup all sensors. You can add additional code to change the sensors e.g.
        # if you want to randomize camera positions
        return super()._setup_sensors(options)

    def _load_lighting(self, options: dict):
        # default code here will setup all lighting. You can add additional code to change the lighting e.g.
        # if you want to randomize lighting in the scene
        return super()._load_lighting(options)

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: dict):
        return {}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return 0

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        # this should be equal to compute_dense_reward / max possible reward
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

    def get_state_dict(self):
        # this function is important in order to allow accurate replaying of trajectories. Make sure to specify any
        # non simulation state related data such as a random 3D goal position you generated
        # alternatively you can skip this part if the environment's rewards, observations, eval etc. are dependent on simulation data only
        # e.g. self.your_custom_actor.pose.p will always give you your actor's 3D position
        state = super().get_state_dict()
        # state["goal_pos"] = add_your_non_sim_state_data_here
        return state

    def set_state_dict(self, state: dict):
        super().set_state_dict(state)


    ###### Some builders

    def build_a_zone(self, width=0.2, length=0.2, color=[1,1,1,1], collision = False, name = "zone"):
        """Build a zone"""
        builder = self.scene.create_actor_builder()

        material = sapien.render.RenderMaterial(base_color=color)
        size = [width, length, 0.01]
        builder.add_box_visual(pose=sapien.Pose([0, 0, 0])  , half_size=size, material=material)
        
        if collision:
            builder.add_box_collision(pose=sapien.Pose([0, 0, 0])  , half_size=size)
        return builder.build_kinematic(name=name)

    def create_box(
            self,
            thickness = 0.01,
            color=[1,1,1,1],
            size = 0.2,
            height = 0.05,
            name="box",
            add_bottom_wall : bool = False,
            initial_pose : Optional[np.ndarray] = None
        ):
        """Build a container"""
        builder = self.scene.create_actor_builder()

        material = sapien.render.RenderMaterial(base_color=color)
        half_size = size/2
        half_thickness = thickness/2

        wall_pose = sapien.Pose([0, -half_size, height/2])  
        wall_half_size = [half_size, half_thickness, height]

        builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
        builder.add_box_visual(pose=wall_pose, half_size=wall_half_size, material=material)

        wall_pose = sapien.Pose([0, half_size, height/2])  
        wall_half_size = [half_size, half_thickness, height]

        builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
        builder.add_box_visual(pose=wall_pose, half_size=wall_half_size, material=material)

        wall_pose = sapien.Pose([-half_size, 0, height/2])  
        wall_half_size = [half_thickness, half_size + half_thickness, height]

        builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
        builder.add_box_visual(pose=wall_pose, half_size=wall_half_size, material=material)

        wall_pose = sapien.Pose([half_size, 0, height/2])
        wall_half_size = [half_thickness, half_size + half_thickness, height]

        builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
        builder.add_box_visual(pose=wall_pose, half_size=wall_half_size, material=material)

        if add_bottom_wall:
            wall_pose = sapien.Pose([0, 0, 0])
            wall_half_size = [half_size+half_thickness, half_size + half_thickness, half_thickness]

            builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
            builder.add_box_visual(pose=wall_pose, half_size=wall_half_size, material=material)

        if initial_pose is not None:
            builder.set_initial_pose(initial_pose)
        return builder.build_kinematic(name=name)