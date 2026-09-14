# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import logging

import requests, torch
from typing import Dict, List, Tuple, Union
import numpy as np

from magma_core.simulation.data_structures import EnvToolContext, ToolExecution
from magma_core.simulation.envs import DefaultEnv
from collections import OrderedDict

LOGGER = logging.getLogger("TRAJECTORY")

class TrajectoryConverter():
    """
    This class allows to compute the steps of action based on the Trajectory computed by the tool from the Task.
    It's initialized with the pose of each controllable agents from environment
    """

    mplib_container_url : str

    agents : Dict # Dict with key as agent names and value as ref to the BaseAgent class from maniskill
    agents_name : List[str]

    max_nb_env : int # The number of parrallel env

    def __init__(
            self,
            env : DefaultEnv, # ref to default env class
            max_nb_env : int,
            mplib_container_url : str,
            planner_init_options : Dict = {}
        ) -> None:
        self.mplib_container_url = mplib_container_url
        self.max_nb_env = max_nb_env
        agent_names = env.agents_name
        self.agents = {}
        if len(agent_names) == 1: # Single Agent
            self.agents[agent_names[0]] = env.agent
        elif len(agent_names) != 0: # Multi Agent
            self.agents = env.agent.agents_dict.copy()
            for a in agent_names:
                if not a in self.agents:
                    raise ValueError(f"Missing {a} agent from official env agents : {self.agents}")
        else:
            raise RuntimeError(f"The env do not exposes correctly agents_name attributes. Make sure to inherits from DefaultEnv.")
        
        self.agents_name = agent_names
        # Keep one controller target per robot and environment, independently of
        # the lifetime of the RobotToolContext that produced it.
        self._last_actions: Dict[str, np.ndarray] = {
            agent_name: self.agents[agent_name].robot.get_qpos().cpu().numpy()[:, :8].copy()
            for agent_name in self.agents_name
        }
        self._init_planner(planner_init_options, env.control_timestep)

    def _init_planner(self, options : Dict, timesteps : float):
        agent_name = next(iter(self.agents))
        link_names = [link.get_name() for link in self.agents[agent_name].robot.get_links()]
        joint_names = [joint.get_name() for joint in self.agents[agent_name].robot.get_active_joints()]

        req = {"link_names": link_names,
                "joint_names": joint_names,
                'control_timesteps': timesteps,
                'base_pose':[0]*7,
                'options': options}

        requests.post(
            f"{self.mplib_container_url}/init",
            json=req
        )
        
    def _single_agent_step(self, existing_env_dict : Dict[int, EnvToolContext]) -> torch.Tensor:
        """Return a torch tensor containing all action for all parrallel environment based on the env dict"""
        agent = self.agents_name[0]
        action_list = self._last_actions[agent].copy()
        for k, tool_infos in existing_env_dict.items():
            if not tool_infos.tool_robots:
                continue

            tool = tool_infos.tool_robots[0]
            if tool.latent_actions:
                action = np.asarray(tool.latent_actions.pop(0), dtype=np.float32)
                action_list[k] = action
                # Persist the emitted target after the tool context disappears.
                self._last_actions[agent][k] = action.copy()

        return torch.tensor(action_list, dtype=torch.float32)
    
    def _multi_agent_step(self, existing_env_dict : Dict[int, EnvToolContext]) -> OrderedDict[str, torch.Tensor]:
        out = OrderedDict()
        for a in self.agents_name: 
            out[a] = self._last_actions[a].copy()
 
        for k, tool_infos in existing_env_dict.items():
            for tool in tool_infos.tool_robots:
                # Failed calls that were never converted have no resolved environment robot.
                if not tool.real_robot_name:
                    continue

                latent = tool.latent_actions
                if latent:
                    action = np.asarray(latent.pop(0), dtype=np.float32)
                    out[tool.real_robot_name][k] = action
                    # Persist the emitted target after the tool context disappears.
                    self._last_actions[tool.real_robot_name][k] = action.copy()

        for a, l in out.items():
            out[a] = torch.tensor(l, dtype=torch.float32)

        return out

    def step(self, existing_env_dict : Dict[int, EnvToolContext]) -> Union[torch.Tensor, OrderedDict[str, torch.Tensor]]:
        """Compute the next maniskill action for a given dict of env with key as env_id and value as EnvToolContext"""
        if len(self.agents) == 1:
            return self._single_agent_step(existing_env_dict)
        return self._multi_agent_step(existing_env_dict)
    
    def _poses_to_action(
            self,
            tool_execution: ToolExecution,
            agent_name: str,
            env_id: int,
            planner_attempt: int,
        ) -> Tuple[bool, List]:
        
        q_pos = self.agents[agent_name].robot.get_qpos()[env_id][:8].cpu().tolist()
        base_pose = self.agents[agent_name].robot.pose[env_id]
        array_base_pose = base_pose.raw_pose.cpu().tolist()[0]
        
        latent_action = []
        waypoint_count = len(tool_execution.poses)
        tool_name = getattr(tool_execution, "function_name", "unknown")
        waypoint_action_counts: List[int] = []

        for waypoint_index, p in enumerate(tool_execution.poses):
            if isinstance(p, str): #Gripper action
                q_pos = q_pos.copy()
                if p == "CLOSE":
                    q_pos[7] = -1
                elif p == "OPEN":
                    q_pos[7] = 1
                elif p == "OK":
                    waypoint_action_counts.append(0)
                    continue
                else:
                    raise ValueError("Unknow string action into poses")
                latent_action.append(q_pos)
                waypoint_action_counts.append(1)
                continue

            result = requests.post(
            f"{self.mplib_container_url}/plan",
            json={
                "pose": np.concatenate([p.p, p.q]).tolist(),
                # The final controller value is a normalized gripper command,
                # not a physical finger joint position understood by MPLib.
                "robot_qpos": q_pos[:7],
                "base_pose": array_base_pose,
                "robot_name": agent_name,
                "env_id": env_id,
                "waypoint_index": waypoint_index,
                "planner_attempt": planner_attempt,
            }).json()
            
            if not result["status"] == "Success":
                LOGGER.warning(
                    "Trajectory waypoint failed env=%d tool=%s robot=%s "
                    "waypoint=%d/%d status=%s waypoint_actions=%s total_actions=%d",
                    env_id,
                    tool_name,
                    agent_name,
                    waypoint_index + 1,
                    waypoint_count,
                    result["status"],
                    waypoint_action_counts,
                    len(latent_action),
                )
                return False, latent_action

            waypoint_action_count = len(result['position'])
            if waypoint_action_count > 0:
                planned_actions = [
                    arm_qpos + [q_pos[7]]
                    for arm_qpos in result['position']
                ]
                q_pos = planned_actions[-1]
                latent_action.extend(planned_actions)
            waypoint_action_counts.append(waypoint_action_count)

        LOGGER.info(
            "Trajectory planned env=%d stage=%s tool=%s robot=%s waypoints=%d "
            "waypoint_actions=%s total_actions=%d",
            env_id,
            getattr(tool_execution, "associated_stage_id", "unknown"),
            tool_name,
            agent_name,
            waypoint_count,
            waypoint_action_counts,
            len(latent_action),
        )

        return True, latent_action

    def transform_poses_in_actions(
            self,
            tool_infos: EnvToolContext,
            env_id: int,
            planner_attempt: int = 0,
        ) -> bool:
        """
        Populate the latent_action dict from the dict of tool_execution of a tool_infos
        Return False if any planning has encounter an error.
        """
        if not tool_infos.tool_robots:
            raise ValueError("Try to compute actions from fully empty toolinfos")

        # A new execution follows an environment state restore. Resynchronize
        # every robot target from that restored qpos before planning new actions.
        for agent_name in self.agents_name:
            current_qpos = self.agents[agent_name].robot.get_qpos()[env_id][:8].cpu().numpy()
            self._last_actions[agent_name][env_id] = current_qpos.copy()
        
        robot_idx = []
        # Track the accumulated duration of each exclusive group in call order.
        execution_group_delays: Dict[str, int] = {}
        out = True
        for tool in tool_infos.tool_robots:
            tool_exec = tool.tool_execution
            if not tool_exec:
                raise ValueError("Try to compute actions from empty toolexecution")
            
            if tool_exec.robot_idx >= len(self.agents_name):
                raise RuntimeError(f"The robot indice ({tool_exec.robot_idx}) is not valid. Verify that you correctly set robot_idx in the ToolExecution return of the tool.")
            
            if tool_exec.robot_idx in robot_idx:
                raise RuntimeError(f"The robot indice ({tool_exec.robot_idx}) was already used in a tool. Verify that you correctly set robot_idx in the ToolExecution return of the tool.")
            robot_idx.append(tool_exec.robot_idx)
            agent_name = self.agents_name[tool_exec.robot_idx]
            if not agent_name in tool.latent_actions:
                tool.latent_actions = []

            b, latent = self._poses_to_action(
                tool_exec,
                agent_name,
                env_id,
                planner_attempt,
            )
            if not b:
                out = False

            execution_group = tool_exec.execution_group
            if execution_group is not None:
                delay = execution_group_delays.get(execution_group, 0)
                action_count = len(latent)
                if delay > 0:
                    # Hold this robot at its restored position until the previous
                    # tools from the same exclusive group have completed.
                    hold_action = self._last_actions[agent_name][env_id]
                    latent = [hold_action.copy() for _ in range(delay)] + latent
                    LOGGER.info(
                        "Trajectory execution-group delay env=%d node=%d tool=%s "
                        "robot=%s group=%s hold_actions=%d final_actions=%d",
                        env_id,
                        tool_infos.node_id,
                        getattr(tool_exec, "function_name", "unknown"),
                        agent_name,
                        execution_group,
                        delay,
                        len(latent),
                    )
                execution_group_delays[execution_group] = delay + action_count

            tool.latent_actions.extend(latent)
            tool.real_robot_name = agent_name
            LOGGER.info(
                "Trajectory queued env=%d node=%d stage=%d tool=%s robot=%s "
                "planning_success=%s queued_actions=%d",
                env_id,
                tool_infos.node_id,
                tool_infos.current_task_stage,
                getattr(tool_exec, "function_name", "unknown"),
                agent_name,
                b,
                len(tool.latent_actions),
            )
        return out
