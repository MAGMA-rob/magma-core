import requests, os, json_numpy, torch
from typing import Dict, List, Tuple, Union
import numpy as np
json_numpy.patch()

from ..data_structures import ToolInfos, ToolExecution
from ..envs import DefaultEnv
from collections import OrderedDict

mplib_container_url = os.getenv("MPLIB_URL", None)
if not mplib_container_url:
    raise ValueError("You need to define the mplib port")

class TrajectoryConverter():
    """
    This class allows to compute the steps of action based on the Trajectory computed by the tool from the Task.
    It's initialized with the pose of each controllable agents from environment
    """

    agents : Dict # Dict with key as agent names and value as ref to the BaseAgent class from maniskill
    agents_name : List[str]

    max_nb_env : int # The number of parrallel env

    def __init__(
            self,
            env : DefaultEnv, # ref to default env class
            max_nb_env : int,
            planner_init_options : Dict = {}
        ) -> None:
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
            f"{mplib_container_url}/init",
            json=req)
        
    def _single_agent_step(self, existing_env_dict : Dict[int, ToolInfos]) -> torch.Tensor:
        """Return a torch tensor containing all action for all parrallel environment based on the env dict"""
        agent = self.agents_name[0]
        qpos = self.agents[agent].robot.get_qpos().cpu().numpy()
        action_list = np.empty((self.max_nb_env, 8), dtype=np.float32)
        action_list[:] = qpos[:, :8]
        for k, tool_infos in existing_env_dict.items():
            if tool_infos.tool_robots and tool_infos.tool_robots[0].latent_actions:
                action_list[k] = tool_infos.tool_robots[0].latent_actions.pop(0)

        return torch.tensor(action_list, dtype=torch.float32)
    
    def _multi_agent_step(self, existing_env_dict : Dict[int, ToolInfos]) -> OrderedDict[str, torch.Tensor]:
        out = OrderedDict()
        for a in self.agents_name: 
            action_list = np.empty((self.max_nb_env, 8), dtype=np.float32)
            qpos = self.agents[a].robot.get_qpos().cpu().numpy()
            action_list[:] = qpos[:, :8]           
            out[a] = action_list
 
        for k, tool_infos in existing_env_dict.items():
            for tool in tool_infos.tool_robots:
                latent = tool.latent_actions
                if latent:
                    out[tool.real_robot_name][k] = latent.pop(0)

        for a, l in out.items():
            out[a] = torch.tensor(l, dtype=torch.float32)

        return out

    def step(self, existing_env_dict : Dict[int, ToolInfos]) -> Union[torch.Tensor, OrderedDict[str, torch.Tensor]]:
        """Compute the next maniskill action for a given dict of env with key as env_id and value as ToolInfos"""
        if len(self.agents) == 1:
            return self._single_agent_step(existing_env_dict)
        return self._multi_agent_step(existing_env_dict)
    
    def _poses_to_action(self, tool_execution : ToolExecution, agent_name : str, env_id : int) -> Tuple[bool, List]:
        
        q_pos = self.agents[agent_name].robot.get_qpos()[env_id][:8].cpu().tolist()
        base_pose = self.agents[agent_name].robot.pose[env_id]
        array_base_pose = base_pose.raw_pose.cpu().tolist()[0]
        
        latent_action = []

        for p in tool_execution.poses:
            if isinstance(p, str): #Gripper action
                q_pos = q_pos.copy()
                if p == "CLOSE":
                    q_pos[7] = -1
                elif p == "OPEN":
                    q_pos[7] = 1
                elif p == "OK":
                    continue
                else:
                    raise ValueError("Unknow string action into poses")
                latent_action.append(q_pos)
                continue

            result = requests.post(
            f"{mplib_container_url}/plan",
            json={
                "pose": np.concatenate([p.p, p.q]).tolist(),
                "robot_qpos": q_pos,
                "base_pose" : array_base_pose
            }).json()
            result = json_numpy.loads(result)

            if not result["status"] == "Success":
                return False, latent_action

            if len(result['position']) > 0:
                q_pos = result['position'][-1]
                latent_action.extend(result['position'])

        return True, latent_action

    def transform_poses_in_actions(self, tool_infos : ToolInfos, env_id : int) -> bool:
        """
        Populate the latent_action dict from the dict of tool_execution of a tool_infos
        Return False if any planning has encounter an error.
        """
        if not tool_infos.tool_robots:
            raise ValueError("Try to compute actions from fully empty toolinfos")
        
        robot_idx = []
        out = True
        for tool in tool_infos.tool_robots:
            tool_exec = tool.tool_execution
            if not tool_exec:
                raise ValueError("Try to compute actions from empty toolexecution")
            
            if tool_exec.robot_idx >= len(self.agents_name):
                raise RuntimeError(f"The robot indice ({tool_exec.robot_idx}) is not valid. Verify that you correctly set robot_idx in the ToolExecution return of the tool.")
            
            if tool_exec.robot_idx in robot_idx:
                raise RuntimeError(f"The robot indice ({tool_exec.robot_idx}) was already used in a tool. Verify that you correctly set robot_idx in the ToolExecution return of the tool.")
            robot_idx.append(robot_idx)
            agent_name = self.agents_name[tool_exec.robot_idx]
            if not agent_name in tool.latent_actions:
                tool.latent_actions = []

            b, latent = self._poses_to_action(tool_exec, agent_name, env_id)
            if not b: out = False
            tool.latent_actions.extend(latent)
            tool.real_robot_name = agent_name
        return out