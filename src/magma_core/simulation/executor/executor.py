# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Any, Dict, List, Optional, Union
from collections import OrderedDict
import torch, logging, copy
import gymnasium as gym
from abc import ABC, abstractmethod

from magma_core.simulation.trajectory import TrajectoryConverter
from magma_core.simulation.data_structures import (
    ActiveStageErrorState,
    ToolExecution,
    EnvToolContext,
    RobotToolContext,
    ToolBatchContext,
)
from magma_core.domain import Call

from magma_core.simulation.tasks import BaseTask
from magma_core.simulation.envs import DefaultEnv
from magma_core.simulation.randomizer import Randomizer, RuntimeRandomizer

from magma_core.workers import LMWorker, LMWorkerPool
from magma_core.utils.text_utils import join_with_and


class ToolsBaseExecutor(ABC):
    """
    Tools executor is the main class responsible for transforming LLM calls into actions using specific defined class.
    """

    nb_env : int
    logger : logging.Logger

    env : DefaultEnv
    _current_env_id : str
    trajectory_converter : TrajectoryConverter

    # Params
    _gui : bool
    _visual_assets : bool
    worker : Optional[Union[LMWorker, LMWorkerPool]]
    planner_endpoint : str

    def __init__(
            self,
            nb_env : int,
            ollama_worker: Optional[Union[LMWorker, LMWorkerPool]],
            planner_endpoint : str,
            gui : bool = False,
            visual_assets: bool = False,
        ):
        self.worker = ollama_worker
        self.nb_env = nb_env
        self._gui = gui
        self._visual_assets = gui or visual_assets
        self._current_env_id = ""

        self.logger = logging.getLogger("EXECUTION")
        self.planner_endpoint = planner_endpoint

    def reset_environment(
            self,
            env_options: Dict[str, Any],
            *,
            seed: Optional[int],
            reconfigure: bool,
        ):
        """Reset the environment with executor-owned runtime options."""
        reset_options = copy.deepcopy(env_options)
        reset_options["use_visual_assets"] = self._visual_assets
        reset_options["reconfigure"] = reconfigure
        return self.env.reset(seed=seed, options=reset_options)

    def _create_envs(
            self,
            maniskill_env_id : str,
            env_options : Dict,
            planner_options : Dict,
            obs_mode: str = "state_dict",
            sim_backend: str = "auto",
        ) -> DefaultEnv:
        """
        Initialize the Executor with a task name class and some arguments for its __init__.
        Tasks must be present in the scenarios.scenarios package.
        """
        if sim_backend == "auto" and not torch.cuda.is_available():
            sim_backend = "cpu"

        
        if self._current_env_id != maniskill_env_id: # We create a new env only if the task required a different env_id then precedently.
            if hasattr(self, "env"):
                self.env.close()
            
            # env init
            if self._gui:
                self.env = gym.make( #type: ignore
                    maniskill_env_id,
                    num_envs = self.nb_env,
                    obs_mode=obs_mode,
                    control_mode="pd_joint_pos",
                    render_mode="human",
                    sim_backend=sim_backend,
                    parallel_in_single_scene=self.nb_env > 1,
                    )
            else:
                self.env = gym.make( #type: ignore
                    maniskill_env_id,
                    num_envs = self.nb_env,
                    obs_mode=obs_mode,
                    control_mode="pd_joint_pos",
                    render_mode="rgb_array",
                    sim_backend=sim_backend,
                )

        self._current_env_id = maniskill_env_id

        self.reset_environment(env_options, seed=0, reconfigure=True)

        # trajectory converter init
        self.trajectory_converter = TrajectoryConverter(
            self.env.unwrapped, #type: ignore
            self.nb_env,
            self.planner_endpoint,
            planner_init_options = planner_options
        )
        a = self.step()
        self.env.step(a) # Steps just for make sure the env state is updated

        return self.env
    
    def _initialize_stage_error_state(
            self,
            task_ref: BaseTask,
            stage_id: int,
            error_state: ActiveStageErrorState,
            obs: Dict,
            env_id: int,
            attributes: Dict[str, Any],
            agent_id: int = 0,
        ) -> ActiveStageErrorState:
        """Lazily select the active errors for one task and stage."""
        if error_state:
            return error_state
        if not task_ref.get_available_stage_errors(stage_id):
            return {}
        return task_ref.initialize_stage_errors(
            stage_id,
            obs,
            attributes,
            env_id,
            agent_id,
        )

    def _compute_tool(
            self,
            *,
            task_ref: BaseTask,
            randomizer: Optional[Union[Randomizer, RuntimeRandomizer]],
            calls: List[Call],
            env_id: int,
            obs: Dict,
            error_state: ActiveStageErrorState,
            stage_id: int,
            current_node_step: int,
            original_log_length: int,
            source_node_id: int,
            attributes: Dict[str, Any],
            previous_tool_calls: int,
            previous_forgiven_tool_calls: int,
            node_id: int = -1,
            logs: Optional[List] = None,
            composite_progress: Optional[Dict[str, Any]] = None,
        ) -> EnvToolContext:
        """Translate and execute tool calls against an explicit task runtime."""
        error_state = self._initialize_stage_error_state(
            task_ref=task_ref,
            stage_id=stage_id,
            error_state=error_state,
            obs=obs,
            env_id=env_id,
            attributes=attributes,
        )

        names = task_ref.get_agent_names()
        active_stage_errors = task_ref.get_active_stage_error(stage_id, error_state)
        unknown_robots = [
            call.target_robot_name
            for call in calls
            if call.target_robot_name not in names
        ]
        if unknown_robots:
            if randomizer is not None:
                randomizer.set_tmp_translation(node_id, None)
            return EnvToolContext(
                node_id=node_id,
                tool_robots=[
                    RobotToolContext(
                        "default",
                        ToolExecution(
                            [],
                            verifier=None,
                            reason=(
                                f"Unknow robot names {join_with_and(unknown_robots)}. "
                                "Please use only knowns robots names"
                            ),
                        ),
                    )
                ],
                error_state=error_state,
                active_stage_errors=active_stage_errors,
                logs=logs,
                stage_log_start_idx=original_log_length,
                task_stage=stage_id,
                source_id=source_node_id,
                agent_step=current_node_step,
                attributes=attributes,
                tool_calls=previous_tool_calls + len(calls),
                forgiven_tool_calls=previous_forgiven_tool_calls,
                composite_progress=composite_progress,
            )

        robot_tools = []
        tool_batch_context = ToolBatchContext()
        public_function_names = []
        for call in calls:
            public_function_names.append(call.name)
            agent_id = names.index(call.target_robot_name)
            if not call.name:
                continue

            function_name = call.name
            arguments = call.arguments
            if randomizer is not None:
                translated_call = randomizer.map_tool_call(function_name, arguments)
                if isinstance(translated_call, ToolExecution):
                    translated_call.robot_idx = agent_id
                    robot_tools.append(
                        RobotToolContext(call.target_robot_name, translated_call)
                    )
                    continue
                function_name, arguments = translated_call

            robot_tools.append(
                RobotToolContext(
                    call.target_robot_name,
                    task_ref.execute_tools(
                        obs,
                        env_id,
                        attributes,
                        function_name,
                        arguments,
                        stage_id,
                        error_state,
                        agent_id,
                        tool_batch_context,
                    ),
                )
            )

        if not robot_tools:
            raise RuntimeError(
                "A fully empty multiple tool call reached the tool executor."
            )

        if randomizer is not None:
            randomizer.set_tmp_translation(node_id, public_function_names)

        return EnvToolContext(
            node_id=node_id,
            tool_robots=robot_tools,
            error_state=error_state,
            active_stage_errors=active_stage_errors,
            logs=logs,
            source_id=source_node_id,
            stage_log_start_idx=original_log_length,
            task_stage=stage_id,
            agent_step=current_node_step,
            attributes=attributes,
            tool_calls=previous_tool_calls + len(calls),
            forgiven_tool_calls=previous_forgiven_tool_calls,
            composite_progress=composite_progress,
        )

    def _pass_to_the_next_stage(
            self,
            task_ref: BaseTask,
            current_stage_id: int,
            env_ids: List[int],
            env_state: Dict,
        ) -> Dict:
        """Apply the stage transition effects for an explicit task runtime."""
        task_ref.call_stage_change_effects(current_stage_id, env_state, env_ids)
        return env_state
    
    @abstractmethod
    def compute_actions(self, tools_call : Dict):
        """
        Take a batch of tools_calls. It's a dict where each key is a node_id associate with a dict.
        In case of GENERATION mode, the value dict contains 'tool' (the action dict) and 'src_id' (the parent node_id).
        In case of EVALUATION mode, the value dict contains directly the action dict.
        Transforms this into a batched sequence of steps per environment.
        """
        raise NotImplementedError("This function must be implemented in child class")

    @abstractmethod
    def step(self) -> Union[torch.Tensor, OrderedDict]:
        """
        Execute the actions in the environment.

        Returns:
            Dict: If SingleAGent env, return a batched tensor of action. For MultiAGent, return an OrderedDict
        """

        raise NotImplementedError("This function must be implemented in child class")

    @abstractmethod
    def verif_ended_tool(self, obs : Dict) -> Dict:
        """
        Verify if some envs have finished executing their ToolsExecution elements from the observation.

        If in GENERATION mode, it will also check if the overall task is finished.
        
        :param obs: Observation of the environment
        :type obs: Dict
        :return: A dict where each finished node_id have its assoiated return status. COuld be empty if no environments have finished.
        :rtype: Dict[int, Dict]
        """
        raise NotImplementedError("This function must be implemented in child class")
