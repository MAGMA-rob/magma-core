from typing import Any, Dict, List, Optional
from copy import deepcopy

from magma_core.simulation.tasks.base_task import BaseTask, required_task_attrs, TaskMetadata
from magma_core.simulation.data_structures import (
    ActiveStageErrorState,
    ToolBatchContext,
    ToolExecution,
)
from magma_core.simulation.errors import BaseError


class BaseBenchmarkTask(BaseTask):

    stages = []
    task_metadata : TaskMetadata = TaskMetadata()

    # must be specified
    benchmark_possible_errors: List[BaseError]

    saved_task_attributes : Dict
    saved_tools_constant : Dict
    saved_env_options : Dict

    def validate(self):
        for attr in required_task_attrs:
            if not hasattr(self, attr):
                raise TypeError(
                    f"Task '{self.__class__.__name__}' must define attribute '{attr}' "
                    f"(either in __init__ or in class)."
                )
        if len(self.stages) > 0:
            raise TypeError("A Bench Task has some stages defined")

        for att_values in self.all_task_attributes.values():
            if att_values == self.agent_names:
                break
        else:
            if self.all_task_attributes.get("known_robots",None) is not None:
                raise TypeError(f"Fail to verify that attributes {self.all_task_attributes} contains correct robots name {self.agent_names}")
            
            self.all_task_attributes["known_robots"] = self.agent_names

        # Benchmark tasks have no internal MAGMA stages, but the rest of the
        # runtime still passes stage_id=0 for tool execution helpers.
        self.NB_STAGES = 1

        if not hasattr(self, "benchmark_possible_errors"):
            self.benchmark_possible_errors = []

        normalized_errors: List[BaseError] = []
        for error in self.benchmark_possible_errors:
            if isinstance(error, BaseError):
                normalized_errors.append(error)
            elif isinstance(error, type) and issubclass(error, BaseError):
                normalized_errors.append(error())
            else:
                raise TypeError(
                    "benchmark_possible_errors must contain BaseError instances "
                    f"or BaseError classes. Got {error!r}."
                )
        # Ensuring different names per error
        error_names = [error.get_name() for error in normalized_errors]
        duplicate_names = sorted({name for name in error_names if error_names.count(name) > 1})
        if duplicate_names:
            raise TypeError(
                "benchmark_possible_errors contains duplicate error names: "
                f"{duplicate_names}. Please keep unique error identifiers."
            )
        self.benchmark_possible_errors = normalized_errors

        self.saved_task_attributes = deepcopy(self.all_task_attributes)
        self.saved_tools_constant = deepcopy(self.tools_constant)
        self.saved_env_options = deepcopy(self.env_options)

    def get_available_stage_errors(self, stage_id: int) -> List[BaseError]:
        # Benchmark runtime errors are declared directly on the benchmark task
        # rather than on internal MAGMA stages.
        return self.benchmark_possible_errors
    
    def reset_stage(self) -> bool:
        self.all_task_attributes = deepcopy(self.saved_task_attributes)
        self.tools_constant = deepcopy(self.saved_tools_constant)
        self.env_options = deepcopy(self.saved_env_options)
        return True

    def initialize_benchmark_variant(self, attributes : Dict, tools_constant : Dict, env_options : Dict):
        self.saved_task_attributes = deepcopy(attributes)
        self.saved_tools_constant = deepcopy(tools_constant)
        self.saved_env_options = deepcopy(env_options)
        self.reset_stage()

    def rewrite_benchmark_task_payload(self, task_data: Dict) -> Dict:
        """
        Rewrite one benchmark task payload according to the currently active
        benchmark variant.

        Benchmark scenarios can override this hook when their JSON tasks are
        canonically authored with placeholder or scenario-specific names that
        must be resolved after `initialize_benchmark_variant(...)`.
        """
        return deepcopy(task_data)

    def execute_tools(
            self,
            obs: Dict,
            env_id: int,
            attributes: Dict[str, Any],
            function_name: str,
            params: Dict,
            stage_id: int,
            error_state: ActiveStageErrorState,
            agent_id: int = 0,
            tool_batch_context: Optional[ToolBatchContext] = None,
        ) -> ToolExecution:
        """
        In this version we do not check any 
        """

        magma_obs = self.build_tool_observation(
            obs,
            attributes,
            agent_id,
            tool_batch_context,
        )
        exec = self._tools.execute_tools(magma_obs, env_id, function_name, params)
        exec.robot_idx = agent_id
        exec._set_execution_info(
            function_name,
            stage_id,
            self.get_active_stage_error(stage_id, error_state),
        )
        return exec
