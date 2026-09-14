from magma_core.simulation.state import RuleRenderer, TaskState
from magma_core.simulation.requests import BaseRequest
from magma_core.simulation.tools import BaseToolsAPI
from magma_core.simulation.data_structures import SituationInit
from magma_core.simulation.tasks.base_task import BaseTask, InitializationParameters, TaskMetadata

from copy import deepcopy
from typing import Any, Dict, List, Optional, Type


class TaskDefinition:

    # Required runtime fields
    starting_state : TaskState
    maniskill_env_id : str
    situation_init : SituationInit
    Tools_cls : Type[BaseToolsAPI]
    active_requests : List[BaseRequest]
    RuleRenderer_cls = RuleRenderer

    # Optional runtime fields
    name : str
    randomized_config_path : str
    initialization_parameters : InitializationParameters
    task_metadata : TaskMetadata
    tools_constant : Dict[str, Any]

    def __init__(
            self,
            situation_init : SituationInit,
            starting_state : Optional[TaskState] = None,
            maniskill_env_id : Optional[str] = None,
            Tools_cls : Optional[Type[BaseToolsAPI]] = None,
            active_requests : Optional[List[BaseRequest]] = None,
            name : Optional[str] = None,
            randomized_config_path : Optional[str] = None,
            initialization_parameters : Optional[InitializationParameters] = None,
            task_metadata : Optional[TaskMetadata] = None,
            tools_constant : Optional[Dict[str, Any]] = None,
        ) -> None:
        self.situation_init = deepcopy(situation_init)

        if starting_state is not None:
            self.starting_state = deepcopy(starting_state)
        elif not hasattr(self, "starting_state"):
            self.starting_state = TaskState()
            self.starting_state.attributes = deepcopy(situation_init.attributes)
            self.starting_state.memory = deepcopy(situation_init.memory)

        if maniskill_env_id is not None:
            self.maniskill_env_id = maniskill_env_id
        if Tools_cls is not None:
            self.Tools_cls = Tools_cls
        if active_requests is not None:
            self.active_requests = deepcopy(active_requests)
        if name is not None:
            self.name = name
        if randomized_config_path is not None:
            self.randomized_config_path = randomized_config_path
        if initialization_parameters is not None:
            self.initialization_parameters = deepcopy(initialization_parameters)
        elif not hasattr(self, "initialization_parameters"):
            self.initialization_parameters = InitializationParameters()
        if task_metadata is not None:
            self.task_metadata = deepcopy(task_metadata)
        elif not hasattr(self, "task_metadata"):
            self.task_metadata = TaskMetadata()
        if tools_constant is not None:
            self.tools_constant = deepcopy(tools_constant)
        elif not hasattr(self, "tools_constant"):
            self.tools_constant = {}
        if not hasattr(self, "randomized_config_path"):
            self.randomized_config_path = ""
        if not hasattr(self, "name"):
            self.name = f"Build from {self.__class__.__name__}"

    def build_default_task(self) -> BaseTask:
        task = BaseTask()
        missing_attrs = [
            attr
            for attr in ("maniskill_env_id", "Tools_cls", "active_requests")
            if not hasattr(self, attr)
        ]
        if missing_attrs:
            raise ValueError(
                f"TaskDefinition '{self.__class__.__name__}' must define: "
                f"{', '.join(missing_attrs)}."
            )

        task.maniskill_env_id = self.maniskill_env_id
        task.Tools_cls = self.Tools_cls
        task.situation_init = deepcopy(self.situation_init)
        task.initialization_parameters = deepcopy(self.initialization_parameters)
        task.task_metadata = deepcopy(self.task_metadata)
        task.randomized_config_path = self.randomized_config_path
        task.tools_constant = deepcopy(self.tools_constant)
        task.name = self.name

        return task
