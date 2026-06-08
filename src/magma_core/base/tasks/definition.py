from ..state import TaskState
from ..user_request import BaseRequest
from ..tools import BaseToolsAPI
from ..tasks_style import TaskStyle
from .base_task import BaseTask

from typing import Any, Dict, List, Optional, Type, Union


class _UnsetType:
    pass


_UNSET = _UnsetType()


def _clone_container(value: Any) -> Any:
    if isinstance(value, dict):
        return value.copy()
    if isinstance(value, list):
        return value.copy()
    return value


class TaskDefinition:

    # required params
    starting_state : TaskState
    env_id : str
    Tools_cls : Type[BaseToolsAPI]
    active_requests : List[BaseRequest]

    def __init__(
            self,
            name : Union[Optional[str], _UnsetType] = _UNSET,
            all_task_attributes : Union[Optional[Dict], _UnsetType] = _UNSET,
            randomized_config_path : Union[str, _UnsetType] = _UNSET,
            approximal_difficulty : Union[str, _UnsetType] = _UNSET,
            styles : Union[List[TaskStyle], _UnsetType] = _UNSET,
            env_options : Union[Dict, _UnsetType] = _UNSET,
            tools_constant : Union[Dict, _UnsetType] = _UNSET
        ) -> None:
        self._init_attr("randomized_config_path", randomized_config_path, "")
        self._init_attr("approximal_difficulty", approximal_difficulty, "Medium")
        self._init_attr("styles", styles, [])
        self._init_attr("all_task_attributes", all_task_attributes, None)
        self._init_attr("env_options", env_options, {})
        self._init_attr("name", name, None)
        self._init_attr("tools_constant", tools_constant, {})

    def _init_attr(self, attr_name: str, value: Any, default: Any) -> None:
        if value is _UNSET:
            value = getattr(self, attr_name, default)
        setattr(self, attr_name, _clone_container(value))

    def build_default_task(self) -> BaseTask:
        task = BaseTask()

        try:
            task.env_id = self.env_id
            task.Tools_cls = self.Tools_cls
        except Exception as e:
            raise ValueError(f"You have probably forget to define one of the needed argument for definition: {self.__class__.__name__}. \
                             \nNeeded arguments: env_id, starting_state, Tools_cls, active_requests") from e

        task.env_options = _clone_container(getattr(self, "env_options", {}))
        task.randomized_config_path = getattr(self, "randomized_config_path", "")
        task.approximal_difficulty = getattr(self, "approximal_difficulty", "Medium")
        task.styles = _clone_container(getattr(self, "styles", []))
        task.tools_constant = _clone_container(getattr(self, "tools_constant", {}))

        all_task_attributes = getattr(self, "all_task_attributes", None)
        if all_task_attributes is None:
            task.all_task_attributes = _clone_container(self.starting_state.attributes)
        else:
            task.all_task_attributes = _clone_container(all_task_attributes)
        
        name = getattr(self, "name", None)
        if name is None:
            task.name = f"Build from {self.__class__.__name__}"
        else:
            task.name = name

        return task
