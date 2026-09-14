# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from magma_core.simulation.errors.base_error import BaseError
from magma_core.simulation.data_structures import ToolErrorSupport

# Exemple of correct tool function signature :
# def get_object_state(self, obs : Dict, env_id : int, params : Dict) -> ToolExecution:
# Where obs is the observation dict from the environment, 
# env_id is the id of the env which use the function 
# and params the argument passed to the function

def _validate_error_type(error_cls: type[BaseError], func_name: str) -> None:
    if not inspect.isclass(error_cls) or not issubclass(error_cls, BaseError):
        raise TypeError(
            f"Each error attached to tool '{func_name}' must be a BaseError type. "
            f"Got {error_cls!r}."
        )

    error_sig = inspect.signature(error_cls)
    for param in error_sig.parameters.values():
        if (
            param.default is inspect._empty
            and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ):
            raise TypeError(
                f"Error type '{error_cls.__name__}' attached to tool '{func_name}' "
                "must be instantiable without arguments."
            )

def _normalize_error_supports(
        errors: List[Union[type[BaseError], ToolErrorSupport]],
        func_name: str,
    ) -> List[ToolErrorSupport]:
    normalized: List[ToolErrorSupport] = []

    for error_decl in errors:
        support = (
            error_decl
            if isinstance(error_decl, ToolErrorSupport)
            else ToolErrorSupport(error_type=error_decl)
        )
        if not support.pre and not support.post:
            raise TypeError(
                f"Error support '{support.error_type.__name__}' attached to tool '{func_name}' "
                "must enable pre and/or post."
            )

        _validate_error_type(support.error_type, func_name)
        normalized.append(support)

    return normalized

def register_tool(
        description: str = "",
        params_spec: Optional[Dict] = None,
        optional: Optional[List] = None,
        errors: Optional[List[Union[type[BaseError], ToolErrorSupport]]] = None,
        execution_group: Optional[str] = None,
        is_detection: bool = False,
    ):

    if not isinstance(is_detection, bool):
        raise TypeError("is_detection must be a bool")

    def decorator(func):
        _params_spec = {} if params_spec is None else params_spec
        _optional = [] if optional is None else optional
        _errors = _normalize_error_supports([] if errors is None else errors, func.__name__)

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Check that the function has exactly 4 parameters: self, obs, env_id, params
        required_args = ["self", "obs", "env_id", "params"]
        if params[:4] != required_args:
            raise TypeError(f"Tool function '{func.__name__}' must have arguments {required_args}, got {params}")

        # Check params_spec
        for name, spec in _params_spec.items():
            if "description" not in spec or "type" not in spec:
                raise TypeError(f"Parameter '{name}' in tool '{func.__name__}' must have 'description' and 'type'.")
            
        # Check that all params in optional exist in param_spec
        for name in _optional:
            if not name in _params_spec:
                raise TypeError(f'{name} from optional is not present inside param_spec.')

        func._tool_meta = {
            "name": func.__name__,
            "description": description,
            "params_spec": _params_spec,
            "optional" : _optional,
            "errors": _errors,
            # Tools sharing this group may later be scheduled as mutually exclusive.
            "execution_group": execution_group,
            "is_detection": is_detection,
        }
        return func

    return decorator
