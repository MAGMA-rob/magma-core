# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from __future__ import annotations
from abc import ABC
from typing import List, Dict, Any, Optional

from magma_core.utils.data_utils import verify_parameters_dict
from ..data_structures import ToolExecution, Observation
from .decorator import ToolErrorSupport

from mani_skill.agents.base_agent import BaseAgent

class Tool:
    """
    Tool Class to initialize a Tool.
    """
    def __init__(
            self,
            name : str,
            description : str,
            params_spec : Dict,
            optional : List,
            func,
            error_types: Optional[List[ToolErrorSupport]] = None,
        ):
        self.name = name
        self.description = description
        self.func = func                # wrapped executor
        self.optional = optional
        self.required_param_spec = params_spec
        self.error_types = [] if error_types is None else list(error_types)
        self.optional_param_spec = {}
        for param in optional:
            self.optional_param_spec[param] = self.required_param_spec.pop(param)

    def __call__(self, instance : BaseToolsAPI, obs : Observation, env_id : int, params : Dict):
        ok, info = verify_parameters_dict(
            params, 
            {name : spec['type'] for name, spec in self.required_param_spec.items()},
            optional= {name : spec['type'] for name, spec in self.optional_param_spec.items()}
        )
        if not ok:
            failed_tool = BaseToolsAPI._return_failed_tool(info)
            failed_tool.compatible_error_supports = list(self.error_types)
            return failed_tool
        tool_execution = self.func(instance, obs, env_id, params)
        tool_execution.compatible_error_supports = list(self.error_types)
        return tool_execution

class BaseToolsAPI(ABC):
    """
    Base Tools API class allowing to register custom tools to be executed in the environment and called by an agent.

    This class will be linked to a BaseTask class. Allowing you to access env_agents.
    """

    registry: Dict[str, Tool] = {}

    env_agents : List[BaseAgent]
    equivalence : Dict[str,int]

    def __init_subclass__(cls):
        super().__init_subclass__()

        # Each subclass gets its own registry, not inherited shared
        cls.registry = {}

        # Scan class methods for @tool
        for attr_name, attr in cls.__dict__.items():
            if hasattr(attr, "_tool_meta"):
                meta = attr._tool_meta
                tool_name = meta["name"] or attr_name
                
                cls.registry[tool_name] = Tool(
                    name=tool_name,
                    description=meta["description"],
                    params_spec=meta["params_spec"],
                    optional=meta["optional"],
                    error_types=meta.get("errors"),
                    func=attr,
                )

    def __init__(self, agents : List[BaseAgent], equivalence : Dict[str,int]):
        self.env_agents = agents
        self.equivalence = equivalence

    def get_agent(self, name : str = '') -> BaseAgent:
        """
        Get the env agent corresponding to the name specified in task.
        If you did not set any specific names to overridee, it must be the real agent name.
        If it's a single agent env, will automatically return the single agent reference.
        If it's a multiple agent env and the name it's not found, it will raise an error.
        """
        if len(self.env_agents) == 1:
            return self.env_agents[0]
        else:
            idx = self.equivalence.get(name,None)
            if idx is None:
                raise RuntimeError("Trying to access a non-existing robot")
            return self.env_agents[idx]
        
    def execute_tools(self, obs: Observation, env_id : int, function_name: str, params: Dict) -> ToolExecution:
        """
        Execute a registered tool
        
        :param obs: observation data from the environment
        :type obs: Observation
        :param env_id: id of the env calling the function
        :type env_id: int
        :param function_name: name of the tool
        :type function_name: str
        :param params: arguments of the tool
        :type params: Dict
        :return: The tool execution data
        :rtype: ToolExecution
        """
        if function_name not in self.registry:
            return BaseToolsAPI._return_failed_tool(
                f"{function_name} is not a known tool"
            )
        return self.registry[function_name](self, obs, env_id, params)
    
    @classmethod
    def _return_failed_tool(
        cls,
        reason : str
    ):
        """
        Helper function to return a failed ToolExecution
        
        :param reason: error message
        :type reason: str
        """
        return ToolExecution(
            poses=[],
            verifier=None,
            reason=reason,
        )
        
    def get_api_description(self):
        """
        Returns the tool API as a list of dictionaries suitable for agents.
        """
        api = []
        for name, tool in self.registry.items():
            params = {}
            tool_api : Dict[str,Any] = {
                "name": tool.name,
                "description": tool.description,
            }
            for pname, pinfo in tool.required_param_spec.items():
                params[pname] = {
                    "description": pinfo.get("description", ""),
                    "type": pinfo.get("type", str).__name__,
                }
            if tool.optional:
                for pname, pinfo in tool.optional_param_spec.items():
                    params[pname] = {
                        "description": pinfo.get("description", ""),
                        "type": pinfo.get("type", str).__name__,
                    }
                tool_api["optional"] = tool.optional

            tool_api["parameters"] = params

            api.append(tool_api)

        return api
