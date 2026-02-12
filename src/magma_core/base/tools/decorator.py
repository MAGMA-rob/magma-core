import inspect
from typing import Dict, List

# Exemple of correct tool function signature :
# def get_object_state(self, obs : Dict, env_id : int, params : Dict) -> ToolExecution:
# Where obs is the observation dict from the environment, 
# env_id is the id of the env which use the function 
# and params the argument passed to the function

def register_tool(description : str ="", params_spec : Dict = {}, optional : List =[]):

    def decorator(func):
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Check that the function has exactly 4 parameters: self, obs, env_id, params
        required_args = ["self", "obs", "env_id", "params"]
        if params[:4] != required_args:
            raise TypeError(f"Tool function '{func.__name__}' must have arguments {required_args}, got {params}")

        # Check params_spec
        for name, spec in params_spec.items():
            if "description" not in spec or "type" not in spec:
                raise TypeError(f"Parameter '{name}' in tool '{func.__name__}' must have 'description' and 'type'.")
            
        # Check that all params in optional exist in param_spec
        for name in optional:
            if not name in params_spec:
                raise TypeError(f'{name} from optional is not present inside param_spec.')

        func._tool_meta = {
            "name": func.__name__,
            "description": description,
            "params_spec": params_spec,
            "optional" : optional
        }
        return func

    return decorator
