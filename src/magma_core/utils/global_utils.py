# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Dict
import torch, json

import importlib
import pkgutil

def slice_obs(obs : Dict, env_ids : torch.tensor) -> Dict:
    """
    Return a sliced copy of the observation dict to keep only env_ids
    
    :param obs: env observation
    :type obs: Dict
    :param env_ids: env ids to keep
    :type env_ids: torch.tensor
    :return: a sliced obs dict
    :rtype: Dict[Any, Any]
    """
    out = {}

    for key, val in obs.items():
        if isinstance(val, Dict):
            out[key] = slice_obs(val, env_ids)
        else:
            out[key] = val[env_ids]

    return out

def add_batch_dim(data):
    """
    Recursively add a batch dimension (dim=0) to all tensors
    in a (possibly nested) dict.
    """
    if isinstance(data, torch.Tensor):
        return data.unsqueeze(0)
    elif isinstance(data, dict):
        return {k: add_batch_dim(v) for k, v in data.items()}
    else:
        raise TypeError(f"Unsupported type: {type(data)}")

def batch_replace_indices(state_env: dict[str, torch.Tensor], srcs : torch.Tensor, tgts : torch.Tensor):
    """Allows to replace indices inside the state_env (Dict) according to pairs from srcs and tgts"""
    if srcs.shape != tgts.shape:
        raise ValueError(f"srcs and tgts must have the same shape : {srcs.shape} != {tgts.shape}")

    for v in state_env.values():
        if isinstance(v, Dict):
            batch_replace_indices(v, srcs, tgts)
        elif isinstance(v, torch.Tensor):
            v[tgts] = v[srcs]
        else:
            raise ValueError(f"Unknow type inside the state_dict : {type(v)}")

def batch_set_value(state_env: Dict, env_ids : torch.Tensor, template : Dict, strict : bool = True):
    """
    Allows to set the ids from env_ids of state_env to the same value as the template.
    If strict is set to false a key is not present in the template, it's ignored.
    """
    for k,v in state_env.items():
        if isinstance(v, Dict):
            if k in template:
                batch_set_value(v, env_ids, template[k])
            elif not strict:
                batch_set_value(v, env_ids, template)
            else:
                raise RuntimeError(f"Unknow key {k} from state_env, which is not in template")
        elif k in template:
            v[env_ids] = template[k]
        else:
            raise RuntimeError(f"Unknow key {k} from state_env, which is not in template")
        

def extract_env_state_val(st, env_id) -> Dict:
    out = {}
    for key, value in st.items():
        if isinstance(value, Dict):
            out[key] = extract_env_state_val(value, env_id)
        else:
            out[key] = value[env_id].clone()
    return out       

def load_module_from_name(pkg, module_name: str):
    """Return the class corresponding to the module_name inside the pkg. Raise ImportError if not found."""
    for _, m_name, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        module = importlib.import_module(m_name)
        if hasattr(module, module_name):
            return getattr(module, module_name)
    raise ImportError(f"Module '{module_name}' not found in {pkg.__path__}.")
