# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Collection, Dict, List, Optional, Tuple
import json
import re
from torch import Tensor

import importlib
import pkgutil

def slice_obs(obs : Dict, env_ids : Tensor) -> Dict:
    """
    Return a sliced copy of the observation dict to keep only env_ids
    
    :param obs: env observation
    :type obs: Dict
    :param env_ids: env ids to keep
    :type env_ids: Tensor
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
    if isinstance(data, Tensor):
        return data.unsqueeze(0)
    elif isinstance(data, dict):
        return {k: add_batch_dim(v) for k, v in data.items()}
    else:
        raise TypeError(f"Unsupported type: {type(data)}")

def batch_replace_indices(state_env: dict[str, Tensor], srcs : Tensor, tgts : Tensor):
    """Allows to replace indices inside the state_env (Dict) according to pairs from srcs and tgts"""
    if srcs.shape != tgts.shape:
        raise ValueError(f"srcs and tgts must have the same shape : {srcs.shape} != {tgts.shape}")

    for v in state_env.values():
        if isinstance(v, Dict):
            batch_replace_indices(v, srcs, tgts)
        elif isinstance(v, Tensor):
            v[tgts] = v[srcs]
        else:
            raise ValueError(f"Unknow type inside the state_dict : {type(v)}")

def batch_set_value(state_env: Dict, env_ids : Tensor, template : Dict, strict : bool = True):
    """
    Allows to set the ids from env_ids of state_env to the same value as the template.
    If strict is set to False, missing or shape-incompatible branches from the
    template are ignored, allowing sparse updates.
    """
    if not isinstance(template, dict):
        raise RuntimeError(f"template must be a dict, got {type(template)}")

    for key, value in state_env.items():
        if key not in template:
            if strict:
                raise RuntimeError(f"Unknow key {key} from state_env, which is not in template")
            continue

        template_value = template[key]
        if isinstance(value, Dict):
            if not isinstance(template_value, dict):
                if strict:
                    raise RuntimeError(
                        f"Key {key} from state_env is a dict while template contains {type(template_value)}"
                    )
                continue
            batch_set_value(value, env_ids, template_value, strict=strict)
        else:
            if isinstance(template_value, dict):
                if strict:
                    raise RuntimeError(
                        f"Key {key} from state_env is not a dict while template contains a dict"
                    )
                continue
            value[env_ids] = template_value


def apply_env_state_updates(state_env: Dict, env_id: int, updates: List) -> bool:
    """
    Apply tool-produced state updates to one env slot of a batched state dict.
    Each update path must point to a tensor leaf in ``state_env``.
    """
    if not updates:
        return False

    for update in updates:
        target = state_env
        for key in update.path:
            if not isinstance(target, dict) or key not in target:
                raise KeyError(f"Unknown env state update path: {update.path}")
            target = target[key]
        if not isinstance(target, Tensor):
            raise TypeError(f"Env state update path must target a tensor: {update.path}")
        value = update.value
        if isinstance(value, Tensor):
            value = value.to(device=target.device, dtype=target.dtype)
        target[env_id] = value

    return True


def restore_disallowed_actor_states(
        state_env: Dict,
        source_state: Dict,
        env_id: int,
        allowed_actor_names: Collection[str],
    ) -> Optional[List[str]]:
    """Restore one env slot's actors outside an explicit allowlist.

    ``None`` reports an incompatible declaration or state layout without
    mutating ``state_env``. An empty list means the protection was valid but no
    actor state differed from the saved source.
    """
    current_actors = state_env.get("actors")
    source_actors = source_state.get("actors")
    if not isinstance(current_actors, dict) or not isinstance(source_actors, dict):
        return None

    allowed_actors = set(allowed_actor_names)
    if not allowed_actors.issubset(current_actors) or not allowed_actors.issubset(source_actors):
        return None
    if set(current_actors) != set(source_actors):
        return None

    replacements = []
    restored_actor_names = []
    for actor_name, source_value in source_actors.items():
        if actor_name in allowed_actors:
            continue
        current_value = current_actors[actor_name]
        if not isinstance(current_value, Tensor) or not isinstance(source_value, Tensor):
            return None
        if current_value[env_id].shape != source_value.shape:
            return None
        prepared_source_value = source_value.to(
            device=current_value.device,
            dtype=current_value.dtype,
        )
        replacements.append((current_value, prepared_source_value))
        if not current_value[env_id].equal(prepared_source_value):
            restored_actor_names.append(actor_name)

    for current_value, source_value in replacements:
        current_value[env_id] = source_value.clone()

    return sorted(restored_actor_names)


def extract_env_state_val(st, env_id) -> Dict:
    out = {}
    for key, value in st.items():
        if isinstance(value, Dict):
            out[key] = extract_env_state_val(value, env_id)
        else:
            out[key] = value[env_id].clone()
    return out       

def _normalize_state_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())

def extract_robot_articulations(articulations: Dict, robot_names: List[str]) -> Tuple[Dict, bool]:
    """
    Return only the articulation entries that correspond to the provided robot
    names. If the mapping cannot be resolved for every robot, the original
    articulations dict is returned with False so callers can safely fall back.
    """
    if not isinstance(articulations, dict) or len(robot_names) == 0:
        return articulations, False

    resolved_keys: List[str] = []
    used_keys = set()

    for robot_name in robot_names:
        if robot_name in articulations:
            resolved_keys.append(robot_name)
            used_keys.add(robot_name)
            continue

        normalized_name = _normalize_state_name(robot_name)
        candidates = [
            key for key in articulations.keys()
            if key not in used_keys and _normalize_state_name(key) == normalized_name
        ]
        if len(candidates) != 1:
            return articulations, False

        resolved_keys.append(candidates[0])
        used_keys.add(candidates[0])

    return {key: articulations[key] for key in resolved_keys}, True

def merge_robot_articulations(
        base_articulations: Dict,
        source_articulations: Dict,
        robot_names: List[str],
    ) -> Tuple[Dict, bool]:
    """
    Keep the full articulation tree from base_articulations, but override only
    the robot articulation entries with values from source_articulations.

    If robot articulations cannot be resolved reliably, fall back to the full
    source articulations to preserve the previous behavior.
    """
    if not isinstance(source_articulations, dict):
        return source_articulations, False
    if not isinstance(base_articulations, dict):
        return source_articulations, False

    robot_articulations, matched = extract_robot_articulations(source_articulations, robot_names)
    if not matched:
        return source_articulations, False

    merged_articulations = base_articulations.copy()
    merged_articulations.update(robot_articulations)
    return merged_articulations, True

def load_module_from_name(pkg, module_name: str):
    """Return the class corresponding to the module_name inside the pkg. Raise ImportError if not found."""
    for _, m_name, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        try:
            module = importlib.import_module(m_name)
            if hasattr(module, module_name):
                return getattr(module, module_name)
        except:
            print(f"[WARN] Failed to import {m_name}")
    raise ImportError(f"Module '{module_name}' not found in {pkg.__path__}.")
