# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Type

AttributeUpdate = Tuple[Literal["ADD", "REMOVE"], Tuple[str, str]]


def apply_att_modif(
    attributes: Dict[str, Any],
    modif_list: Sequence[AttributeUpdate],
) -> None:
    """Apply ordered ADD/REMOVE updates to list-valued attributes."""
    for action, (att_name, att_val) in modif_list:
        if action not in {"ADD", "REMOVE"}:
            raise ValueError(f"Unknown attribute update action {action!r}.")
        if att_name not in attributes:
            raise ValueError(
                f"Trying to modify attribute {att_name!r} from existing "
                f"attributes {sorted(attributes)}."
            )
        if not isinstance(attributes[att_name], list):
            raise ValueError("Only list attributes can be updated for now.")
        if action == "ADD":
            attributes[att_name].append(att_val)
        else:
            attributes[att_name].remove(att_val)

def verify_parameters_dict(params: Dict, allowed_types: Dict[str, Type], optional : Dict[str, Type] = {}) -> Tuple[bool, str]:
    """Verify a params dict:
    - No unexpected keys
    - All required keys present
    - Each value matches expected type

    allowed_types: dict mapping {key: expected_type}
    """
    if not isinstance(params, dict):
        return False, f"Arguments must be a dict, got {type(params).__name__}."

    params_keys = set(params.keys())
    allowed_keys = set(allowed_types.keys())
    optional_keys = set(optional.keys())

    # unexpected keys
    unexpected = params_keys - allowed_keys - optional_keys
    if unexpected:
        return False, f"Unknown arguments: {', '.join(unexpected)}"

    # missing keys
    missing = allowed_keys - params_keys
    if missing:
        return False, f"Missing arguments: {', '.join(missing)}"

    # type checking
    r=""
    for key, expected_type in allowed_types.items():
        if not isinstance(params[key], expected_type):
            r+=f"Argument '{key}' must be of type {expected_type.__name__}, got {type(params[key]).__name__}. "

    for key, expected_type in optional.items():
        if key in params and not isinstance(params[key], expected_type):
           r+=f"Argument '{key}' must be of type {expected_type.__name__}, got {type(params[key]).__name__}. " 
    
    if r!= "":
        return False, r

    return True, ""

def verify_key(config : Dict, key_name : str, possible_values : Optional[List[Any]] = None) -> Any:
        """
        Verify the presence of the keys in the dict.
        Return the value associate with the key if existing, else Raise Value Error.
        """
        key_val = config.get(key_name, None)
        if not key_val:
            raise ValueError(f"{key_name} key must be specified and != None")
        if possible_values:
            if not key_val in possible_values:
                raise ValueError(f"{key_name} has un unknow value {key_val}. Should be in {possible_values}")
        
        return key_val
