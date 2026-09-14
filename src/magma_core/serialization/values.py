from __future__ import annotations

import numbers
from pathlib import Path
from typing import Any, Callable

import torch


_TYPE_KEY = "__magma_type__"


def encode_value(
    value: Any, *, encode_special: Callable[[Any], Any] | None = None,
) -> Any:
    """Convert supported MAGMA values to an unambiguous JSON value."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, Path):
        return {_TYPE_KEY: "path", "value": str(value)}
    if isinstance(value, torch.Tensor):
        return {
            _TYPE_KEY: "tensor",
            "dtype": str(value.dtype).removeprefix("torch."),
            "value": value.detach().cpu().tolist(),
        }

    if isinstance(value, tuple):
        return {
            _TYPE_KEY: "tuple",
            "value": [encode_value(item, encode_special=encode_special) for item in value],
        }
    if isinstance(value, list):
        return [encode_value(item, encode_special=encode_special) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            return {
                _TYPE_KEY: "mapping",
                "value": [
                    [
                        encode_value(key, encode_special=encode_special),
                        encode_value(item, encode_special=encode_special),
                    ]
                    for key, item in value.items()
                ],
            }
        if _TYPE_KEY in value:
            raise ValueError(f"{_TYPE_KEY!r} is reserved by the MAGMA value codec")
        return {
            key: encode_value(item, encode_special=encode_special)
            for key, item in value.items()
        }
    if encode_special is not None:
        return encode_special(value)
    raise TypeError(f"Unsupported JSON value {value!r} ({type(value).__name__})")


def decode_value(
    value: Any, *, decode_special: Callable[[dict], Any] | None = None,
) -> Any:
    """Reconstruct a value produced by :func:`encode_value`."""

    if isinstance(value, list):
        return [decode_value(item, decode_special=decode_special) for item in value]
    if not isinstance(value, dict):
        return value
    if _TYPE_KEY not in value:
        return {
            key: decode_value(item, decode_special=decode_special)
            for key, item in value.items()
        }

    value_type = value[_TYPE_KEY]
    if value_type == "path":
        _require_tag_keys(value, {_TYPE_KEY, "value"})
        if not isinstance(value["value"], str):
            raise TypeError("Serialized path value must be a string")
        return Path(value["value"])
    if value_type == "tensor":
        _require_tag_keys(value, {_TYPE_KEY, "dtype", "value"})
        if not isinstance(value["dtype"], str) or not hasattr(torch, value["dtype"]):
            raise ValueError(f"Unknown torch dtype {value['dtype']!r}")
        dtype = getattr(torch, value["dtype"])
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"{value['dtype']!r} is not a torch dtype")
        return torch.tensor(value["value"], dtype=dtype)
    if value_type == "tuple":
        _require_tag_keys(value, {_TYPE_KEY, "value"})
        if not isinstance(value["value"], list):
            raise TypeError("Serialized tuple value must be a list")
        return tuple(decode_value(item, decode_special=decode_special) for item in value["value"])
    if value_type == "mapping":
        _require_tag_keys(value, {_TYPE_KEY, "value"})
        if not isinstance(value["value"], list):
            raise TypeError("Serialized mapping value must be a list")
        decoded = {}
        for pair in value["value"]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise TypeError("Every serialized mapping entry must be a pair")
            key = decode_value(pair[0], decode_special=decode_special)
            decoded[key] = decode_value(pair[1], decode_special=decode_special)
        return decoded
    if decode_special is not None:
        return decode_special(value)
    raise ValueError(f"Unknown MAGMA serialized value type {value_type!r}")


def _require_tag_keys(value: dict, expected: set[str]) -> None:
    if set(value) != expected:
        raise ValueError(
            f"Serialized {value.get(_TYPE_KEY)!r} value keys mismatch: "
            f"expected={sorted(expected)}, got={sorted(value)}"
        )
