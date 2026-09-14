from __future__ import annotations

import importlib
import inspect
from types import UnionType
from typing import Callable, Any, Dict, Literal, Type, Union, get_args, get_origin, get_type_hints

from magma_core.serialization.values import decode_value, encode_value


def qualified_type_name(value_type: Type[Any]) -> str:
    return f"{value_type.__module__}:{value_type.__qualname__}"


def canonical_type_name(type_name: str) -> str:
    """Resolve constructor paths stored before the core namespace separation."""
    module_name, qualified_name = type_name.split(":", 1)
    legacy_modules = {
        "magma_core.base.data_structures.agent_call": "magma_core.domain.agent_call",
    }
    module_name = legacy_modules.get(module_name, module_name)
    if module_name.startswith("magma_core.base."):
        module_name = module_name.replace("magma_core.base.", "magma_core.simulation.", 1)
    return f"{module_name}:{qualified_name}"


def build_spec(
    instance: Any, arguments: Dict[str, Any], *,
    encode: Callable[[Any], Any] = encode_value,
) -> Dict[str, Any]:
    if not isinstance(arguments, dict):
        raise TypeError("Serialized constructor arguments must be a dictionary")
    return {
        "type": qualified_type_name(type(instance)),
        "arguments": encode(arguments),
    }


def load_spec(
    spec: Dict[str, Any],
    expected_base: Type[Any],
    *, decode: Callable[[Any], Any] = decode_value,
) -> tuple[Type[Any], Dict[str, Any]]:
    if not isinstance(spec, dict):
        raise TypeError("Object spec must be a dictionary")
    if set(spec) != {"type", "arguments"}:
        raise ValueError("Object spec must contain exactly 'type' and 'arguments'")
    type_name = spec["type"]
    if not isinstance(type_name, str) or ":" not in type_name:
        raise ValueError("Object spec type must use the 'module:ClassName' format")
    if not isinstance(spec["arguments"], dict):
        raise TypeError("Object spec arguments must be a dictionary")

    module_name, qualified_name = canonical_type_name(type_name).split(":", 1)
    try:
        value: Any = importlib.import_module(module_name)
        for part in qualified_name.split("."):
            value = getattr(value, part)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"Cannot load object type {type_name!r}: {error}") from error
    if not inspect.isclass(value) or not issubclass(value, expected_base):
        raise TypeError(
            f"{type_name!r} is not a {expected_base.__name__} subclass"
        )
    arguments = decode(spec["arguments"])
    if not isinstance(arguments, dict):
        raise TypeError("Decoded object arguments must be a dictionary")
    return value, arguments


def extract_constructor_arguments(instance: Any) -> Dict[str, Any]:
    signature = inspect.signature(type(instance).__init__)
    arguments: Dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if not hasattr(instance, name):
            raise ValueError(
                f"{type(instance).__name__} cannot use default serialization: "
                f"constructor argument {name!r} is not stored under the same name"
            )
        arguments[name] = getattr(instance, name)
    return arguments


def validate_constructor_arguments(
    object_type: Type[Any],
    arguments: Dict[str, Any],
) -> None:
    signature = inspect.signature(object_type.__init__)
    try:
        signature.bind(None, **arguments)
    except TypeError as error:
        raise TypeError(
            f"Invalid constructor arguments for {object_type.__name__}: {error}"
        ) from error
    try:
        annotations = get_type_hints(object_type.__init__)
    except (NameError, TypeError):
        annotations = {}
    for name, value in arguments.items():
        annotation = annotations.get(name, signature.parameters[name].annotation)
        if not _matches_annotation(value, annotation):
            raise TypeError(
                f"Argument {name!r} of {object_type.__name__} expects "
                f"{annotation!r}, got {type(value).__name__}"
            )


def construct_from_spec(
    spec: Dict[str, Any],
    expected_base: Type[Any],
    *, decode: Callable[[Any], Any] = decode_value,
) -> Any:
    object_type, arguments = load_spec(spec, expected_base, decode=decode)
    validate_constructor_arguments(object_type, arguments)
    try:
        return object_type(**arguments)
    except Exception as error:
        raise ValueError(
            f"Cannot reconstruct {qualified_type_name(object_type)!r}: {error}"
        ) from error


def _matches_annotation(value: Any, annotation: Any) -> bool:
    if annotation in {Any, inspect.Parameter.empty}:
        return True
    origin = get_origin(annotation)
    if origin in {Union, UnionType}:
        return any(_matches_annotation(value, option) for option in get_args(annotation))
    if origin is Literal:
        return value in get_args(annotation)
    if origin in {list, tuple}:
        element_types = get_args(annotation)
        expected_container = list if origin is list else tuple
        return isinstance(value, expected_container) and (
            not element_types
            or all(_matches_annotation(item, element_types[0]) for item in value)
        )
    if origin is dict:
        key_value_types = get_args(annotation)
        return isinstance(value, dict) and (
            not key_value_types
            or all(
                _matches_annotation(key, key_value_types[0])
                and _matches_annotation(item, key_value_types[1])
                for key, item in value.items()
            )
        )
    if annotation is type(None):
        return value is None
    if annotation in {str, bool, int, float}:
        return type(value) is annotation
    return isinstance(value, annotation) if inspect.isclass(annotation) else True
