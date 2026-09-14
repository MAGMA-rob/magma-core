"""Simulation serialization, including support for legacy constructor paths."""

from functools import partial

from magma_core.serialization import spec
from magma_core.serialization.spec import (
    canonical_type_name,
    extract_constructor_arguments,
    qualified_type_name,
    validate_constructor_arguments,
)
from .values import decode_value, encode_value

# Bind the simulation codec while sharing constructor validation with core.
build_spec = partial(spec.build_spec, encode=encode_value)
load_spec = partial(spec.load_spec, decode=decode_value)
construct_from_spec = partial(spec.construct_from_spec, decode=decode_value)

__all__ = [
    "canonical_type_name",
    "build_spec", "construct_from_spec", "decode_value", "encode_value",
    "extract_constructor_arguments", "load_spec", "qualified_type_name",
    "validate_constructor_arguments",
]
