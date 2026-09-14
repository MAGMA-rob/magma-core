"""Public serialization primitives shared by MAGMA packages."""

from magma_core.serialization.spec import (
    build_spec,
    construct_from_spec,
    canonical_type_name,
    extract_constructor_arguments,
    load_spec,
    qualified_type_name,
    validate_constructor_arguments,
)
from magma_core.serialization.values import decode_value, encode_value

__all__ = [
    "canonical_type_name",
    "build_spec",
    "construct_from_spec",
    "decode_value",
    "encode_value",
    "extract_constructor_arguments",
    "load_spec",
    "qualified_type_name",
    "validate_constructor_arguments",
]
