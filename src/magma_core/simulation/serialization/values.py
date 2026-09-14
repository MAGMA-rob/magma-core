"""Extended value codec for simulation instructions, transitions, and poses."""

from typing import Any

from magma_core.serialization.values import (
    _require_tag_keys,
    decode_value as decode_core_value,
    encode_value as encode_core_value,
)


def encode_value(value: Any) -> Any:
    return encode_core_value(value, encode_special=_encode_simulation_value)


def decode_value(value: Any) -> Any:
    return decode_core_value(value, decode_special=_decode_simulation_value)


def _encode_simulation_value(value: Any) -> Any:
    # These types themselves use this codec, so resolve them after module initialization.
    from magma_core.simulation.data_structures.situation import Instruction
    from magma_core.simulation.stage.environment_transition import BaseStageEnvironmentTransition

    if isinstance(value, Instruction):
        return {"__magma_type__": "instruction", "spec": value.to_spec()}
    if isinstance(value, BaseStageEnvironmentTransition):
        return {"__magma_type__": "environment_transition", "spec": value.to_spec()}
    if value.__class__.__module__.startswith("sapien") and hasattr(value, "p") and hasattr(value, "q"):
        return {
            "__magma_type__": "sapien_pose",
            "p": encode_value(list(value.p)),
            "q": encode_value(list(value.q)),
        }
    raise TypeError(f"Unsupported JSON value {value!r} ({type(value).__name__})")


def _decode_simulation_value(value: dict) -> Any:
    value_type = value["__magma_type__"]
    if value_type == "instruction":
        _require_tag_keys(value, {"__magma_type__", "spec"})
        from magma_core.simulation.data_structures.situation import Instruction

        return Instruction.from_spec(value["spec"])
    if value_type == "environment_transition":
        _require_tag_keys(value, {"__magma_type__", "spec"})
        from magma_core.simulation.stage.environment_transition import BaseStageEnvironmentTransition

        return BaseStageEnvironmentTransition.from_spec(value["spec"])
    if value_type == "sapien_pose":
        _require_tag_keys(value, {"__magma_type__", "p", "q"})
        import sapien

        return sapien.Pose(p=decode_value(value["p"]), q=decode_value(value["q"]))
    raise ValueError(f"Unknown MAGMA serialized value type {value_type!r}")
