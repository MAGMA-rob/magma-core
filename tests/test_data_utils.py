import pytest

from magma_core.utils.data_utils import (
    apply_att_modif,
    verify_key,
    verify_parameters_dict,
)


def test_apply_att_modif_add_and_remove():
    attributes = {"known_objects": ["box", "cup"]}
    apply_att_modif(attributes, [("ADD", ("known_objects", "bottle"))])
    apply_att_modif(attributes, [("REMOVE", ("known_objects", "cup"))])
    assert attributes["known_objects"] == ["box", "bottle"]


def test_apply_att_modif_raises_on_unknown_field():
    with pytest.raises(ValueError, match="Trying to modify attributes"):
        apply_att_modif({}, [("ADD", ("missing", "x"))])


def test_apply_att_modif_raises_on_non_list_field():
    with pytest.raises(ValueError, match="only remove list attributes"):
        apply_att_modif({"count": 1}, [("ADD", ("count", "x"))])


def test_verify_parameters_dict_success_with_optional():
    ok, msg = verify_parameters_dict(
        params={"name": "robot", "retry": 2},
        allowed_types={"name": str},
        optional={"retry": int},
    )
    assert ok is True
    assert msg == ""


def test_verify_parameters_dict_rejects_unknown_key():
    ok, msg = verify_parameters_dict(
        params={"name": "robot", "unexpected": 1},
        allowed_types={"name": str},
    )
    assert ok is False
    assert "Unknown arguments" in msg


def test_verify_parameters_dict_rejects_missing_key():
    ok, msg = verify_parameters_dict(params={}, allowed_types={"name": str})
    assert ok is False
    assert "Missing arguments" in msg


def test_verify_parameters_dict_rejects_bad_types():
    ok, msg = verify_parameters_dict(
        params={"name": 1, "retry": "two"},
        allowed_types={"name": str},
        optional={"retry": int},
    )
    assert ok is False
    assert "Argument 'name' must be of type str" in msg
    assert "Argument 'retry' must be of type int" in msg


def test_verify_key_happy_path_and_allowed_values():
    value = verify_key({"mode": "safe"}, "mode", possible_values=["safe", "fast"])
    assert value == "safe"


def test_verify_key_raises_when_missing_or_invalid():
    with pytest.raises(ValueError, match="must be specified"):
        verify_key({}, "mode")
    with pytest.raises(ValueError, match="unknow value"):
        verify_key({"mode": "turbo"}, "mode", possible_values=["safe"])
