# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

"""
Coverage map for this file:

- Attribute mutation helpers.
- Parameter validation for required and optional tool arguments.
- Strict key extraction with allowed-value guards.
"""

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


def test_verify_parameters_dict_success_with_optional():
    ok, msg = verify_parameters_dict(
        params={"name": "robot", "retry": 2},
        allowed_types={"name": str},
        optional={"retry": int},
    )
    assert ok is True
    assert msg == ""


def test_verify_parameters_dict_reports_invalid_arguments():
    invalid_cases = [
        ({"name": "robot", "unexpected": 1}, "Unknown arguments"),
        ({}, "Missing arguments"),
        ({"name": 1, "retry": "two"}, "must be of type"),
        ("name=robot", "Arguments must be a dict"),
    ]

    for params, expected_message in invalid_cases:
        ok, msg = verify_parameters_dict(
            params=params,
            allowed_types={"name": str},
            optional={"retry": int},
        )
        assert ok is False
        assert expected_message in msg


def test_verify_key_happy_path_and_allowed_values():
    value = verify_key({"mode": "safe"}, "mode", possible_values=["safe", "fast"])
    assert value == "safe"


def test_verify_key_raises_when_missing_or_invalid():
    with pytest.raises(ValueError, match="must be specified"):
        verify_key({}, "mode")
    with pytest.raises(ValueError, match="unknow value"):
        verify_key({"mode": "turbo"}, "mode", possible_values=["safe"])
