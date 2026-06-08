# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import pytest

from magma_core.base.errors import BaseError


class DummyError(BaseError):
    required_key = ["required"]
    optional_key = ["optional"]

    def initialize(self, obs, env_id):
        return {}

    def get_description(self, arguments):
        return ""


def test_validate_arguments_accepts_required_and_optional_keys():
    DummyError().validate_arguments({"required": ["x"], "optional": 1})


def test_validate_arguments_rejects_invalid_keys():
    invalid_inputs = [
        None,
        {},
        {"optional": 1},
        {"required": ["x"], "unexpected": True},
    ]

    for arguments in invalid_inputs:
        with pytest.raises(RuntimeError):
            DummyError().validate_arguments(arguments)
