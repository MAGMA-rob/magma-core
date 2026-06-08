# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import pytest

from magma_core.configs import BackendConfig, MAGMAConfig


def _make_config(nb_branch: int) -> MAGMAConfig:
    return MAGMAConfig(
        backends={
            "default": BackendConfig(
                type="ollama",
                endpoint="http://localhost:11434",
                default_model="test-model",
                headers={},
            )
        },
        generate={
            "mode": "single",
            "nb_branch": nb_branch,
            "nb_env": 2,
            "nb_max_update": 2,
        },
        benchmark={},
        magma_agent_address="http://localhost:8888",
        magma_planner_address="http://localhost:8000",
    )


def test_verify_allows_single_branch_generation() -> None:
    _make_config(nb_branch=1).verify()


def test_verify_rejects_zero_branch_generation() -> None:
    with pytest.raises(ValueError, match="nb_branch.*>= 1"):
        _make_config(nb_branch=0).verify()
