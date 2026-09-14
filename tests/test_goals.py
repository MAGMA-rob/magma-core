# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

"""
Coverage map for this file:

- Positive placement goals across raw and component observations.
- Negative / bounded goals with strict and soft failure modes.

These tests need a real ``torch`` install. The lightweight test stub is only
meant to keep import-time dependencies optional elsewhere in the suite.
"""

import pytest

torch = pytest.importorskip("torch")

if getattr(torch, "__codex_stub__", False):
    pytest.skip("requires a real torch install for tensor operations", allow_module_level=True)

from magma_core.simulation.goals import At, AtLeastCountAt, ExactCountAt, MaxAt, NotAt


def _entry(pose, use_components: bool):
    if use_components:
        return {"pose": pose}
    return pose


@pytest.mark.parametrize("use_components", [False, True])
def test_at_supports_raw_and_component_observations(use_components: bool):
    obs = {
        "extra": {
            "cup": _entry(torch.tensor([[0.00, 0.00, 0.00], [0.30, 0.30, 0.00]]), use_components),
            "plate": _entry(torch.tensor([[0.02, 0.01, 0.00], [0.00, 0.00, 0.00]]), use_components),
        }
    }

    result = At("cup", "plate", thresh=0.05).verify(obs)

    assert result.tolist() == [1, 0]


@pytest.mark.parametrize("use_components", [False, True])
def test_count_at_counts_objects_across_formats(use_components: bool):
    obs = {
        "extra": {
            "cup_1": _entry(torch.tensor([[0.00, 0.00, 0.00], [0.00, 0.00, 0.00]]), use_components),
            "cup_2": _entry(torch.tensor([[0.02, 0.01, 0.00], [0.40, 0.40, 0.00]]), use_components),
            "tray": _entry(torch.tensor([[0.01, 0.01, 0.00], [0.00, 0.00, 0.00]]), use_components),
        }
    }

    result = AtLeastCountAt(["cup_1", "cup_2"], "tray", minimum=2).verify(obs)

    assert result.tolist() == [1, 0]


@pytest.mark.parametrize("use_components", [False, True])
def test_exact_count_at_requires_exact_quantity(use_components: bool):
    obs = {
        "extra": {
            "cup_1": _entry(torch.tensor([[0.00, 0.00, 0.00], [0.00, 0.00, 0.00]]), use_components),
            "cup_2": _entry(torch.tensor([[0.02, 0.01, 0.00], [0.40, 0.40, 0.00]]), use_components),
            "tray": _entry(torch.tensor([[0.01, 0.01, 0.00], [0.00, 0.00, 0.00]]), use_components),
        }
    }

    result = ExactCountAt(["cup_1", "cup_2"], "tray", expected=1).verify(obs)

    assert result.tolist() == [0, 1]


@pytest.mark.parametrize("use_components", [False, True])
def test_not_at_reports_failures_with_component_observations(use_components: bool):
    obs = {
        "extra": {
            "cup": _entry(torch.tensor([[0.01, 0.01, 0.00], [0.40, 0.40, 0.00]]), use_components),
            "sink": _entry(torch.tensor([[0.00, 0.00, 0.00], [0.00, 0.00, 0.00]]), use_components),
            "trash": _entry(torch.tensor([[1.00, 1.00, 0.00], [1.00, 1.00, 0.00]]), use_components),
        }
    }

    strict_result = NotAt("cup", ["sink", "trash"], strict=True).verify(obs)
    soft_result = NotAt("cup", ["sink", "trash"], strict=False).verify(obs)

    assert strict_result.tolist() == [-1, 1]
    assert soft_result.tolist() == [0, 1]


@pytest.mark.parametrize("use_components", [False, True])
def test_max_at_supports_component_format_and_soft_failures(use_components: bool):
    obs = {
        "extra": {
            "cup_1": _entry(torch.tensor([[0.00, 0.00, 0.00], [0.40, 0.40, 0.00]]), use_components),
            "cup_2": _entry(torch.tensor([[0.02, 0.01, 0.00], [0.35, 0.35, 0.00]]), use_components),
            "tray": _entry(torch.tensor([[0.01, 0.01, 0.00], [0.00, 0.00, 0.00]]), use_components),
        }
    }

    strict_result = MaxAt(["cup_1", "cup_2"], "tray", maximum=1, strict=True).verify(obs)
    soft_result = MaxAt(["cup_1", "cup_2"], "tray", maximum=1, strict=False).verify(obs)

    assert strict_result.tolist() == [-1, 1]
    assert soft_result.tolist() == [0, 1]
