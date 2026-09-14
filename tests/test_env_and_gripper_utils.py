# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

"""
Coverage map for this file:

- Geometry helpers used by goal verification.
- Random manufacturing-order generation.
- Gripper proximity helpers.

These tests need a real ``torch`` install. The lightweight test stub is good
enough for import-only tests but not for tensor math.
"""

import pytest

torch = pytest.importorskip("torch")

if getattr(torch, "__codex_stub__", False):
    pytest.skip("requires a real torch install for tensor operations", allow_module_level=True)

from magma_core.simulation.utils.env_utils import craft_random_manu_order, is_object_inside_target
from magma_core.simulation.utils.gripper_utils import find_object_in_gripper, is_object_in_gripper
from magma_core.utils.global_utils import restore_disallowed_actor_states


def test_is_object_inside_target_single_and_batch():
    obj = torch.tensor([0.0, 0.0, 0.0])
    target = torch.tensor([0.05, 0.05, 1.0])
    assert is_object_inside_target(obj, target, thresh=0.1, keep_tensor=False) is True

    obj_b = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    target_b = torch.tensor([[0.01, 0.01, 0.0], [0.0, 0.0, 0.0]])
    out = is_object_inside_target(obj_b, target_b, thresh=0.05)
    assert out.tolist() == [True, False]


def test_craft_random_manu_order_shape_and_charset():
    order = craft_random_manu_order(8)
    assert len(order) == 8
    assert order.isalnum()
    with pytest.raises(ValueError, match="at least 2 characters"):
        craft_random_manu_order(1)


def test_find_object_in_gripper_and_is_object_in_gripper():
    tcp = torch.tensor([0.0, 0.0, 0.0])
    objects = {
        "far": torch.tensor([1.0, 1.0, 0.0]),
        "near": torch.tensor([0.01, 0.0, 0.01]),
    }
    found = find_object_in_gripper(tcp, objects, threshold=0.03, z_dist_threshold=0.05)
    assert found == "near"
    assert is_object_in_gripper(tcp, objects["near"], threshold=0.03) is True
    assert is_object_in_gripper(tcp, objects["far"], threshold=0.03) is False


def test_restore_disallowed_actor_states_preserves_allowed_and_articulations():
    source_state = {
        "actors": {
            "allowed": torch.tensor([1.0, 0.0]),
            "restored": torch.tensor([2.0, 0.0]),
        },
        "articulations": {"robot": torch.tensor([3.0, 0.0])},
    }
    state_env = {
        "actors": {
            "allowed": torch.tensor([[10.0, 1.0]]),
            "restored": torch.tensor([[20.0, 2.0]]),
        },
        "articulations": {"robot": torch.tensor([[30.0, 3.0]])},
    }

    restored = restore_disallowed_actor_states(
        state_env,
        source_state,
        env_id=0,
        allowed_actor_names={"allowed"},
    )

    assert restored == ["restored"]
    assert state_env["actors"]["allowed"].tolist() == [[10.0, 1.0]]
    assert state_env["actors"]["restored"].tolist() == [[2.0, 0.0]]
    assert state_env["articulations"]["robot"].tolist() == [[30.0, 3.0]]


def test_restore_disallowed_actor_states_fails_open_without_mutation():
    source_state = {"actors": {"cube": torch.tensor([1.0])}}
    state_env = {"actors": {"cube": torch.tensor([[2.0]])}}

    restored = restore_disallowed_actor_states(
        state_env,
        source_state,
        env_id=0,
        allowed_actor_names={"unknown"},
    )

    assert restored is None
    assert state_env["actors"]["cube"].tolist() == [[2.0]]
