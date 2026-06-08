# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

"""
Coverage map for this file:

- Recomputing a derived ``TaskState`` from a base snapshot.
- Filtering outdated constraints while preserving base constraints order.
"""

from magma_core.base.constraints import BaseConstraint
from magma_core.base.state import TaskState


class AreaAssignmentConstraint(BaseConstraint):
    def __init__(self, obj: str, area: str) -> None:
        super().__init__()
        self.obj = obj
        self.area = area

    def apply(self, state: TaskState):
        super().apply(state)
        state.relations["object_area"][self.obj] = self.area

    def outdated(self, state: TaskState) -> bool:
        return self.area not in state.attributes["target_areas"]


def test_recompute_from_base_replays_only_non_outdated_constraints_in_order():
    base_state = TaskState()
    base_state.attributes = {
        "objects": ["obj_1", "obj_2"],
        "target_areas": ["area_1", "area_2"],
    }
    base_state.relations["object_type"]["obj_1"] = "fragile"

    current_state = base_state.clone()
    first = AreaAssignmentConstraint("obj_1", "area_1")
    second = AreaAssignmentConstraint("obj_2", "area_2")
    third = AreaAssignmentConstraint("obj_1", "area_2")

    first.apply(current_state)
    second.apply(current_state)
    third.apply(current_state)

    current_state.attributes["target_areas"].remove("area_1")

    recomputed_state = current_state.recompute_from_base(base_state)

    assert recomputed_state.attributes["target_areas"] == ["area_2"]
    assert recomputed_state.relations["object_type"] == {"obj_1": "fragile"}
    assert recomputed_state.relations["object_area"] == {
        "obj_2": "area_2",
        "obj_1": "area_2",
    }
    assert recomputed_state.constraints_history == [second, third]
    assert base_state.attributes["target_areas"] == ["area_1", "area_2"]
    assert base_state.relations["object_area"] == {}


def test_recompute_from_base_keeps_base_constraints_without_replaying_them():
    base_state = TaskState()
    base_state.attributes = {
        "objects": ["obj_1", "obj_2"],
        "target_areas": ["area_1", "area_2"],
    }
    base_constraint = AreaAssignmentConstraint("obj_1", "area_1")
    base_constraint.apply(base_state)

    current_state = base_state.clone()
    extra_constraint = AreaAssignmentConstraint("obj_2", "area_2")
    extra_constraint.apply(current_state)

    recomputed_state = current_state.recompute_from_base(base_state)

    assert recomputed_state.relations["object_area"] == {
        "obj_1": "area_1",
        "obj_2": "area_2",
    }
    assert recomputed_state.constraints_history == [base_constraint, extra_constraint]
