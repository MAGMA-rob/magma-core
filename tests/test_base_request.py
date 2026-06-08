# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

"""
Coverage map for this file:

- Default ``BaseRequest`` hooks and abstract behavior.
- Constraint-request stage generation and replay onto ``TaskState``.
- Attribute-modification requests, including incompatible schema rejection.
"""

import pytest

from magma_core.base.constraints import BaseConstraint
from magma_core.base.data_structures import Situation, UserInstruction
from magma_core.base.stage import BaseTaskStage, ConstraintBaseStage
from magma_core.base.state import TaskState
from magma_core.base.user_request import (
    BaseAttributesModifRequest,
    BaseConstraintRequest,
    BaseRequest,
)


class EchoStage(BaseTaskStage):
    target_steps = 1
    acceptance_steps = 0

    def __init__(self, instruction: str):
        super().__init__([], True, "Echo stage")
        self.situation = Situation(
            instruction=UserInstruction(instruction),
            memory=[],
            preserved_memory_indices=[],
            attributes={},
            flag_answer_to_user=False,
        )
        self.verification_prompt = "ok"


class EchoRequest(BaseRequest):
    def create_stages(self, state: TaskState):
        return [EchoStage("repeat the message")]


class AreaAssignmentConstraint(BaseConstraint):
    def __init__(self, obj: str, area: str) -> None:
        super().__init__()
        self.obj = obj
        self.area = area

    def apply(self, state: TaskState):
        super().apply(state)
        state.relations["object_area"][self.obj] = self.area


class RememberAreaRequest(BaseConstraintRequest):
    def initialize_constraints(self, state: TaskState):
        obj = state.attributes["objects"][0]
        area = state.attributes["target_areas"][-1]
        self.constraints = [AreaAssignmentConstraint(obj, area)]
        self.constraint_msg = f"Always place {obj} in {area}"


class MissingMessageConstraintRequest(BaseConstraintRequest):
    def initialize_constraints(self, state: TaskState):
        self.constraints = []


class RestrictAreasRequest(BaseAttributesModifRequest):
    def __init__(self) -> None:
        super().__init__({"target_areas": []})

    def create_stages(self, state: TaskState):
        self.att_state = {
            "objects": list(state.attributes["objects"]),
            "target_areas": [state.attributes["target_areas"][-1]],
        }
        return [EchoStage("update available areas")]


class InvalidAttributeTypeRequest(BaseAttributesModifRequest):
    def __init__(self) -> None:
        super().__init__({"target_areas": "area_1"})

    def create_stages(self, state: TaskState):
        return []


def test_base_request_default_hooks_are_noops():
    state = TaskState()
    request = EchoRequest()

    stages = request.create_stages(state)

    assert len(stages) == 1
    assert isinstance(stages[0], EchoStage)
    assert request.sampling_weight(state) == 1.0
    assert request.force_state_recompute() is False
    assert request.apply_request(state) is state


def test_base_request_create_stages_must_be_implemented():
    with pytest.raises(NotImplementedError):
        BaseRequest().create_stages(TaskState())


def test_constraint_request_builds_constraint_stage_from_state():
    state = TaskState()
    state.memory = ["user already mentioned the preferred area"]
    state.attributes = {
        "objects": ["cup_1"],
        "target_areas": ["left_zone", "right_zone"],
    }

    request = RememberAreaRequest()
    out = request.create_stages(state)

    assert len(out) == 1
    assert isinstance(out[0], ConstraintBaseStage)
    assert out[0].situation.memory == state.memory
    assert out[0].situation.memory is not state.memory
    assert out[0].situation.attributes == state.attributes
    assert out[0].situation.attributes is not state.attributes
    assert out[0].situation.instruction.get_content() == "Always place cup_1 in right_zone"


def test_constraint_request_apply_request_replays_constraints():
    state = TaskState()
    state.attributes = {
        "objects": ["cup_1"],
        "target_areas": ["left_zone", "right_zone"],
    }
    request = RememberAreaRequest()

    request.create_stages(state)
    updated = request.apply_request(state)

    assert updated is state
    assert state.relations["object_area"] == {"cup_1": "right_zone"}
    assert state.constraints_history == request.constraints


def test_constraint_request_requires_constraint_message():
    with pytest.raises(ValueError, match="constraint_msg"):
        MissingMessageConstraintRequest().create_stages(TaskState())


def test_attributes_modif_request_updates_state_and_requires_recompute():
    state = TaskState()
    state.attributes = {
        "objects": ["cup_1"],
        "target_areas": ["left_zone", "right_zone"],
    }
    request = RestrictAreasRequest()

    stages = request.create_stages(state)
    updated = request.apply_request(state)

    assert len(stages) == 1
    assert isinstance(stages[0], EchoStage)
    assert request.sampling_weight(state) == 1
    assert request.force_state_recompute() is True
    assert updated.attributes == {
        "objects": ["cup_1"],
        "target_areas": ["right_zone"],
    }
    assert updated.constraints_history[-1].__class__.__name__ == "AttributesModifConstraint"


def test_attributes_modif_request_sampling_weight_rejects_incompatible_types():
    state = TaskState()

    with pytest.raises(ValueError, match="Incompatible type"):
        InvalidAttributeTypeRequest().sampling_weight(state)
