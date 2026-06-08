# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import pytest

from magma_core.base.data_structures import UserInstruction
from magma_core.base.stage import ModifAttributesBaseStage


def _attribute_stage(mode: str) -> ModifAttributesBaseStage:
    return ModifAttributesBaseStage(
        mode=mode,
        instruction=UserInstruction("update areas"),
        val_name="area1",
        att_name="target_areas",
        memory=[],
        preserved_memory_indices=[],
        attributes={"target_areas": ["area1"]},
        flag_answer_to_user=True,
    )


def test_modif_attributes_stage_describes_add_goal():
    stage = _attribute_stage("ADD")

    assert "to add area1 to its target_areas" in stage.stage_goal_description


def test_modif_attributes_stage_describes_remove_goal():
    stage = _attribute_stage("REMOVE")

    assert "to remove area1 from its target_areas" in stage.stage_goal_description


def test_modif_attributes_stage_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported attribute modification mode"):
        _attribute_stage("UPDATE")
