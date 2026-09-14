# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import pytest

from magma_core.simulation.data_structures import StageInput, UserInstruction
from magma_core.simulation.stage import ModifAttributesBaseStage


def _attribute_stage(mode: str) -> ModifAttributesBaseStage:
    stage_input = StageInput(
        instruction=UserInstruction("update areas"),
        flag_answer_to_user=True,
    )
    return ModifAttributesBaseStage(
        mode=mode,
        stage_input=stage_input,
        val_name="area1",
        att_name="target_areas",
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
