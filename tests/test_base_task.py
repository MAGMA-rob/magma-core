# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import List
import pytest

from magma_core.base.tasks import (
    BaseTask,
    BaseTaskStage,
)
from magma_core.base.data_structures import (
    Situation,
    StageState,
    UserInstruction
)

class DummyStage(BaseTaskStage):
    def __init__(self):
        self.situation = Situation(
            instruction=UserInstruction("dummy"),
            memory=[],
            preserved_memory_indices=[],
            attributes={"a": 1},
            flag_answer_to_user=False
        )
        self.target_steps = 3
        self.acceptance_steps = 2
        self.additive_stage = True
        self.reset_at_end = False
        self.verification_prompt = None


def test_base_task_is_abstract():
    with pytest.raises(TypeError):
        b = BaseTask()
        b.validate()

def test_verif_stage_raises_when_out_of_bounds():
    class DummyTask(BaseTask):
        def __init__(self):
            super().__init__()
            self.stages = [DummyStage()]
            self.NB_STAGES = 1

    task = DummyTask()
    
    with pytest.raises(ValueError):
        task._verif_stage(5)

    with pytest.raises(ValueError):
        task._verif_stage(-1)

def test_get_stage_state_transitions():
    stage = DummyStage()
    stage.target_steps = 3
    stage.acceptance_steps = 2

    class DummyTask(BaseTask):
        def __init__(self):
            super().__init__()
            self.stages = [stage]
            self.NB_STAGES = 1

    task = DummyTask()

    assert task.get_stage_state(0, 0) == StageState.OPTIMAL
    assert task.get_stage_state(0, 3) == StageState.OPTIMAL
    assert task.get_stage_state(0, 4) == StageState.ACCEPTABLE
    assert task.get_stage_state(0, 10) == StageState.EXCEEDED

def test_is_stage_text_only():
    stage = DummyStage()
    stage.verification_prompt = "text"
    
    class DummyTask(BaseTask):
        def __init__(self):
            super().__init__()
            self.stages = [stage]
            self.NB_STAGES = 1

    task = DummyTask()
    assert task.is_stage_text_only(0) is True

def test_combine_stage_verif_scores():
    stage = DummyStage()

    class DummyTask(BaseTask):
        def __init__(self):
            super().__init__()
            self.stages = [stage]
            self.NB_STAGES = 1

    task = DummyTask()
    assert task.combine_stage_verif_scores(0, -1, -1) == -1
    assert task.combine_stage_verif_scores(0, 1, -1) == -1
    assert task.combine_stage_verif_scores(0, 1, 1) == 1
    assert task.combine_stage_verif_scores(0, 1, 0) == 0
    assert task.combine_stage_verif_scores(0, 0, 0) == 0
    assert task.combine_stage_verif_scores(0, 0, -1) == -1
