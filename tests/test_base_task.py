# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

"""
Coverage map for this file:

- Core ``BaseTask`` validation guards.
- Stage-state boundary computation.
- Small stage metadata helpers such as text-only and linked-to-previous.
"""

import pytest

from magma_core.base.data_structures import EmptyInstruction, Situation, StageState, ToolExecution, ToolResult, UserInstruction
from magma_core.base.errors import BaseError
from magma_core.base.stage import AskingBaseStage, BaseTaskStage
from magma_core.base.tasks import BaseTask
from magma_core.base.tools import BaseToolsAPI, register_tool


class DummyStage(BaseTaskStage):
    def __init__(self):
        super().__init__(goals=[], reset_at_end=False, stage_goal_description="dummy stage")
        self.situation = Situation(
            instruction=UserInstruction("dummy"),
            memory=[],
            preserved_memory_indices=[],
            attributes={"a": 1},
            flag_answer_to_user=False,
        )
        self.target_steps = 3
        self.acceptance_steps = 2
        self.additive_stage = True
        self.verification_prompt = None


class DummyTask(BaseTask):
    def __init__(self, stage: BaseTaskStage):
        super().__init__()
        self.env_id = "dummy-env"
        self.name = "dummy-task"
        self.stages = [stage]
        self.NB_STAGES = 1
        self.Tools_cls = DummyToolsAPI
        self.styles = []
        self.approximal_difficulty = "easy"


class EmptyInstructionStage(BaseTaskStage):
    target_steps = 1
    acceptance_steps = 0

    def __init__(self):
        super().__init__(goals=[], reset_at_end=False, stage_goal_description="empty stage")
        self.situation = Situation(
            instruction=EmptyInstruction(),
            memory=[],
            preserved_memory_indices=[],
            attributes={"a": 1},
            flag_answer_to_user=False,
        )

    def verif_log_completion(self, stage_log, full_log):
        return 1


class LogOnlyStage(BaseTaskStage):
    target_steps = 1
    acceptance_steps = 0

    def __init__(self):
        super().__init__(goals=[], reset_at_end=False, stage_goal_description="log only stage")
        self.situation = Situation(
            instruction=UserInstruction("dummy"),
            memory=[],
            preserved_memory_indices=[],
            attributes={"a": 1},
            flag_answer_to_user=False,
        )

    def verif_log_completion(self, stage_log, full_log):
        return 1


class DummyToolsAPI(BaseToolsAPI):
    @register_tool(description="Ping a dummy tool")
    def ping(self, obs, env_id, params):
        return ToolExecution(
            poses=["OK"],
            verifier=lambda _env: ToolResult(ok=True),
        )


class FirstDummyError(BaseError):
    recovery_extra_steps = 1

    def initialize(self, obs, env_id):
        return {"env_id": env_id, "name": self.get_name()}

    def get_description(self, arguments):
        return ""


class SecondDummyError(BaseError):
    recovery_extra_steps = 2

    def initialize(self, obs, env_id):
        return {"env_id": env_id, "name": self.get_name()}

    def get_description(self, arguments):
        return ""


def test_base_task_validate_raises_for_missing_required_attrs():
    with pytest.raises(TypeError):
        task = BaseTask()
        task.validate()


def test_verif_stage_raises_when_out_of_bounds():
    task = DummyTask(DummyStage())

    with pytest.raises(ValueError):
        task._verif_stage(5)

    with pytest.raises(ValueError):
        task._verif_stage(-1)


def test_get_stage_state_transitions():
    task = DummyTask(DummyStage())

    assert task.get_stage_state(0, 0) == StageState.OPTIMAL
    assert task.get_stage_state(0, 3) == StageState.OPTIMAL
    assert task.get_stage_state(0, 4) == StageState.ACCEPTABLE
    assert task.get_stage_state(0, 10) == StageState.EXCEEDED


def test_get_stage_state_handles_zero_acceptance_boundary():
    stage = DummyStage()
    stage.acceptance_steps = 0
    task = DummyTask(stage)

    assert task.get_stage_state(0, 3) == StageState.EXCEEDED_OPTIMAL
    assert task.get_stage_state(0, 4) == StageState.EXCEEDED


def test_is_stage_text_only():
    stage = DummyStage()
    stage.verification_prompt = "text"
    task = DummyTask(stage)

    assert task.is_stage_text_only(0) is True


def test_combine_stage_verif_scores():
    task = DummyTask(DummyStage())

    assert task.combine_stage_verif_scores(0, -1, -1) == -1
    assert task.combine_stage_verif_scores(0, 1, -1) == -1
    assert task.combine_stage_verif_scores(0, 1, 1) == 1
    assert task.combine_stage_verif_scores(0, 1, 0) == 0
    assert task.combine_stage_verif_scores(0, 0, 0) == 0
    assert task.combine_stage_verif_scores(0, 0, -1) == -1


def test_validate_auto_links_empty_instruction_stage() -> None:
    stage = EmptyInstructionStage()
    assert stage.linked_to_prev is False

    stage.validate(["robot"])

    assert stage.linked_to_prev is True
    assert stage.get_description()["linked_to_prev"] is True


def test_asking_base_stage_defaults_remain_text_only() -> None:
    stage = AskingBaseStage("Where is the mug?", "The mug is on the table.", [], {"a": 1})

    stage.validate(["robot"])

    assert stage.target_steps == 1
    assert stage.acceptance_steps == 0
    assert stage.allow_tools_before_answer is False
    assert stage.get_description()["text_only"] is True
    assert stage.get_description()["allow_tools_before_answer"] is False


def test_asking_base_stage_supports_tool_assisted_configuration() -> None:
    stage = AskingBaseStage(
        "Where is the mug?",
        "The mug is on the table.",
        [],
        {"a": 1},
        allow_tools_before_answer=True,
        allowed_tools=["ping"],
    )
    stage.target_steps = 3
    stage.acceptance_steps= 2

    stage.validate(["robot"])

    assert stage.target_steps == 3
    assert stage.acceptance_steps == 2
    assert stage.allow_tools_before_answer is True
    assert stage.allowed_tools == ["ping"]
    assert stage.get_description()["allow_tools_before_answer"] is True


def test_validate_rejects_allowed_tools_without_allow_tools_flag() -> None:
    stage = AskingBaseStage(
        "Where is the mug?",
        "The mug is on the table.",
        [],
        {"a": 1},
        allowed_tools=["ping"],
    )

    with pytest.raises(TypeError, match="allowed_tools"):
        stage.validate(["robot"])


def test_validate_rejects_tools_before_answer_without_verification_prompt() -> None:
    stage = LogOnlyStage()
    stage.allow_tools_before_answer = True

    with pytest.raises(TypeError, match="verification_prompt"):
        stage.validate(["robot"])


def test_initialize_stage_errors_supports_multiple_active_errors() -> None:
    stage = LogOnlyStage()
    stage.possible_errors = [FirstDummyError(), SecondDummyError()]
    stage.min_active_errors = 2
    stage.max_active_errors = 2
    task = DummyTask(stage)
    task.validate()

    error_state = task.initialize_stage_errors(stage_id=0, obs={}, env_id=7)

    assert error_state == {
        "FirstDummyError": {"env_id": 7, "name": "FirstDummyError"},
        "SecondDummyError": {"env_id": 7, "name": "SecondDummyError"},
    }
    assert task.get_stage_recovery_extra_steps(0, error_state) == 3


def test_initialize_stage_errors_clamps_count_to_possible_errors() -> None:
    stage = LogOnlyStage()
    stage.possible_errors = [FirstDummyError()]
    stage.min_active_errors = 2
    stage.max_active_errors = 3
    task = DummyTask(stage)
    task.validate()

    error_state = task.initialize_stage_errors(stage_id=0, obs={}, env_id=7)

    assert error_state == {
        "FirstDummyError": {"env_id": 7, "name": "FirstDummyError"},
    }


def test_validate_rejects_invalid_error_sampling_bounds() -> None:
    stage = LogOnlyStage()
    stage.possible_errors = [FirstDummyError(), SecondDummyError()]
    stage.min_active_errors = 2
    stage.max_active_errors = 1

    with pytest.raises(TypeError, match="min_active_errors"):
        stage.validate(["robot"])


def test_execute_tools_blocks_stage_disallowed_tool() -> None:
    stage = AskingBaseStage(
        "Where is the mug?",
        "The mug is on the table.",
        [],
        {"a": 1},
        allow_tools_before_answer=True,
        allowed_tools=["other_tool"],
    )
    task = DummyTask(stage)
    task._tools = DummyToolsAPI([], {})
    task.agent_names = ["robot"]

    result = task.execute_tools(
        obs={},
        env_id=0,
        function_name="ping",
        params={},
        stage_id=0,
        error_state={},
    )

    assert result.has_error() is True
    assert result.must_fail_stage is True
    assert "not allowed in this stage" in result.reason
