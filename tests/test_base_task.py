# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

"""
Coverage map for this file:

- Core ``BaseTask`` validation guards.
- Stage-state boundary computation.
- Small stage metadata helpers such as text-only and linked-to-previous.
"""

import pytest
import torch

from magma_core.simulation.data_structures import (
    EmptyInstruction,
    SituationInit,
    StageInput,
    StageState,
    StageSuccess,
    ToolBatchContext,
    ToolExecution,
    ToolResult,
    UserInstruction,
)
from magma_core.simulation.errors import BaseError
from magma_core.simulation.stage import (
    AskingBaseStage,
    BaseTaskStage,
    CompletionAnswerStage,
    StageErrorParameters,
    StageGlobalParameters,
    TextOnlyValidationMode,
)
from magma_core.simulation.tasks import BaseTask
from magma_core.simulation.tools import BaseToolsAPI, register_tool


requires_real_torch = pytest.mark.skipif(
    getattr(torch, "__codex_stub__", False),
    reason="Vectorized state assertions require a real torch installation",
)


class DummyStage(BaseTaskStage):
    target_tool_calls = 3
    max_tool_calls = 5

    def __init__(self):
        stage_input = StageInput(
            instruction=UserInstruction("dummy"),
            flag_answer_to_user=False,
        )
        super().__init__(
            goals=[],
            stage_goal_description="dummy stage",
            stage_input=stage_input,
            global_parameters=StageGlobalParameters(additive_stage=True),
        )


class SlotInitializationStage(DummyStage):
    def verif_log_completion(self, *args, **kwargs):
        return True

    def build_init_state(self, env_state, env_ids):
        env_state["value"][env_ids] = 1
        return env_state


class DummyTask(BaseTask):
    def __init__(self, stage: BaseTaskStage):
        super().__init__()
        self.maniskill_env_id = "dummy-env"
        self.name = "dummy-task"
        self.stages = [stage]
        self.NB_STAGES = 1
        self.Tools_cls = DummyToolsAPI
        self.situation_init = SituationInit(attributes={"a": 1})


class EmptyInstructionStage(BaseTaskStage):
    target_tool_calls = 0
    max_tool_calls = None

    def __init__(self):
        stage_input = StageInput(
            instruction=EmptyInstruction(),
            flag_answer_to_user=False,
        )
        super().__init__(
            goals=[],
            stage_goal_description="empty stage",
            stage_input=stage_input,
        )

    def verif_log_completion(self, stage_log, full_log):
        return 1


class LogOnlyStage(BaseTaskStage):
    target_tool_calls = 1
    max_tool_calls = 1

    def __init__(self):
        stage_input = StageInput(
            instruction=UserInstruction("dummy"),
            flag_answer_to_user=False,
        )
        super().__init__(
            goals=[],
            stage_goal_description="log only stage",
            stage_input=stage_input,
        )

    def verif_log_completion(self, stage_log, full_log):
        return 1


class DummyToolsAPI(BaseToolsAPI):
    @register_tool(description="Ping a dummy tool")
    def ping(self, obs, env_id, params):
        return ToolExecution(
            poses=["OK"],
            verifier=lambda _env: ToolResult(ok=True),
            context={"tool_batch_context": obs.tool_batch_context},
        )

    @register_tool(description="Inspect dummy state", is_detection=True)
    def inspect(self, obs, env_id, params):
        return ToolExecution(
            poses=["OK"],
            verifier=lambda _env: ToolResult(ok=True),
        )


def test_completion_answer_stage_uses_say_only_validation() -> None:
    stage = CompletionAnswerStage(["inspect", "inspect"])

    assert stage.get_text_only_validation() is TextOnlyValidationMode.SAY_ONLY
    assert stage.get_allowed_tools() == ["inspect"]
    assert stage.get_target_tool_calls() == 1
    assert stage.get_max_tool_calls() == 3
    assert stage.get_stage_input().linked_to_prev is True
    assert stage.get_stage_input().flag_answer_to_user is False


def test_completion_answer_stage_allows_zero_or_one_detection_as_optimal() -> None:
    stage = CompletionAnswerStage(["inspect"])

    assert stage.get_stage_state(0, StageSuccess.FINISH, []) is StageState.OPTIMAL
    assert stage.get_stage_state(1, StageSuccess.FINISH, []) is StageState.OPTIMAL
    assert stage.get_stage_state(2, StageSuccess.FINISH, []) is StageState.ACCEPTABLE
    assert stage.get_stage_state(3, StageSuccess.ONGOING, []) is StageState.EXCEEDED


class FirstDummyError(BaseError):
    def initialize(self, obs, env_id):
        return {"env_id": env_id, "name": self.get_name()}

    def get_description(self, arguments):
        return ""


class SecondDummyError(BaseError):
    def initialize(self, obs, env_id):
        return {"env_id": env_id, "name": self.get_name()}

    def get_description(self, arguments):
        return ""


def test_base_task_validate_raises_for_missing_required_attrs():
    with pytest.raises(TypeError):
        task = BaseTask()
        task.validate()


def test_detection_tool_metadata_enforces_an_empty_actor_allowlist():
    tools = DummyToolsAPI([], {})

    execution = tools.execute_tools(None, 0, "inspect", {})

    assert DummyToolsAPI.registry["ping"].is_detection is False
    assert DummyToolsAPI.registry["inspect"].is_detection is True
    assert execution.allowed_moving_actors == []


def test_register_tool_rejects_non_boolean_detection_metadata():
    with pytest.raises(TypeError, match="is_detection"):
        register_tool(is_detection="yes")


def test_tool_batch_context_reservations_are_namespaced_and_idempotent():
    context = ToolBatchContext()

    assert context.try_reserve("tray_a", 0, "cube_a") is True
    assert context.try_reserve("tray_a", 0, "cube_a") is True
    assert context.try_reserve("tray_a", 0, "cube_b") is False
    assert context.try_reserve("tray_b", 0, "cube_b") is True
    assert context.get_reservations("tray_a") == {0: "cube_a"}
    assert context.get_reservations("tray_b") == {0: "cube_b"}


def test_base_task_propagates_explicit_tool_batch_context():
    task = DummyTask(DummyStage())
    task._tools = DummyToolsAPI([], {})
    task.initialization_parameters.agent_names = ["robot"]
    context = ToolBatchContext()

    first = task.execute_tools(
        obs={},
        env_id=0,
        attributes={},
        function_name="ping",
        params={},
        stage_id=0,
        error_state={},
        tool_batch_context=context,
    )
    second = task.execute_tools(
        obs={},
        env_id=0,
        attributes={},
        function_name="ping",
        params={},
        stage_id=0,
        error_state={},
        tool_batch_context=context,
    )

    assert first.context["tool_batch_context"] is context
    assert second.context["tool_batch_context"] is context
    assert task.build_tool_observation({}, {}).tool_batch_context is not context


def test_stage_request_type_defaults_to_unknown_and_cannot_be_reassigned():
    stage = DummyStage()

    assert stage.get_request_type() == "unknown"
    assert stage.get_description()["request_type"] == "unknown"

    stage.set_request_type(" EchoRequest ")

    assert stage.get_request_type() == "EchoRequest"
    assert stage.get_description()["request_type"] == "EchoRequest"
    stage.set_request_type("EchoRequest")
    with pytest.raises(RuntimeError, match="already assigned"):
        stage.set_request_type("AnotherRequest")


def test_initialize_task_replaces_prevalidated_default_robot_placeholder():
    task = DummyTask(LogOnlyStage())
    task.validate()

    task.initialize_task(
        agents={"robot": object()},
        agents_name=["robot"],
        env_state={},
        nb_env=1,
        build_first_stage=False,
    )

    assert task.get_agent_names() == ["robot"]
    assert task.get_init_attributes()["known_robots"] == ["robot"]


@requires_real_torch
def test_initialize_task_can_target_selected_environment_slots():
    task = DummyTask(SlotInitializationStage())
    env_state = {"value": torch.zeros((3, 1))}

    initialized = task.initialize_task(
        agents={"robot": object()},
        agents_name=["robot"],
        env_state=env_state,
        nb_env=3,
        env_ids=[1],
    )

    assert initialized["value"].tolist() == [[0.0], [1.0], [0.0]]
    assert task._default_env_state["value"].tolist() == [0.0]


@requires_real_torch
def test_initialize_task_defaults_to_every_environment_slot():
    task = DummyTask(SlotInitializationStage())
    env_state = {"value": torch.zeros((3, 1))}

    initialized = task.initialize_task(
        agents={"robot": object()},
        agents_name=["robot"],
        env_state=env_state,
        nb_env=3,
    )

    assert initialized["value"].tolist() == [[1.0], [1.0], [1.0]]


@pytest.mark.parametrize("env_ids", [[], [0, 0], [-1], [3], [True]])
@requires_real_torch
def test_initialize_task_rejects_invalid_environment_slots(env_ids):
    task = DummyTask(SlotInitializationStage())

    with pytest.raises(ValueError):
        task.initialize_task(
            agents={"robot": object()},
            agents_name=["robot"],
            env_state={"value": torch.zeros((3, 1))},
            nb_env=3,
            env_ids=env_ids,
        )


def test_verif_stage_raises_when_out_of_bounds():
    task = DummyTask(DummyStage())

    with pytest.raises(ValueError):
        task._verif_stage(5)

    with pytest.raises(ValueError):
        task._verif_stage(-1)


def test_get_stage_state_transitions():
    task = DummyTask(DummyStage())

    assert task.get_stage_state(0, 0, 0, StageSuccess.ONGOING) == StageState.OPTIMAL
    assert task.get_stage_state(0, 3, 0, StageSuccess.FINISH) == StageState.OPTIMAL
    assert task.get_stage_state(0, 4, 0, StageSuccess.FINISH) == StageState.ACCEPTABLE
    assert task.get_stage_state(0, 5, 0, StageSuccess.ONGOING) == StageState.EXCEEDED


def test_get_stage_state_handles_unknown_target():
    stage = DummyStage()
    stage.target_tool_calls = None
    task = DummyTask(stage)

    assert task.get_stage_state(0, 3, 0, StageSuccess.FINISH) == StageState.UNASSESSED
    assert task.get_stage_state(0, 5, 0, StageSuccess.ONGOING) == StageState.EXCEEDED


def test_stage_derives_max_tool_calls_from_target() -> None:
    class DefaultMaxStage(DummyStage):
        target_tool_calls = 3
        max_tool_calls = None

    stage = DefaultMaxStage()

    assert stage.max_tool_calls == 6


def test_stage_rejects_missing_target_and_max_tool_calls() -> None:
    class UnboundedStage(DummyStage):
        target_tool_calls = None
        max_tool_calls = None

    with pytest.raises(ValueError, match="cannot both be None"):
        UnboundedStage()


def test_is_stage_text_only():
    stage = DummyStage()
    stage.global_parameters.verification_prompt = "text"
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
    assert stage.is_linked_to_prev() is True

    stage.validate(["robot"])

    assert stage.is_linked_to_prev() is True
    assert stage.get_description()["linked_to_prev"] is True


def test_asking_base_stage_defaults_remain_text_only() -> None:
    stage = AskingBaseStage("Where is the mug?", "The mug is on the table.")

    stage.validate(["robot"])

    assert stage.target_tool_calls == 1
    assert stage.max_tool_calls == 1
    assert stage.allows_tools_before_answer() is False
    assert stage.get_description()["text_only"] is True
    assert stage.get_description()["allow_tools_before_answer"] is False


def test_asking_base_stage_supports_tool_assisted_configuration() -> None:
    stage = AskingBaseStage(
        "Where is the mug?",
        "The mug is on the table.",
        allow_tools_before_answer=True,
        allowed_tools=["ping"],
    )
    stage.target_tool_calls = 3
    stage.max_tool_calls = 5

    stage.validate(["robot"])

    assert stage.target_tool_calls == 3
    assert stage.max_tool_calls == 5
    assert stage.allows_tools_before_answer() is True
    assert stage.get_allowed_tools() == ["ping"]
    assert stage.get_description()["allow_tools_before_answer"] is True


def test_validate_rejects_allowed_tools_without_allow_tools_flag() -> None:
    stage = AskingBaseStage(
        "Where is the mug?",
        "The mug is on the table.",
        allowed_tools=["ping"],
    )

    with pytest.raises(TypeError, match="allowed_tools"):
        stage.validate(["robot"])


def test_validate_rejects_tools_before_answer_without_verification_prompt() -> None:
    stage = LogOnlyStage()
    stage.global_parameters.allow_tools_before_answer = True

    with pytest.raises(TypeError, match="verification_prompt"):
        stage.validate(["robot"])


def test_initialize_stage_errors_supports_multiple_active_errors() -> None:
    stage = LogOnlyStage()
    stage.error_parameters = StageErrorParameters(
        possible_errors=[FirstDummyError(), SecondDummyError()],
        min_active_errors=2,
        max_active_errors=2,
    )
    task = DummyTask(stage)
    task.validate()

    error_state = task.initialize_stage_errors(stage_id=0, obs={}, attributes={}, env_id=7)

    assert error_state == {
        "FirstDummyError": {"env_id": 7, "name": "FirstDummyError"},
        "SecondDummyError": {"env_id": 7, "name": "SecondDummyError"},
    }


def test_initialize_stage_errors_clamps_count_to_possible_errors() -> None:
    stage = LogOnlyStage()
    stage.error_parameters = StageErrorParameters(
        possible_errors=[FirstDummyError()],
        min_active_errors=2,
        max_active_errors=3,
    )
    task = DummyTask(stage)
    task.validate()

    error_state = task.initialize_stage_errors(stage_id=0, obs={}, attributes={}, env_id=7)

    assert error_state == {
        "FirstDummyError": {"env_id": 7, "name": "FirstDummyError"},
    }


def test_validate_rejects_invalid_error_sampling_bounds() -> None:
    stage = LogOnlyStage()
    stage.error_parameters = StageErrorParameters(
        possible_errors=[FirstDummyError(), SecondDummyError()],
        min_active_errors=2,
        max_active_errors=1,
    )

    with pytest.raises(TypeError, match="min_active_errors"):
        stage.validate(["robot"])


def test_execute_tools_blocks_stage_disallowed_tool() -> None:
    stage = AskingBaseStage(
        "Where is the mug?",
        "The mug is on the table.",
        allow_tools_before_answer=True,
        allowed_tools=["other_tool"],
    )
    task = DummyTask(stage)
    task._tools = DummyToolsAPI([], {})
    task.initialization_parameters.agent_names = ["robot"]

    result = task.execute_tools(
        obs={},
        env_id=0,
        attributes={},
        function_name="ping",
        params={},
        stage_id=0,
        error_state={},
    )

    assert result.has_error() is True
    assert result.must_fail_stage is True
    assert "not allowed in this stage" in result.reason
