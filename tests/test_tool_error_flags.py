# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

"""
Coverage map for this file:

- Error-flag derivation on tool executions before and after verification.
- Small helper semantics on ``ToolStatus``.
- The default failure mode applied by ``ToolExecution.fail``.
"""

import pytest
import torch

from magma_core.domain import Call
from magma_core.simulation.data_structures import (
    EmptyInstruction,
    ToolErrorFlag,
    ToolExecution,
    EnvToolContext,
    ToolResult,
    RobotToolContext,
    RobotToolStatus,
    StageInput,
    StageSuccess,
    ToolStatus,
    UserInstruction,
)
from magma_core.simulation.agents import ValidAgentAnswer
from magma_core.simulation.skills import (
    BaseSkill,
    CallTick,
    MessTick,
    SkillExecutionContext,
    SkillManager,
    SkillRegistration,
    SkillStateRef,
    SkillSpec,
)


class _SingleToolAPI:
    def get_tools_with_real_names(self):
        return [
            ({
                "name": "move",
                "description": "Move an object",
                "parameters": {},
            }, "move"),
            ({
                "name": "primitive",
                "description": "Run one primitive step",
                "parameters": {},
            }, "primitive"),
        ]

    def get_skill_vocabulary_translator(self):
        return None


def _build_skill_manager() -> SkillManager:
    manager = SkillManager()
    manager.build_api(_SingleToolAPI())
    return manager


class _TwoStepSkill(BaseSkill):
    spec = SkillSpec(
        name="two_step",
        description="Run two primitive steps",
        required_tools=["primitive"],
        argument_schema={},
    )

    def __init__(self, arguments):
        super().__init__(arguments)
        self.completed_steps = 0

    def start(self):
        return CallTick("primitive", {"step": 1})

    def tick(self, status):
        self.completed_steps += 1
        if self.completed_steps == 1:
            return CallTick("primitive", {"step": 2})
        return MessTick("two-step skill completed", True, ToolErrorFlag.NONE)


def _execution_context(
    stage_id: int,
    flag_answer_to_user: bool = False,
) -> SkillExecutionContext:
    return SkillExecutionContext(
        stage_id=stage_id,
        attributes={
            "known_robots": ["robot"],
            "robot_statuses": {"robot": "FREE"},
        },
        flag_answer_to_user=flag_answer_to_user,
    )


def _register(
    manager: SkillManager,
    node_id: int,
    answer: ValidAgentAnswer,
    state_ref: SkillStateRef,
    stage_id: int,
    flag_answer_to_user: bool = False,
) -> None:
    manager.register({
        node_id: SkillRegistration(
            answer=answer,
            execution_context=_execution_context(
                stage_id,
                flag_answer_to_user=flag_answer_to_user,
            ),
            state_ref=state_ref,
        )
    })


def _finish_with_next_input() -> ToolStatus:
    return ToolStatus(
        robots_status=[RobotToolStatus(
            robot_name="robot",
            mess="move completed",
            result=True,
            error_flag=ToolErrorFlag.NONE,
        )],
        error_descriptions=[""],
        stage_id=0,
        attributes={
            "known_robots": ["robot"],
            "robot_statuses": {"robot": "FREE"},
        },
        stage_success=StageSuccess.FINISH,
        next_input=StageInput(UserInstruction("new instruction"), False),
        tool_calls=1,
    )


def _reach_deferred_input(
    manager: SkillManager,
    captured_answer: ValidAgentAnswer,
) -> SkillStateRef:
    state_ref = manager.create_initial_state(_execution_context(0))
    initial = ValidAgentAnswer(1, 1, "", [Call("move", {}, "robot")])
    _register(manager, 1, initial, state_ref, 0)
    initial_tick = manager.tick({})
    assert initial_tick.results[1].request.agent_step_id == 1

    finished_status = _finish_with_next_input()
    feedback_tick = manager.tick({1: finished_status})
    feedback = feedback_tick.results[1].status_event
    assert feedback is not None
    assert isinstance(
        feedback.status.next_input.instruction,
        EmptyInstruction,
    )
    assert finished_status.next_input.instruction.get_content() == "new instruction"

    _register(manager, 2, captured_answer, feedback.state_ref, 1)
    publish_tick = manager.tick({})
    transition = publish_tick.results[2].deferred_input
    assert transition is not None
    assert transition.stage_input.instruction.get_content() == "new instruction"
    return transition.state_ref


def _build_tool_infos(tool_execution: ToolExecution) -> EnvToolContext:
    return EnvToolContext(
        node_id=0,
        task_stage=0,
        agent_step=0,
        stage_log_start_idx=0,
        source_id=0,
        tool_robots=[RobotToolContext("default", tool_execution)],
        error_state={},
        attributes={},
        tool_calls=1,
        forgiven_tool_calls=0,
    )


def test_tool_infos_reports_bad_call_for_invalid_tool_call() -> None:
    infos = _build_tool_infos(
        ToolExecution(
            poses=[],
            verifier=None,
            reason="bad call",
        )
    )

    status = infos.build_tool_status()

    assert status.robots_status[0].error_flag == ToolErrorFlag.BAD_CALL
    assert status.tool_calls == 1
    assert status.forgiven_tool_calls == 0


def test_tool_execution_copies_allowed_moving_actors() -> None:
    actor_names = ["cube"]

    execution = ToolExecution(
        poses=["OK"],
        verifier=None,
        allowed_moving_actors=actor_names,
    )
    actor_names.append("plate")

    assert execution.allowed_moving_actors == ["cube"]


def test_tool_context_unions_actor_allowlists_only_when_all_are_declared() -> None:
    first = ToolExecution(
        poses=["OK"],
        verifier=None,
        allowed_moving_actors=["cube_a"],
    )
    second = ToolExecution(
        poses=["OK"],
        verifier=None,
        allowed_moving_actors=["cube_b", "cube_a"],
    )
    infos = EnvToolContext(
        node_id=0,
        task_stage=0,
        agent_step=0,
        stage_log_start_idx=0,
        source_id=0,
        tool_robots=[
            RobotToolContext("robot_a", first),
            RobotToolContext("robot_b", second),
        ],
        error_state={},
        attributes={},
        tool_calls=2,
        forgiven_tool_calls=0,
    )

    assert infos.get_allowed_moving_actors() == {"cube_a", "cube_b"}

    second.allowed_moving_actors = None

    assert infos.get_allowed_moving_actors() is None


def test_tool_infos_reports_planner_injection_and_success_flags() -> None:
    planner_execution = ToolExecution(poses=["OK"], verifier=None)
    planner_infos = _build_tool_infos(planner_execution)
    planner_infos.tool_robots[0].tool_result = ToolResult(ok=False, reason="missed target")

    injected_execution = ToolExecution(poses=["OK"], verifier=None)
    injected_execution.injection_applied = True
    injected_infos = _build_tool_infos(injected_execution)
    injected_infos.tool_robots[0].tool_result = ToolResult(ok=False, reason="injected failure")

    recovered_execution = ToolExecution(poses=["OK"], verifier=None)
    recovered_execution.injection_applied = True
    recovered_infos = _build_tool_infos(recovered_execution)
    recovered_infos.tool_robots[0].tool_result = ToolResult(ok=True, reason="recovered")

    planner_status = planner_infos.build_tool_status()
    injected_status = injected_infos.build_tool_status()
    recovered_status = recovered_infos.build_tool_status()

    assert planner_status.robots_status[0].error_flag == ToolErrorFlag.PLANNER_ERROR
    assert planner_status.forgiven_tool_calls == 1
    assert injected_status.robots_status[0].error_flag == ToolErrorFlag.INJECTION_ERROR
    assert injected_status.forgiven_tool_calls == 1
    assert recovered_status.robots_status[0].error_flag == ToolErrorFlag.NONE
    assert recovered_status.forgiven_tool_calls == 0


@pytest.mark.parametrize(
    ("tensor_value", "expected"),
    [
        (torch.tensor(True), True),
        (torch.tensor(False), False),
        (torch.tensor([1]), True),
        (torch.tensor([0]), False),
    ],
)
@pytest.mark.skipif(
    getattr(torch, "__codex_stub__", False),
    reason="The lightweight torch test stub does not preserve tensor values",
)
def test_tool_result_normalizes_single_value_tensor(
    tensor_value: torch.Tensor,
    expected: bool,
) -> None:
    result = ToolResult(ok=tensor_value)

    assert result.ok is expected
    assert isinstance(result.ok, bool)


@pytest.mark.parametrize("invalid_value", [1, 0, "true", None])
def test_tool_result_rejects_non_boolean_values(
    invalid_value: object,
) -> None:
    with pytest.raises(TypeError, match="must be a bool"):
        ToolResult(ok=invalid_value)


@pytest.mark.skipif(
    getattr(torch, "__codex_stub__", False),
    reason="The lightweight torch test stub does not implement tensor item errors",
)
def test_tool_result_rejects_multi_value_tensor() -> None:
    with pytest.raises(TypeError, match="exactly one value"):
        ToolResult(ok=torch.tensor([True, False]))


def test_tool_status_and_execution_helpers() -> None:
    status = ToolStatus(
        robots_status=[RobotToolStatus(
            robot_name="default",
            mess="planner failed",
            result=False,
            error_flag=ToolErrorFlag.PLANNER_ERROR,
        )],
        error_descriptions=[""],
        stage_id=0,
        attributes={},
    )

    assert status.is_full_of_this_flag(ToolErrorFlag.PLANNER_ERROR)
    assert status.should_restore_source_env_state()

    non_planner_status = ToolStatus(
        robots_status=[RobotToolStatus(
            robot_name="default",
            mess="bad call",
            result=False,
            error_flag=ToolErrorFlag.BAD_CALL,
        )],
        error_descriptions=[""],
        stage_id=0,
        attributes={},
    )

    assert not non_planner_status.should_restore_source_env_state()

    execution = ToolExecution(poses=["OK"], verifier=None)

    execution.fail("invalid")

    assert execution.reason == "invalid"
    assert execution.has_error()


def test_delayed_tool_is_executed_only_after_empty_follow_up() -> None:
    manager = _build_skill_manager()
    suspended = ValidAgentAnswer(1, 7, "", [Call("move", {}, "robot")])
    state_ref = _reach_deferred_input(manager, suspended)

    empty = ValidAgentAnswer(2, 8, "", [])
    _register(manager, 3, empty, state_ref, 1)
    resume_tick = manager.tick({})

    result = resume_tick.results[3]
    request = result.request
    assert request is not None
    assert request.source_node_id == 2
    assert request.agent_step_id == 7
    assert [call.name for call in request.get_tool_calls()] == ["move"]

    result = ToolStatus(
        robots_status=[RobotToolStatus(
            robot_name="robot",
            mess="resumed move completed",
            result=True,
            error_flag=ToolErrorFlag.NONE,
        )],
        error_descriptions=[""],
        stage_id=1,
        attributes=_execution_context(1).attributes,
    )
    feedback = manager.tick({3: result}).results[3].status_event
    assert feedback is not None
    assert feedback.executed_answer == suspended


def test_next_instruction_is_not_delayed_before_additional_answer() -> None:
    manager = _build_skill_manager()
    state_ref = manager.create_initial_state(_execution_context(0))
    answer = ValidAgentAnswer(1, 1, "", [Call("move", {}, "robot")])
    _register(
        manager,
        1,
        answer,
        state_ref,
        0,
        flag_answer_to_user=True,
    )
    initial_tick = manager.tick({})
    assert initial_tick.results[1].request is not None

    finished_status = _finish_with_next_input()
    transition = manager.tick({1: finished_status}).results[1].status_event

    assert transition is not None
    assert transition.status.next_input is not None
    assert isinstance(
        transition.status.next_input.instruction,
        UserInstruction,
    )
    assert transition.status.next_input.instruction.get_content() == "new instruction"


def test_concurrent_tool_is_rejected_until_suspended_tool_is_cancelled() -> None:
    manager = _build_skill_manager()
    suspended = ValidAgentAnswer(1, 7, "", [Call("move", {}, "robot")])
    state_ref = _reach_deferred_input(manager, suspended)

    concurrent = ValidAgentAnswer(2, 8, "", [Call("move", {}, "robot")])
    _register(manager, 3, concurrent, state_ref, 1)
    rejected_tick = manager.tick({})

    rejected_event = rejected_tick.results[3].status_event
    assert rejected_event is not None
    assert rejected_event.executed_answer == concurrent
    rejected = rejected_event.status.robots_status[0]
    assert rejected.error_flag == ToolErrorFlag.BAD_CALL
    assert "cancel_current_action" in rejected.mess

    cancel = ValidAgentAnswer(
        3,
        9,
        "",
        [Call("cancel_current_action", {}, "robot")],
    )
    _register(manager, 4, cancel, rejected_event.state_ref, 1)
    cancelled_tick = manager.tick({})

    cancelled_event = cancelled_tick.results[4].status_event
    assert cancelled_event is not None
    assert cancelled_event.executed_answer == cancel
    cancelled = cancelled_event.status.robots_status[0]
    assert cancelled.result is True
    assert "was cancelled" in cancelled.mess

    _register(
        manager,
        5,
        ValidAgentAnswer(2, 10, "", []),
        state_ref,
        1,
    )
    resumed = manager.tick({}).results[5].request
    assert resumed is not None
    assert [call.name for call in resumed.get_tool_calls()] == ["move"]


def test_unknown_tool_is_reported_as_bad_call() -> None:
    manager = _build_skill_manager()
    state_ref = manager.create_initial_state(_execution_context(0))
    answer = ValidAgentAnswer(
        1,
        1,
        "",
        [Call("unknown_tool", {}, "robot")],
    )

    _register(manager, 1, answer, state_ref, 0)
    result = manager.tick({}).results[1].status_event

    assert result is not None
    assert result.executed_answer == answer
    assert len(result.status.robots_status) == 1
    rejected = result.status.robots_status[0]
    assert rejected.robot_name == "robot"
    assert rejected.result is False
    assert rejected.error_flag == ToolErrorFlag.BAD_CALL
    assert rejected.mess == "unknown_tool is not a known tool."
    assert rejected.mess_is_public is True


def test_delayed_say_can_be_resumed_or_replaced() -> None:
    resumed_manager = _build_skill_manager()
    suspended_say = ValidAgentAnswer(1, 4, "I will continue.", [])
    state_ref = _reach_deferred_input(resumed_manager, suspended_say)
    _register(
        resumed_manager,
        3,
        ValidAgentAnswer(2, 5, "", []),
        state_ref,
        1,
    )

    resumed = resumed_manager.tick({}).results[3].request
    assert resumed is not None
    assert resumed.get_say() == "I will continue."
    assert resumed.agent_step_id == 4

    replaced_manager = _build_skill_manager()
    state_ref = _reach_deferred_input(replaced_manager, suspended_say)
    _register(
        replaced_manager,
        3,
        ValidAgentAnswer(2, 5, "Updated answer.", []),
        state_ref,
        1,
    )

    replaced = replaced_manager.tick({}).results[3].request
    assert replaced is not None
    assert replaced.get_say() == "Updated answer."
    assert replaced.agent_step_id == 5


def test_failed_say_keeps_suspended_tool_for_coaching_reinjection() -> None:
    manager = _build_skill_manager()
    suspended = ValidAgentAnswer(1, 7, "", [Call("move", {}, "robot")])
    state_ref = _reach_deferred_input(manager, suspended)
    _register(
        manager,
        3,
        ValidAgentAnswer(2, 8, "I cannot continue.", []),
        state_ref,
        1,
    )
    request = manager.tick({}).results[3].request
    assert request is not None
    assert request.get_say() == "I cannot continue."

    failed = ToolStatus(
        robots_status=[],
        error_descriptions=[],
        stage_id=1,
        attributes=_execution_context(1).attributes,
        stage_success=StageSuccess.FAILED,
        failure_reason="Rejected answer",
    )
    failed_event = manager.tick({3: failed}).results[3].status_event
    assert failed_event is not None
    assert failed_event.status is failed

    _register(
        manager,
        4,
        ValidAgentAnswer(3, 9, "", []),
        failed_event.state_ref,
        1,
    )
    resumed = manager.tick({}).results[4].request
    assert resumed is not None
    assert resumed.agent_step_id == 7
    assert [call.name for call in resumed.get_tool_calls()] == ["move"]


def test_failed_resumed_tool_restores_suspended_work_for_coaching() -> None:
    manager = _build_skill_manager()
    suspended = ValidAgentAnswer(1, 7, "", [Call("move", {}, "robot")])
    state_ref = _reach_deferred_input(manager, suspended)
    _register(manager, 3, ValidAgentAnswer(2, 8, "", []), state_ref, 1)
    request = manager.tick({}).results[3].request
    assert request is not None

    failed = ToolStatus(
        robots_status=[RobotToolStatus(
            robot_name="robot",
            mess="move rejected",
            result=False,
            error_flag=ToolErrorFlag.BAD_CALL,
        )],
        error_descriptions=[""],
        stage_id=1,
        attributes=_execution_context(1).attributes,
        stage_success=StageSuccess.FAILED,
        failure_reason="Rejected resumed action",
    )
    failed_event = manager.tick({3: failed}).results[3].status_event
    assert failed_event is not None
    assert manager.get_suspended_answer(failed_event.state_ref) == suspended

    manager.resume(4, failed_event.state_ref, _execution_context(1))
    retried = manager.tick({}).results[4].request
    assert retried is not None
    assert retried.agent_step_id == 7
    assert [call.name for call in retried.get_tool_calls()] == ["move"]


def test_successful_interruption_answer_carries_tool_into_empty_stage() -> None:
    manager = _build_skill_manager()
    suspended = ValidAgentAnswer(1, 7, "", [Call("move", {}, "robot")])
    state_ref = _reach_deferred_input(manager, suspended)
    _register(
        manager,
        3,
        ValidAgentAnswer(2, 8, "Here are the requested details.", []),
        state_ref,
        1,
    )
    request = manager.tick({}).results[3].request
    assert request is not None
    assert request.get_say() == (
        "Here are the requested details."
    )

    answer_status = ToolStatus(
        robots_status=[],
        error_descriptions=[],
        stage_id=1,
        attributes=_execution_context(1).attributes,
        stage_success=StageSuccess.FINISH,
        next_input=StageInput(EmptyInstruction(), False),
    )
    transition_tick = manager.tick({3: answer_status})

    assert not transition_tick.has_execution()
    transition = transition_tick.results[3].status_event
    assert transition is not None
    assert transition.status is answer_status
    assert transition.has_suspended_work
    assert manager.get_suspended_answer(transition.state_ref) == suspended

    manager.resume(4, transition.state_ref, _execution_context(2))
    resumed = manager.tick({}).results[4].request
    assert resumed is not None
    assert resumed.source_node_id == 3
    assert resumed.agent_step_id == 7
    assert [call.name for call in resumed.get_tool_calls()] == ["move"]


def test_empty_stage_after_say_requires_suspended_work() -> None:
    manager = _build_skill_manager()
    state_ref = manager.create_initial_state(_execution_context(0))
    _register(
        manager,
        1,
        ValidAgentAnswer(1, 1, "Nothing is suspended.", []),
        state_ref,
        0,
    )
    request = manager.tick({}).results[1].request
    assert request is not None
    assert request.get_say() == "Nothing is suspended."
    answer_status = ToolStatus(
        robots_status=[],
        error_descriptions=[],
        stage_id=0,
        attributes=_execution_context(0).attributes,
        stage_success=StageSuccess.FINISH,
        next_input=StageInput(EmptyInstruction(), False),
    )

    transition = manager.tick({1: answer_status}).results[1].status_event
    assert transition is not None
    assert not transition.has_suspended_work


def test_internal_skill_call_is_carried_directly_to_next_instruction() -> None:
    manager = SkillManager([_TwoStepSkill])
    manager.build_api(_SingleToolAPI())
    state_ref = manager.create_initial_state(_execution_context(0))
    original = ValidAgentAnswer(1, 6, "", [Call("two_step", {}, "robot")])
    _register(manager, 1, original, state_ref, 0)
    request = manager.tick({}).results[1].request
    assert request is not None
    assert request.get_tool_calls()[0].name == "primitive"

    transition_tick = manager.tick({1: _finish_with_next_input()})

    transition = transition_tick.results[1].status_event
    assert transition is not None
    assert (
        transition.status.next_input.instruction.get_content()
        == "new instruction"
    )

    _register(
        manager,
        2,
        ValidAgentAnswer(1, 7, "", []),
        transition.state_ref,
        1,
    )
    resumed = manager.tick({}).results[2].request
    assert resumed is not None
    assert resumed.agent_step_id == 6
    assert resumed.get_tool_calls()[0].name == "primitive"
    assert resumed.get_tool_calls()[0].arguments == {"step": 2}

    completed = ToolStatus(
        robots_status=[RobotToolStatus(
            robot_name="robot",
            mess="primitive completed",
            result=True,
            error_flag=ToolErrorFlag.NONE,
        )],
        error_descriptions=[""],
        stage_id=1,
        attributes=_execution_context(1).attributes,
    )
    feedback = manager.tick({2: completed}).results[2].status_event
    assert feedback is not None
    assert feedback.executed_answer == original


def test_empty_answer_is_serialized_without_placeholder() -> None:
    answer = ValidAgentAnswer(1, 1, "", [])

    assert answer.to_dict()["say"] == ""
    assert answer.is_empty()
