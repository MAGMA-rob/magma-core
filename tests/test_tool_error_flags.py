# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

"""
Coverage map for this file:

- Error-flag derivation on tool executions before and after verification.
- Small helper semantics on ``ToolStatus``.
- The default failure mode applied by ``ToolExecution.fail``.
"""

from magma_core.base.data_structures import (
    ToolErrorFlag,
    ToolExecution,
    ToolInfos,
    ToolResult,
    ToolRobot,
    ToolStatus,
)


def _build_tool_infos(tool_execution: ToolExecution) -> ToolInfos:
    return ToolInfos(
        node_id=0,
        task_stage=0,
        agent_step=0,
        stage_log_start_idx=0,
        source_id=0,
        tool_robots=[ToolRobot("default", tool_execution)],
        error_state={},
    )


def test_tool_infos_reports_bad_call_for_invalid_tool_call() -> None:
    infos = _build_tool_infos(
        ToolExecution(
            poses=[],
            verifier=None,
            reason="bad call",
        )
    )

    assert infos.get_error_flags() == [ToolErrorFlag.BAD_CALL]


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

    assert planner_infos.get_error_flags() == [ToolErrorFlag.PLANNER_ERROR]
    assert injected_infos.get_error_flags() == [ToolErrorFlag.INJECTION_ERROR]
    assert recovered_infos.get_error_flags() == [ToolErrorFlag.NONE]


def test_tool_status_and_execution_helpers() -> None:
    status = ToolStatus(
        result=[False],
        error_flags=[ToolErrorFlag.PLANNER_ERROR],
        stage_id=0,
    )

    assert status.has_single_error_flag(ToolErrorFlag.PLANNER_ERROR)
    assert status.has_single_planner_error()
    assert status.all_error_flags_are(ToolErrorFlag.PLANNER_ERROR)

    non_planner_status = ToolStatus(
        result=[False],
        error_flags=[ToolErrorFlag.BAD_CALL],
        stage_id=0,
    )

    assert not non_planner_status.has_single_planner_error()

    execution = ToolExecution(poses=["OK"], verifier=None)

    execution.fail("invalid")

    assert execution.failure_flag == ToolErrorFlag.BAD_CALL
    assert execution.has_error()
