# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import pytest

from magma_core.simulation.data_structures import StageInput, UserInstruction
from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest


class EchoStage(BaseTaskStage):
    target_tool_calls = 0
    max_tool_calls = None

    def __init__(self, instruction: str):
        super().__init__(
            [],
            "Echo stage",
            StageInput(UserInstruction(instruction), False),
        )
        self.global_parameters.verification_prompt = "ok"


class EchoRequest(BaseRequest[str]):
    def sample_parameters(self, state: TaskState) -> str:
        return "repeat the message"

    def create_stages(
        self,
        state: TaskState,
        parameters: str,
    ) -> list[BaseTaskStage]:
        return [EchoStage(parameters)]


def test_base_request_contract_and_default_hooks():
    state = TaskState()
    request = EchoRequest()
    parameters = request.sample_parameters(state)

    stages = request.create_stages(state, parameters)

    assert len(stages) == 1
    assert isinstance(stages[0], EchoStage)
    assert request.sampling_weight(state) == 1.0
    assert request.force_state_recompute() is False
    assert request.apply_request(state, parameters) is state


def test_base_request_sampling_and_stage_creation_must_be_implemented():
    request = BaseRequest()
    state = TaskState()
    with pytest.raises(NotImplementedError):
        request.sample_parameters(state)
    with pytest.raises(NotImplementedError):
        request.create_stages(state, None)
