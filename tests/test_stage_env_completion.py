import pytest

torch = pytest.importorskip("torch")

if getattr(torch, "__codex_stub__", False):
    pytest.skip(
        "requires a real torch install for tensor operations",
        allow_module_level=True,
    )

from magma_core.simulation.data_structures import StageInput, UserInstruction
from magma_core.simulation.goals import BaseGoal
from magma_core.simulation.stage import BaseTaskStage


class FixedGoal(BaseGoal):
    def __init__(self, result):
        super().__init__()
        self.result = result

    def verify(self, obs):
        return self.result


class VerificationStage(BaseTaskStage):
    target_tool_calls = 0
    max_tool_calls = 0

    def __init__(self, goals):
        super().__init__(
            goals=goals,
            stage_goal_description="verification stage",
            stage_input=StageInput(
                instruction=UserInstruction("verify"),
                flag_answer_to_user=False,
            ),
        )

    def verif_log_completion(self, stage_log, full_log):
        return 1


def test_env_completion_aggregates_goal_results_without_agent_qpos():
    stage = VerificationStage(
        [
            FixedGoal(torch.tensor([1, 1, 0], dtype=torch.int32)),
            FixedGoal(torch.tensor([1, -1, 1], dtype=torch.int32)),
        ]
    )

    result = stage._verif_env_completion(
        {"agent": {"panda-0": {}, "panda-1": {}}}
    )

    assert result.tolist() == [1, -1, 0]


@pytest.mark.parametrize(
    "obs",
    [
        {
            "agent": {"qpos": torch.zeros((2, 8))},
            "extra": {"table": {"pose": torch.zeros((2, 7))}},
        },
        {
            "agent": {
                "panda-0": {"qpos": torch.zeros((2, 9))},
                "panda-1": {"qpos": torch.zeros((2, 9))},
            },
            "extra": {},
        },
    ],
)
def test_env_completion_without_goals_supports_agent_layouts(obs):
    result = VerificationStage([])._verif_env_completion(obs)

    assert result.tolist() == [1, 1]


def test_env_completion_without_goals_requires_reference_tensor():
    with pytest.raises(ValueError, match="reference tensor"):
        VerificationStage([])._verif_env_completion({"agent": {}})
