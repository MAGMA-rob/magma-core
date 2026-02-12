from magma_core.protocol.payload.coaching_payload import (
    BuildRepeatPayload,
    CleanMemoryPayload,
    InformUserPayload,
)
from magma_core.protocol.payload.dataset_build_payload import (
    CompareLeafPayload,
    EvaluateLeafPayload,
)
from magma_core.protocol.payload.user_sim_payload import (
    JudgePayload,
    ParaphrasingPayload,
    SimulateUserPayload,
)
from magma_core.protocol.registry import ExternalRequestType, PROMPT_REGISTRY


def test_prompt_registry_contains_core_prompt_types():
    assert PROMPT_REGISTRY[ExternalRequestType.JUDGE]
    assert PROMPT_REGISTRY[ExternalRequestType.SIMULATE_USER]
    assert PROMPT_REGISTRY[ExternalRequestType.REPEAT]
    assert isinstance(PROMPT_REGISTRY[ExternalRequestType.FIX_SUBOPTIMAL], str)


def test_judge_and_simulate_user_payload_to_dict():
    judge = JudgePayload(rule="must be short", model_answer="ok", id=1)
    simulate = SimulateUserPayload(user_scenario="be strict", model_answer="answer", id=2)
    assert judge.to_dict()["rule"] == "must be short"
    assert "verdict" in judge.to_dict()["exp"]
    assert simulate.to_dict() == {"user_scenario": "be strict", "model_answer": "answer"}


def test_paraphrasing_payload_filters_status_and_limits_previous_variants():
    payload = ParaphrasingPayload(
        queries=["Q1", "Q2"],
        real_conversation=[
            {"author": "user", "content": "hello"},
            {"author": "status", "content": "hidden"},
            {"author": "assistant", "content": "hi"},
        ],
        previous_variant=["a", "b", "c", "d", "e", "f"],
        id=3,
    )
    data = payload.to_dict()
    assert data["template_answer"] == "Q2"
    assert len(data["previous_variants"]) == 5
    assert "status:" not in data["real_conversation"]


def test_build_repeat_and_inform_user_payload():
    model_answer = {"think": "t", "say": "s", "action": {"name": "pick"}}
    status = {"error": "collision", "previous_tool_call": {"name": "pick"}}
    repeat = BuildRepeatPayload(model_answer=model_answer, status_return=status, id=4)
    repeat_data = repeat.to_dict()
    assert "action" not in repeat_data["template"]
    assert repeat_data["tool_call"] == {"name": "pick"}
    assert repeat_data["error_message"] == "collision"

    inform = InformUserPayload(status_return=status, is_dual_agent=True, id=5)
    inform_data = inform.to_dict()
    assert "Looking at my memory" in inform_data["is_dual_agent"]


def test_clean_memory_payload_modes():
    clean = CleanMemoryPayload(
        memory=["a", "b"],
        preserved_indices=[1],
        list_constraints=None,
        id=6,
    )
    clean_data = clean.to_dict()
    assert clean_data["memory"] == "0. a\nX. b\n"

    gen = CleanMemoryPayload(
        memory=["a"],
        preserved_indices=[],
        list_constraints=["must keep IDs"],
        id=7,
    )
    gen_data = gen.to_dict()
    assert "constraints" in gen_data
    assert "must keep IDs" in gen_data["constraints"]


def test_evaluate_leaf_and_compare_leaf_payload_get_data():
    answer = {"think": "reasoning", "say": "result"}
    situations = [{"id": 1}]

    evaluate = EvaluateLeafPayload(
        user_instruction="u",
        task_description="t",
        stage_description="s",
        answer=answer,
        situations=situations,
        success=False,
        id=8,
    )
    eval_dict = evaluate.to_dict()
    assert "failed" in eval_dict["failtext"]
    assert evaluate.get_data()["answers"] == answer

    compare = CompareLeafPayload(
        user_instruction="u",
        task_description="t",
        stage_description="s",
        answers=[answer, answer],
        success=True,
        situations=situations,
        id=9,
    )
    compare_dict = compare.to_dict()
    assert "[ANSWER 0]" in compare_dict["answers"]
    assert compare.get_data()["situations"] == situations
