from concurrent.futures import Future
import logging
import threading
from unittest.mock import Mock

import pytest

from magma_core.configs import BackendConfig
from magma_core.protocol.payload.diagnosis_payload import (
    CoachDiagnosisPayload,
    SimilarCaseSelectionPayload,
    SuboptimalDiagnosisPayload,
)
from magma_core.protocol.payload.user_sim_payload import JudgePayload
from magma_core.workers.human_coaching import (
    HumanCoachingError,
    HumanCoachingSkip,
    HumanCoachingWorker,
)
from magma_core.workers.pool import LMWorkerPool
from magma_core.workers.worker import LMWorker


def _make_backend(name: str) -> BackendConfig:
    return BackendConfig(
        type="ollama",
        endpoint=f"http://{name}.example.com",
        default_model="model",
        headers={},
    )


class _FakeLMWorker:
    instances = []

    def __init__(self, backend_config: BackendConfig, backend_name=None) -> None:
        self.backend_config = backend_config
        self.backend_name = backend_name
        self.__class__.instances.append(self)

    def run_payload(self, payload, fut):
        if payload.model_answer == "raise":
            raise RuntimeError("worker failed")
        fut.set_result((payload.id, self.backend_name, payload.model_answer))


@pytest.fixture(autouse=True)
def _reset_fake_workers():
    _FakeLMWorker.instances = []
    yield
    _FakeLMWorker.instances = []


def test_worker_pool_rejects_empty_backend_config() -> None:
    with pytest.raises(ValueError, match="at least one backend"):
        LMWorkerPool({})


def test_worker_pool_submits_payload_and_runs_callback(monkeypatch) -> None:
    monkeypatch.setattr("magma_core.workers.pool.LMWorker", _FakeLMWorker)
    pool = LMWorkerPool(
        {
            "backend_a": _make_backend("backend_a"),
        }
    )
    try:
        callback_results = []
        callback_called = threading.Event()

        payload = JudgePayload("rule", "answer", id=1)

        def record_result(fut):
            callback_results.append(fut.result())
            callback_called.set()

        future = pool.submit(payload, callback=record_result)

        expected = (payload.id, "backend_a", "answer")
        assert pool.get_capacity() == 1
        assert future.result(timeout=2.0) == expected
        assert callback_called.wait(timeout=2.0)
        assert callback_results == [expected]
    finally:
        pool.close()


def test_worker_pool_surfaces_worker_errors(monkeypatch) -> None:
    monkeypatch.setattr("magma_core.workers.pool.LMWorker", _FakeLMWorker)
    pool = LMWorkerPool(
        {
            "backend_a": _make_backend("backend_a"),
        }
    )
    try:
        future = pool.submit(JudgePayload("rule", "raise", id=1), callback=None)

        with pytest.raises(RuntimeError, match="worker failed"):
            future.result(timeout=2.0)
    finally:
        pool.close()


def test_lm_worker_embeds_validated_case_in_diagnosis_prompt() -> None:
    worker = LMWorker.__new__(LMWorker)
    worker.client = Mock()
    worker.client.send_messages.return_value = {"content": "current output"}
    worker.logger = logging.getLogger("TEST.WORKER")
    worker.backend_name = "test"
    worker.backend_type = "ollama"
    worker.backend_endpoint = "http://localhost:11434"
    worker.backend_default_model = "test-model"
    payload = CoachDiagnosisPayload(
        stage_goal="Current goal",
        trajectory="CURRENT TRAJECTORY",
        failure_type="failure",
        id=1,
    )
    payload.set_coaching_examples([{
        "stage_goal": "Reference goal",
        "trajectory": [{
            "diagnosable": True,
            "input_type": "INSTRUCTION",
            "input_content": "REFERENCE TRAJECTORY CONTENT",
            "answer": "CALLS: robot.pick()",
            "execution_results": [],
        }],
        "diagnosis": {
            "decision_index": 2,
            "reason": "Reference reason",
            "expected_decision": "Reference repair",
        },
        "component_diagnosis": {
            "diagnosis": {"component": "dispatcher"},
        },
    }])
    future = Future()

    worker.run_payload(payload, future)

    messages = worker.client.send_messages.call_args.args[1]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "VALIDATED SIMILAR CASE" in messages[0]["content"]
    assert "REFERENCE TRAJECTORY CONTENT" in messages[0]["content"]
    assert '"reason": "Reference reason"' in messages[0]["content"]
    assert "CURRENT TRAJECTORY" in messages[0]["content"]
    assert "component" not in messages[0]["content"]
    assert future.result() == (payload.id, "current output")

    plain_payload = CoachDiagnosisPayload(
        stage_goal="Current goal",
        trajectory="CURRENT TRAJECTORY",
        failure_type="failure",
        id=2,
    )
    plain_prompt = plain_payload.prompt_template.format(**plain_payload.to_dict())
    assert "VALIDATED SIMILAR CASE" not in plain_prompt


def test_similar_case_selection_payload_is_compact() -> None:
    trajectory = [{
        "diagnosable": True,
        "input_type": "INSTRUCTION",
        "input_content": "current instruction",
        "answer": "CALLS: robot.move()",
        "execution_results": [],
    }]
    payload = SimilarCaseSelectionPayload(
        stage_goal="Current goal",
        trajectory=trajectory,
        cases=[{
            "stage_goal": "Stored goal",
            "trajectory": trajectory,
            "diagnosis": {
                "decision_index": 1,
                "reason": "Stored reason",
                "expected_decision": "Stored repair",
            },
            "component_diagnosis": {"private": "must not appear"},
        }],
        id=1,
    )

    prompt = payload.prompt_template.format(**payload.to_dict())

    assert payload.max_tokens > 0
    assert "CASE 1" in prompt
    assert "Stored reason" in prompt
    assert "must not appear" not in prompt
    assert "Return only a JSON array" in prompt


def test_suboptimal_diagnosis_payload_formats_context_and_optional_example() -> None:
    trajectory = [{
        "diagnosable": True,
        "input_type": "INSTRUCTION",
        "input_content": "sort objects",
        "answer": "CALLS: robot.move",
        "execution_results": [],
    }]
    example = [{
        "diagnosable": True,
        "input_type": "INSTRUCTION",
        "input_content": "reference instruction",
        "answer": "CALLS: robot.pick",
        "execution_results": [],
    }]
    payload = SuboptimalDiagnosisPayload(
        task_description="Task description",
        stage_goal="Stage goal",
        permanent_rules=["Keep red objects on the table"],
        trajectory=trajectory,
        example_trajectory=example,
        id=3,
    )

    prompt = payload.prompt_template.format(**payload.to_dict())

    assert "Task description" in prompt
    assert "Stage goal" in prompt
    assert "- Keep red objects on the table" in prompt
    assert "CALLS: robot.move" in prompt
    assert "CALLS: robot.pick" in prompt
    assert payload.to_human_dict()["trajectory_steps"] == trajectory

    without_example = SuboptimalDiagnosisPayload(
        task_description="Task description",
        stage_goal="Stage goal",
        permanent_rules=[],
        trajectory=trajectory,
        example_trajectory=None,
        id=4,
    )
    assert (
        without_example.to_dict()["example_trajectory"]
        == "No validated efficient trajectory is available."
    )


def test_human_coaching_worker_returns_payload_result(monkeypatch) -> None:
    health_response = Mock()
    health_response.raise_for_status.return_value = None
    monkeypatch.setattr("magma_core.workers.human_coaching.requests.get", Mock(return_value=health_response))

    coaching_response = Mock()
    coaching_response.raise_for_status.return_value = None
    coaching_response.json.side_effect = lambda: {
        "request_id": coaching_response.request_id,
        "status": "completed",
        "output": "fixed answer",
        "force_save_example": True,
    }

    posted_request = {}

    def post_request(*args, **kwargs):
        posted_request.update(kwargs["json"])
        coaching_response.request_id = kwargs["json"]["request_id"]
        return coaching_response

    monkeypatch.setattr("magma_core.workers.human_coaching.requests.post", post_request)
    worker = HumanCoachingWorker("http://localhost:8890")
    payload = JudgePayload("rule", "answer", id=7)
    payload.allow_force_save_example = True

    assert worker.get_capacity() == 1
    assert worker.submit(payload, callback=None).result(timeout=2) == (7, "fixed answer")
    assert payload.force_save_example is True
    assert posted_request["allow_force_save_example"] is True


@pytest.mark.parametrize(
    ("status", "message", "exception_type"),
    [
        ("skipped", "operator skipped coaching", HumanCoachingSkip),
        ("aborted", "operator stopped generation", HumanCoachingError),
    ],
)
def test_human_coaching_worker_surfaces_non_completed_response(
    monkeypatch,
    status,
    message,
    exception_type,
) -> None:
    health_response = Mock()
    health_response.raise_for_status.return_value = None
    monkeypatch.setattr("magma_core.workers.human_coaching.requests.get", Mock(return_value=health_response))

    coaching_response = Mock()
    coaching_response.raise_for_status.return_value = None

    def response_body():
        return {
            "request_id": coaching_response.request_id,
            "status": status,
            "error": message,
        }

    coaching_response.json.side_effect = response_body

    def post_request(*args, **kwargs):
        coaching_response.request_id = kwargs["json"]["request_id"]
        return coaching_response

    monkeypatch.setattr("magma_core.workers.human_coaching.requests.post", post_request)
    worker = HumanCoachingWorker("http://localhost:8890")
    future = worker.submit(JudgePayload("rule", "answer", id=8), callback=None)

    with pytest.raises(exception_type, match=message):
        future.result(timeout=2)
