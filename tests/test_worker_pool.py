import threading

import pytest

from magma_core.configs import BackendConfig
from magma_core.protocol.payload.user_sim_payload import JudgePayload
from magma_core.workers.pool import LMWorkerPool


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


def test_worker_pool_surfaces_worker_errors(monkeypatch) -> None:
    monkeypatch.setattr("magma_core.workers.pool.LMWorker", _FakeLMWorker)
    pool = LMWorkerPool(
        {
            "backend_a": _make_backend("backend_a"),
        }
    )
    future = pool.submit(JudgePayload("rule", "raise", id=1), callback=None)

    with pytest.raises(RuntimeError, match="worker failed"):
        future.result(timeout=2.0)
