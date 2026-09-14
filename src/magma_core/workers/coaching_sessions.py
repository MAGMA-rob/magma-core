"""Session-scoped workers and HTTP integration for specialized coaching."""
from contextlib import contextmanager
from dataclasses import dataclass
from concurrent.futures import Future
from threading import Condition, Lock
from typing import Any, Callable, Iterator, Protocol

from magma_core.configs import BackendConfig
from magma_core.protocol.agent_coaching import (
    CoachingLog, CoachingSessionConfig, SpecializedCoachingRequest, SpecializedCoachingResponse,
)
from magma_core.protocol.payload import BasePayload
from .base import PayloadWorker
from .pool import LMWorkerPool
from .human_coaching import HumanCoachingWorker


class CoachingSessionConflict(ValueError):
    pass


@dataclass
class CoachingSession:
    config: CoachingSessionConfig
    worker: LMWorkerPool | HumanCoachingWorker | None
    active: int = 0
    closing: bool = False


class CoachingSessions:
    def __init__(self, supported: bool) -> None:
        self.supported = supported
        self.sessions: dict[str, CoachingSession] = {}
        self.condition = Condition()

    def configure(self, run_id: str, config: CoachingSessionConfig) -> None:
        with self.condition:
            existing = self.sessions.get(run_id)
            if existing is not None:
                if existing.closing or existing.config != config:
                    raise CoachingSessionConflict("Coaching session already has a different configuration or is closing")
                return
            worker = None
            if self.supported:
                if config.provider == "human":
                    worker = HumanCoachingWorker(config.endpoint or "", config.connect_timeout)
                else:
                    backends = {name: BackendConfig(**value.model_dump()) for name, value in config.backends.items()}
                    worker = LMWorkerPool(backends)
            self.sessions[run_id] = CoachingSession(config.model_copy(deep=True), worker)

    @contextmanager
    def acquire(self, run_id: str) -> Iterator[CoachingSession]:
        with self.condition:
            session = self.sessions.get(run_id)
            if session is None or session.closing:
                raise KeyError("Unknown or closing coaching session")
            session.active += 1
        try:
            yield session
        finally:
            with self.condition:
                session.active -= 1
                self.condition.notify_all()

    def close(self, run_id: str) -> None:
        with self.condition:
            session = self.sessions.get(run_id)
            if session is None:
                return
            if session.closing:
                self.condition.wait_for(lambda: self.sessions.get(run_id) is not session)
                return
            session.closing = True
            self.condition.wait_for(lambda: session.active == 0)
        try:
            if session.worker is not None:
                session.worker.close()
        finally:
            with self.condition:
                del self.sessions[run_id]
                self.condition.notify_all()

    def close_all(self) -> None:
        with self.condition:
            run_ids = list(self.sessions)
        for run_id in run_ids:
            self.close(run_id)


class RequestCoachingWorker:
    """Explicitly attach one request's collector to every submitted payload."""
    def __init__(self, worker: PayloadWorker, request: SpecializedCoachingRequest, capture: bool) -> None:
        self.worker = worker
        self.request = request
        self.capture = capture
        self.logs: list[CoachingLog] = []
        self.lock = Lock()

    def record(self, log: CoachingLog) -> None:
        if self.capture:
            with self.lock:
                self.logs.append(log)

    def get_capacity(self) -> int:
        return self.worker.get_capacity()

    def submit(self, payload: BasePayload, callback: Any) -> Future:
        payload.log_sink = self.record
        payload.log_type = self.request.kind
        payload.log_context = f"- Run: `{self.request.run_id}`\n- Request: `{self.request.request_id}`"
        return self.worker.submit(payload, callback)


class CoachingService(Protocol):
    def process(self, request: SpecializedCoachingRequest) -> SpecializedCoachingResponse: ...


def mount_coaching_routes(
    app: Any,
    sessions: CoachingSessions,
    service_factory: Callable[[PayloadWorker], CoachingService] | None,
    unavailable_reason: str | None = None,
) -> None:
    # FastAPI is only required in the agent process, not by core worker clients.
    from fastapi import HTTPException, Response
    import asyncio

    @app.put("/v1/coaching/sessions/{run_id}", status_code=204)
    async def configure(run_id: str, config: CoachingSessionConfig):
        try:
            await asyncio.to_thread(sessions.configure, run_id, config)
        except CoachingSessionConflict as error:
            raise HTTPException(409, str(error)) from error
        except Exception as error:
            raise HTTPException(503, "Unable to initialize the coaching provider; verify backend connectivity from the agent") from error
        return Response(status_code=204)

    @app.delete("/v1/coaching/sessions/{run_id}", status_code=204)
    async def close(run_id: str):
        await asyncio.to_thread(sessions.close, run_id)
        return Response(status_code=204)

    def process(request: SpecializedCoachingRequest) -> SpecializedCoachingResponse:
        try:
            with sessions.acquire(request.run_id) as session:
                if service_factory is None or session.worker is None:
                    raise HTTPException(503, unavailable_reason or "Specialized coaching is unavailable")
                worker = RequestCoachingWorker(session.worker, request, session.config.capture_logs)
                try:
                    result = service_factory(worker).process(request)
                    result.validate_request(request)
                except Exception as error:
                    result = SpecializedCoachingResponse(request_id=request.request_id, status="error", reason=str(error))
                if result.status != "corrected" or not worker.logs:
                    worker.record(CoachingLog(coaching_type=request.kind, content=(
                        f"# Coaching {result.status}\n\nRun: {request.run_id}\nRequest: {request.request_id}\n"
                        f"Node: {request.target_step_id}\n\n{result.reason or 'Completed'}\n"
                    )))
                result.logs = worker.logs + result.logs
                return result
        except KeyError as error:
            raise HTTPException(404, "Unknown or closing coaching session") from error

    @app.post("/v1/coaching", response_model=SpecializedCoachingResponse)
    async def coaching(request: SpecializedCoachingRequest):
        return await asyncio.to_thread(process, request)
