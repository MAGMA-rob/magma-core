from __future__ import annotations

import copy
from concurrent.futures import Future
from queue import Queue
import threading
from pathlib import Path
from typing import Any, Callable, Tuple
import re
from magma_core.protocol.agent_coaching import CoachingLog
from uuid import uuid4
from datetime import datetime

import requests

from magma_core.protocol import HumanCoachingRequest, HumanCoachingResponse
from magma_core.protocol.payload import BasePayload


class HumanCoachingError(RuntimeError):
    pass


class HumanCoachingSkip(RuntimeError):
    pass


class HumanCoachingWorker:
    def __init__(self, endpoint: str, connect_timeout: float = 10, log_sink: Callable[[CoachingLog], Path | None] | None = None) -> None:
        self.log_sink = log_sink
        self.closed = False
        self.endpoint = endpoint.rstrip("/")
        self.connect_timeout = connect_timeout
        self.pending_call: Queue[Tuple[BasePayload, Future] | None] = Queue()
        self.thread = threading.Thread(target=self._periodic_call, daemon=True)
        self._thread_started = False
        self._thread_start_lock = threading.Lock()
        try:
            response = requests.get(
                f"{self.endpoint}/health",
                timeout=self.connect_timeout,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise HumanCoachingError(
                f"Human coaching server {self.endpoint!r} is unavailable: {error}"
            ) from error

    def get_capacity(self) -> int:
        return 1

    def _periodic_call(self) -> None:
        while True:
            item = self.pending_call.get()
            if item is None:
                return
            payload, future = item
            self.run_payload(payload, future)

    def run_payload(self, payload: BasePayload, future: Future) -> None:
        request_id = str(uuid4())
        rendered_prompt = ""
        raw_output = ""
        try:
            payload_data = payload.to_dict()
            prompt_data = copy.deepcopy(payload_data)
            old_messages = prompt_data.pop("old_messages", [])
            rendered_prompt = payload.prompt_template.format(**prompt_data)
            if old_messages:
                rendered_prompt = f"Previous messages:\n{old_messages}\n\n{rendered_prompt}"
            request = HumanCoachingRequest(
                request_id=request_id,
                payload_id=payload.id,
                payload_type=payload.__class__.__name__,
                payload=payload.to_human_dict(),
                rendered_prompt=rendered_prompt,
                allow_force_save_example=payload.allow_force_save_example,
            )
            response = requests.post(
                f"{self.endpoint}/v1/coaching/requests",
                json=request.model_dump(),
                timeout=(self.connect_timeout, None),
            )
            response.raise_for_status()
            raw_output = response.text
            result = HumanCoachingResponse.model_validate(response.json())
            if result.request_id != request_id:
                raise HumanCoachingError("Human coaching response request_id does not match")
            if result.status == "skipped":
                raise HumanCoachingSkip(result.error or "Human coaching was skipped")
            if result.status == "aborted":
                raise HumanCoachingError(result.error or "Human coaching was aborted")
            if result.force_save_example and not request.allow_force_save_example:
                raise HumanCoachingError(
                    "Human coaching tried to force-save an ineligible example"
                )
            payload.force_save_example = result.force_save_example
            self._record_log(payload, request_id, rendered_prompt, raw_output)
            future.set_result((payload.id, result.output))
        except Exception as error:
            self._record_log(payload, request_id, rendered_prompt, raw_output, str(error))
            if not isinstance(error, (HumanCoachingError, HumanCoachingSkip)):
                error = HumanCoachingError(
                    f"Human coaching request {request_id} failed for "
                    f"{payload.__class__.__name__} payload {payload.id}: {error}"
                )
            future.set_exception(error)

    def submit(self, payload: BasePayload, callback: Any) -> Future:
        if payload.log_sink is None:
            payload.log_sink = self.log_sink
        with self._thread_start_lock:
            if self.closed:
                raise RuntimeError("Human coaching worker is closed")
            if not self._thread_started:
                self.thread.start()
                self._thread_started = True
        future = Future()
        if callback is not None:
            future.add_done_callback(callback)
        with self._thread_start_lock:
            if self.closed:
                raise RuntimeError("Human coaching worker is closed")
            self.pending_call.put((payload, future))
        return future

    def _record_log(self, payload: BasePayload, request_id: str, prompt: str, output: str, error: str = "") -> None:
        if payload.log_sink is None or not payload.debug_log:
            return
        log_type = payload.log_type or re.sub(r"(?<!^)(?=[A-Z])", "_", payload.__class__.__name__).lower().removesuffix("_payload")
        payload.log_sink(CoachingLog(coaching_type=log_type, content=(
            f"# Human coaching\n\nTimestamp: {datetime.now().isoformat()}\nProvider: {self.endpoint}\nNode: {payload.id}\nRequest: {request_id}\n{payload.log_context}\n"
            f"\n## Prompt\n{prompt}\n\n## Response\n{output}\n\n## Error\n{error}\n"
        )))

    def close(self) -> None:
        with self._thread_start_lock:
            if self.closed:
                return
            self.closed = True
            self.pending_call.put(None)
        if self._thread_started and self.thread is not threading.current_thread():
            self.thread.join()
