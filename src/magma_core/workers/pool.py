from __future__ import annotations

from concurrent.futures import Future
from typing import Any, Dict, List, Tuple
from queue import Queue
import logging
import threading

from magma_core.configs import BackendConfig
from magma_core.protocol.payload import BasePayload

from .worker import LMWorker


class LMWorkerPool:
    """
    Dispatch payloads across multiple backend workers through a shared queue.

    The pool is the public entry point used by the generation pipeline. Each
    backend owns one internal LMWorker instance, but all payloads first go
    through the same FIFO queue.
    """

    def __init__(self, backends: Dict[str, BackendConfig]) -> None:
        if len(backends) == 0:
            raise ValueError("LMWorkerPool requires at least one backend")

        self.logger = logging.getLogger("WORKER.POOL")
        self.pending_call: Queue[Tuple[BasePayload, Future]] = Queue()
        self._workers: List[Tuple[str, LMWorker]] = []
        self._threads: List[threading.Thread] = []
        self.thread_running = True

        for backend_name, backend_config in backends.items():
            self._workers.append((backend_name, LMWorker(backend_config, backend_name=backend_name)))
            thread = threading.Thread(
                target=self._periodic_call,
                args=(backend_name, self._workers[-1][1]),
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def get_capacity(self) -> int:
        return len(self._workers)

    def _periodic_call(self, backend_name: str, worker: LMWorker) -> None:
        while self.thread_running:
            payload, fut = self.pending_call.get()
            try:
                worker.run_payload(payload, fut)
            except Exception as exc:
                self.logger.exception(
                    "Worker pool thread for backend=%s failed while processing payload=%s id=%s",
                    backend_name,
                    payload.__class__.__name__,
                    payload.id,
                )
                if not fut.done():
                    fut.set_exception(exc)

    def submit(self, payload: BasePayload, callback: Any) -> Future:
        fut = Future()
        if callback is not None:
            fut.add_done_callback(callback)
        self.pending_call.put((payload, fut))
        return fut
