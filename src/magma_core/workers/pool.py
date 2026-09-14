from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from typing import Callable, Any, Dict, List, Tuple
from queue import Empty, Queue
import logging
import threading

from magma_core.configs import BackendConfig
from magma_core.protocol.payload import BasePayload

from .worker import LMWorker
from magma_core.protocol.agent_coaching import CoachingLog


class LMWorkerPool:
    """
    Dispatch payloads across multiple backend workers through a shared queue.

    The pool is the public entry point used by the generation pipeline. Each
    backend owns one internal LMWorker instance, but all payloads first go
    through the same FIFO queue.
    """

    def __init__(self, backends: Dict[str, BackendConfig], log_sink: Callable[[CoachingLog], Path | None] | None = None) -> None:
        self.log_sink = log_sink
        if len(backends) == 0:
            raise ValueError("LMWorkerPool requires at least one backend")

        self.logger = logging.getLogger("WORKER.POOL")
        self.pending_call: Queue[Tuple[BasePayload, Future] | None] = Queue()
        self._workers: List[Tuple[str, LMWorker]] = []
        self._threads: List[threading.Thread] = []
        self.thread_running = True
        self._shutdown_lock = threading.Lock()

        # Validate every backend before starting threads, so a later connection
        # failure cannot leave a partially initialized pool running.
        for backend_name, backend_config in backends.items():
            self._workers.append((backend_name, LMWorker(backend_config, backend_name=backend_name)))
        for backend_name, worker in self._workers:
            thread = threading.Thread(
                target=self._periodic_call,
                args=(backend_name, worker),
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def get_capacity(self) -> int:
        return len(self._workers)

    def _periodic_call(self, backend_name: str, worker: LMWorker) -> None:
        while True:
            item = self.pending_call.get()
            if item is None:
                return
            payload, fut = item
            if not fut.set_running_or_notify_cancel():
                continue
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
        if payload.log_sink is None:
            payload.log_sink = self.log_sink
        fut = Future()
        if callback is not None:
            fut.add_done_callback(callback)
        with self._shutdown_lock:
            if not self.thread_running:
                raise RuntimeError("Worker pool is closed")
            self.pending_call.put((payload, fut))
        return fut

    def close(self) -> None:
        pending = []
        with self._shutdown_lock:
            if self.thread_running:
                self.thread_running = False
                while True:
                    try:
                        item = self.pending_call.get_nowait()
                    except Empty:
                        break
                    if item is not None:
                        pending.append(item[1])
                for _ in self._threads:
                    self.pending_call.put(None)
        # Cancellation runs callbacks: do not hold the shutdown lock here.
        for future in pending:
            future.cancel()
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join()
