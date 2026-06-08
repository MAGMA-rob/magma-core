# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Dict, List, Any, Tuple, Optional
from queue import Queue
import threading
import copy
import json
import logging
import re
import traceback
from concurrent.futures import Future
from datetime import datetime
from pathlib import Path

from magma_core._clients import LLMClientBase, ClientFactory
from magma_core.protocol.payload import BasePayload
from magma_core.configs import BackendConfig
from magma_core.protocol.registry import PROMPT_REGISTRY, ExternalRequestType

class LMWorker:
    """
    Define a worker for asking an external model to complete/generate/correct data.

    Each Worker is associated with a client. It sends to the client the submitted payload and return to the main thread via a callback the answer.
    Each worker has its own queue.
    """

    pending_call : Queue[Tuple[BasePayload,Future]]
    client : LLMClientBase

    # COACHING LOGGING AND DEBUG Set To TRUE to log each coaching request
    DEBUG_LOG_COACHING = False
    DEBUG_LOG_DIR = "output/_coaching_logs"
    _DEBUG_COACHING_TYPES = {
        ExternalRequestType.JUDGE,
        ExternalRequestType.BAD_CALL_FIX,
        ExternalRequestType.BAD_CALL_DIAGNOSE,
        ExternalRequestType.FAILURE_TEXT_ONLY_ANSWER_DIAGNOSE,
        ExternalRequestType.FAILURE_TEXT_ONLY_ANSWER_FIX,
        ExternalRequestType.FAILURE_TEXT_ONLY_FIX,
        ExternalRequestType.MISSING_ACTION_DIAGNOSE,
        ExternalRequestType.MISSING_ACTION_FIX
    }
    _debug_log_counter = 0
    _debug_log_lock = threading.Lock()
    
    def __init__(self, backendConfig : BackendConfig, backend_name: Optional[str] = None) -> None:
        self.pending_call = Queue()
        self.client = ClientFactory.create_client(backendConfig)
        self.client.test_server()
        self.logger = logging.getLogger("WORKER")
        self.backend_name = backend_name
        self.backend_type = backendConfig.type
        self.backend_endpoint = backendConfig.endpoint
        self.backend_default_model = backendConfig.default_model

        self.thread = threading.Thread(target=self._periodic_call, daemon=True)
        self.thread_running = False
        self._thread_start_lock = threading.Lock()
        self._thread_started = False

    @classmethod
    def _should_debug_log_coaching(cls, payload: BasePayload) -> bool:
        return cls.DEBUG_LOG_COACHING and payload.augment_type in cls._DEBUG_COACHING_TYPES

    @staticmethod
    def _slugify_debug_value(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("._-").lower()
        return slug or "unknown"

    def _get_backend_metadata(self) -> Dict[str, str]:
        backend_name = getattr(self, "backend_name", None) or ""
        backend_type = getattr(self, "backend_type", None) or "unknown"
        backend_endpoint = getattr(self, "backend_endpoint", None) or "unknown"
        backend_default_model = getattr(self, "backend_default_model", None) or "unknown"
        backend_label = backend_name or backend_endpoint
        return {
            "name": backend_name,
            "type": backend_type,
            "endpoint": backend_endpoint,
            "default_model": backend_default_model,
            "label": backend_label,
        }

    @classmethod
    def _next_debug_log_path(cls, payload: BasePayload, backend_label: str) -> Path:
        with cls._debug_log_lock:
            log_idx = cls._debug_log_counter
            cls._debug_log_counter += 1

        slug = re.sub(r"(?<!^)(?=[A-Z])", "_", payload.__class__.__name__).lower()
        backend_slug = cls._slugify_debug_value(backend_label)
        log_dir = Path(cls.DEBUG_LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / f"{log_idx:06d}_{backend_slug}_{slug}-{payload.id}.md"

    @staticmethod
    def _dump_debug_json(value: Any) -> str:
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def _format_debug_block(cls, value: Any) -> Tuple[str, str]:
        if isinstance(value, str):
            return "text", value
        return "json", cls._dump_debug_json(value)
    
    def get_capacity(self) -> int:
        return 1

    def _write_coaching_debug_log(
        self,
        payload: BasePayload,
        *,
        payload_dict: Optional[Dict[str, Any]],
        prompt: Optional[str],
        raw_response: Any,
        result: Any,
        error: Optional[BaseException] = None,
        error_traceback: Optional[str] = None,
    ) -> Optional[Path]:
        if not self._should_debug_log_coaching(payload):
            return None

        try:
            backend = self._get_backend_metadata()
            requested_model = payload.model if payload.model is not None else "<default>"
            resolved_model = payload.model if payload.model is not None else backend["default_model"]
            path = self._next_debug_log_path(payload, backend["label"])
            timestamp = datetime.now().isoformat(timespec="seconds")
            augment_type = f"{payload.augment_type.name} ({int(payload.augment_type)})"

            content = [
                "# LMWorker Coaching Debug Log",
                "",
                "## Metadata",
                f"- Payload class: `{payload.__class__.__name__}`",
                f"- Augment type: `{augment_type}`",
                f"- Timestamp: `{timestamp}`",
                f"- Backend name: `{backend['name'] or '<unnamed>'}`",
                f"- Backend type: `{backend['type']}`",
                f"- Backend endpoint: `{backend['endpoint']}`",
                f"- Requested model: `{requested_model}`",
                f"- Resolved model: `{resolved_model}`",
                "",
                "## payload.to_dict()",
                "```json",
                self._dump_debug_json(payload_dict if payload_dict is not None else {}),
                "```",
                "",
                "## Final Prompt",
                "```text",
                prompt if prompt is not None else "",
                "```",
                "",
            ]

            if error is not None:
                content.extend([
                    "## Error",
                    "```text",
                    f"{error.__class__.__name__}: {error}",
                    "```",
                    "",
                ])
                if error_traceback is not None:
                    content.extend([
                        "## Traceback",
                        "```text",
                        error_traceback,
                        "```",
                        "",
                    ])

            raw_response_lang, raw_response_body = self._format_debug_block(raw_response)
            content.extend([
                "## Raw Response",
                f"```{raw_response_lang}",
                raw_response_body,
                "```",
                "",
            ])

            path.write_text("\n".join(content), encoding="utf-8")
            return path
        except Exception as log_error:
            self.logger.warning(
                "Failed to write coaching debug log for payload=%s id=%s: %s",
                payload.__class__.__name__,
                payload.id,
                log_error,
            )
            return None

    def _periodic_call(self):
        self.thread_running = True
        while self.thread_running:
            payload, fut = self.pending_call.get()
            self.run_payload(payload, fut)

    def run_payload(self, payload: BasePayload, fut: Future) -> None:
        payload_dict = None
        old_messages: List[Dict[str, Any]] = []
        prompt = None
        messages: List[Dict[str, Any]] = []
        raw_response = None
        result = None
        backend = self._get_backend_metadata()
        resolved_model = payload.model if payload.model is not None else backend["default_model"]
        try:
            prompt = PROMPT_REGISTRY[payload.augment_type]
            payload_dict = payload.to_dict()
            p_dict = copy.deepcopy(payload_dict)
            old_messages = p_dict.pop("old_messages", [])
            messages = copy.deepcopy(old_messages)
            prompt = prompt.format(**p_dict)
            messages.append({"role":"user","content":prompt})
            raw_response = self.client.send_messages(
                payload.model,
                messages,
                payload.max_tokens,
                keep_messages=True
            )
            if payload.keep_message:
                messages.append(raw_response)
                result = messages
            else:
                result = raw_response['content']
            debug_log_path = self._write_coaching_debug_log(
                payload,
                payload_dict=payload_dict,
                prompt=prompt,
                raw_response=raw_response,
                result=result,
            )
            if self._should_debug_log_coaching(payload):
                self.logger.info(
                    "Coaching payload handled by backend=%s endpoint=%s type=%s model=%s payload=%s id=%s debug_log=%s",
                    backend["label"],
                    backend["endpoint"],
                    backend["type"],
                    resolved_model,
                    payload.augment_type.name,
                    payload.id,
                    str(debug_log_path) if debug_log_path is not None else "n/a",
                )
            fut.set_result((payload.id, result))
        except Exception as e:
            debug_log_path = self._write_coaching_debug_log(
                payload,
                payload_dict=payload_dict,
                prompt=prompt,
                raw_response=raw_response,
                result=result,
                error=e,
                error_traceback=traceback.format_exc(),
            )
            if self._should_debug_log_coaching(payload):
                self.logger.warning(
                    "Coaching payload failed on backend=%s endpoint=%s type=%s model=%s payload=%s id=%s debug_log=%s error=%s",
                    backend["label"],
                    backend["endpoint"],
                    backend["type"],
                    resolved_model,
                    payload.augment_type.name,
                    payload.id,
                    str(debug_log_path) if debug_log_path is not None else "n/a",
                    e,
                )
            fut.set_exception(e)


    def submit(self, payload : BasePayload, callback):
        """
        Submit a payload to the worker. If a callback is defined, it will call it when th result is complete.
        
        :param payload: The Payload Request
        :type payload: BasePayload
        :param callback: A callback function
        """
        with self._thread_start_lock:
            if not self._thread_started:
                self.thread.start()
                self._thread_started = True
        fut = Future()
        if callback is not None:
            fut.add_done_callback(callback)
        self.pending_call.put((payload,fut))
        return fut
        
