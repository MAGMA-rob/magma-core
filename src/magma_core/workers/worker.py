from typing import Dict, List, Any, Tuple, Optional
from queue import Queue
import threading
from concurrent.futures import Future

from magma_core._clients import LLMClientBase, ClientFactory
from magma_core.protocol.payload import BasePayload
from magma_core.configs import BackendConfig
from magma_core.protocol.registry import PROMPT_REGISTRY

class LMWorker:
    """
    Define a worker for asking an external model to complete/generate/correct data.

    Each Worker is associated with a client. It sends to the client the submitted payload and return to the main thread via a callback the answer.
    Each worker has its own queue.
    """

    pending_call : Queue[Tuple[BasePayload,Future]]
    client : LLMClientBase
    
    def __init__(self, backendConfig : BackendConfig) -> None:
        self.pending_call = Queue()
        self.client = ClientFactory.create_client(backendConfig)
        self.client.test_server()

        self.thread = threading.Thread(target=self._periodic_call, daemon=True)
        self.thread_running = False

    def _periodic_call(self):
        self.thread_running = True
        while self.thread_running:
            payload, fut = self.pending_call.get()
            try:
                prompt = PROMPT_REGISTRY[payload.augment_type]
                p_dict = payload.to_dict()
                messages = p_dict.pop("old_messages",[])
                prompt = prompt.format(**p_dict)
                messages.append({"role":"user","content":prompt})
                r = self.client.send_messages(
                    payload.model,
                    messages,
                    payload.max_tokens,
                    keep_messages=True
                )
                if payload.keep_message:
                    messages.append(r)
                    result = messages
                else:
                    result = r['content']
                fut.set_result((payload.id, result))
            except Exception as e:
                fut.set_exception(e)


    def submit(self, payload : BasePayload, callback):
        """
        Submit a payload to the worker. If a callback is defined, it will call it when th result is complete.
        
        :param payload: The Payload Request
        :type payload: BasePayload
        :param callback: A callback function
        """
        if not self.thread_running:
            self.thread.start()
        fut = Future()
        if callback is not None:
            fut.add_done_callback(callback)
        self.pending_call.put((payload,fut))
        return fut
        
