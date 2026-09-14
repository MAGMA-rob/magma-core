from typing import Any, Protocol

from concurrent.futures import Future

from magma_core.protocol.payload import BasePayload


class PayloadWorker(Protocol):
    def submit(self, payload: BasePayload, callback: Any) -> Future:
        ...

    def get_capacity(self) -> int:
        ...
