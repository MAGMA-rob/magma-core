from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from magma_core.configs import BackendConfig

class LLMClientBase(ABC):
    """
    Base class to create a custom LLM CLient to handle payload sended by the workers for coaching, user simulation, curriculum etc...
    """

    def __init__(self, config: BackendConfig, server_instance : int = 0):
        self.config = config
        self.server_instance = server_instance

    @abstractmethod
    def send_prompt(
        self,
        model: Optional[str],
        prompt: str,
        max_tokens: int
    ) -> str:
        ...

    @abstractmethod
    def send_messages(
        self,
        model: Optional[str],
        messages: List[Dict[str, Any]],
        max_tokens: int,
        keep_messages: bool = False
    ) -> Any:
        ...

    def test_server(self):
        self.send_prompt(
            self.config.default_model,
            "Hello how are you?",
            max_tokens=50
        )