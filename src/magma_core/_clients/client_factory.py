from typing import Type
from .client_base import LLMClientBase
from .ollama_client import OllamaClient

from magma_core.configs.config import BackendConfig

CLIENT_REGISTRY: dict[str, Type[LLMClientBase]] = {
    "ollama" : OllamaClient
}

class ClientFactory:

    @staticmethod
    def create_client(backend_config : BackendConfig):
        if backend_config.type not in CLIENT_REGISTRY:
            raise ValueError(f"Unknown LM Backend {backend_config.type}. Avalaible are : {','.join(CLIENT_REGISTRY.keys())}")

        ClientCLS = CLIENT_REGISTRY[backend_config.type]
        return ClientCLS(**backend_config.to_dict())