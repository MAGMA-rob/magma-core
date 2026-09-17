# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Dict, Optional, Any
from urllib.parse import urlparse
import yaml
from pathlib import Path
from dataclasses import dataclass, field

@dataclass(frozen=True)
class BackendConfig:
    type: str
    endpoint: str
    default_model: str
    headers: Dict[str,str]
    timeout: float = 30
    max_retry: int = 3

@dataclass(frozen=True)
class CoachingExamplesConfig:
    register_successful: bool = False
    use_for_automatic: bool = False
    cache_dir: str = "cache"


@dataclass(frozen=True)
class CoachingConfig:
    enabled: bool
    provider: str
    endpoint: Optional[str] = None
    connect_timeout: float = 10
    examples: CoachingExamplesConfig = field(default_factory=CoachingExamplesConfig)

@dataclass
class MAGMAConfig:
    backends: Dict[str,BackendConfig]
    coaching: CoachingConfig
    generate: Dict[str, Any]
    benchmark: Dict[str, Any]
    magma_agent_address: str
    magma_planner_address: str
    magma_agent_timeout: float = 360

    @staticmethod
    def load(path : Optional[Path] = None, accept_no_backend : bool = False) -> "MAGMAConfig":
        default_path = Path(__file__).parent / "default_config.yaml"

        with open(default_path) as f:
            data : Dict = yaml.safe_load(f)

        if path is not None and path.exists():
            with open(path) as f:
                user_data = yaml.safe_load(f)
            data.update(user_data)

        generate = data.get("generate", {})
        if "coaching" in generate:
            raise ValueError(
                "generate.coaching is no longer supported; use the top-level coaching section"
            )

        coaching_data = data.get("coaching")
        if not isinstance(coaching_data, dict):
            raise ValueError("The top-level coaching section must be an object")
        coaching_data = dict(coaching_data)
        examples_data = coaching_data.get("examples", {})
        if not isinstance(examples_data, dict):
            raise ValueError("coaching.examples must be an object")
        coaching_data["examples"] = CoachingExamplesConfig(**examples_data)
        data["coaching"] = CoachingConfig(**coaching_data)

        if accept_no_backend:
            data["backends"] = {}

        backend_dict = data.get("backends",{})

        backends = {}
        for instance, info in backend_dict.items():
            backends[instance] = BackendConfig(**info)

        data["backends"] = backends
        return MAGMAConfig(**data)
    
    def override_with_dict(self, arg_dict : Dict):
        """
        Override the configuration with runtime arguments
        """

        def _recursive_override(base : Dict, override : Any):
            for k, v in override.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    _recursive_override(base[k], v)
                else:
                    base[k] = v
        
        if "generate" in arg_dict:
            _recursive_override(self.generate, arg_dict["generate"])
        
        if "benchmark" in arg_dict:
            _recursive_override(self.benchmark, arg_dict["benchmark"])

        if "coaching" in arg_dict:
            coaching = {
                "enabled": self.coaching.enabled,
                "provider": self.coaching.provider,
                "endpoint": self.coaching.endpoint,
                "connect_timeout": self.coaching.connect_timeout,
                "examples": {
                    "register_successful": self.coaching.examples.register_successful,
                    "use_for_automatic": self.coaching.examples.use_for_automatic,
                    "cache_dir": self.coaching.examples.cache_dir,
                },
            }
            _recursive_override(coaching, arg_dict["coaching"])
            coaching["examples"] = CoachingExamplesConfig(**coaching["examples"])
            self.coaching = CoachingConfig(**coaching)
        
        # Backend activation
        active_backends = arg_dict.get("active_backends", [])
        if active_backends:
            self.backends = {
                b: self.backends[b]
                for b in active_backends
                if b in self.backends
            }

        # Server override
        if arg_dict.get("magma_agent_address"):
            self.magma_agent_address = arg_dict["magma_agent_address"]

        if arg_dict.get("magma_agent_timeout") is not None:
            self.magma_agent_timeout = arg_dict["magma_agent_timeout"]

    def verify(self, accept_no_backend: bool = False):
        """
        Raise a ValueError if the config is not valid
        """

        if self.coaching.provider not in {"llm", "human"}:
            raise ValueError("The coaching provider must be either 'llm' or 'human'")

        if len(self.backends) == 0 and not accept_no_backend:
            raise ValueError(
                "There is no backend specified for generation services such as judging and user simulation"
            )

        if self.coaching.connect_timeout <= 0:
            raise ValueError("The coaching connect_timeout must be greater than 0")

        if self.magma_agent_timeout <= 0:
            raise ValueError("The magma_agent_timeout must be greater than 0")

        if not self.coaching.examples.cache_dir.strip():
            raise ValueError("The coaching cache_dir must be non-empty")

        if (
            self.coaching.enabled
            and self.coaching.provider == "llm"
            and len(self.backends) == 0
        ):
            raise ValueError("LLM coaching requires at least one backend")

        if self.coaching.enabled and self.coaching.provider == "human":
            endpoint = self.coaching.endpoint or ""
            parsed_endpoint = urlparse(endpoint)
            if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.netloc:
                raise ValueError("Human coaching requires a valid HTTP(S) endpoint")
        

        if self.generate['nb_branch'] < 1:
            raise ValueError(f"The nb_branch parameters must be >= 1")
        
        if self.generate['nb_env'] < 1:
            raise ValueError(f"The nb_env parameters must be > 1")
        
        if self.generate['nb_max_update'] <= 1:
            raise ValueError(f"The nb_max_update parameters must be > 1")
