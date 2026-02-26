# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Dict, Optional, Any
import yaml
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class BackendConfig:
    type: str
    endpoint: str
    default_model: str
    headers: Dict[str,str]
    timeout: float = 30
    max_retry: int = 3

@dataclass
class MAGMAConfig:
    backends: Dict[str,BackendConfig]
    generate: Dict[str, Any]
    benchmark: Dict[str, Any]
    magma_agent_address: str
    magma_planner_address: str

    @staticmethod
    def load(path : Optional[Path] = None) -> "MAGMAConfig":
        default_path = Path(__file__).parent / "default_config.yaml"

        with open(default_path) as f:
            data : Dict = yaml.safe_load(f)

        if path is not None and path.exists():
            with open(path) as f:
                user_data = yaml.safe_load(f)
            data.update(user_data)

        backend_dict = data.get("backends",{})

        if len(list(backend_dict.keys())) == 0:
            raise KeyError("Backends config must at least define one valid backend to use for coaching, user sim and curriculum")

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

    def verify(self):
        """
        Raise a ValueError if the config is not valid
        """

        if len(self.backends) == 0:
            raise ValueError("There is no backend specified. You must at least have one for UserSim")
        
        if self.generate["mode"] != "single" and self.generate["mode"] != "dual":
            raise ValueError(f"Unknow mode {self.generate['mode']}. Acceptable : single, dual")
        
        if self.generate['nb_branch'] <= 1:
            raise ValueError(f"The nb_branch parameters must be > 1")
        
        if self.generate['nb_env'] <= 1:
            raise ValueError(f"The nb_env parameters must be > 1")
        
        if self.generate['nb_max_update'] <= 1:
            raise ValueError(f"The nb_max_update parameters must be > 1")