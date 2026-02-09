from typing import Dict, Optional
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

    def to_dict(self) -> Dict:
        return {
            "endpoint" : self.endpoint,
            "timeout" : self.timeout,
            "max_retry": self.max_retry,
            "headers": self.headers,
            "default_model": self.default_model
        }

class MAGMAConfig:
    backends: Dict[int,BackendConfig]

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
    
