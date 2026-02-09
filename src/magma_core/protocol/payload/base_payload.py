from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional, Type
from ..registry import ExternalRequestType

class BasePayload(ABC):
    augment_type : ExternalRequestType
    model : Optional[str] # Set to None to use the default worker one.
    max_tokens : int
    id : int
    keep_message : bool

    def __init__(self, augment_type : ExternalRequestType, id :int, max_tokens = 5000, model : Optional[str] = None) -> None:
        super().__init__()
        self.augment_type = augment_type
        self.max_tokens = max_tokens
        self.model = model
        self.id = id
        self.keep_message = False

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()
    
    def get_data(self) -> Dict:
        raise NotImplementedError()