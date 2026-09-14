# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Any, ClassVar, Dict, List, Optional, Tuple

from magma_core.protocol.agent_coaching import CoachingLog


class BasePayload(ABC):
    prompt_template: ClassVar[str]
    debug_log: ClassVar[bool] = False
    model: Optional[str]  # Set to None to use the default worker one.
    max_tokens: int
    id: int
    keep_message: bool
    allow_force_save_example: bool
    force_save_example: bool

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        prompt_template = getattr(cls, "prompt_template", None)
        if not isinstance(prompt_template, str) or len(prompt_template.strip()) == 0:
            raise TypeError(
                f"{cls.__name__} must define a non-empty prompt_template"
            )

    def __init__(
        self,
        id: int,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.log_sink: Callable[[CoachingLog], Path | None] | None = None
        self.log_type: str | None = None
        self.log_context: str = ""
        self.max_tokens = max_tokens
        self.model = model
        self.id = id
        self.keep_message = False
        self.allow_force_save_example = False
        self.force_save_example = False

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def to_human_dict(self) -> Dict[str, Any]:
        return self.to_dict()

    def set_coaching_examples(
        self,
        examples: List[Dict[str, Any]],
    ) -> None:
        """Attach phase-appropriate coaching examples when supported."""

    def extract_coaching_example_phase(
        self,
        output: str,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Return a compact validated cache phase for this payload."""
        return None
    
    def get_data(self) -> Dict[str, Any]:
        raise NotImplementedError()
