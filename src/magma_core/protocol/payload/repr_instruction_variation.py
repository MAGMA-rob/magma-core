# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import json
from typing import Any, ClassVar, Dict, Optional

from .base_payload import BasePayload
from magma_core._prompts.dataset.paraphrasing import (
  REPR_INSTRUCTION_VARIATIONS, get_repr_variation_guidance,
  get_repr_variation_output_format
)

class ReprInstructionVariationPayload(BasePayload):
    """
    Payload to generate semantic-preserving syntactic variations of an
    instruction for representation-update data augmentation.
    """

    USER_UTTERANCE_MODE: ClassVar[str] = "user_utterance"
    FEEDBACK_VARIATION_MODE: ClassVar[str] = "feedback_variation"
    prompt_template: ClassVar[str] = REPR_INSTRUCTION_VARIATIONS

    def __init__(
        self,
        instruction: Any,
        nb_variations: int,
        id: int,
        variation_mode: str,
        max_tokens: int = 5000,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(id, max_tokens, model)
        self.instruction = instruction
        self.nb_variations = nb_variations
        self.variation_mode = variation_mode

    def _format_instruction_for_prompt(self) -> str:
        if isinstance(self.instruction, str):
            return self.instruction
        return json.dumps(self.instruction, ensure_ascii=False, indent=2, default=str)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction": self._format_instruction_for_prompt(),
            "nb_variations": self.nb_variations,
            "variation_mode": self.variation_mode,
            "variation_guidance": get_repr_variation_guidance(self.variation_mode),
            "output_format": get_repr_variation_output_format(self.variation_mode),
        }

    def get_data(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "nb_variations": self.nb_variations,
            "variation_mode": self.variation_mode,
        }
