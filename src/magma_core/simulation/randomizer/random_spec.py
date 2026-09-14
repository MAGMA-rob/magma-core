# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

# Author : Loan BERNAT
# 2 BSD

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RandomizationSpec:
    """Serializable description of one randomization setup.

    The generator builds this object once, then the runtime wrapper consumes it
    without doing any additional random choice.

    Attributes:
        tool_equivalence: Mapping from randomized tool names to the real runtime
            tool name and randomized-to-real parameter mapping.
        attributes_equivalence: Mapping from real task attribute values to the
            names exposed to the LLM.
        reversed_attributes_equivalence: Reverse mapping used when converting a
            model tool call back into environment values.
        tools: Tool schema already rewritten with randomized names/descriptions.
        attribute_keys_order: Final order in which attribute groups should be
            exposed to the LLM.
        attribute_values_order: Final order of values inside each attribute
            group after randomization/deterministic variation selection.
    """

    tool_equivalence: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    attributes_equivalence: Dict[str, str] = field(default_factory=dict)
    reversed_attributes_equivalence: Dict[str, str] = field(default_factory=dict)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    attribute_keys_order: List[str] = field(default_factory=list)
    attribute_values_order: Dict[str, List[str]] = field(default_factory=dict)
