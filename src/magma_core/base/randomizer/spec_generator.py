# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

# Author : Loan BERNAT
# 2 BSD

import random
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..tasks import BaseTask
from .random_spec import RandomizationSpec


class SpecGenerator:
    """Builds `RandomizationSpec` objects from a task randomization YAML file.

    This class owns all generation-time choices:
    - fully random spec generation for a single run
    - deterministic indexed variation generation for benchmarking

    The output spec is then passed to `ToolRandomizerWrapper`, which only
    applies the prepared mapping at runtime.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Create an isolated RNG so spec generation stays reproducible."""
        self._random = random.Random(seed)

    def generate_random_spec(self, task_ref: BaseTask) -> RandomizationSpec:
        """Build one spec using random choices for every configurable option."""
        config = self._load_config(task_ref)
        return self._build_spec(task_ref, config, indexed_variation=None)

    def generate_N_spec(
        self,
        task_ref: BaseTask,
        n: Optional[int] = None,
    ) -> List[RandomizationSpec]:
        """Build `n` deterministic specs by indexing config choices.

        Variation `i` picks the `i`th value for each configurable list, with
        modulo wrapping when some lists are shorter than others.
        """
        config = self._load_config(task_ref)

        if n is None:
            n = self._infer_nb_variations(config)

        if n <= 0:
            return []

        return [
            self._build_spec(task_ref, config, indexed_variation=variation_idx)
            for variation_idx in range(n)
        ]

    def _load_config(self, task_ref: BaseTask) -> Dict[str, Any]:
        """Load the YAML randomization config declared by the task."""
        if task_ref.randomized_config_path == "":
            raise TypeError(f"The {task_ref.name} has no randomized file path set")

        with open(task_ref.randomized_config_path, "r") as f:
            return yaml.safe_load(f)

    def _build_spec(
        self,
        task_ref: BaseTask,
        config: Dict[str, Any],
        indexed_variation: Optional[int],
    ) -> RandomizationSpec:
        """Assemble a full spec for one random or deterministic variation."""
        tool_config = config.get("tools", None)
        attributes_config = config.get("attributes", {})

        if tool_config is None:
            raise ValueError("No tool randomization detected.")

        tools, tool_equivalence = self._build_tools_spec(
            tool_config,
            task_ref.get_tools(),
            indexed_variation=indexed_variation,
        )

        (
            attributes_equivalence,
            reversed_attributes_equivalence,
            attribute_keys_order,
            attribute_values_order,
        ) = self._build_attributes_spec(
            attributes_config,
            task_ref.all_task_attributes,
            indexed_variation=indexed_variation,
        )

        return RandomizationSpec(
            tool_equivalence=tool_equivalence,
            attributes_equivalence=attributes_equivalence,
            reversed_attributes_equivalence=reversed_attributes_equivalence,
            tools=tools,
            attribute_keys_order=attribute_keys_order,
            attribute_values_order=attribute_values_order,
        )

    def _build_tools_spec(
        self,
        tool_config: Dict[str, Any],
        original_tools: List[Dict[str, Any]],
        indexed_variation: Optional[int],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Build the randomized public tool schema and reverse lookup table."""
        tools: List[Dict[str, Any]] = []
        tools_equivalence: Dict[str, Dict[str, Any]] = {}

        for tool in original_tools:
            tool_augmentation = tool_config.get(tool["name"], None)

            if not tool_augmentation:
                print(f"[RANDOMIZER] Unknow function {tool['name']}")

                # If a tool has no config entry, keep it visible exactly as-is.
                parameters_dict = {
                    param_name: {
                        "description": param_val["description"],
                        "type": param_val["type"],
                    }
                    for param_name, param_val in tool["parameters"].items()
                }
                tools.append(
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": parameters_dict,
                    }
                )
                tools_equivalence[tool["name"]] = {
                    "name": tool["name"],
                    "parameters": {k: k for k in tool["parameters"]},
                }
                continue

            name = self._select(tool_augmentation["name"], indexed_variation)
            description = self._select(
                tool_augmentation["description"],
                indexed_variation,
            )

            parameters_dict = {}
            param_equivalence = {}
            tool_parameters_cfg = tool_augmentation.get("parameters", {})
            for param, param_val in tool["parameters"].items():
                possible_param_val = tool_parameters_cfg.get(param, None)
                if not possible_param_val:
                    print(
                        f"[RANDOMIZER] Unknow parameter {param} for function {tool['name']}"
                    )
                    # Missing parameter config falls back to the original schema.
                    p_name = param
                    p_desc = param_val["description"]
                else:
                    p_name = self._select(possible_param_val["name"], indexed_variation)
                    p_desc = self._select(
                        possible_param_val["description"],
                        indexed_variation,
                    )

                parameters_dict[p_name] = {
                    "description": p_desc,
                    "type": param_val["type"],
                }
                param_equivalence[p_name] = param

            tools.append(
                {
                    "name": name,
                    "description": description,
                    "parameters": parameters_dict,
                }
            )
            tools_equivalence[name] = {
                "name": tool["name"],
                "parameters": param_equivalence,
            }

        return tools, tools_equivalence

    def _build_attributes_spec(
        self,
        attributes_config: Dict[str, Any],
        task_attributes: Dict[str, Any],
        indexed_variation: Optional[int],
    ) -> Tuple[Dict[str, str], Dict[str, str], List[str], Dict[str, List[str]]]:
        """Build attribute renaming plus the final ordering exposed to the LLM.

        `attributes_equivalence` controls value translation, while
        `attribute_keys_order` and `attribute_values_order` preserve the exact
        presentation order selected during generation.
        """
        attributes_equivalence: Dict[str, str] = {}
        attribute_values_order: Dict[str, List[str]] = {}
        attribute_keys_order = list(task_attributes.keys())

        if indexed_variation is None:
            # Single-spec generation keeps the previous randomized order.
            self._random.shuffle(attribute_keys_order)

        if not attributes_config:
            print("[RANDOMIZER] No config attributes detected. Shuffle as default.")

        for att_name, att_val in task_attributes.items():
            if att_name == "known_robots":
                continue

            if not isinstance(att_val, list):
                continue

            possible_values = attributes_config.get(att_name, None)
            if possible_values is None:
                if attributes_config:
                    print(
                        f"[RANDOMIZER] Warn : No randomization for attributes {att_name}"
                    )
                selected_values = list(att_val)
            else:
                self._validate_attribute_config(att_name, att_val, possible_values)
                selected_values = list(
                    self._select(possible_values, indexed_variation)[: len(att_val)]
                )
                for i, val in enumerate(att_val):
                    attributes_equivalence[val] = selected_values[i]

            if indexed_variation is None:
                # Keep random generation behavior for the visible value order too.
                attribute_values_order[att_name] = self._shuffled(selected_values)
            else:
                # Benchmark variations keep the selected config order untouched.
                attribute_values_order[att_name] = selected_values

        reversed_attributes_equivalence = {
            randomized_name: real_name
            for real_name, randomized_name in attributes_equivalence.items()
        }

        return (
            attributes_equivalence,
            reversed_attributes_equivalence,
            attribute_keys_order,
            attribute_values_order,
        )

    def _validate_attribute_config(
        self,
        att_name: str,
        att_val: List[Any],
        possible_values: List[Any],
    ) -> None:
        """Check that each configured attribute variation is structurally valid."""
        for i, config_val in enumerate(possible_values):
            if not isinstance(config_val, list):
                raise TypeError(
                    f"There is a config value ({i}) for attributes {att_name} "
                    f"which is not a List ({type(config_val)})"
                )
            if len(config_val) < len(att_val):
                raise TypeError(
                    f"There is a config value ({i}) for attributes {att_name} "
                    f"with a length ({len(config_val)}) < to task attributes ({len(att_val)})"
                )

    def _select(self, options: List[Any], indexed_variation: Optional[int]) -> Any:
        """Pick one option randomly or by deterministic variation index."""
        if len(options) == 0:
            raise ValueError("Randomization config contains an empty list of options.")

        if indexed_variation is None:
            return self._random.choice(options)

        return options[indexed_variation % len(options)]

    def _shuffled(self, values: List[str]) -> List[str]:
        """Return a shuffled copy without mutating the source list."""
        out = list(values)
        self._random.shuffle(out)
        return out

    def _infer_nb_variations(self, config: Dict[str, Any]) -> int:
        """Infer a reasonable default number of benchmark variations.

        The maximum configurable list length is used so every list contributes
        at least one explicit indexed choice before wrapping.
        """
        lengths: List[int] = []

        tools_cfg = config.get("tools", {})
        for tool_cfg in tools_cfg.values():
            lengths.extend(self._extract_lengths(tool_cfg))

        attributes_cfg = config.get("attributes", {})
        for possible_values in attributes_cfg.values():
            if isinstance(possible_values, list) and len(possible_values) > 0:
                lengths.append(len(possible_values))

        return max(lengths, default=1)

    def _extract_lengths(self, value: Any) -> List[int]:
        """Recursively collect configurable list lengths from the YAML tree."""
        lengths: List[int] = []

        if isinstance(value, list):
            if len(value) > 0 and all(not isinstance(item, list) for item in value):
                lengths.append(len(value))
            for item in value:
                lengths.extend(self._extract_lengths(item))
            return lengths

        if isinstance(value, dict):
            for item in value.values():
                lengths.extend(self._extract_lengths(item))

        return lengths
