# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

# Author : Loan BERNAT
# 2 BSD

import json
import re
from typing import Dict, List, Optional, Tuple, Union

from ..data_structures import (
    EmptyInstruction,
    Situation,
    StatusReturn,
    TemplateInstruction,
    ToolErrorFlag,
    ToolExecution,
    ToolStatus,
    UserInstruction,
)
from .random_spec import RandomizationSpec

class RuntimeRandomizer():
    """Runtime adapter that applies a pre-generated `RandomizationSpec`.

    This class does not generate randomness. Its job is to:
    - expose randomized tools/attributes to the LLM
    - translate LLM tool calls back to real runtime calls
    - translate environment messages/situations back to the randomized view
    """

    tools: List[Dict]
    tools_equivalence: Dict
    attributes_equivalence: Dict
    reversed_attributes_equivalence: Dict
    memory: List[str]
    instruction: str
    tmp_translation: Dict

    def __init__(self, spec: Optional[RandomizationSpec] = None) -> None:
        """Create an empty wrapper, optionally hydrated from a generated spec."""
        self._reset_runtime_state()
        if spec is not None:
            self.initialize_randomizer(spec)

    def _reset_runtime_state(self) -> None:
        """Clear all runtime caches before loading a new spec."""
        self.tools = []
        self.tools_equivalence = {}
        self.attributes_equivalence = {}
        self.reversed_attributes_equivalence = {}
        self.attribute_keys_order = []
        self.attribute_values_order = {}
        self.original_attributes_listed = set()
        self._param_patterns = {}
        self._pattern_to_env = None
        self._pattern_to_llm = None
        self.tmp_translation = {}

    def initialize_randomizer(self, spec: RandomizationSpec) -> None:
        """Load a generated spec and rebuild the runtime lookup helpers."""
        self._reset_runtime_state()

        self.tools = list(spec.tools)
        self.tools_equivalence = dict(spec.tool_equivalence)
        self.attributes_equivalence = dict(spec.attributes_equivalence)
        self.reversed_attributes_equivalence = dict(spec.reversed_attributes_equivalence)
        self.attribute_keys_order = list(spec.attribute_keys_order)
        self.attribute_values_order = {
            att_name: list(values)
            for att_name, values in spec.attribute_values_order.items()
        }
        self.original_attributes_listed = set(self.attributes_equivalence.keys())

        self._initialize_param_patterns()
        self._initialize_attribute_patterns()

    def _initialize_param_patterns(self) -> None:
        """Precompute regex replacements for tool argument error messages."""
        for randomized_tool_name, tool_mapping in self.tools_equivalence.items():
            reversed_p = {
                real_name: randomized_param_name
                for randomized_param_name, real_name in tool_mapping["parameters"].items()
            }
            if not reversed_p:
                continue

            pattern = re.compile(
                r"\b(" + "|".join(map(re.escape, reversed_p.keys())) + r")\b"
            )
            self._param_patterns[randomized_tool_name] = (pattern, reversed_p)

    def _initialize_attribute_patterns(self) -> None:
        """Precompute regex replacements for env<->LLM text translation."""
        if self.reversed_attributes_equivalence:
            self._pattern_to_env = re.compile(
                r"\b("
                + "|".join(map(re.escape, self.reversed_attributes_equivalence.keys()))
                + r")\b"
            )

        if self.attributes_equivalence:
            self._pattern_to_llm = re.compile(
                r"\b("
                + "|".join(map(re.escape, self.attributes_equivalence.keys()))
                + r")\b"
            )

    def get_tools(self) -> List[Dict]:
        """Return the randomized tool schema that should be shown to the LLM."""
        return self.tools

    def get_randomized_attributes(self, attributes: Dict) -> Dict:
        """Translate task attributes into the randomized/public representation.

        Only list-valued attributes are renamed/reordered. Non-list values are
        passed through unchanged for now.
        """
        out = {}
        for att_name, att_val in attributes.items():
            if att_name == "known_robots" or not isinstance(att_val, list):
                out[att_name] = att_val
                continue

            randomized_values = []
            for val in att_val:
                if val in self.original_attributes_listed:
                    randomized_val = self.attributes_equivalence.get(val, None)
                else:
                    randomized_val = val
                randomized_values.append(randomized_val)

            out[att_name] = self._order_attribute_values(att_name, randomized_values)

        return self._order_attribute_keys(out)

    def _order_attribute_values(
        self,
        att_name: str,
        values: List[str],
    ) -> List[str]:
        """Reorder one attribute list according to the loaded spec."""
        target_order = self.attribute_values_order.get(att_name, None)
        if not target_order:
            return values

        order_index = {value: idx for idx, value in enumerate(target_order)}
        ordered = sorted(
            enumerate(values),
            key=lambda item: (order_index.get(item[1], len(order_index)), item[0]),
        )
        return [value for _, value in ordered]

    def _order_attribute_keys(self, attributes: Dict) -> Dict:
        """Reorder attribute groups according to the loaded spec."""
        if not self.attribute_keys_order:
            return attributes

        order_index = {
            key: idx
            for idx, key in enumerate(self.attribute_keys_order)
        }
        ordered_keys = sorted(
            attributes.keys(),
            key=lambda key: (order_index.get(key, len(order_index)),),
        )
        return {key: attributes[key] for key in ordered_keys}

    def get_randomized_situation(self, original_situation: Situation) -> Situation:
        """Clone a situation while translating every user-visible string.

        This is used when the LLM should only ever see the randomized/public
        naming, even though the underlying task still uses the real names.
        """
        attributes = self.get_randomized_attributes(original_situation.attributes)

        if isinstance(original_situation.instruction, UserInstruction):
            new_instruction_content = self.traduce_attributes_to_llm(
                original_situation.instruction.get_content()
            )
            new_instruction = UserInstruction(new_instruction_content)
        elif isinstance(original_situation.instruction, StatusReturn):
            new_instruction_content = self.traduce_attributes_to_llm(
                original_situation.instruction.get_content()
            )
            new_instruction = StatusReturn(json.loads(new_instruction_content))
        elif isinstance(original_situation.instruction, TemplateInstruction):
            new_template_str = self.traduce_attributes_to_llm(
                json.dumps(original_situation.instruction.template)
            )
            new_context_str = self.traduce_attributes_to_llm(
                original_situation.instruction.context
            )
            new_instruction = TemplateInstruction(
                json.loads(new_template_str),
                new_context_str,
            )
        else:
            new_instruction = EmptyInstruction()

        new_memory = [
            self.traduce_attributes_to_llm(mem)
            for mem in original_situation.memory
        ]
        new_user_scenario = self.traduce_attributes_to_llm(
            original_situation.user_scenario
        )

        return original_situation.copy_with(
            memory=new_memory,
            attributes=attributes,
            instruction=new_instruction,
            user_scenario=new_user_scenario,
        )

    def map_tool_call(
        self,
        func_name: str,
        params: Dict,
    ) -> Union[ToolExecution, Tuple[str, Dict]]:
        """Translate a randomized LLM tool call into a real runtime call.

        Returns:
            Either `(real_function_name, real_params)` when translation
            succeeds, or a `ToolExecution` containing the validation error that
            should be surfaced to the LLM.
        """
        if func_name not in self.tools_equivalence:
            return ToolExecution(
                [],
                None,
                reason=f"{func_name} is not a known tool."
            )

        if not isinstance(params, dict):
            return ToolExecution(
                [],
                None,
                reason=f"Arguments for {func_name} must be a dict, got {type(params).__name__}."
            )

        mapping = self.tools_equivalence[func_name]["parameters"]
        new_func_name = self.tools_equivalence[func_name]["name"]

        new_params, err_param_names, err_param_values = {}, [], []
        for p, val in params.items():
            if p in mapping:
                param_mapped, unk = self._recursive_arguments_replacement(val)
                if len(unk) > 0:
                    err_param_values.extend(unk)
                else:
                    new_params[mapping[p]] = param_mapped
            else:
                err_param_names.append(p)

        if err_param_names:
            invalid = ", ".join(err_param_names)
            return ToolExecution(
                [],
                None,
                reason=f"{invalid} are not valid arguments"
            )

        if err_param_values:
            invalid = ", ".join(err_param_values)
            return ToolExecution(
                [],
                None,
                reason=f"Unknow attributes: {invalid} are not valid attributes"
            )

        return new_func_name, new_params

    def traduce_attributes_to_env(self, text: str) -> str:
        """Replace randomized attribute names with their real env names."""
        if self._pattern_to_env is not None:
            return self._pattern_to_env.sub(
                lambda m: self.reversed_attributes_equivalence[m.group(0)],
                text,
            )
        return text

    def traduce_attributes_to_llm(self, text: str) -> str:
        """Replace real attribute names with the randomized public names."""
        if self._pattern_to_llm is not None:
            return self._pattern_to_llm.sub(
                lambda m: self.attributes_equivalence[m.group(0)],
                text,
            )
        return text

    def _replace_missing_arg(self, text: str, func_name: str) -> str:
        """Rewrite runtime parameter names back to the randomized ones."""
        pattern_data = self._param_patterns.get(func_name, None)
        if pattern_data is None:
            return text

        pattern, reversed_p = pattern_data
        return pattern.sub(lambda m: reversed_p[m.group(0)], text)

    def _recursive_arguments_replacement(self, argument):
        """Translate nested tool arguments from randomized values to real ones."""
        unknown_arg = []

        def inner(argument):
            if isinstance(argument, list):
                return [inner(a) for a in argument]

            if isinstance(argument, dict):
                out = {}
                for k, v in argument.items():
                    out[inner(k)] = inner(v)
                return out

            att = self.reversed_attributes_equivalence.get(argument, None)
            if not att:
                if argument in self.original_attributes_listed:
                    unknown_arg.append(argument)
                    return "_"
                return argument
            return att

        att = inner(argument)

        return att, unknown_arg

    def _traduce_reason(self, reason: Optional[str], node_id: int) -> Optional[str]:
        """Translate a runtime error/status message back to the LLM view."""
        if reason is not None:
            src_names = self.tmp_translation.pop(node_id, None)
            if isinstance(src_names, list):
                if len(src_names) > 1:
                    raise NotImplementedError(
                        "Randomization for multi-robot tool is not yet implemented"
                    )
                src_names = src_names[0]

            if reason.startswith("Unknow attributes:"):
                return reason

            if reason.startswith("Missing arguments:") or reason.startswith("Argument"):
                if src_names is None:
                    return reason
                return self._replace_missing_arg(reason, src_names)

            return self.traduce_attributes_to_llm(reason)
        return reason

    def _traduce_att_modif(self, modif: List[Tuple[str, Tuple[str, str]]]):
        """Translate attribute modification payloads back to public names."""
        tmp = []
        for action, tupl in modif:
            att, value = tupl
            tmp.append((action, (att, self.traduce_attributes_to_llm(value))))
        return tmp

    def traduce_end_gen(self, out: Dict[int, ToolStatus]) -> Dict[int, ToolStatus]:
        """Translate generation-mode tool results into the randomized view."""
        if out and (self.attributes_equivalence or self.tools_equivalence):
            for node_id, status in out.items():
                status.mess = self._traduce_reason(status.mess, node_id)
                if status.next_situation is not None:
                    status.next_situation = self.get_randomized_situation(
                        status.next_situation
                    )
                status.attributes_modif = self._traduce_att_modif(status.attributes_modif)
        return out

    def traduce_end_eval(self, out: Dict[int, Dict]) -> Dict[int, Dict]:
        """Translate evaluation-mode tool results into the randomized view."""
        if out and (self.attributes_equivalence or self.tools_equivalence):
            for node_id, status in out.items():
                status["reason"] = self._traduce_reason(status.get("reason", None), node_id)
                status["att_modif"] = self._traduce_att_modif(status["att_modif"])
        return out
