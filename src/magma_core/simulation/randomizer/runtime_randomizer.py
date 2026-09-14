# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

# Author : Loan BERNAT
# 2 BSD

import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

from magma_core.simulation.data_structures import (
    EmptyInstruction,
    Instruction,
    Situation,
    StageInput,
    StatusReturn,
    TemplateInstruction,
    ToolExecution,
    ToolStatus,
    UserInstruction,
)
from magma_core.simulation.randomizer.random_spec import RandomizationSpec

class RuntimeRandomizer():
    """Runtime adapter that applies a pre-generated `RandomizationSpec`.

    This class does not generate randomness. Its job is to:
    - expose randomized tools/attributes to the LLM
    - translate LLM tool calls back to real runtime calls
    - translate environment messages/situations back to the randomized view
    """

    tools: List[Dict]
    tools_equivalence: Dict
    reversed_tools_equivalence: Dict
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
        self.reversed_tools_equivalence = {}
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

        self.tools = deepcopy(spec.tools)
        self.tools_equivalence = dict(spec.tool_equivalence)
        self.reversed_tools_equivalence = {
            tool_mapping["name"]: {
                "name": public_tool_name,
                "parameters": {
                    real_parameter_name: public_parameter_name
                    for public_parameter_name, real_parameter_name
                    in tool_mapping["parameters"].items()
                },
            }
            for public_tool_name, tool_mapping in self.tools_equivalence.items()
        }
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

        for tool in self.tools:
            tool["description"] = self.traduce_attributes_to_llm(
                tool.get("description", "")
            )
            for parameter in tool.get("parameters", {}).values():
                parameter["description"] = self.traduce_attributes_to_llm(
                    parameter.get("description", "")
                )

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
                r"(?<![A-Za-z0-9])((?:"
                + "|".join(
                    map(
                        re.escape,
                        sorted(
                            self.reversed_attributes_equivalence,
                            key=len,
                            reverse=True,
                        ),
                    )
                )
                + r"))(?![A-Za-z0-9])"
            )

        if self.attributes_equivalence:
            self._pattern_to_llm = re.compile(
                r"(?<![A-Za-z0-9])((?:"
                + "|".join(
                    map(
                        re.escape,
                        sorted(
                            self.attributes_equivalence,
                            key=len,
                            reverse=True,
                        ),
                    )
                )
                + r"))(?![A-Za-z0-9])"
            )

    def get_tools(self) -> List[Dict]:
        """Return the randomized tool schema that should be shown to the LLM."""
        return self.tools

    def get_real_tool_name(self, public_tool_name: str) -> str:
        return self.tools_equivalence[public_tool_name]["name"]

    def get_tmp_translation(self, node_id: int):
        return self.tmp_translation[node_id]

    def set_tmp_translation(self, node_id: int, name) -> None:
        self.tmp_translation[node_id] = name

    def translate_skill_arguments_to_runtime(self, arguments: Dict) -> Dict:
        translated, unknown_attributes = self._recursive_arguments_replacement(arguments)
        if unknown_attributes:
            raise ValueError(
                self._format_unknown_attributes_reason(unknown_attributes)
            )
        return translated

    def translate_skill_call_to_public(
            self,
            tool_name: str,
            arguments: Dict,
        ) -> Tuple[str, Dict]:
        tool_mapping = self.reversed_tools_equivalence[tool_name]
        parameter_mapping = tool_mapping["parameters"]
        translated_arguments = {
            parameter_mapping.get(name, name): self._recursive_arguments_to_public(value)
            for name, value in arguments.items()
        }
        return tool_mapping["name"], translated_arguments

    def _recursive_arguments_to_public(self, argument: Any) -> Any:
        if isinstance(argument, list):
            return [self._recursive_arguments_to_public(value) for value in argument]
        if isinstance(argument, dict):
            return {
                self._recursive_arguments_to_public(key): (
                    self._recursive_arguments_to_public(value)
                )
                for key, value in argument.items()
            }
        return self.attributes_equivalence.get(argument, argument)

    def get_randomized_attributes(self, attributes: Dict) -> Dict:
        """Translate task attributes into the randomized/public representation.

        List-valued attributes keep their configured presentation order. Other
        nested values are translated recursively without changing their shape.
        """
        out = {}
        for att_name, att_val in attributes.items():
            if att_name == "known_robots":
                out[att_name] = att_val
                continue

            randomized_value = self._recursive_arguments_to_public(att_val)
            if isinstance(randomized_value, list):
                randomized_value = self._order_attribute_values(
                    att_name,
                    randomized_value,
                )
            out[att_name] = randomized_value

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
    
    def get_randomized_instruction(self, instruction : Instruction) -> Instruction:
        if isinstance(instruction, UserInstruction):
            new_instruction_content = self.traduce_attributes_to_llm(
                instruction.get_content()
            )
            new_instruction = UserInstruction(
                new_instruction_content,
                timestamp=instruction.get_timestamp(),
                has_constraint=instruction.has_constraint,
            )
        elif isinstance(instruction, StatusReturn):
            new_instruction_content = self.traduce_attributes_to_llm(
                instruction.get_content()
            )
            new_instruction = StatusReturn(
                json.loads(new_instruction_content),
                timestamp=instruction.get_timestamp(),
            )
        elif isinstance(instruction, TemplateInstruction):
            new_template_str = self.traduce_attributes_to_llm(
                json.dumps(instruction.template)
            )
            new_context_str = self.traduce_attributes_to_llm(
                instruction.context
            )
            new_instruction = TemplateInstruction(
                json.loads(new_template_str),
                new_context_str,
                timestamp=instruction.get_timestamp(),
            )
        else:
            new_instruction = EmptyInstruction()
        
        return new_instruction

    def get_randomized_situation(self, original_situation: Situation) -> Situation:
        """Clone a situation while translating every user-visible string.

        This is used when the LLM should only ever see the randomized/public
        naming, even though the underlying task still uses the real names.
        """
        attributes = self.get_randomized_attributes(original_situation.attributes)

        new_instruction = self.get_randomized_instruction(original_situation.instruction)

        memory_content = self.traduce_attributes_to_llm(
            json.dumps(original_situation.memory)
        )
        new_memory = json.loads(memory_content)

        return original_situation.copy_with(
            memory=new_memory,
            attributes=attributes,
            instruction=new_instruction,
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
            return ToolExecution(
                [],
                None,
                reason=self._format_unknown_attributes_reason(err_param_values),
                reason_is_public=True,
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

    def _recursive_arguments_replacement(
        self,
        argument: Any,
    ) -> Tuple[Any, List[str]]:
        """Translate nested tool arguments from randomized values to real ones."""
        unknown_arg: List[str] = []

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
                    return argument
                return argument
            return att

        att = inner(argument)

        return att, unknown_arg

    def _format_unknown_attributes_reason(
        self,
        unknown_attributes: List[str],
    ) -> str:
        unique_attributes = list(dict.fromkeys(unknown_attributes))
        invalid = ", ".join(unique_attributes)
        verb = "was" if len(unique_attributes) == 1 else "were"
        return (
            f"{invalid} {verb} not detected."
        )

    def _traduce_reason(
            self,
            reason: str,
            public_tool_name: Optional[str],
            reason_is_public: bool = False,
        ) -> str:
        """Translate a runtime error/status message back to the LLM view."""
        if reason_is_public:
            return reason

        if reason.startswith("Missing arguments:") or reason.startswith("Argument"):
            if public_tool_name is None:
                return reason
            return self._replace_missing_arg(reason, public_tool_name)

        return self.traduce_attributes_to_llm(reason)

    def _traduce_att_modif(self, modif: List[Tuple[str, Tuple[str, str]]]):
        """Translate attribute modification payloads back to public names."""
        tmp = []
        for action, tupl in modif:
            att, value = tupl
            tmp.append((action, (att, self.traduce_attributes_to_llm(value))))
        return tmp

    def traduce_end(self, out: Dict[int, ToolStatus]) -> Dict[int, ToolStatus]:
        """Translate generation-mode tool results into the randomized view."""
        if out and (self.attributes_equivalence or self.tools_equivalence):
            for node_id, status in out.items():
                stored_tool_names = self.tmp_translation.pop(node_id, None)
                if isinstance(stored_tool_names, list):
                    public_tool_names = stored_tool_names
                elif isinstance(stored_tool_names, str):
                    public_tool_names = [stored_tool_names]
                else:
                    public_tool_names = []

                for index, robot_status in enumerate(status.robots_status):
                    public_tool_name = (
                        public_tool_names[index]
                        if index < len(public_tool_names)
                        else None
                    )
                    robot_status.mess = self._traduce_reason(
                        robot_status.mess,
                        public_tool_name,
                        robot_status.mess_is_public,
                    )
                if status.next_input is not None:
                    status.next_input = StageInput(
                        self.get_randomized_instruction(status.next_input.instruction),
                        status.next_input.flag_answer_to_user,
                        status.next_input.linked_to_prev
                    )
                if status.attributes is not None:
                    status.attributes = self.get_randomized_attributes(status.attributes)
        return out
