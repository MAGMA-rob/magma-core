# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import pytest

from magma_core.simulation.data_structures import (
    RobotToolStatus,
    StatusReturn,
    ToolErrorFlag,
    ToolExecution,
    ToolStatus,
    UserInstruction,
)
from magma_core.simulation.randomizer import Randomizer
from magma_core.simulation.randomizer.random_spec import RandomizationSpec
from magma_core.simulation.randomizer.runtime_randomizer import RuntimeRandomizer
from magma_core.simulation.randomizer.spec_generator import SpecGenerator


def test_runtime_randomizer_rejects_non_dict_tool_arguments() -> None:
    randomizer = RuntimeRandomizer(
        RandomizationSpec(
            tool_equivalence={
                "public_tool": {
                    "name": "real_tool",
                    "parameters": {"public_arg": "real_arg"},
                }
            }
        )
    )

    out = randomizer.map_tool_call("public_tool", "public_arg=value")

    assert isinstance(out, ToolExecution)
    assert out.failure_flag == ToolErrorFlag.BAD_CALL
    assert out.reason == "Arguments for public_tool must be a dict, got str."


def test_runtime_randomizer_tracks_tool_translation_per_execution_id() -> None:
    randomizer = RuntimeRandomizer(RandomizationSpec())

    randomizer.set_tmp_translation(4, ["public_tool"])

    assert randomizer.get_tmp_translation(4) == ["public_tool"]
def test_runtime_randomizer_translates_multi_robot_results_per_tool() -> None:
    randomizer = RuntimeRandomizer(
        RandomizationSpec(
            tool_equivalence={
                "public_first": {
                    "name": "real_first",
                    "parameters": {"public_first_arg": "real_first_arg"},
                },
                "public_second": {
                    "name": "real_second",
                    "parameters": {"public_second_arg": "real_second_arg"},
                },
            },
            attributes_equivalence={"banana_1": "pear_1"},
            reversed_attributes_equivalence={"pear_1": "banana_1"},
        )
    )
    randomizer.tmp_translation[7] = ["public_first", "public_second"]
    status = ToolStatus(
        robots_status=[
            RobotToolStatus(
                robot_name="robot1",
                result=False,
                mess="Missing arguments: real_first_arg",
                error_flag=ToolErrorFlag.BAD_CALL,
            ),
            RobotToolStatus(
                robot_name="robot2",
                result=True,
                mess="Moved banana_1",
                error_flag=ToolErrorFlag.NONE,
            ),
        ],
        stage_id=0,
        attributes={},
        error_descriptions=[],
    )

    out = randomizer.traduce_end({7: status})

    assert out[7].robots_status[0].mess == "Missing arguments: public_first_arg"
    assert out[7].robots_status[1].mess == "Moved pear_1"
    assert 7 not in randomizer.tmp_translation


def test_spec_generator_rejects_duplicate_public_tool_names() -> None:
    generator = SpecGenerator(seed=0)
    original_tools = [
        {"name": "first", "description": "", "parameters": {}},
        {"name": "second", "description": "", "parameters": {}},
    ]
    config = {
        "first": {"name": ["duplicate"], "description": [""]},
        "second": {"name": ["duplicate"], "description": [""]},
    }

    with pytest.raises(ValueError, match="Duplicate randomized tool name"):
        generator._build_tools_spec(config, original_tools, indexed_variation=0)


def test_spec_generator_rejects_duplicate_public_parameter_names() -> None:
    generator = SpecGenerator(seed=0)
    original_tools = [
        {
            "name": "move",
            "description": "",
            "parameters": {
                "object": {"description": "", "type": "str"},
                "target": {"description": "", "type": "str"},
            },
        }
    ]
    config = {
        "move": {
            "name": ["move"],
            "description": [""],
            "parameters": {
                "object": {"name": ["value"], "description": [""]},
                "target": {"name": ["value"], "description": [""]},
            },
        }
    }

    with pytest.raises(ValueError, match="Duplicate randomized parameter name"):
        generator._build_tools_spec(config, original_tools, indexed_variation=0)


def test_spec_generator_rejects_ambiguous_attribute_aliases() -> None:
    generator = SpecGenerator(seed=0)

    with pytest.raises(ValueError, match="maps to both"):
        generator._build_attributes_spec(
            {"objects": [["fruit", "fruit"]]},
            {"objects": ["banana", "apple"]},
            indexed_variation=0,
        )


def test_spec_generator_rejects_json_unsafe_attribute_aliases() -> None:
    generator = SpecGenerator(seed=0)

    with pytest.raises(ValueError, match="unsafe for JSON translation"):
        generator._build_attributes_spec(
            {"objects": [['unsafe"value']]},
            {"objects": ["banana"]},
            indexed_variation=0,
        )


def test_randomizer_rejects_variation_index_at_upper_bound() -> None:
    randomizer = Randomizer(nb_variation=1, seed=0)

    with pytest.raises(RuntimeError, match="outside the variation range"):
        randomizer.set_variation_index(1)


def test_spec_generator_expands_semantic_attribute_forms() -> None:
    generator = SpecGenerator(seed=0)

    attributes, reversed_attributes, _, _ = generator._build_attributes_spec(
        {},
        {
            "food_types": ["banana", "apple"],
            "dishware_types": ["plate"],
        },
        indexed_variation=0,
        semantic_attributes_config={
            "food_types": {
                "variations": [["mango", "orange"], ["lemon", "apricot"]],
                "forms": ["{value}", "{value}s", "{value}_1", "{value}_2"],
            },
            "dishware_types": {
                "variations": [["bowl"], ["saucer"]],
                "forms": ["{value}", "{value}_1"],
            },
        },
    )

    assert attributes["banana"] == "mango"
    assert attributes["bananas"] == "mangos"
    assert attributes["banana_1"] == "mango_1"
    assert attributes["banana_2"] == "mango_2"
    assert attributes["apple_1"] == "orange_1"
    assert attributes["plate_1"] == "bowl_1"
    assert reversed_attributes["mango_1"] == "banana_1"


def test_spec_generator_random_semantic_selection_is_seeded() -> None:
    task_attributes = {"food_types": ["banana", "apple"]}
    semantic_config = {
        "food_types": {
            "variations": [["mango", "orange"], ["lemon", "apricot"]],
            "forms": ["{value}"],
        }
    }

    first = SpecGenerator(seed=7)._build_attributes_spec(
        {},
        task_attributes,
        indexed_variation=None,
        semantic_attributes_config=semantic_config,
    )[0]
    second = SpecGenerator(seed=7)._build_attributes_spec(
        {},
        task_attributes,
        indexed_variation=None,
        semantic_attributes_config=semantic_config,
    )[0]

    assert first == second
    assert (first["banana"], first["apple"]) in {
        ("mango", "orange"),
        ("lemon", "apricot"),
    }


def test_runtime_randomizer_applies_semantic_aliases_on_public_surfaces() -> None:
    spec = RandomizationSpec(
        tool_equivalence={
            "pick_item": {
                "name": "take",
                "parameters": {"item": "name"},
            }
        },
        attributes_equivalence={
            "banana": "mango",
            "bananas": "mangos",
            "banana_1": "mango_1",
            "sponge": "cloth",
        },
        reversed_attributes_equivalence={
            "mango": "banana",
            "mangos": "bananas",
            "mango_1": "banana_1",
            "cloth": "sponge",
        },
        tools=[
            {
                "name": "pick_item",
                "description": "Take banana_1 after using the sponge.",
                "parameters": {
                    "item": {
                        "description": "The banana to take.",
                        "type": "str",
                    }
                },
            }
        ],
    )
    randomizer = RuntimeRandomizer(spec)

    instruction = randomizer.get_randomized_instruction(
        UserInstruction("Move banana_1 with the other bananas.")
    )
    status = randomizer.get_randomized_instruction(
        StatusReturn({"infos": "Detected banana_1", "objects": {"banana_1": {}}})
    )
    attributes = randomizer.get_randomized_attributes(
        {
            "active_objects": ["banana_1", "banana_3"],
            "object_counts": {"banana": {"banana_1": 1}},
        }
    )
    mapped_call = randomizer.map_tool_call(
        "pick_item",
        {"item": {"primary": "mango_1", "fallback": ["banana_3"]}},
    )

    assert instruction.get_content() == "Move mango_1 with the other mangos."
    assert status.status == {
        "infos": "Detected mango_1",
        "objects": {"mango_1": {}},
    }
    assert attributes == {
        "active_objects": ["mango_1", "banana_3"],
        "object_counts": {"mango": {"mango_1": 1}},
    }
    assert mapped_call == (
        "take",
        {"name": {"primary": "banana_1", "fallback": ["banana_3"]}},
    )
    assert randomizer.get_tools()[0]["description"] == (
        "Take mango_1 after using the cloth."
    )
    assert randomizer.get_tools()[0]["parameters"]["item"]["description"] == (
        "The mango to take."
    )
    assert spec.tools[0]["description"] == "Take banana_1 after using the sponge."


def test_runtime_randomizer_rejects_canonical_names_without_placeholder() -> None:
    randomizer = RuntimeRandomizer(
        RandomizationSpec(
            tool_equivalence={
                "pick_item": {
                    "name": "take",
                    "parameters": {"item": "name"},
                }
            },
            attributes_equivalence={"glass_1": "tumbler_1"},
            reversed_attributes_equivalence={"tumbler_1": "glass_1"},
        )
    )

    direct_result = randomizer.map_tool_call(
        "pick_item",
        {"item": "glass_1"},
    )

    assert isinstance(direct_result, ToolExecution)
    assert direct_result.reason == "glass_1 was not detected."
    assert direct_result.reason_is_public is True

    status = ToolStatus(
        robots_status=[
            RobotToolStatus(
                robot_name="panda",
                result=False,
                mess=direct_result.reason,
                error_flag=ToolErrorFlag.BAD_CALL,
                mess_is_public=direct_result.reason_is_public,
            )
        ],
        stage_id=0,
        attributes={},
        error_descriptions=[""],
    )
    randomizer.tmp_translation[3] = "pick_item"

    translated_status = randomizer.traduce_end({3: status})

    assert translated_status[3].robots_status[0].mess == (
        "glass_1 was not detected."
    )

    physical_status = ToolStatus(
        robots_status=[
            RobotToolStatus(
                robot_name="panda",
                result=False,
                mess="glass_1 was not detected.",
                error_flag=ToolErrorFlag.BAD_CALL,
            )
        ],
        stage_id=0,
        attributes={},
        error_descriptions=[""],
    )
    randomizer.tmp_translation[4] = "pick_item"

    translated_physical_status = randomizer.traduce_end({4: physical_status})

    assert translated_physical_status[4].robots_status[0].mess == (
        "tumbler_1 was not detected."
    )

    with pytest.raises(ValueError, match="glass_1 was not detected"):
        randomizer.translate_skill_arguments_to_runtime(
            {"item": {"primary": "glass_1"}}
        )


@pytest.mark.parametrize(
    ("task_attributes", "semantic_config", "error_match"),
    [
        (
            {},
            {
                "food_types": {
                    "variations": [["mango"]],
                    "forms": ["{value}"],
                }
            },
            "must reference a list",
        ),
        (
            {"food_types": ["banana", "apple"]},
            {
                "food_types": {
                    "variations": [["mango"]],
                    "forms": ["{value}"],
                }
            },
            "has length 1, expected 2",
        ),
        (
            {"food_types": ["banana"]},
            {
                "food_types": {
                    "variations": [["mango"]],
                    "forms": ["{value}_{index}"],
                }
            },
            "Invalid form",
        ),
        (
            {"food_types": ["banana"], "objects": ["mango"]},
            {
                "food_types": {
                    "variations": [["mango"]],
                    "forms": ["{value}"],
                }
            },
            "collides with a canonical attribute value",
        ),
        (
            {"food_types": ["banana"]},
            {
                "food_types": {
                    "variations": [['unsafe"value']],
                    "forms": ["{value}"],
                }
            },
            "must be a non-empty ASCII identifier",
        ),
    ],
)
def test_spec_generator_rejects_invalid_semantic_attributes(
    task_attributes: dict,
    semantic_config: dict,
    error_match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error_match):
        SpecGenerator(seed=0)._build_attributes_spec(
            {},
            task_attributes,
            indexed_variation=0,
            semantic_attributes_config=semantic_config,
        )


def test_semantic_attributes_contribute_to_inferred_variation_count() -> None:
    config = {
        "tools": {},
        "semantic_attributes": {
            "food_types": {
                "variations": [
                    ["mango", "orange"],
                    ["lemon", "apricot"],
                    ["peach", "avocado"],
                ],
                "forms": ["{value}"],
            }
        },
    }

    assert SpecGenerator(seed=0)._infer_nb_variations(config) == 3
