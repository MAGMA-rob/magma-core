# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

"""
Coverage map for this file:

- JSON extraction and memorizer text round-trips.
- Disk serialization helper.
- User-facing formatting for executor returns and fake failures.
- Small parsing helpers such as wildcard extraction and auto-casting.
"""

import json

import pytest

from magma_core.utils.text_utils import (
    auto_cast,
    build_fake_execution_fail,
    build_model_return_from_executor,
    extract_json_from_answer,
    save_list_of_data_to_file,
    star_extractor,
    transform_json_to_memorizer_output,
    transform_memorizer_output_to_json,
)


def test_extract_json_from_answer_with_and_without_marker():
    text = "noise START {\"ok\": true, \"value\": 3} trailing"
    assert extract_json_from_answer(text, start_marker="START") == {"ok": True, "value": 3}
    assert extract_json_from_answer("{\"a\": 1}") == {"a": 1}
    with pytest.raises(ValueError, match="No JSON object found"):
        extract_json_from_answer("no braces here")


def test_memorizer_transform_round_trip():
    original = {"add": ["A", "B"], "remove": ["1", "2"]}
    text = transform_json_to_memorizer_output(original)
    assert "ADD A" in text
    assert "REMOVE 1" in text
    back = transform_memorizer_output_to_json(text)
    assert back == original


def test_memorizer_transform_nothing():
    assert transform_memorizer_output_to_json("NOTHING") == {"add": [], "remove": []}


def test_save_list_of_data_to_file(tmp_path):
    target = tmp_path / "data.json"
    save_list_of_data_to_file(str(target), [{"a": 1}, {"b": 2}])
    assert json.loads(target.read_text()) == [{"a": 1}, {"b": 2}]


def test_build_model_return_single_tool():
    action = {"name": "pick", "arguments": {"item": "box"}}
    ok = build_model_return_from_executor(action, [True], "done")
    err = build_model_return_from_executor(action, [False], "failed")
    assert ok["infos"] == "pick succeed : done"
    assert err["error"] == "pick fails : failed"
    assert ok["previous_tool_call"] == action


def test_build_fake_execution_fail_single_tool():
    single_action = {"name": "pick", "arguments": {}}
    single_status = {"infos": "pick succeed : ok", "previous_tool_call": single_action}
    out_single = build_fake_execution_fail(single_action, single_status, "planner down")
    assert out_single["error"] == "pick fails : planner down"
    assert out_single["previous_tool_call"] == single_action


def test_star_extractor_and_auto_cast():
    names = ["obj_a", "obj_b", "robot_1"]
    assert star_extractor(names, "obj_*") == ["obj_a", "obj_b"]
    assert auto_cast("true") is True
    assert auto_cast("10") == 10
    assert auto_cast("3.14") == 3.14
    assert auto_cast("1,2,false") == [1, 2, False]
