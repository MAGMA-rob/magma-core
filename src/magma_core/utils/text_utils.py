# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import re, json, copy
from typing import Any, Optional, List, Dict
import fnmatch

def join_with_and(values: List[str]) -> str:
    if len(values) == 0:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"

def is_or_are(values: List[str]) -> str:
    return "is" if len(values) == 1 else "are"

def stringify_history_content(content: Any, drop_previous_tool_call: bool = False) -> str:
    cleaned = content
    if drop_previous_tool_call:
        try:
            if isinstance(cleaned, str):
                cleaned = json.loads(cleaned)
            elif isinstance(cleaned, dict):
                cleaned = copy.deepcopy(cleaned)
            if isinstance(cleaned, dict):
                cleaned.pop("previous_tool_call", None)
        except Exception:
            cleaned = content

    if isinstance(cleaned, (dict, list)):
        return json.dumps(cleaned, ensure_ascii=True)
    if cleaned is None:
        return ""
    return str(cleaned)


def format_history_message(author: str, content: Any, timestamp: float) -> Dict[str, Any]:
    return {
        "author": author,
        "content": stringify_history_content(
            content,
            drop_previous_tool_call=author == "SYSTEM",
        ),
        "timestamp": timestamp,
    }


def format_model_history_content(say: Any, action: Any = None) -> str:
    say_text = stringify_history_content(say)
    if action in (None, {}, [], ""):
        return say_text

    action_text = stringify_history_content(action)
    if not say_text:
        return action_text
    return f"{say_text}\n{action_text}"


def format_model_history_message(say: Any, action: Any, timestamp: float) -> Dict[str, Any]:
    return {
        "author": "MODEL",
        "content": format_model_history_content(say, action),
        "timestamp": timestamp,
    }

def extract_json_from_answer(text: str, start_marker : Optional[str] = None) -> dict:
    """
    Extracts and parses the JSON object appearing after start_marker
    """
    tail = text
    if start_marker is not None:
        try:
            tail = text.split(start_marker, 1)[1]
        except IndexError:
            pass
    
    start = tail.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found after {start_marker}")
    
    depth = 0
    end = None

    for i, ch in enumerate(tail[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        raise ValueError("Unbalanced braces in JSON object")

    json_str = tail[start:end]

    return json.loads(json_str)

def transform_memorizer_output_to_json(data : str) -> Dict:
    r_id = []
    add_statement = []
    if not "NOTHING" in data:
        for line in data.splitlines():
            if not line.strip():
                continue  # skip empty lines

            cmd, args = line.split(maxsplit=1)

            if "REMOVE" in cmd:
                r_id.extend(args.split(','))

            elif "ADD" in cmd:
                add_statement.append(args)

    return {
        "add" : add_statement,
        "remove" : r_id
    }

def transform_json_to_memorizer_output(data : Dict) -> str:
    s=""
    for add in data['add']:
        s+="ADD " + add + "\n"
    for remove in data['remove']:
        s+="REMOVE " + remove + "\n"
    return s

def save_list_of_data_to_file(path_to_file : str, datas : List):
    N = len(datas)
    with open(path_to_file, "w+") as f:
        f.write("[\n")
        for i, item in enumerate(datas):
            f.write(json.dumps(item))
            if i != N-1:
                f.write(",\n")
        f.write(']')

def build_fake_execution_fail(action : Dict, original_status_dict : Dict, error_mess : Optional[str]) -> Dict:
    if not error_mess:
        error_mess = "The motion planner have encounter a temporary error."
    info = original_status_dict.get('infos',None)
    out = {}
    name = action.get("name", None)
    if info is None:
        if name is None:
            #Multi-Robot call
            for robot in original_status_dict.keys():
                if robot == "previous_tool_call": continue
                name = action[robot].get('name','unknwon tool name')
                out[robot] = f"{name} fails : {error_mess}"
        else:
            print("WARNING MULTI-TOOL call but they got a name : ", original_status_dict, " and action ", action)
            return original_status_dict
    else:
        if name is None:
            key, value = next(iter(action.items()))
            name = f"Robot {key} - tool {value.get('name','unknwon tool name')}"
        out["error"] = f"{name} fails : {error_mess}"
    
    out["previous_tool_call"] = original_status_dict["previous_tool_call"]
    return out

def star_extractor(env_actors : List, value : str) -> List[str]:
    return [name for name in env_actors if fnmatch.fnmatch(name, value)]

def auto_cast(value: str):
    # Bool
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"

    # Int
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # List (comma-separated)
    if "," in value:
        return [auto_cast(v) for v in value.split(",")]

    # Fallback
    return value
