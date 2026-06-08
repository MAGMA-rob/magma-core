# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.data_structures import ToolErrorFlag, ToolExecution
from magma_core.base.randomizer.random_spec import RandomizationSpec
from magma_core.base.randomizer.runtime_randomizer import RuntimeRandomizer


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
