# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _ensure_sapien_stub() -> None:
    try:
        import sapien  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        sapien = types.ModuleType("sapien")

        class Pose:
            def __init__(self, *args, **kwargs):
                pass

        sapien.Pose = Pose
        sys.modules["sapien"] = sapien


def _ensure_mani_skill_stub() -> None:
    try:
        import mani_skill.agents.base_agent  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        mani_skill = types.ModuleType("mani_skill")
        agents = types.ModuleType("mani_skill.agents")
        base_agent = types.ModuleType("mani_skill.agents.base_agent")

        class BaseAgent:
            pass

        base_agent.BaseAgent = BaseAgent
        agents.base_agent = base_agent
        mani_skill.agents = agents

        sys.modules["mani_skill"] = mani_skill
        sys.modules["mani_skill.agents"] = agents
        sys.modules["mani_skill.agents.base_agent"] = base_agent


def _ensure_torch_stub() -> None:
    try:
        import torch  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        torch = types.ModuleType("torch")
        torch_tensor = types.ModuleType("torch._tensor")

        class Tensor:
            def __init__(self, *args, **kwargs):
                self.shape = (0,)
                self.device = "cpu"

            def item(self):
                return 0

        def _tensor(*args, **kwargs):
            return Tensor()

        torch.Tensor = Tensor
        torch_tensor.Tensor = Tensor
        torch.__codex_stub__ = True
        torch_tensor.__codex_stub__ = True
        torch.tensor = _tensor
        torch.ones = _tensor
        torch.minimum = lambda a, b: a
        torch.zeros_like = _tensor
        torch.int32 = "int32"

        sys.modules["torch"] = torch
        sys.modules["torch._tensor"] = torch_tensor


_ensure_sapien_stub()
_ensure_mani_skill_stub()
_ensure_torch_stub()
