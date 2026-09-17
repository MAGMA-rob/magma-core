# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import pytest
import yaml

from magma_core.configs import (
    BackendConfig,
    CoachingConfig,
    CoachingExamplesConfig,
    MAGMAConfig,
)


def _make_config(nb_branch: int) -> MAGMAConfig:
    return MAGMAConfig(
        backends={
            "default": BackendConfig(
                type="ollama",
                endpoint="http://localhost:11434",
                default_model="test-model",
                headers={},
            )
        },
        coaching=CoachingConfig(enabled=True, provider="llm"),
        generate={
            "mode": "single",
            "nb_branch": nb_branch,
            "nb_env": 2,
            "nb_max_update": 2,
        },
        benchmark={},
        magma_agent_address="http://localhost:8888",
        magma_planner_address="http://localhost:8000",
    )


def test_verify_allows_single_branch_generation() -> None:
    _make_config(nb_branch=1).verify()


def test_verify_rejects_zero_branch_generation() -> None:
    with pytest.raises(ValueError, match="nb_branch.*>= 1"):
        _make_config(nb_branch=0).verify()


def test_verify_accepts_human_coaching_without_backend_when_allowed() -> None:
    config = _make_config(nb_branch=1)
    config.backends = {}
    config.coaching = CoachingConfig(
        enabled=True,
        provider="human",
        endpoint="http://localhost:8890",
    )

    config.verify(accept_no_backend=True)


def test_verify_rejects_invalid_human_coaching_endpoint() -> None:
    config = _make_config(nb_branch=1)
    config.coaching = CoachingConfig(enabled=True, provider="human")

    with pytest.raises(ValueError, match="valid HTTP"):
        config.verify()


def test_verify_rejects_unknown_coaching_provider() -> None:
    config = _make_config(nb_branch=1)
    config.coaching = CoachingConfig(enabled=True, provider="unknown")

    with pytest.raises(ValueError, match="provider"):
        config.verify()


def test_verify_rejects_non_positive_agent_timeout() -> None:
    config = _make_config(nb_branch=1)
    config.magma_agent_timeout = 0

    with pytest.raises(ValueError, match="magma_agent_timeout"):
        config.verify()


def test_verify_allows_disabled_human_coaching_without_endpoint() -> None:
    config = _make_config(nb_branch=1)
    config.coaching = CoachingConfig(enabled=False, provider="human")

    config.verify()


def test_loads_coaching_example_configuration(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "coaching": {
                    "enabled": True,
                    "provider": "llm",
                    "examples": {
                        "register_successful": True,
                        "use_for_automatic": True,
                        "cache_dir": "custom/examples",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    config = MAGMAConfig.load(config_path)

    assert config.coaching.examples == CoachingExamplesConfig(
        register_successful=True,
        use_for_automatic=True,
        cache_dir="custom/examples",
    )


def test_verify_rejects_empty_coaching_example_directory() -> None:
    config = _make_config(nb_branch=1)
    config.coaching = CoachingConfig(
        enabled=True,
        provider="llm",
        examples=CoachingExamplesConfig(cache_dir=""),
    )

    with pytest.raises(ValueError, match="cache_dir"):
        config.verify()


def test_override_updates_only_selected_coaching_example_fields() -> None:
    config = _make_config(nb_branch=1)

    config.override_with_dict(
        {
            "coaching": {
                "examples": {
                    "register_successful": True,
                }
            }
        }
    )

    assert config.coaching.examples.register_successful is True
    assert config.coaching.examples.use_for_automatic is False
    assert config.coaching.examples.cache_dir == "cache"


def test_load_rejects_legacy_generate_coaching(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "generate": {
                    "mode": "single",
                    "nb_branch": 1,
                    "nb_env": 2,
                    "nb_max_update": 2,
                    "coaching": True,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="generate.coaching"):
        MAGMAConfig.load(config_path)
