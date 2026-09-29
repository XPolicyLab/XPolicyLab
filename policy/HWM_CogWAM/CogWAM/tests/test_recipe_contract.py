"""The shipped config must be the frozen recipe, and the contract must bite.

`cogwam/recipe.py` and `configs/*.yaml` were derived independently from the same
source run. If they ever drift apart, one of them is wrong -- these tests are
what notices.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from cogwam.recipe import EXPECTED, RECIPE_PROFILE, config_fingerprint, validate_recipe
from cogwam.training.config import apply_config_compat

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "cogwam_robodojo_h25_eventmem_dino_multilayer_50k.yaml"

# The config resolves storage locations from the environment. The values do not
# matter here -- the contract validates backbone *identity* separately, and
# deliberately does not pin paths.
_ENV = {
    "COGWAM_RUN_ROOT": "/tmp/cogwam-run",
    "COGWAM_BASE_VLM": "/tmp/cogwam-vlm",
    "COGWAM_DINO_MODEL": "/tmp/cogwam-dino",
    "COGWAM_DATA_ROOT": "/tmp/cogwam-data",
}


@pytest.fixture
def cfg(monkeypatch: pytest.MonkeyPatch):
    for key, value in _ENV.items():
        monkeypatch.setenv(key, value)
    return apply_config_compat(OmegaConf.load(CONFIG_PATH))


def test_shipped_config_satisfies_the_contract(cfg) -> None:
    record = validate_recipe(cfg)
    assert record["validated"] is True
    assert record["profile"] == RECIPE_PROFILE
    assert record["global_batch_size"] == 768
    assert record["max_train_steps"] == 50000


def test_fingerprint_is_stable(cfg) -> None:
    """The fingerprint is written next to every checkpoint; it must not drift."""
    assert config_fingerprint(cfg) == "f58cc99c47e5cd3480b637d35ef4bdf0008887471afe0936be1e6a85fdf506e2"


def test_batch_topology_is_enforced(cfg) -> None:
    """A different global batch is a different recipe, not a variant of it."""
    cfg.trainer.expected_global_batch_size = 512
    with pytest.raises(ValueError, match="expected_global_batch_size"):
        validate_recipe(cfg)


def test_event_memory_cadence_is_enforced(cfg) -> None:
    """Semantic offset must stay the negative of the replan interval."""
    cfg.datasets.vla_data.text_annotations.event_memory.semantic_offset = -12
    with pytest.raises(ValueError, match="semantic_offset"):
        validate_recipe(cfg)


def test_dino_layer_selection_is_pinned(cfg) -> None:
    """The multi-layer fusion is the one knob this recipe exists to isolate."""
    cfg.framework.dino.dino_layers = [5, 7, 9, 11]
    with pytest.raises(ValueError, match="dino_layers"):
        validate_recipe(cfg)


def test_action_horizon_matches_query_bank(cfg) -> None:
    """num_action_queries and action_horizon are bound by the query bank's RoPE table."""
    assert EXPECTED["framework.planner.num_action_queries"] == EXPECTED["framework.action_model.action_horizon"]
    assert EXPECTED["datasets.vla_data.action_horizon"] == EXPECTED["framework.action_model.action_horizon"]


def test_skip_flag_marks_the_run_as_not_a_reproduction(cfg, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COGWAM_SKIP_RECIPE_VALIDATION", "1")
    cfg.trainer.max_train_steps = 10
    with pytest.warns(UserWarning, match="NOT a reproduction run"):
        record = validate_recipe(cfg)
    assert record["validated"] is False
    assert "trainer.max_train_steps" in record["mismatches"]


def test_contract_pins_no_absolute_paths() -> None:
    """Upstream froze storage paths into the contract, which made it untravelable."""
    for path, value in EXPECTED.items():
        assert not (isinstance(value, str) and value.startswith("/")), f"{path} pins an absolute path: {value!r}"
    assert os.sep == "/" or True  # documents the assumption above
