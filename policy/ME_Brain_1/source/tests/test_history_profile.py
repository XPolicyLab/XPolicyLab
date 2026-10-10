import dataclasses
import json

import pytest

from focus_vlwa.configs.model import FocusVLWAConfig, load_model_config


def test_history_profile_roundtrip(tmp_path):
    config = FocusVLWAConfig(history_mode="head_history", max_token_len=400)
    (tmp_path / "model_config.json").write_text(json.dumps(dataclasses.asdict(config)))
    assert load_model_config(tmp_path) == config


def test_missing_model_configuration_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="model_config.json"):
        load_model_config(tmp_path)


def test_release_model_configuration(tmp_path):
    (tmp_path / "model_config.json").write_text(json.dumps({"world_model_loss_weight": 0.0}))
    config = load_model_config(tmp_path)
    assert config.history_mode == "head_history"
    assert config.world_model_loss_weight == 0.0
    assert config.action_horizon == 50


def test_invalid_profile_rejected():
    with pytest.raises(ValueError, match="history_mode"):
        FocusVLWAConfig(history_mode="silent_fallback")


@pytest.mark.parametrize("mode", ["event_grid", "current_frame"])
@pytest.mark.parametrize("saved", [False, True])
def test_removed_history_profile_is_rejected(tmp_path, saved, mode):
    if saved:
        (tmp_path / "model_config.json").write_text(json.dumps({"history_mode": mode}))
    with pytest.raises(ValueError, match="history_mode"):
        load_model_config(tmp_path) if saved else FocusVLWAConfig(history_mode=mode)
