import pytest

from focus_vlwa.configs.model import FocusVLWAConfig, get_expert_config


def test_released_experts_have_matching_depth() -> None:
    language = get_expert_config("gemma_2b")
    action = get_expert_config("gemma_300m")
    assert language.depth == action.depth == 18
    assert action.width == 1024


def test_world_model_horizon_is_validated() -> None:
    with pytest.raises(ValueError, match="world_model_horizon"):
        FocusVLWAConfig(world_model_horizon=269)
