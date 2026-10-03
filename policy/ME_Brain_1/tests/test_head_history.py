import numpy as np


def test_live_head_history_uses_each_control_step_and_resets():
    from XPolicyLab.policy.ME_Brain_1.hist_live import (
        LiveHeadHistoryTracker,
        hist_from_obs,
    )

    tracker = LiveHeadHistoryTracker()
    for step in range(54):
        obs = {
            "images": {
                camera: np.full((224, 224, 3), step, np.uint8)
                for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist")
            }
        }
        tracker.stamp_obs(obs, 8)
    assert obs["hist_mask"].sum() == 2
    np.testing.assert_array_equal(obs["hist_images"][-2:, 0, 0, 0], [0, 25])
    assert "hist_cells" not in hist_from_obs(obs)
    tracker.reset_all()
    tracker.stamp_obs(obs, 8)
    assert obs["hist_mask"].sum() == 0


def test_removed_history_shape_is_rejected():
    import pytest
    from XPolicyLab.policy.ME_Brain_1.hist_live import hist_from_obs

    with pytest.raises(ValueError, match="20 full head-history"):
        hist_from_obs({"hist_images": np.zeros((21, 3, 84, 84, 3), np.uint8)})


def test_client_uses_head_history_and_rejects_removed_profiles(monkeypatch):
    import pytest
    from XPolicyLab.policy.ME_Brain_1.deploy import _history_tracker
    from XPolicyLab.policy.ME_Brain_1.hist_live import LiveHeadHistoryTracker

    monkeypatch.delenv("FOCUS_VLWA_HISTORY_MODE", raising=False)
    assert isinstance(_history_tracker(), LiveHeadHistoryTracker)
    for mode in ("current_frame", "event_grid"):
        monkeypatch.setenv("FOCUS_VLWA_HISTORY_MODE", mode)
        with pytest.raises(ValueError):
            _history_tracker()


def test_adapter_requires_full_head_history():
    import pytest
    from XPolicyLab.policy.ME_Brain_1.model import Model

    model = Model.__new__(Model)
    model.action_type, model.robot_action_dim_info, model.task_name = (
        "joint",
        None,
        "stack bowls",
    )
    image = np.zeros((224, 224, 3), np.uint8)
    observation = {
        "state": np.zeros(14),
        "images": {
            key: image for key in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        },
    }
    with pytest.raises(ValueError, match="Missing client-stamped history"):
        model.update_obs(observation)
    observation.update(
        hist_images=np.zeros((20, 224, 224, 3), np.uint8), hist_mask=np.zeros(20)
    )
    model.update_obs(observation)
    assert model.observation_window["hist_images"].shape == (1, 20, 224, 224, 3)
    model.reset()
    assert model.observation_window is None
