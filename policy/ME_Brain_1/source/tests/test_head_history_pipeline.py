"""Verify the full-frame input contract through data and inference boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from focus_vlwa.configs.model import FocusVLWAConfig
from focus_vlwa.data.dataset import FocusVLWAPostTrainingDataset, collate_focus_vlwa_batch
from focus_vlwa.data.normalization import NormStats
from focus_vlwa.inference.policy import FocusVLWAPolicy


def components():
    stats = NormStats(mean=np.zeros(14), std=np.ones(14), q01=-np.ones(14), q99=np.ones(14))
    return dict(
        config=FocusVLWAConfig(),
        norm_stats={"state": stats, "actions": stats},
        tokenizer=SimpleNamespace(tokenize=lambda *args: (np.zeros(4, np.int64), np.ones(4, bool))),
    )


def test_dataset_and_inference_history_contract():
    values = components()
    images = {key: np.full((224, 224, 3), 100, np.uint8) for key in ("cam_high", "cam_left_wrist", "cam_right_wrist")}
    history = np.zeros((20, 224, 224, 3), np.uint8)
    history[-1] = 100
    mask = np.zeros(20, np.float32)
    mask[-1] = 1
    sample = {
        "observation.state": np.zeros(14),
        "action": np.zeros((50, 14)),
        "task": "stack bowls",
        **{f"observation.images.{k}": v for k, v in images.items()},
        "s1": np.zeros((270, 32)),
        "s1_mask": np.ones(270),
        "event_action": np.zeros(14),
        "event_action_mask": np.ones(1),
    }
    observation = {"state": np.zeros(14), "images": images, "prompt": "stack bowls"}
    for row in (sample, observation):
        row.update(hist_images=history, hist_mask=mask)
    dataset = FocusVLWAPostTrainingDataset.__new__(FocusVLWAPostTrainingDataset)
    dataset.source = [sample]
    dataset.model_config = values["config"]
    dataset.norm_stats = values["norm_stats"]
    dataset.tokenizer = values["tokenizer"]
    batch, actions = collate_focus_vlwa_batch([dataset[0]])
    policy = FocusVLWAPolicy.__new__(FocusVLWAPolicy)
    policy.config = values["config"]
    policy.norm_stats = values["norm_stats"]
    policy.tokenizer = values["tokenizer"]
    policy.device = torch.device("cpu")
    inferred, _ = policy._prepare_observation(observation)
    assert actions.shape == (1, 50, 32)
    assert batch.history_images.shape == (1, 20, 224, 224, 3)
    torch.testing.assert_close(batch.history_images, inferred.history_images)
    torch.testing.assert_close(batch.history_mask, inferred.history_mask)
    assert batch.history_mask.sum() == 1
    with pytest.raises(ValueError):
        policy._prepare_observation({**observation, "hist_images": np.zeros((21, 3, 84, 84, 3), np.uint8)})
