import numpy as np
import torch

from evals.common.hdf5_replay import (
    HDF5ReplayConfig,
    _replay_prompts,
    _to_observation_batch,
)


def test_hdf5_replay_builds_explicit_public_schema():
    video = np.full((4, 8, 6, 3), 255, dtype=np.uint8)
    observation = _to_observation_batch(
        high=video,
        left=video,
        right=video,
        state={
            "state.left_joint": np.zeros((1, 7), dtype=np.float32),
            "state.right_joint": np.ones((1, 7), dtype=np.float32),
        },
        embodiment_id=7,
    )

    assert observation.images.shape == (1, 4, 3, 3, 8, 6)
    assert observation.head_view.shape == (1, 1, 3, 8, 6)
    assert observation.proprioception.shape == (1, 1, 14)
    assert torch.all(observation.images == 1)
    assert observation.embodiment_id.item() == 7


def test_hdf5_replay_creates_mode_specific_negative_cfg_prompts():
    auto = _replay_prompts(
        HDF5ReplayConfig("episode.hdf5", "auto", "pick", 1),
        batch_size=1,
    )
    interactive = _replay_prompts(
        HDF5ReplayConfig("episode.hdf5", "interactive", "pick", 1),
        batch_size=1,
    )

    assert auto.negative_vlm_text == [""]
    assert interactive.negative_language_instruction == [""]
