from __future__ import annotations

import h5py
import numpy as np

from evals.common.backends import (
    HDF5EvaluationAdapter,
    HDF5EvaluationEnvironment,
)
from evals.common.suite import EvaluationTask


def _write_episode(path) -> None:
    frames = np.zeros((3, 12, 16, 3), dtype=np.uint8)
    frames[1:] = 127
    with h5py.File(path, "w") as stream:
        stream.create_dataset("observation.camera.head", data=frames)
        stream.create_dataset("observation.camera.left", data=frames + 1)
        stream.create_dataset("observation.camera.right", data=frames + 2)
        stream.create_dataset(
            "observation.eef6d",
            data=np.arange(60, dtype=np.float32).reshape(3, 20),
        )
        stream.create_dataset("is_exec", data=np.ones(3, dtype=np.bool_))


def test_generated_hdf5_episode_replays_through_common_backend(tmp_path):
    episode = tmp_path / "episode.hdf5"
    _write_episode(episode)
    environment = HDF5EvaluationEnvironment(
        episode,
        use_history=True,
        num_history_frames=2,
    )
    adapter = HDF5EvaluationAdapter(embodiment_id=4)

    native = environment.reset(EvaluationTask("replay", "replay episode"), seed=7)
    observation = adapter.observation(native)
    next_step = environment.step(np.zeros((2, 10), dtype=np.float32))
    environment.close()

    assert observation.images.shape[:3] == (1, 2, 3)
    assert observation.proprioception.shape == (1, 1, 20)
    assert observation.embodiment_id.item() == 4
    assert next_step[0][4] == 1
