import dataclasses

import jax
import numpy as np

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_extract_sf_identity_uses_openpi_sf_key_fields():
    batch = {
        "image": {"cam_high": np.zeros((2, 224, 224, 3), dtype=np.float32)},
        "episode_index": np.array([3, 4], dtype=np.int64),
        "frame_index": np.array([30, 40], dtype=np.int64),
    }

    identity = _data_loader.extract_sf_identity(batch, dataset_uid=7)

    assert identity["dataset_uid"].tolist() == [7, 7]
    assert identity["episode_index"].tolist() == [3, 4]
    assert identity["step_index"].tolist() == [30, 40]


def test_create_data_loader_returns_sf_identity_when_cache_enabled():
    config = _config.get_config("debug")
    config = dataclasses.replace(
        config,
        sf_cache_enable=True,
        sf_cache_mode="readonly",
        sf_cache_dir="/tmp/nonexistent-sf-cache",
        sf_dataset_uid=5,
    )

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=1)
    observation, actions, sf_identity = next(iter(loader))

    assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
    assert next(iter(observation.images.values())).shape[0] == config.batch_size
    assert sf_identity["dataset_uid"].tolist() == [5] * config.batch_size
    assert sf_identity["episode_index"].tolist() == [0] * config.batch_size
    assert sf_identity["step_index"].shape[0] == config.batch_size


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
