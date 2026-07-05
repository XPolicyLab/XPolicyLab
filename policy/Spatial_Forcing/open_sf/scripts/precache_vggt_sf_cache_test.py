from types import SimpleNamespace

import numpy as np
import torch

from openpi.models import model as _model
from openpi.models_pytorch import sf_offline_cache
from scripts import precache_vggt_sf_cache


class _FakeExtractor:
    def __call__(self, observation, *, reference_tokens, sf_identity=None):
        del observation, sf_identity
        values = torch.arange(2 * reference_tokens * 6, dtype=torch.float32)
        return SimpleNamespace(
            targets=values.reshape(2, reference_tokens, 6).numpy(),
            mask=np.ones((2, reference_tokens), dtype=bool),
        )


def _observation(batch_size=2):
    return _model.Observation(
        images={"cam_high": np.zeros((batch_size, 224, 224, 3), dtype=np.float32)},
        image_masks={"cam_high": np.ones((batch_size,), dtype=bool)},
        state=np.zeros((batch_size, 32), dtype=np.float32),
        image_padding_mask={"cam_high": np.ones((batch_size, 224, 224), dtype=bool)},
    )


def _config(cache_dir):
    return SimpleNamespace(
        sf_cache_dir=str(cache_dir),
        sf_cache_save_dtype="bf16",
        sf_cache_chunk_size=4,
        sf_cache_strict_shape=True,
        sf_cache_overwrite=False,
        sf_dataset_uid=0,
        vggt_dim=3,
    )


def test_precache_batch_writes_missing_vggt_features(tmp_path):
    sf_identity = {
        "dataset_uid": torch.tensor([7, 7]),
        "episode_index": torch.tensor([2, 2]),
        "step_index": torch.tensor([9, 10]),
    }

    stats = precache_vggt_sf_cache.precache_batch(
        extractor=_FakeExtractor(),
        observation=_observation(),
        sf_identity=sf_identity,
        config=_config(tmp_path),
        reference_tokens=4,
        overwrite=False,
    )

    assert stats == {"existing": 0, "written": 2, "failed": 0}
    key = sf_offline_cache.make_cache_key(7, 2, 10)
    loaded = sf_offline_cache.load_cached_tensor(
        tmp_path,
        key,
        cache_dtype="bf16",
        chunk_size=4,
        expected_shape=(4, 6),
        strict_shape=True,
    )
    assert loaded is not None
    assert loaded.dtype == torch.bfloat16


def test_precache_batch_skips_existing_cache(tmp_path):
    config = _config(tmp_path)
    key = sf_offline_cache.make_cache_key(7, 2, 9)
    sf_offline_cache.save_cached_tensor(
        tmp_path,
        key,
        torch.ones((4, 6), dtype=torch.float32),
        cache_dtype="bf16",
        chunk_size=4,
    )
    sf_identity = {
        "dataset_uid": torch.tensor([7]),
        "episode_index": torch.tensor([2]),
        "step_index": torch.tensor([9]),
    }

    stats = precache_vggt_sf_cache.precache_batch(
        extractor=_FakeExtractor(),
        observation=_observation(batch_size=1),
        sf_identity=sf_identity,
        config=config,
        reference_tokens=4,
        overwrite=False,
    )

    assert stats == {"existing": 1, "written": 0, "failed": 0}


def test_prepare_precache_config_forces_readwrite_mode(tmp_path, monkeypatch):
    base = SimpleNamespace(
        sf_cache_enable=False,
        sf_cache_mode="readonly",
        sf_cache_miss_policy="error",
        sf_cache_dir=None,
        sf_cache_save_dtype="bf16",
        sf_cache_chunk_size=128,
        sf_cache_strict_shape=True,
        sf_cache_overwrite=False,
    )
    monkeypatch.setenv("SF_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SF_CACHE_MODE", "readwrite")

    config = precache_vggt_sf_cache.prepare_precache_config(base)

    assert config.sf_cache_enable is True
    assert config.sf_cache_mode == "readwrite"
    assert config.sf_cache_miss_policy == "online_compute"
    assert config.sf_cache_dir == str(tmp_path)
