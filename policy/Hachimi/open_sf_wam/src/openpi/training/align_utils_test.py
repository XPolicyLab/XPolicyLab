import types

import numpy as np
import torch

from openpi.models import model as _model
from openpi.training import align_utils


def _write_bf16_cache(cache_dir, *, dataset_uid, episode_index, step_index, tensor, chunk_size):
    chunk_idx = step_index // chunk_size
    slot_idx = step_index % chunk_size
    base = (
        cache_dir
        / f"ds_{dataset_uid}"
        / f"ep_{episode_index}"
        / f"bf16_chunk_{chunk_idx:08d}"
    )
    base.parent.mkdir(parents=True, exist_ok=True)
    mask_path = base.with_suffix(".mask")
    mask = bytearray(mask_path.read_bytes()) if mask_path.exists() else bytearray(chunk_size)
    mask[slot_idx] = 1
    mask_path.write_bytes(bytes(mask))

    frame = torch.as_tensor(tensor).to(dtype=torch.bfloat16).contiguous()
    slot_bytes = frame.numel() * 2
    data_path = base.with_suffix(".bf16bin")
    payload = bytearray(data_path.read_bytes()) if data_path.exists() else bytearray(chunk_size * slot_bytes)
    payload[slot_idx * slot_bytes : (slot_idx + 1) * slot_bytes] = frame.view(torch.int16).numpy().tobytes()
    data_path.write_bytes(bytes(payload))


def test_offline_align_feature_extractor_reads_openpi_sf_chunked_cache(tmp_path):
    config = types.SimpleNamespace(
        sf_cache_dir=str(tmp_path),
        sf_cache_save_dtype="bf16",
        sf_cache_chunk_size=4,
        sf_cache_strict_shape=True,
        sf_cache_miss_policy="error",
        vggt_dim=3,
        ignore_img_padding_area=False,
    )
    first = np.arange(24, dtype=np.float32).reshape(4, 6)
    second = np.arange(24, 48, dtype=np.float32).reshape(4, 6)
    _write_bf16_cache(tmp_path, dataset_uid=7, episode_index=2, step_index=9, tensor=first, chunk_size=4)
    _write_bf16_cache(tmp_path, dataset_uid=7, episode_index=2, step_index=10, tensor=second, chunk_size=4)
    observation = _model.Observation(
        images={"cam_high": np.zeros((2, 224, 224, 3), dtype=np.float32)},
        image_masks={"cam_high": np.array([True, True])},
        state=np.zeros((2, 32), dtype=np.float32),
        image_padding_mask={"cam_high": np.ones((2, 224, 224), dtype=bool)},
    )
    sf_identity = {
        "dataset_uid": np.array([7, 7], dtype=np.int64),
        "episode_index": np.array([2, 2], dtype=np.int64),
        "step_index": np.array([9, 10], dtype=np.int64),
    }

    features = align_utils.OfflineAlignFeatureExtractor(config)(observation, reference_tokens=4, sf_identity=sf_identity)

    np.testing.assert_allclose(features.targets, np.stack([first, second]), rtol=0.01, atol=0.02)
    assert features.targets.dtype == np.float32
    assert features.mask.tolist() == [[True, True, True, True], [True, True, True, True]]


def test_offline_align_feature_extractor_errors_on_missing_cache(tmp_path):
    config = types.SimpleNamespace(
        sf_cache_dir=str(tmp_path),
        sf_cache_save_dtype="bf16",
        sf_cache_chunk_size=4,
        sf_cache_strict_shape=True,
        sf_cache_miss_policy="error",
        vggt_dim=3,
        ignore_img_padding_area=False,
    )
    observation = _model.Observation(
        images={"cam_high": np.zeros((1, 224, 224, 3), dtype=np.float32)},
        image_masks={"cam_high": np.array([True])},
        state=np.zeros((1, 32), dtype=np.float32),
        image_padding_mask={"cam_high": np.ones((1, 224, 224), dtype=bool)},
    )
    sf_identity = {
        "dataset_uid": np.array([7], dtype=np.int64),
        "episode_index": np.array([2], dtype=np.int64),
        "step_index": np.array([9], dtype=np.int64),
    }

    with np.testing.assert_raises_regex(ValueError, "SF cache miss.*dataset_uid=7.*episode_index=2.*step_index=9"):
        align_utils.OfflineAlignFeatureExtractor(config)(observation, reference_tokens=4, sf_identity=sf_identity)
