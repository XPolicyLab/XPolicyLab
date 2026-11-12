"""Optional CPU integration test with real OpenPI transforms and LeRobot v3 IO.

Run with the data dependencies in requirements-cpu-sft-data.txt. Only the
external SentencePiece asset is replaced; no checkpoint or download is needed.
"""

import numpy as np
import pytest
import torch

from XPolicyLab.policy.Pi_05.rlinf import sft


@pytest.fixture
def data_case(tmp_path, monkeypatch):
    monkeypatch.setenv("PI05_SFT_ACTION_DIM", "14")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.syspath_prepend(str(sft.POLICY_DIR / "openpi" / "src"))
    monkeypatch.syspath_prepend(
        str(sft.POLICY_DIR / "openpi" / "packages" / "openpi-client" / "src")
    )
    pytest.importorskip("lerobot")
    pytest.importorskip("flax")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from openpi.models import tokenizer
    from openpi.shared import normalize

    class LocalTokenizer:
        def __init__(self, max_len):
            self.max_len = max_len

        def tokenize(self, prompt, state=None):
            assert prompt == "stack the bowls"
            assert state is not None  # Pi05's discrete state tokens are enabled.
            return np.ones(self.max_len, dtype=np.int32), np.ones(
                self.max_len, dtype=bool
            )

    monkeypatch.setattr(tokenizer, "PaligemmaTokenizer", LocalTokenizer)
    root = tmp_path / "lerobot"
    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": None},
        "action": {"dtype": "float32", "shape": (14,), "names": None},
        **{
            f"observation.images.{cam}": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            }
            for cam in cameras
        },
    }
    dataset = LeRobotDataset.create(
        "test/stack_bowls", root=root, fps=10, features=features, use_videos=False
    )
    for index in range(3):
        dataset.add_frame(
            {
                "task": "stack the bowls",
                "observation.state": np.ones(14, dtype=np.float32),
                "action": np.full(14, 3 + index, dtype=np.float32),
                **{
                    f"observation.images.{cam}": np.broadcast_to(
                        np.array([255, 0, 128], dtype=np.uint8), (224, 224, 3)
                    ).copy()
                    for cam in cameras
                },
            }
        )
    dataset.save_episode()
    dataset.finalize()
    stats = normalize.NormStats(
        mean=np.zeros(14), std=np.ones(14), q01=np.full(14, -2.0), q99=np.full(14, 2.0)
    )
    normalize.save(tmp_path / "stats", {"state": stats, "actions": stats})
    cfg = sft.compose_config("RoboDojo", "stack_bowls", "arx_x5", "joint", 0, "0")
    cfg.data.train_data_paths = str(root)
    cfg.data.num_workers = 0
    cfg.actor.model.openpi_data.norm_stats_path = str(
        tmp_path / "stats" / "norm_stats.json"
    )
    return cfg


def test_lerobot_window_rgb_delta_quantile_and_padding(data_case):
    from openpi.training import data_loader
    from rlinf.data.datasets.openpi import get_official_openpi_sft_num_batches

    config = sft.build_train_config(data_case, 1)
    data_config = config.data.create(config.assets_dirs, config.model)
    raw = data_loader.create_torch_dataset(data_config, 50, config.model)
    np.testing.assert_allclose(raw[0]["action"][:, 0], [3, 4] + [5] * 48)
    transformed = data_loader.transform_dataset(raw, data_config)[0]
    # Joint delta (3-1)/2 = 1, while absolute grippers 3/2 = 1.5.
    np.testing.assert_allclose(
        transformed["actions"][0, [0, 6, 7, 13]], [1, 1.5, 1, 1.5], atol=2e-6
    )
    assert transformed["actions"].shape == (50, 32)
    assert np.count_nonzero(transformed["actions"][:, 14:]) == 0
    loader, _ = sft.build_dataloader(data_case, 1)
    observation, actions = next(iter(loader))
    assert actions.shape == (1, 50, 32)
    # JSON stats produce float64 NumPy arrays; native Pi0 casts actions to FP32.
    assert actions.is_floating_point() and torch.isfinite(actions).all()
    assert observation.tokenized_prompt.shape == (1, 200)
    assert observation.images["base_0_rgb"].shape == (1, 3, 224, 224)
    rgb = observation.images["base_0_rgb"][0, :, 0, 0]
    torch.testing.assert_close(rgb, torch.tensor([1.0, -1.0, 128 / 255 * 2 - 1]))
    from rlinf.models.embodiment.openpi.modules.model import (
        Observation,
        preprocess_observation,
    )

    native = preprocess_observation(
        Observation.from_observation_like(observation), train=False
    )
    torch.testing.assert_close(native.images["base_0_rgb"][0, 0, 0], rgb)
    assert get_official_openpi_sft_num_batches(loader) == 3


def test_data_model_horizon_mismatch_fails_before_training(data_case):
    data_case.actor.model.openpi.action_horizon = 10
    with pytest.raises(ValueError, match="action_horizon"):
        sft.build_train_config(data_case, 1)


def test_cpu_norm_stats_match_delta_semantics(data_case, tmp_path):
    from openpi.shared import normalize

    target = tmp_path / "computed" / "norm_stats.json"
    data_case.actor.model.openpi_data.norm_stats_path = str(target)
    assert sft.compute_norm_stats(data_case) == target
    stats = normalize.load(target.parent)
    np.testing.assert_allclose(stats["state"].mean, np.ones(14))
    # Grippers stay absolute; joints subtract the current state of 1.
    assert stats["actions"].mean[6] - stats["actions"].mean[0] == pytest.approx(1)
    with pytest.raises(FileExistsError):
        sft.compute_norm_stats(data_case)


def test_checkpoint_zero_with_multidigit_steps(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(sft.POLICY_DIR / "openpi" / "src"))
    monkeypatch.syspath_prepend(str(sft.POLICY_DIR / "openpi" / "packages" / "openpi-client" / "src"))
    from XPolicyLab.policy.Pi_05.model import _resolve_pi05_model_root

    config = {"model_path": str(tmp_path), "checkpoint_num": "0"}
    (tmp_path / "1000" / "assets").mkdir(parents=True)
    assert _resolve_pi05_model_root(config) == tmp_path / "1000"
    (tmp_path / "0" / "assets").mkdir(parents=True)
    assert _resolve_pi05_model_root(config) == tmp_path / "0"
