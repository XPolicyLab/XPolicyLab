"""Verify dataset rejection and the portable training launch contract."""

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
from XPolicyLab.policy.ME_Brain_1 import process_data

POLICY_DIR = Path(__file__).resolve().parents[1]


def make_dataset(path, version="v3.0", supervised=True):
    features = {
        "observation.state": {"shape": [14]},
        "action": {"shape": [14]},
        "episode_index": {"shape": [1]},
        "timestamp": {"shape": [1]},
        "task": {"shape": [1]},
    }
    for camera in process_data.CAMERAS:
        key = f"observation.images.{camera}"
        features[key] = {"shape": [480, 640, 3], "dtype": "video"}
        video = path / "videos" / key / "episode_000000.mp4"
        video.parent.mkdir(parents=True)
        video.touch()
    if supervised:
        features.update({
            "s1": {"shape": [270, 32]},
            "s1_mask": {"shape": [270]},
            "event_action": {"shape": [14]},
            "event_action_mask": {"shape": [1]},
        })
    info = {"codebase_version": version, "fps": 25, "total_frames": 100,
            "total_episodes": 1, "features": features}
    (path / "meta").mkdir()
    (path / "meta/info.json").write_text(json.dumps(info))
    (path / "data").mkdir()
    (path / "data/episode_000000.parquet").touch()
    return path


def test_plain_official_export_is_rejected_before_output_creation(tmp_path):
    dataset = make_dataset(tmp_path / "official", supervised=False)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="missing aligned world-model supervision"):
        process_data.prepare_datasets([dataset], output)
    assert not output.exists()


def test_prepared_shards_are_linked_without_changing_targets(tmp_path):
    first = make_dataset(tmp_path / "first")
    second = make_dataset(tmp_path / "second")
    original_info = (first / "meta/info.json").read_bytes()
    output = tmp_path / "prepared"
    manifest = process_data.prepare_datasets([first, second], output)
    assert (output / "shards/0000").resolve() == first
    assert (output / "shards/0001").resolve() == second
    assert (first / "meta/info.json").read_bytes() == original_info
    assert process_data.resolve_datasets([manifest["dataset_pattern"]]) == [first, second]
    with pytest.raises(FileExistsError):
        process_data.prepare_datasets([second], output)
    assert (output / "shards/0000").resolve() == first


def test_incompatible_readers_are_rejected_without_output(tmp_path):
    datasets = [make_dataset(tmp_path / "v2", "v2.1"), make_dataset(tmp_path / "v3")]
    with pytest.raises(ValueError, match="Do not mix"):
        process_data.prepare_datasets(datasets, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("change, message", [
    ({"fps": 50}, "25 FPS"),
    ({"codebase_version": "v2.0"}, "v2.1 or v3.0"),
])
def test_invalid_metadata_is_rejected(tmp_path, change, message):
    dataset = make_dataset(tmp_path / "dataset")
    info_file = dataset / "meta/info.json"
    info = json.loads(info_file.read_text())
    info.update(change)
    info_file.write_text(json.dumps(info))
    with pytest.raises(ValueError, match=message):
        process_data.validate_dataset(dataset)


@pytest.mark.parametrize("devices, stage", [("0", "joint"), ("0,2", "frozen")])
def test_train_launch_preserves_seed_and_stage_without_starting_jobs(tmp_path, devices, stage):
    dataset = make_dataset(tmp_path / "dataset")
    checkpoint = tmp_path / "init"
    asset = checkpoint / "assets/arx_x5_sim"
    asset.mkdir(parents=True)
    (asset / "norm_stats.json").write_text("{}")
    (checkpoint / "model_config.json").write_text("{}")
    (checkpoint / "model.safetensors").touch()
    env = {**os.environ, "FOCUS_VLWA_INIT_CHECKPOINT": str(checkpoint),
           "FOCUS_VLWA_DATASET": str(dataset), "FOCUS_VLWA_PYTHON": sys.executable,
           "FOCUS_VLWA_DRY_RUN": "1", "FOCUS_VLWA_STAGE": stage}
    env.pop("FOCUS_VLWA_LEROBOT_REPOS", None)
    result = subprocess.run(
        ["bash", str(POLICY_DIR / "train.sh"), "RoboDojo", "test-portable", "arx_x5", "joint", "17", devices],
        cwd=tmp_path, env=env, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--seed 17" in result.stdout
    assert f"--stage {stage}" in result.stdout
    assert "RoboDojo-test-portable-arx_x5-joint-17" in result.stdout
    assert ("--nproc-per-node 2" in result.stdout) == (devices == "0,2")
    assert not (POLICY_DIR / "checkpoints/RoboDojo-test-portable-arx_x5-joint-17").exists()


def test_training_parser_applies_frozen_defaults_and_seed(monkeypatch, tmp_path):
    from focus_vlwa.scripts.post_train import build_parser, run_from_args

    captured = []
    trainer = types.ModuleType("focus_vlwa.post_training.trainer")
    trainer.run_post_training = captured.append
    monkeypatch.setitem(sys.modules, trainer.__name__, trainer)
    args = build_parser().parse_args([
        "--dataset", "prepared", "--init-checkpoint", str(tmp_path),
        "--norm-stats-dir", str(tmp_path), "--output-dir", str(tmp_path),
        "--stage", "frozen", "--seed", "17",
    ])
    run_from_args(args)
    assert captured[0].seed == 17
    assert captured[0].freeze_world_model_expert
    assert captured[0].world_model_loss_weight == 0.0
    assert captured[0].num_steps == 10000
