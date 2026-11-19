"""What the adapter reads from a checkpoint directory."""

import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("sana_robot")

from XPolicyLab.policy.SANA_Robot import runtime  # noqa: E402

META = {
    "format": "sana_robot_zip_v1",
    "fps": 25.0,
    "embodiment": "dual-arm RoboDojo ARX-X5 robot with parallel grippers",
    "action_mode": "robot_base_eef",
    "views": [
        {"name": "cam_high", "slot": 0, "description": "overhead external camera"},
        {"name": "cam_left_wrist", "slot": 2, "description": "left wrist-mounted camera"},
        {"name": "cam_right_wrist", "slot": 3, "description": "right wrist-mounted camera"},
    ],
    "state_slots": [0, 1, 2, 16],
    "action_slots": [0, 1, 2, 16],
}
TRAIN_CONFIG = {
    "vae": "checkpoints/LTX-2.3-Diffusers",
    "text_encoder": "checkpoints/gemma-2-2b-it",
    "policy": {"spatial_layout": "t_shape", "actions_per_chunk": 32, "sliding_window_chunks": 1},
    "data": {"view_sizes": [[384, 480], [192, 256], [192, 256]], "chunk_rows": 32, "video_frame_stride": 4},
}


@pytest.fixture
def repository(tmp_path) -> Path:
    """A checkout with the two public models and the RoboDojo recipes."""
    root = tmp_path / "Sana"
    for folder in ("checkpoints/LTX-2.3-Diffusers", "checkpoints/gemma-2-2b-it", "configs/sana_robot"):
        (root / folder).mkdir(parents=True)
    for policy_type, recipe in runtime.RECIPES.items():
        config = dict(TRAIN_CONFIG, policy=dict(TRAIN_CONFIG["policy"], recipe=policy_type))
        (root / recipe).write_text(yaml.safe_dump(config))
    return root


def run_directory(tmp_path, steps=(250, 500), config=TRAIN_CONFIG) -> Path:
    """A run directory as ``train.sh`` leaves it."""
    root = tmp_path / "RoboDojo-cotrain-arx_x5-ee-0"
    for step in steps:
        (root / "checkpoints" / f"step_{step}").mkdir(parents=True)
        (root / "checkpoints" / f"step_{step}" / "model_ema.pth").write_text(str(step))
    (root / "checkpoints" / "latest").symlink_to(f"step_{steps[-1]}")
    if config is not None:
        (root / "config.yaml").write_text(yaml.safe_dump(config))
    (root / "meta.json").write_text(json.dumps(META))
    (root / "normalization.json").write_text("{}")
    return root


def test_a_run_directory_serves_its_latest_step(tmp_path, repository):
    root = run_directory(tmp_path)
    run = runtime.resolve_run({"policy_type": "ar"}, root, repository)
    assert run.weights == (root / "checkpoints/step_500/model_ema.pth").resolve()
    assert run.normalization == root / "normalization.json" and run.meta == root / "meta.json"
    assert run.vae == repository / "checkpoints/LTX-2.3-Diffusers"
    assert run.text_encoder == repository / "checkpoints/gemma-2-2b-it"
    assert run.policy == TRAIN_CONFIG["policy"]
    assert (run.view_sizes, run.chunk_size, run.video_frame_stride) == ([[384, 480], [192, 256], [192, 256]], 32, 4)


def test_checkpoint_step_and_checkpoint_file_choose_the_weights(tmp_path, repository):
    root = run_directory(tmp_path)
    run = runtime.resolve_run({"checkpoint_step": 250}, root, repository)
    assert run.weights == root / "checkpoints/step_250/model_ema.pth"
    run = runtime.resolve_run({"checkpoint_file": "checkpoints/step_250/model_ema.pth"}, root, repository)
    assert run.weights == root / "checkpoints/step_250/model_ema.pth"
    with pytest.raises(FileNotFoundError, match="checkpoint_step"):
        runtime.resolve_run({"checkpoint_step": 750}, root, repository)


def test_a_released_directory_holds_one_weights_file(tmp_path, repository):
    root = tmp_path / "sana_robot_ar_robodojo"
    root.mkdir()
    (root / "sana_robot_ar.pth").write_text("weights")
    (root / "meta.json").write_text(json.dumps(META))
    (root / "normalization.json").write_text("{}")
    run = runtime.resolve_run({"policy_type": "base"}, root, repository)
    assert run.weights == root / "sana_robot_ar.pth"
    assert run.policy["recipe"] == "base"          # no config.yaml: the repository's recipe of the policy type
    assert runtime.resolve_run({"policy_type": "ar"}, root / "sana_robot_ar.pth", repository).weights == root / "sana_robot_ar.pth"
    (root / "second.pth").write_text("weights")
    with pytest.raises(FileNotFoundError, match="exactly one"):
        runtime.resolve_run({}, root, repository)


def test_deploy_keys_come_before_the_directory_and_the_training_config(tmp_path, repository, monkeypatch):
    root = run_directory(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    for name in ("normalization.json", "meta.json", "vae", "gemma"):
        (elsewhere / name).mkdir() if "." not in name else (elsewhere / name).write_text("{}")
    cfg = {
        "normalization_path": str(elsewhere / "normalization.json"),
        "dataset_meta_path": str(elsewhere / "meta.json"),
        "vae_path": str(elsewhere / "vae"),
        "policy": {"spatial_layout": "2x2"},
        "view_sizes": [[256, 320]],
        "chunk_size": 16,
    }
    monkeypatch.setenv(runtime.TEXT_ENCODER_ENV, str(elsewhere / "gemma"))
    run = runtime.resolve_run(cfg, root, repository)
    assert run.normalization == elsewhere / "normalization.json" and run.meta == elsewhere / "meta.json"
    assert run.vae == elsewhere / "vae" and run.text_encoder == elsewhere / "gemma"
    assert run.policy == {"spatial_layout": "2x2"}
    assert (run.view_sizes, run.chunk_size, run.video_frame_stride) == ([[256, 320]], 16, 4)


def test_the_description_and_the_normalization_fall_back_to_the_training_dataset(tmp_path, repository):
    data = tmp_path / "data"
    (data / "sana_robot_zip").mkdir(parents=True)
    (data / "sana_robot_zip" / "meta.json").write_text(json.dumps(META))
    (data / "normalization.json").write_text("{}")
    config = dict(TRAIN_CONFIG, data=dict(TRAIN_CONFIG["data"], path=str(data / "sana_robot_zip"), normalization=str(data / "normalization.json")))
    root = run_directory(tmp_path, config=config)
    (root / "meta.json").unlink()
    (root / "normalization.json").unlink()
    run = runtime.resolve_run({}, root, repository)
    assert run.meta == data / "sana_robot_zip" / "meta.json" and run.normalization == data / "normalization.json"


def test_missing_pieces_are_named(tmp_path, repository):
    root = run_directory(tmp_path)
    (root / "meta.json").unlink()
    with pytest.raises(FileNotFoundError, match="dataset_meta_path"):
        runtime.resolve_run({}, root, repository)
    (root / "meta.json").write_text(json.dumps(META))
    with pytest.raises(FileNotFoundError, match="vae_path"):
        runtime.resolve_run({}, root, tmp_path / "no_checkout")


def test_no_policy_arguments_anywhere_is_an_error(tmp_path):
    root = run_directory(tmp_path, config=None)
    with pytest.raises(FileNotFoundError, match="policy key"):
        runtime.resolve_run({"vae_path": str(root), "text_encoder_path": str(root)}, root, tmp_path / "no_checkout")


def test_the_description_dataset_opens_as_a_dataset_without_episodes(tmp_path):
    from sana_robot.data.storage import open_storage

    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps(META))
    directory = runtime.description_dataset(meta)
    storage = open_storage(directory.name)
    assert storage.episodes == []
    assert storage.info.fps == 25.0 and [view.name for view in storage.info.views] == [view["name"] for view in META["views"]]
    assert [view.slot for view in storage.info.views] == [0, 2, 3]
    assert storage.info.state_mask.nonzero()[0].tolist() == [0, 1, 2, 16]
    directory.cleanup()
    assert not Path(directory.name).exists()


def test_build_deployment_passes_the_keys_that_are_set(tmp_path, repository, monkeypatch):
    import sana_robot.deploy.runtime.deployment as deployment

    built, streams = [], []
    monkeypatch.setattr(deployment, "Deployment", lambda config: built.append(config) or config)
    monkeypatch.setattr(runtime, "use_causal_vae", streams.append)
    run = runtime.resolve_run({}, run_directory(tmp_path), repository)
    cfg = {"policy_type": "ar", "steps": 10, "n_action_steps": 24, "async_delay_steps": 0, "optimized": False, "precision": None,
           "video_flow_shift": None, "action_flow_shift": 1.0, "device": None}
    config = runtime.build_deployment(cfg, run, "/dataset")
    assert streams == built == [config]          # the deployment gets the stream of SANA's causal VAE
    defaults = deployment.DeployConfig()
    assert (config.policy_type, config.checkpoint, config.dataset) == ("ar", str(run.weights), "/dataset")
    assert (config.steps, runtime.setting(config, "n_action_steps"), config.optimized) == (10, 24, False)
    assert runtime.setting(config, "async_delay_steps") == 0
    assert (config.precision, config.video_flow_shift, config.device) == (defaults.precision, defaults.video_flow_shift, defaults.device)
    assert config.policy == TRAIN_CONFIG["policy"] and config.video_frame_stride == 4
    assert runtime.setting(config, "chunk_size") == 32
    assert config.normalization == str(run.normalization) and config.vae == str(run.vae)


def test_sana_root_reads_the_key_then_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv(runtime.SANA_ROOT_ENV, raising=False)
    assert runtime.sana_root({}) == runtime.DEFAULT_SANA_ROOT.resolve()
    monkeypatch.setenv(runtime.SANA_ROOT_ENV, str(tmp_path / "env"))
    assert runtime.sana_root({}) == tmp_path / "env"
    assert runtime.sana_root({"sana_robot_root": str(tmp_path / "key")}) == tmp_path / "key"


def test_settings_are_read_under_the_packages_names_before_and_after_its_renaming(tmp_path, repository):
    earlier = type("Earlier", (), {"chunk_rows": 32, "period_rows": 24, "delay_rows": 8, "steps": 4})
    renamed = type("Renamed", (), {"chunk_size": 32, "n_action_steps": 24, "async_delay_steps": 8, "steps": 4})
    for owner in (earlier, renamed):
        values = [runtime.setting(owner, name) for name in ("chunk_size", "n_action_steps", "async_delay_steps", "steps")]
        assert values == [32, 24, 8, 4]
    assert runtime.package_name(earlier, "n_action_steps") == "period_rows"
    assert runtime.package_name(renamed, "n_action_steps") == "n_action_steps"
    # a training config written after the renaming names the chunk's length chunk_size
    config = dict(TRAIN_CONFIG, data={"view_sizes": [[384, 480]], "chunk_size": 16, "video_frame_stride": 4})
    assert runtime.resolve_run({}, run_directory(tmp_path, config=config), repository).chunk_size == 16
