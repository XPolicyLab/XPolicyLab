import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evals.agilex.evaluate import (
    AgileXVisualPromptConfig,
    RealRobotRunner,
    RealRobotRunnerConfig,
    _prepare_visual_prompt,
    validate_agilex_transform_bundle,
)
from evals.agilex.observation_adapter import AgileXObservationAdapter
from evals.common.checkpoint_runtime import CheckpointTransformBundle
from worldscape_policy.cli.config_composition import load_composed_config
from worldscape_policy.cli.config_profiles import resolve_config_profiles
from worldscape_policy.cli.evaluate import _load_config
from worldscape_policy.types import WorldActionOutput


class _Transform:
    transforms = ()


def _bundle(tag="agilex", *, state=None, action=None):
    transform = _Transform()
    if state is not None:
        sizes = (3, 6, 1, 3, 6, 1)
        transform.embodiment = SimpleNamespace(
            state_fields=tuple(
                SimpleNamespace(key=key, size=size)
                for key, size in zip(state, sizes, strict=True)
            ),
            action_fields=tuple(
                SimpleNamespace(key=key, size=size)
                for key, size in zip(action, sizes, strict=True)
            ),
        )
        transform.image_input_range = "uint8"
    return CheckpointTransformBundle(transform, tag, 33, 64, 32)


def _raw(value=1):
    cameras = tuple(
        np.full((1, 4, 5, 3), value + index, dtype=np.uint8)
        for index in range(3)
    )
    state = {
        "state.left_pos": np.zeros((1, 3), dtype=np.float32),
        "state.left_rot6d": np.array([[1, 0, 0, 1, 0, 0]], dtype=np.float32),
        "state.left_gripper": np.zeros((1, 1), dtype=np.float32),
        "state.right_pos": np.zeros((1, 3), dtype=np.float32),
        "state.right_rot6d": np.array([[1, 0, 0, 1, 0, 0]], dtype=np.float32),
        "state.right_gripper": np.zeros((1, 1), dtype=np.float32),
    }
    return (*cameras, state, value)


def test_benchmark_configs_are_consolidated_by_platform():
    root = Path(__file__).resolve().parents[2] / "configs"
    platform_configs = {"agilex.yaml", "libero.yaml", "robotwin2.yaml"}
    expected_posttrain = {
        "agilex.yaml",
        "common_wan22.yaml",
        "libero.yaml",
        "robotwin2.yaml",
    }

    assert {
        path.name for path in (root / "posttrain").glob("*.yaml")
    } == expected_posttrain
    assert {
        path.name for path in (root / "eval").glob("*.yaml")
    } == platform_configs


@pytest.mark.parametrize(
    ("mode", "stage", "offload"),
    [
        ("zero2", 2, False),
        ("zero2_offload", 2, True),
        ("zero3", 3, False),
    ],
)
def test_agilex_deepspeed_profiles(mode, stage, offload, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("DEEPSPEED_MODE", mode)
    config = resolve_config_profiles(
        load_composed_config(root / "configs" / "posttrain" / "agilex.yaml")
    )

    zero = config.distributed.deepspeed_config.zero_optimization
    assert zero.stage == stage
    assert ("offload_optimizer" in zero) is offload
    assert config.selectors.deepspeed_mode == mode


@pytest.mark.parametrize("benchmark", ["libero", "robotwin2"])
@pytest.mark.parametrize(
    ("mode", "prompt", "dataset_name", "event_memory_frozen", "schedule_enabled"),
    [
        ("interactive", "demo", "worldscape_lerobot_demo", True, False),
        ("auto", "goal", "worldscape_lerobot_goal", False, True),
    ],
)
def test_lerobot_benchmark_profiles_resolve(
    benchmark,
    mode,
    prompt,
    dataset_name,
    event_memory_frozen,
    schedule_enabled,
    monkeypatch,
):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("WSP_MODE", mode)
    monkeypatch.setenv("VISUAL_PROMPT", prompt)
    monkeypatch.setenv("NATIVE_DATASET_NAME", dataset_name)
    monkeypatch.setenv("DATA_ROOT", "/data")
    monkeypatch.setenv("PRETRAINED_MODEL_PATH", "/checkpoint")
    monkeypatch.setenv("WAN_CKPT_DIR", "/wan")
    monkeypatch.setenv("CLIP_CKPT_DIR", "/wan-image")
    monkeypatch.setenv("Qwen_CKPT_DIR", "/qwen")
    monkeypatch.setenv("TOKENIZER_DIR", "/tokenizer")

    config = resolve_config_profiles(
        load_composed_config(
            root / "configs" / "posttrain" / f"{benchmark}.yaml"
        )
    )

    assert config.model.expected_mode == mode
    assert config.data_loader.mode == mode
    assert config.data_loader.dataset_name == dataset_name
    assert config.data_loader.shuffle is False
    assert "temporal_packing" not in config.data_loader.dataset_kwargs
    assert "history_window" not in config.data_loader.dataset_kwargs
    assert config.freeze.config.event_memory is event_memory_frozen
    assert config.prompt_schedule.enabled is schedule_enabled
    assert config.objective.semantic_forcing_weight == (
        0.001 if mode == "auto" else 0.0
    )
    assert config.model.tokenizer_path == "/tokenizer"
    assert config.model.text_encoder_pretrained_path == (
        "/wan/models_t5_umt5-xxl-enc-bf16.pth"
    )
    if mode == "auto":
        assert config.model.image_encoder_pretrained_path == (
            "/wan-image/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
        )
    assert config.batch_adapter.action_dim == 32
    assert config.batch_adapter.state_dim == 64


@pytest.mark.parametrize("benchmark", ["libero", "robotwin2"])
@pytest.mark.parametrize(
    ("mode", "stage", "offload"),
    [("zero2", 2, False), ("zero2_offload", 2, True), ("zero3", 3, False)],
)
def test_lerobot_benchmark_deepspeed_profiles(
    benchmark, mode, stage, offload, monkeypatch
):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("DEEPSPEED_MODE", mode)
    config = resolve_config_profiles(
        load_composed_config(
            root / "configs" / "posttrain" / f"{benchmark}.yaml"
        )
    )

    zero = config.distributed.deepspeed_config.zero_optimization
    assert zero.stage == stage
    assert ("offload_optimizer" in zero) is offload
    assert config.selectors.deepspeed_mode == mode


@pytest.mark.parametrize("benchmark", ["libero", "robotwin2"])
def test_lerobot_benchmarks_reject_unsupported_none_dataset_pair(
    benchmark, monkeypatch
):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("VISUAL_PROMPT", "none")
    monkeypatch.setenv("NATIVE_DATASET_NAME", "worldscape_lerobot_demo")
    with pytest.raises(ValueError, match="requires visual_prompt"):
        resolve_config_profiles(
            load_composed_config(
                root / "configs" / "posttrain" / f"{benchmark}.yaml"
            )
        )


@pytest.mark.parametrize(
    ("config", "kind"),
    [
        ({"kind": "text", "source": "none"}, "text"),
        (
            {
                "kind": "goal",
                "source": "hdf5",
                "context_frames": 1,
                "ctx_head_only": True,
            },
            "goal",
        ),
        (
            {
                "kind": "uniform",
                "source": "hdf5",
                "context_frames": 50,
                "ctx_head_only": True,
            },
            "uniform",
        ),
    ],
)
def test_strict_recipe_conditions_parse(config, kind):
    assert AgileXVisualPromptConfig.from_config(config).kind == kind


@pytest.mark.parametrize(
    (
        "task",
        "mode",
        "prompt",
        "dataset_name",
        "eval_kind",
        "event_memory_frozen",
        "schedule_enabled",
    ),
    [
        (
            "fold-shirt-text",
            "auto",
            "none",
            "worldscape_hdf5_text",
            "text",
            False,
            True,
        ),
        (
            "build-block-goal",
            "interactive",
            "goal",
            "worldscape_hdf5_goal",
            "goal",
            True,
            False,
        ),
        (
            "build-block-demo",
            "interactive",
            "demo",
            "worldscape_hdf5_demo",
            "uniform",
            True,
            False,
        ),
        (
            "shell-game-demo",
            "interactive",
            "demo",
            "worldscape_hdf5_demo",
            "uniform",
            True,
            False,
        ),
        (
            "build-block-goal-auto",
            "auto",
            "goal",
            "worldscape_hdf5_goal",
            "goal",
            False,
            True,
        ),
        (
            "build-block-demo-auto",
            "auto",
            "demo",
            "worldscape_hdf5_demo",
            "uniform",
            False,
            True,
        ),
        (
            "shell-game-demo-auto",
            "auto",
            "demo",
            "worldscape_hdf5_demo",
            "uniform",
            False,
            True,
        ),
    ],
)
def test_common_agilex_profiles_resolve_train_eval_pair(
    task,
    mode,
    prompt,
    dataset_name,
    eval_kind,
    event_memory_frozen,
    schedule_enabled,
    monkeypatch,
):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("WSP_TASK", task)
    monkeypatch.setenv("WSP_MODE", mode)
    monkeypatch.setenv("VISUAL_PROMPT", prompt)
    monkeypatch.setenv("NATIVE_DATASET_NAME", dataset_name)
    monkeypatch.setenv("DATA_ROOT", "/data")
    monkeypatch.setenv("PRETRAINED_MODEL_PATH", "/checkpoint")
    monkeypatch.setenv("OUTPUT_DIR", "/output")
    monkeypatch.setenv("WAN_CKPT_DIR", "/wan")
    monkeypatch.setenv("CLIP_CKPT_DIR", "/wan-image")
    monkeypatch.setenv("Qwen_CKPT_DIR", "/qwen")
    monkeypatch.setenv("TOKENIZER_DIR", "/tokenizer")
    monkeypatch.setenv("VLM_TOKEN_DIM", "2560")
    monkeypatch.setenv("WORLDSCAPE_CHECKPOINT", "/checkpoint")
    monkeypatch.setenv("WORLDSCAPE_HDF5_EPISODE", "/episode.hdf5")
    monkeypatch.setenv("WSP_GOAL_IMAGE", "/goal.png")
    training = resolve_config_profiles(
        load_composed_config(root / "configs" / "posttrain" / "agilex.yaml")
    )
    evaluation = _load_config(root / "configs" / "eval" / "agilex.yaml")

    assert training.selectors.task == task
    assert training.model.expected_mode == mode
    assert training.data_loader.mode == mode
    assert training.data_loader.dataset_name == dataset_name
    assert training.freeze.config.event_memory is event_memory_frozen
    assert training.prompt_schedule.enabled is schedule_enabled
    assert training.objective.planning_ce_weight == 0
    assert evaluation["mode"] == mode
    assert evaluation["tasks"][0]["id"] == task
    assert AgileXVisualPromptConfig.from_config(
        evaluation["visual_prompt"]
    ).kind == eval_kind
    assert "adapter" not in evaluation["backend_config"]


def test_agilex_eval_resolves_demo_transport_settings(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("WSP_MODE", "interactive")
    monkeypatch.setenv("VISUAL_PROMPT", "demo")
    monkeypatch.setenv("WORLDSCAPE_CHECKPOINT", "/checkpoint")
    monkeypatch.delenv("WSP_SERVER_PORT", raising=False)
    monkeypatch.delenv("WSP_NODE_NAME", raising=False)
    monkeypatch.setenv("WSP_DEMO_SOURCE", "transport")
    monkeypatch.setenv("WSP_DEMO_POLL_TRANSPORT", "true")
    monkeypatch.setenv("OBSERVATION_TIMEOUT_S", "22.5")

    evaluation = _load_config(
        root / "configs" / "eval" / "agilex.yaml"
    )

    prompt = AgileXVisualPromptConfig.from_config(evaluation["visual_prompt"])
    assert prompt.source == "transport"
    assert prompt.poll_transport is True
    backend = evaluation["backend_config"]
    assert backend["port"] == 11451
    assert backend["node_name"] == "WSP"
    assert (
        evaluation["backend_config"]["observation_timeout_s"] == 22.5
    )



@pytest.mark.parametrize(
    ("script", "task", "mode", "prompt", "dataset_name"),
    [
        (
            "posttrain/posttrain_agilex_fold_shirt_text.sh",
            "fold-shirt-text",
            "auto",
            "none",
            "worldscape_hdf5_text",
        ),
        (
            "posttrain/posttrain_agilex_build_block_goal.sh",
            "build-block-goal",
            "interactive",
            "goal",
            "worldscape_hdf5_goal",
        ),
        (
            "posttrain/posttrain_agilex_build_block_demo.sh",
            "build-block-demo",
            "interactive",
            "demo",
            "worldscape_hdf5_demo",
        ),
        (
            "posttrain/posttrain_agilex_shell_game_demo.sh",
            "shell-game-demo",
            "interactive",
            "demo",
            "worldscape_hdf5_demo",
        ),
    ],
)
def test_posttrain_task_launchers_export_only_common_selectors(
    script,
    task,
    mode,
    prompt,
    dataset_name,
    tmp_path,
):
    root = Path(__file__).resolve().parents[2]
    capture = tmp_path / "capture.sh"
    capture.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s|%s|%s|%s\\n" "$WSP_TASK" "$WSP_MODE" '
        '"$VISUAL_PROMPT" "$NATIVE_DATASET_NAME" > "$CAPTURE_FILE"\n'
    )
    capture.chmod(0o755)
    output = tmp_path / "selectors.txt"
    wan = tmp_path / "wan"
    tokenizer = tmp_path / "tokenizer"
    wan.mkdir()
    tokenizer.mkdir()
    (wan / "weights").touch()
    (tokenizer / "tokenizer.json").touch()
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "episodes.jsonl").write_text('{"episode_index": 0}\n')
    environment = os.environ.copy()
    environment.update(
        WSP_TRAIN=str(capture),
        CAPTURE_FILE=str(output),
        DATA_ROOT=str(tmp_path),
        PRETRAINED_MODEL_PATH="/checkpoint",
        WAN_CKPT_DIR=str(wan),
        CLIP_CKPT_DIR=str(tmp_path / "wan-image"),
        TOKENIZER_DIR=str(tokenizer),
        Qwen_CKPT_DIR=str(tmp_path / "qwen"),
        WSP_AUTO_DOWNLOAD="false",
    )

    subprocess.run(
        [str(root / "recipes" / script)],
        check=True,
        env=environment,
    )

    assert output.read_text().strip() == "|".join(
        (task, mode, prompt, dataset_name)
    )


@pytest.mark.parametrize(
    ("script", "config_name", "task"),
    [
        ("posttrain/posttrain_libero.sh", "libero.yaml", "libero"),
        ("posttrain/posttrain_robotwin2.sh", "robotwin2.yaml", "robotwin2"),
    ],
)
def test_benchmark_launchers_use_checkout_torchrun_contract(
    script, config_name, task, tmp_path
):
    root = Path(__file__).resolve().parents[2]
    capture = tmp_path / "python"
    capture.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$@" > "$CAPTURE_FILE"\n'
        'printf "%s|%s|%s|%s\\n" "$WSP_TASK" "$NUM_GPUS" '
        '"$DEEPSPEED_MODE" "$PYTHONPATH" >> "$CAPTURE_FILE"\n'
    )
    capture.chmod(0o755)
    output = tmp_path / "launch.txt"
    wan = tmp_path / "wan"
    tokenizer = tmp_path / "tokenizer"
    wan.mkdir()
    tokenizer.mkdir()
    (wan / "weights").touch()
    (tokenizer / "tokenizer.json").touch()
    environment = os.environ.copy()
    environment.pop("WSP_TRAIN", None)
    environment.pop("DEEPSPEED_MODE", None)
    environment.pop("NUM_GPUS", None)
    environment.update(
        WSP_PYTHON=str(capture),
        CAPTURE_FILE=str(output),
        CUDA_VISIBLE_DEVICES="3,5",
        DATA_ROOT=str(tmp_path),
        ZSCORE_STATS_PATH=str(tmp_path / "dataset_stats.json"),
        SUBTASK_EPISODE_MAP_PATH=str(tmp_path / "episode_map.json"),
        WAN_CKPT_DIR=str(wan),
        TOKENIZER_DIR=str(tokenizer),
        WSP_AUTO_DOWNLOAD="false",
    )

    subprocess.run([str(root / "recipes" / script)], check=True, env=environment)

    lines = output.read_text().splitlines()
    assert lines[:2] == ["-m", "torch.distributed.run"]
    assert "--nproc_per_node" in lines
    assert lines[lines.index("--nproc_per_node") + 1] == "2"
    assert lines[lines.index("--config") + 1] == str(
        root / "configs" / "posttrain" / config_name
    )
    task_name, num_gpus, deepspeed_mode, pythonpath = lines[-1].split("|", 3)
    assert (task_name, num_gpus, deepspeed_mode) == (task, "2", "zero2")
    assert pythonpath.split(":")[:2] == [str(root / "src"), str(root)]


@pytest.mark.parametrize(
    ("script", "task", "mode", "prompt"),
    [
        (
            "eval/eval_agilex_fold_shirt_text.sh",
            "fold-shirt-text",
            "auto",
            "none",
        ),
        (
            "eval/eval_agilex_build_block_goal.sh",
            "build-block-goal",
            "interactive",
            "goal",
        ),
        (
            "eval/eval_agilex_build_block_demo.sh",
            "build-block-demo",
            "interactive",
            "demo",
        ),
        (
            "eval/eval_agilex_shell_game_demo.sh",
            "shell-game-demo",
            "interactive",
            "demo",
        ),
    ],
)
def test_eval_task_launchers_export_only_common_selectors(
    script,
    task,
    mode,
    prompt,
    tmp_path,
):
    root = Path(__file__).resolve().parents[2]
    capture = tmp_path / "capture.sh"
    capture.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s|%s|%s\\n" "$WSP_TASK" "$WSP_MODE" '
        '"$VISUAL_PROMPT" > "$CAPTURE_FILE"\n'
    )
    capture.chmod(0o755)
    output = tmp_path / "selectors.txt"
    environment = os.environ.copy()
    environment.update(
        WSP_EVAL=str(capture),
        CAPTURE_FILE=str(output),
        FOLD_SHIRT_TEXT_EVAL_MODEL_PATH=str(tmp_path / "model"),
        BUILD_BLOCK_GOAL_EVAL_MODEL_PATH=str(tmp_path / "model"),
        BUILD_BLOCK_DEMO_EVAL_MODEL_PATH=str(tmp_path / "model"),
        SHELL_GAME_DEMO_EVAL_MODEL_PATH=str(tmp_path / "model"),
    )

    subprocess.run(
        [str(root / "recipes" / script)],
        check=True,
        env=environment,
    )

    assert output.read_text().strip() == "|".join((task, mode, prompt))


def test_build_block_demo_eval_selects_transport_source_by_backend(tmp_path):
    root = Path(__file__).resolve().parents[2]
    capture = tmp_path / "capture.sh"
    capture.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s|%s|%s|%s\\n" "$AGILEX_TRANSPORT" "$WSP_DEMO_SOURCE" '
        '"$WSP_DEMO_POLL_TRANSPORT" "$WSP_SERVER_PORT" > "$CAPTURE_FILE"\n'
    )
    capture.chmod(0o755)
    output = tmp_path / "transport.txt"
    base_environment = os.environ.copy()
    for name in ("WSP_DEMO_SOURCE", "WSP_DEMO_POLL_TRANSPORT", "WSP_SERVER_PORT"):
        base_environment.pop(name, None)
    base_environment.update(
        WSP_EVAL=str(capture),
        CAPTURE_FILE=str(output),
        BUILD_BLOCK_DEMO_EVAL_MODEL_PATH=str(tmp_path / "model"),
    )

    expected = {
        "manifold": "manifold|transport|true|11451",
        "hdf5": "hdf5|hdf5|false|11451",
    }
    for transport, values in expected.items():
        environment = base_environment.copy()
        environment["AGILEX_TRANSPORT"] = transport
        if transport == "hdf5":
            environment["WORLDSCAPE_HDF5_EPISODE"] = str(
                tmp_path / "episode.hdf5"
            )
        subprocess.run(
            [
                str(
                    root
                    / "recipes"
                    / "eval"
                    / "eval_agilex_build_block_demo.sh"
                )
            ],
            check=True,
            env=environment,
        )
        assert output.read_text().strip() == values



def test_recipe_validation_rejects_implicit_goal_and_nonpositive_uniform_length():
    with pytest.raises(ValueError, match="explicit"):
        AgileXVisualPromptConfig.from_config(
            {
                "kind": "goal",
                "source": "first_observation",
                "context_frames": 1,
            }
        )
    with pytest.raises(ValueError, match="positive"):
        AgileXVisualPromptConfig.from_config(
            {"kind": "uniform", "source": "hdf5", "context_frames": 0}
        )


def test_text_goal_and_demo_prompts_have_exact_one_shot_shapes():
    raw = _raw()
    text = AgileXObservationAdapter(
        _bundle(), mode="auto", instruction="task", visual_input_range="uint8", device="cpu"
    )
    assert text(raw).prompts.goal_images is None
    assert text(raw).prompts.demo_videos is None

    goal = AgileXObservationAdapter(
        _bundle(),
        mode="auto",
        instruction="task",
        visual_input_range="uint8",
        device="cpu",
        goal_image=np.zeros((4, 5, 3), dtype=np.uint8),
    )
    assert goal(raw).prompts.goal_images.shape == (1, 1, 3, 4, 5)
    assert goal(raw).prompts.goal_images is None

    for views in (1, 3):
        demo_array = np.zeros((50, views, 4, 5, 3), dtype=np.uint8)
        demo = AgileXObservationAdapter(
            _bundle(),
            mode="auto",
            instruction="task",
            visual_input_range="uint8",
            device="cpu",
            demo_video=demo_array,
        )
        assert demo(raw).prompts.demo_videos.shape == (1, 50, views, 3, 4, 5)
        assert demo(raw).prompts.demo_videos is None


class _ContextRobot:
    def __init__(self):
        self.polls = 0
        self.reads = 0

    def reset_episode(self):
        pass

    def read_observation(self, **_kwargs):
        self.reads += 1
        raw = list(_raw(self.reads))
        raw[3]["state.left_pos"][0, 0] = self.reads
        return tuple(raw)

    def try_read_context_video(self, frames):
        self.polls += 1
        if self.polls <= 2:
            value = tuple(
                np.full((frames, 4, 5, 3), self.polls + index, dtype=np.uint8)
                for index in range(3)
            )
            return value, True
        return None, False

    def send_end_pose_action(self, *_args):
        pass


class _Runtime:
    def __init__(self):
        self.observations = []
        self.policy = torch.nn.Linear(1, 1)
        self.pending = None
        self.resets = 0
        self.prompts = []

    @property
    def has_pending_prediction(self):
        return self.pending is not None

    def reset(self, mode):
        self.mode = mode
        self.pending = None
        self.resets += 1

    def predict(self, **kwargs):
        self.prompts.append(kwargs["prompts"])
        self.observations.append(kwargs["observation"])
        self.pending = WorldActionOutput(action=torch.zeros(1, 1, 20))
        return self.pending

    def commit(self, output):
        assert output is self.pending
        self.pending = None

    def discard(self):
        self.pending = None


class _Actions:
    def __call__(self, _output):
        side = [[0, 0, 0, 0, 0, 0, 1, 0]]
        return side, side


def test_transport_context_reupload_resets_session_and_memory():
    runtime = _Runtime()
    adapter = AgileXObservationAdapter(
        _bundle(), mode="auto", instruction="task", visual_input_range="uint8", device="cpu"
    )
    RealRobotRunner(runtime, _ContextRobot(), adapter, _Actions()).run(
        RealRobotRunnerConfig(
            "auto",
            max_steps=2,
            rollout_steps=1,
            context_poll=True,
            context_frames=50,
            ctx_head_only=True,
        ),
        generator=torch.Generator(),
    )
    assert runtime.resets == 3  # episode reset plus two uploaded sessions
    assert [prompt.demo_videos.shape[2] for prompt in runtime.prompts] == [1, 1]
    assert [
        int(observation.images[0, 0, 0, 0, 0, 0])
        for observation in runtime.observations
    ] == [2, 4]
    assert [
        float(observation.proprioception[0, 0, 0])
        for observation in runtime.observations
    ] == [2.0, 4.0]
    assert adapter.session_id.endswith("ctx-2")


def test_hdf5_context_preload_uniformly_subsamples_and_orders_views():
    class Robot:
        def read_context_video(self, frames):
            assert frames == 6
            return tuple(
                np.full((7, 8, 9, 3), index, dtype=np.uint8)
                for index in (10, 20, 30)
            )

    prompt = AgileXVisualPromptConfig.from_config(
        {
            "kind": "uniform",
            "source": "hdf5",
            "context_frames": 6,
            "ctx_head_only": False,
        }
    )
    goal, demo = _prepare_visual_prompt(prompt, Robot(), uploaded_visual_prompt=None)
    assert goal is None
    assert isinstance(demo, tuple)
    assert [camera.shape for camera in demo] == [(6, 160, 320, 3)] * 3
    assert [int(camera[0, 0, 0, 0]) for camera in demo] == [10, 20, 30]


def test_transform_bundle_requires_lctx_embodiment_and_exact_field_order():
    state = (
        "state.left_pos",
        "state.left_rot6d",
        "state.left_gripper",
        "state.right_pos",
        "state.right_rot6d",
        "state.right_gripper",
    )
    action = tuple(key.replace("state.", "action.") for key in state)
    validate_agilex_transform_bundle(
        _bundle(state=state, action=action), action_mode="eef"
    )
    with pytest.raises(ValueError, match="field order"):
        validate_agilex_transform_bundle(
            _bundle(state=state[::-1], action=action), action_mode="eef"
        )
    with pytest.raises(ValueError, match="agilex"):
        validate_agilex_transform_bundle(
            _bundle("libero", state=state, action=action), action_mode="eef"
        )
    with pytest.raises(ValueError, match="visual_input_range"):
        validate_agilex_transform_bundle(
            _bundle(state=state, action=action),
            action_mode="eef",
            visual_input_range="zero_one",
        )
