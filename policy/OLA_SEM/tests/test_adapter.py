from __future__ import annotations

import json
import types
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import cv2

from XPolicyLab.policy.OLA_SEM.model import (
    Model,
    compose_three_view_rgb,
    pack_joint_state,
    unpack_joint_actions,
)
from XPolicyLab.policy.OLA_SEM.robotwin_eval_client import (
    build_policy_shim,
    native_to_standard,
    parse_result_file,
    standard_action_to_qpos,
    task_config_video_override,
)
from XPolicyLab.policy.OLA_SEM import summarize_local_eval
from XPolicyLab.policy.OLA_SEM.ola_sem.data.robotwin2.robotwin_data_convert.robotwin_converter import (
    RobotWinConverter,
)


DIM_INFO = {"arm_dim": [6, 6], "ee_dim": [1, 1]}


def standard_obs(value=0.0):
    return {
        "instruction": "hang the mug",
        "vision": {
            "cam_head": {"color": np.zeros((480, 640, 3), dtype=np.uint8)},
            "cam_left_wrist": {"color": np.zeros((240, 320, 3), dtype=np.uint8)},
            "cam_right_wrist": {"color": np.zeros((240, 320, 3), dtype=np.uint8)},
        },
        "state": {
            "left_arm_joint_state": np.full(6, value, dtype=np.float32),
            "left_ee_joint_state": np.array([value + 1], dtype=np.float32),
            "right_arm_joint_state": np.full(6, value + 2, dtype=np.float32),
            "right_ee_joint_state": np.array([value + 3], dtype=np.float32),
        },
    }


def test_image_and_state_mapping():
    obs = standard_obs()
    assert compose_three_view_rgb(obs).shape == (360, 320, 3)
    state = pack_joint_state(obs, DIM_INFO)
    assert state.shape == (14,)
    assert state[6] == 1 and state[7] == 2 and state[13] == 3


def test_vendored_converter_uses_shared_rgb_decode():
    rgb = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [12, 34, 56]]],
        dtype=np.uint8,
    )
    ok, encoded = cv2.imencode(".png", rgb)
    assert ok
    converter = RobotWinConverter.__new__(RobotWinConverter)
    np.testing.assert_array_equal(converter.decode_compressed_image(encoded), rgb)


def test_action_round_trip():
    packed = np.arange(16 * 14, dtype=np.float32).reshape(16, 14)
    actions = unpack_joint_actions(packed, DIM_INFO)
    assert len(actions) == 16
    np.testing.assert_array_equal(standard_action_to_qpos(actions[3]), packed[3])


def test_robotwin_observation_mapping():
    qpos = np.arange(14, dtype=np.float32)
    native = {
        "observation": {
            "head_camera": {"rgb": np.zeros((240, 320, 3), dtype=np.uint8)},
            "left_camera": {"rgb": np.zeros((240, 320, 3), dtype=np.uint8)},
            "right_camera": {"rgb": np.zeros((240, 320, 3), dtype=np.uint8)},
        },
        "joint_action": {"vector": qpos},
    }
    mapped = native_to_standard(native, "task")
    np.testing.assert_array_equal(pack_joint_state(mapped, DIM_INFO), qpos)


class FakePolicy:
    def __init__(self):
        self.instructions = []
        self.updated = []
        self.recorded = []
        self.obs_cache = deque()
        self.action_cache = deque()
        self.current_state = None
        self.current_state_norm = None
        self.is_first_step = True
        self.prev_action = None
        self.real_qpos_history = deque()

    def set_instruction(self, value):
        self.instructions.append(value)

    def record_executed_qpos(self, obs):
        self.recorded.append(obs["joint_action"]["vector"].copy())

    def update_obs(self, obs):
        self.updated.append(obs)

    def get_action(self):
        return np.zeros((16, 14), dtype=np.float32)


def test_history_feedback_and_reset():
    model = Model.__new__(Model)
    model.robot_action_dim_info = DIM_INFO
    model.policy = FakePolicy()
    model._has_observation = False
    model._last_obs = None
    model.update_obs(standard_obs(0))
    model.update_obs(standard_obs(10))
    assert len(model.policy.updated) == 2
    assert len(model.policy.recorded) == 1
    assert len(model.get_action()) == 16
    model.reset()
    assert not model._has_observation
    assert len(model.policy.real_qpos_history) == 0


def test_rejects_multi_environment_batch():
    model = Model.__new__(Model)
    model.robot_action_dim_info = DIM_INFO
    model.policy = FakePolicy()
    model._has_observation = False
    model._last_obs = None
    with pytest.raises(NotImplementedError):
        model.update_obs_batch([standard_obs(), standard_obs()])


def test_rejects_wrong_history_metadata(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "common": {"action_chunk_size": 16},
                "flow_source": {
                    "mode": "gaussian",
                    "video_mode": "gaussian",
                    "action_noise_std": 0.02,
                    "history_length": 16,
                },
            }
        )
    )
    model = Model.__new__(Model)
    model.checkpoint_root = tmp_path
    with pytest.raises(ValueError, match="Incompatible history-flow metadata"):
        model._validate_history_metadata()


def test_constructor_uses_real_checkpoint_metadata(monkeypatch):
    monkeypatch.setattr(Model, "_load_policy", lambda self: FakePolicy())
    checkpoint = (
        "/data/user/wsong890/user68/cjy/Motus/checkpoints/"
        "robotwin_lap_history_flow_clean/robotwin_clean_lap_history_flow_clean/"
        "checkpoint_step_30000"
    )
    model = Model(
        {
            "bench_name": "RoboTwin",
            "task_name": "hanging_mug",
            "ckpt_name": checkpoint,
            "env_cfg_type": "aloha_agilex",
            "action_type": "joint",
            "seed": 42,
            "wan_path": "/data/user/wsong890/user68/cjy/Motus/pretrained_models/Wan2.2-TI2V-5B",
            "vlm_path": "/data/user/wsong890/user68/cjy/Motus/pretrained_models/Qwen3-VL-2B-Instruct",
        }
    )
    assert model.checkpoint_root.name == "checkpoint_step_30000"
    assert model.checkpoint_model_dir.name == "pytorch_model"


@pytest.mark.parametrize(
    ("env_cfg_type", "action_type"),
    [("aloha_agilex", "ee"), ("arx_x5", "joint")],
)
def test_constructor_rejects_unsupported_contract(env_cfg_type, action_type):
    with pytest.raises(ValueError, match="supports only"):
        Model({"env_cfg_type": env_cfg_type, "action_type": action_type})


def test_robotwin_shim_executes_chunk_and_returns_feedback():
    qpos = np.zeros(14, dtype=np.float32)
    native = {
        "observation": {
            name: {"rgb": np.zeros((240, 320, 3), dtype=np.uint8)}
            for name in ("head_camera", "left_camera", "right_camera")
        },
        "joint_action": {"vector": qpos},
    }

    class Client:
        def __init__(self):
            self.calls = []

        def call(self, func_name, obs=None):
            self.calls.append((func_name, obs))
            if func_name == "get_action":
                return unpack_joint_actions(np.zeros((2, 14), dtype=np.float32), DIM_INFO)

    class TaskEnv:
        eval_success = False
        take_action_cnt = 0
        step_lim = 10

        def get_instruction(self):
            return "task"

        def get_obs(self):
            return native

        def take_action(self, action, action_type):
            assert action_type == "qpos"
            assert action.shape == (14,)
            self.take_action_cnt += 1

    client = Client()
    shim = build_policy_shim(client)
    task = TaskEnv()
    shim.eval(task, client, native)
    assert task.take_action_cnt == 2
    assert [name for name, _ in client.calls] == [
        "update_obs",
        "get_action",
        "update_obs",
    ]


def test_video_override_is_process_local(tmp_path, monkeypatch):
    robotwin_root = tmp_path / "RoboTwin"
    config_path = robotwin_root / "task_config" / "demo_clean.yml"
    config_path.parent.mkdir(parents=True)
    original = "eval_video_log: true\nrender_freq: 0\n"
    config_path.write_text(original, encoding="utf-8")
    robotwin_eval = types.ModuleType("fake_robotwin_eval")
    monkeypatch.chdir(robotwin_root)

    with task_config_video_override(
        robotwin_eval, robotwin_root, "demo_clean", enabled=False
    ):
        rendered = robotwin_eval.open(
            Path("task_config/demo_clean.yml"), encoding="utf-8"
        ).read()
        assert "eval_video_log: false" in rendered

    assert not hasattr(robotwin_eval, "open")
    assert config_path.read_text(encoding="utf-8") == original


def test_result_file_validation(tmp_path):
    result = tmp_path / "_result.txt"
    result.write_text(
        "Timestamp: 2026-08-31 00:00:00\n\n"
        "Instruction Type: unseen\n\n0.73",
        encoding="utf-8",
    )
    assert parse_result_file(result, 100) == {
        "episodes": 100,
        "success_count": 73,
        "success_rate": 0.73,
    }
    result.write_text("1.01", encoding="utf-8")
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        parse_result_file(result, 100)


def test_local_50_task_summary(tmp_path):
    tasks_file = Path(__file__).parents[1] / "eval_tasks_50.txt"
    tasks = summarize_local_eval.load_tasks(tasks_file)
    assert len(tasks) == 50

    run_dir = tmp_path / "local_eval"
    for child in ("elements", "logs", "status"):
        (run_dir / child).mkdir(parents=True)
    result_file = tmp_path / "_result.txt"
    result_file.write_text("0.5\n", encoding="utf-8")

    for task in tasks:
        stem = f"clean_{task}"
        (run_dir / "status" / f"{stem}.exit_code").write_text(
            "0\n", encoding="utf-8"
        )
        (run_dir / "elements" / f"{stem}.json").write_text(
            json.dumps(
                {
                    "task": task,
                    "task_config": "demo_clean",
                    "episodes": 100,
                    "success_count": 50,
                    "success_rate": 0.5,
                    "result_file": str(result_file),
                }
            ),
            encoding="utf-8",
        )

    records = summarize_local_eval.build_records(run_dir, tasks, ["clean"], 100)
    assert len(records) == 50
    assert all(record["status"] == "passed" for record in records)
    assert summarize_local_eval.write_outputs(run_dir, records, ["clean"])
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["conditions"]["clean"]["macro_average_success_rate"] == 0.5
    assert summary["conditions"]["clean"]["aggregate_success_rate"] == 0.5

    (run_dir / "status" / f"clean_{tasks[0]}.exit_code").write_text(
        "1\n", encoding="utf-8"
    )
    records = summarize_local_eval.build_records(run_dir, tasks, ["clean"], 100)
    assert records[0]["status"] == "failed"
    assert not summarize_local_eval.write_outputs(run_dir, records, ["clean"])
