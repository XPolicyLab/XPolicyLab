"""The adapter's episode clock, on the real ``ChunkRunner`` of ``sana_robot`` over a controller that only counts."""

import json

import numpy as np
import pytest
import torch

pytest.importorskip("sana_robot")

from sana_robot.data.format.episode import View  # noqa: E402
from sana_robot.deploy.async_exec.runner import ChunkRunner  # noqa: E402

import XPolicyLab.policy.SANA_Robot.model as adapter  # noqa: E402
from XPolicyLab.policy.SANA_Robot import runtime  # noqa: E402

CHUNK_ROWS, FRAME_STRIDE = 32, 4


class CountingController:
    """A controller whose chunk generated at row r holds, in the first two joints of row i, the episode row r + i and the chunk's index."""

    def __init__(self, period_rows, delay_rows):
        self.chunk_rows, self.frame_stride = CHUNK_ROWS, FRAME_STRIDE
        self.delay_rows = delay_rows
        self.period_rows = CHUNK_ROWS - delay_rows if period_rows is None else period_rows
        self.window_rows = int(np.gcd(CHUNK_ROWS, self.period_rows))
        self.events = []
        self.chunks = 0

    def _chunk(self, row, window, state):
        actions = torch.zeros(1, CHUNK_ROWS, 80)
        actions[0, :, 0] = row + torch.arange(CHUNK_ROWS)
        actions[0, :, 1] = self.chunks
        self.chunks += 1
        return actions

    def open(self, window, state, state_mask, action_mask):
        self.chunks = 0
        self.events.append(("open", 0, float(window[0][0, 0, 0, 0]), float(state[0, 0])))
        return self._chunk(0, window, state)

    def observe(self, row, window):
        self.events.append(("observe", row, float(window[0][0, -1, 0, 0])))

    def generate(self, row, window, state, state_mask, action_mask, running=None):
        self.events.append(("generate", row, float(window[0][0, -1, 0, 0]), float(state[0, 0])))
        return self._chunk(row, window, state)


class CountingRobot:
    """A robot interface that passes the first pixel of every frame and the state row through."""

    action_mask = torch.ones(1, CHUNK_ROWS, 80, dtype=torch.bool)

    class info:
        views = (View("cam_high", 0, "head"), View("cam_left_wrist", 2, "left"), View("cam_right_wrist", 3, "right"))

    def frames(self, images):
        return tuple(torch.full((3, 1, 2, 2), float(image[0, 0, 0])) for image in images)

    def state(self, row):
        return torch.as_tensor(np.asarray(row, dtype=np.float32)).reshape(1, -1), torch.ones(1, 80, dtype=torch.bool)

    def actions(self, normalized):
        return normalized[0].numpy()


class CountingDeployment:
    def __init__(self, period_rows, delay_rows):
        self.robot = CountingRobot()
        self.config = type("Config", (), dict(policy_type="ar", steps=10, period_rows=period_rows, delay_rows=delay_rows, optimized=False, precision="bf16"))
        self.runners = []
        self.arguments = (period_rows, delay_rows)

    def runner(self, instruction):
        runner = ChunkRunner(CountingController(*self.arguments), self.robot)
        runner.instruction = instruction
        self.runners.append(runner)
        return runner


@pytest.fixture
def build(tmp_path, monkeypatch):
    """Return ``build(period_rows, delay_rows, **keys)``, which builds the adapter on a counting deployment."""
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("vae", "gemma"):
        (checkpoint / name).mkdir()
    (checkpoint / "model.pth").write_text("weights")
    (checkpoint / "normalization.json").write_text("{}")
    (checkpoint / "meta.json").write_text(json.dumps({}))
    monkeypatch.setattr(adapter, "get_robot_action_dim_info", lambda env_cfg_type: {"arm_dim": [6, 6], "ee_dim": [1, 1]})
    models = []

    def build(period_rows=None, delay_rows=0, **keys):
        deployment = CountingDeployment(period_rows, delay_rows)
        monkeypatch.setattr(runtime, "build_deployment", lambda model_cfg, run, dataset: deployment)
        cfg = dict(
            policy_name="SANA_Robot", bench_name="RoboDojo", task_name="stack_bowls", ckpt_name="unused", env_cfg_type="arx_x5",
            action_type="joint", seed=0, checkpoint_path=str(checkpoint), vae_path=str(checkpoint / "vae"),
            text_encoder_path=str(checkpoint / "gemma"), policy={"spatial_layout": "t_shape"},
        )
        cfg.update(keys)
        models.append(adapter.Model(cfg))
        return models[-1], deployment

    yield build
    for model in models:
        model.close()


def observation(row, instruction="stack the bowls"):
    """The observation after ``row`` executed actions: the head camera shows the row, the first left joint holds it."""
    color = np.full((4, 6, 3), row % 256, dtype=np.uint8)
    color.setflags(write=False)
    return {
        "vision": {camera: {"color": color} for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
        "state": {
            "left_arm_joint_state": np.array([row / 1000.0, 0, 0, 0, 0, 0]),
            "left_ee_joint_state": np.array([1.0]),
            "right_arm_joint_state": np.zeros(6),
            "right_ee_joint_state": np.array([1.0]),
        },
        "instruction": instruction,
    }


def run_episode(model, rows, instruction="stack the bowls"):
    """Run the loop of ``deploy.py`` for ``rows`` actions; returns per executed action (the episode row, the chunk) its values say it is."""
    model.reset()
    executed, row = [], 0
    while row < rows:
        model.update_obs(observation(row, instruction))
        actions = model.get_action()
        for index, action in enumerate(actions):
            executed.append((int(round(float(action["left_arm_joint_state"][0]))), int(round(float(action["left_arm_joint_state"][1])))))
            row += 1
            if row >= rows or index + 1 == len(actions):
                break
            model.update_obs(observation(row, instruction))
    return executed


def test_whole_chunks_run_one_after_another(build):
    model, deployment = build()
    executed = run_episode(model, 96)
    model.close()                      # the worker has done everything that was handed over
    assert [row for row, _chunk in executed] == list(range(96))
    assert [chunk for _row, chunk in executed] == [0] * 32 + [1] * 32 + [2] * 32
    events = deployment.runners[0].controller.events
    assert [(name, row) for name, row, *_ in events] == [("open", 0), ("generate", 32), ("generate", 64)]
    # a chunk is generated from the frame and the state observed at its first row
    assert [(event[2], event[3]) for event in events] == [(0.0, 0.0), (32.0, pytest.approx(0.032)), (64.0, pytest.approx(0.064))]


def test_a_shorter_period_replans_inside_the_chunk(build):
    model, deployment = build(period_rows=24)
    executed = run_episode(model, 72)
    model.close()                      # the worker has done everything that was handed over
    assert [row for row, _chunk in executed] == list(range(72))
    assert [chunk for _row, chunk in executed] == [0] * 24 + [1] * 24 + [2] * 24
    events = [(name, row) for name, row, *_ in deployment.runners[0].controller.events]
    assert events == [("open", 0), ("observe", 8), ("observe", 16), ("generate", 24), ("observe", 32), ("observe", 40), ("generate", 48), ("observe", 56), ("observe", 64)]


def test_a_delay_keeps_the_running_chunk_while_the_next_is_generated(build):
    model, deployment = build(period_rows=24, delay_rows=8)
    executed = run_episode(model, 80)
    model.close()                      # the worker has done everything that was handed over
    assert [row for row, _chunk in executed] == list(range(80))
    # chunk 0 runs its first P + d rows; chunk k, generated at row k P, takes over at k P + d
    assert [chunk for _row, chunk in executed] == [0] * 32 + [1] * 24 + [2] * 24
    generated = [row for name, row, *_ in deployment.runners[0].controller.events if name == "generate"]
    assert generated == [24, 48, 72]


def test_every_action_of_a_chunk_must_run_before_the_next_request(build):
    model, _deployment = build()
    model.reset()
    model.update_obs(observation(0))
    assert len(model.get_action()) == 32
    for row in range(1, 5):
        model.update_obs(observation(row))
    with pytest.raises(RuntimeError, match="after 32 executed actions"):
        model.get_action()


def test_get_action_needs_an_observation(build):
    model, _deployment = build()
    model.reset()
    with pytest.raises(RuntimeError, match="needs an observation"):
        model.get_action()


def test_episodes_restart_the_runner_and_an_instruction_change_replaces_it(build):
    model, deployment = build(period_rows=24)
    first = run_episode(model, 30)
    model.reset()
    model.reset()                      # RoboDojo resets twice between episodes
    second = run_episode(model, 48)
    assert len(deployment.runners) == 1
    assert [row for row, _chunk in second] == list(range(48)) and first == second[:30]
    opened = [event for event in deployment.runners[0].controller.events if event[0] == "open"]
    assert len(opened) == 2
    run_episode(model, 8, instruction="hang the mugs")
    assert [runner.instruction for runner in deployment.runners] == ["stack the bowls", "hang the mugs"]
    assert deployment.runners[0].worker._shutdown and not deployment.runners[1].worker._shutdown


def test_the_noise_seed_follows_the_seed_and_the_episode(build):
    model, _deployment = build(seed=3)
    draws = []
    for _episode in range(2):
        run_episode(model, 4)
        draws.append(float(torch.rand(())))
    again, _deployment = build(seed=3)
    run_episode(again, 4)
    assert draws[0] != draws[1] and float(torch.rand(())) == draws[0]


def test_the_batch_calls_serve_one_environment(build):
    model, _deployment = build()
    model.reset()
    model.update_obs_batch([observation(0)])
    (actions,) = model.get_action_batch([0])
    assert len(actions) == 32
    with pytest.raises(ValueError, match="one environment"):
        model.update_obs_batch([observation(1), observation(1)])
    with pytest.raises(ValueError, match="one environment"):
        model.get_action_batch([0, 1])


def test_end_effector_actions_and_the_reported_pose_check(build, capsys):
    model, _deployment = build(action_type="ee")
    model.reset()
    first = observation(0)
    row = model.robodojo.state_row(first["state"])
    for side in ("left", "right"):
        first["state"][f"{side}_ee_pose"] = model.robodojo.transform_to_pose(model.world_from_base[side] @ model.robodojo.flange_transform(row, side))
    model.update_obs(first)
    with pytest.raises(ValueError, match="finite"):      # the counting chunk holds no rotation
        model.get_action()
    assert "left 0.00 mm / 0.00 deg" in capsys.readouterr().out


def test_unknown_action_type_and_camera_count_are_refused(build):
    with pytest.raises(ValueError, match="action_type"):
        build(action_type="delta")
    with pytest.raises(ValueError, match="one camera per view"):
        build(cameras=["cam_head"])
    model, _deployment = build(cameras=["cam_right_wrist", "cam_left_wrist", "cam_head"])
    assert model.cameras == ["cam_right_wrist", "cam_left_wrist", "cam_head"]
