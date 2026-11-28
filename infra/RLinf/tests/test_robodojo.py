"""CPU contract tests; the simulator boundary is a deterministic test double."""

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from XPolicyLab.infra.RLinf.envs.robodojo import runtime
from XPolicyLab.infra.RLinf.envs.robodojo.extension import ENV_ID
from XPolicyLab.infra.RLinf.envs.robodojo.env import (
    RoboDojoEnv,
    pack_observations,
)
from XPolicyLab.infra.RLinf.envs.robodojo.process import SimulatorProcess
from XPolicyLab.utils.process_data import unpack_robot_state

DIMS = {"arm_dim": [6, 6], "ee_dim": [1, 1]}


def observation(value=0):
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 255  # red must stay red
    return {
        "instruction": "Stack the bowls.",
        "state": unpack_robot_state(
            np.full(14, value, dtype=np.float32), "joint", DIMS
        ),
        "vision": {
            k: {"color": rgb.copy()}
            for k in ("cam_head", "cam_left_wrist", "cam_right_wrist")
        },
    }


class Backend:
    metadata = {"robot_action_dim_info": DIMS, "num_layouts": 7}

    def __init__(self, script):
        self.script, self.calls, self.closed = script, [], False

    def snapshot(self, row=None):
        row = row or {}
        n = len(self.layout_ids)
        return {
            "observations": [observation(self.index) for _ in range(n)],
            "reward": row.get("reward", [0.0] * n),
            "official_score": row.get("official_score", [0.0] * n),
            "success": row.get("success", [False] * n),
            "failed": [False] * n,
            "truncated": row.get("truncated", [False] * n),
        }

    def reset(self, layout_ids):
        self.layout_ids = layout_ids
        self.index = 0
        return self.snapshot()

    def step(self, actions, env_indices):
        self.calls.append((deepcopy(actions), list(env_indices)))
        row = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        return self.snapshot(row)

    def close(self):
        self.closed = True


def env(script=None, **kwargs):
    cfg = OmegaConf.create(dict(max_episode_steps=4, **kwargs))
    backend = Backend(script or [{}])
    return RoboDojoEnv(cfg, 2, backend=backend), backend


def test_partial_success_masks_actions_and_caches_terminal_obs():
    task, backend = env([{"reward": [1, 0], "success": [True, False]}, {}, {}, {"truncated": [False, True]}])
    task.reset()
    actions = torch.arange(2 * 3 * 14).reshape(2, 3, 14).float()
    obs, rewards, term, trunc, infos = task.chunk_step(actions)
    assert [indices for _, indices in backend.calls] == [[0, 1], [1], [1]]
    np.testing.assert_array_equal(
        backend.calls[0][0][0]["left_arm_joint_state"], np.arange(6)
    )
    assert backend.calls[0][0][0]["left_ee_joint_state"][0] == 6
    assert term[0].all() and not term[1].any() and not trunc.any()
    assert rewards.sum() == 1 and rewards[0, 0] == 1
    assert (
        obs[-1]["states"][0, 0] == 1
    )  # terminal frame, not later shared-physics frames
    assert obs[-1]["states"][1, 0] == 3
    assert infos[-1]["final_observation"] is obs[-1]
    _, rewards, _, trunc, _ = task.chunk_step(actions)
    assert rewards.sum() == 0
    assert trunc[1].all()
    assert len(backend.calls) == 4
    task.close()
    assert backend.closed


def test_forwards_each_native_reward_and_masks_finished_slots():
    task, _ = env([
        {"reward": [0.25, 0.0], "official_score": [85, 15]},
        {"reward": [0.0, -0.5], "success": [True, False]},
        {"reward": [1.0, 0.75], "truncated": [False, True]},
    ])
    task.reset()
    _, rewards, term, trunc, infos = task.chunk_step(torch.zeros(2, 3, 14))
    assert rewards.tolist() == [[0.25, 0.0, 0.0], [0.0, -0.5, 0.75]]
    assert term.tolist() == [[False, True, True], [False, False, False]]
    assert trunc.tolist() == [[False, False, False], [False, False, True]]
    assert infos[-1]["episode"]["return"].tolist() == [0.25, 0.25]
    task.reset()
    assert task.returns.tolist() == [0.0, 0.0]


def test_requires_explicit_native_reward():
    task, backend = env()
    task.reset()
    row = backend.snapshot({"success": [True, False]})
    del row["reward"]
    backend.step = lambda actions, indices: row
    with pytest.raises(KeyError, match="reward"):
        task.chunk_step(torch.zeros(2, 1, 14))


def test_reset_group_ids_and_no_partial_reset():
    task, backend = env(group_size=2)
    task.reset(seed=12)
    assert backend.layout_ids[0] == backend.layout_ids[1]
    ids = backend.layout_ids.copy()
    task.reset(seed=12)
    assert backend.layout_ids == ids
    with pytest.raises(NotImplementedError):
        task.reset(env_ids=[0])


@pytest.mark.parametrize("slots,group,layouts,epochs", [(256, 1, 110, 4), (64, 8, 110, 8), (128, 1, 55, 3)])
def test_layouts_cover_the_full_pool_across_workers(slots, group, layouts, epochs):
    ranks = 8
    workers = []
    for rank in range(ranks):
        backend = Backend([{}])
        backend.metadata = {**backend.metadata, "num_layouts": layouts}
        n = slots // ranks
        cfg = OmegaConf.create(dict(max_episode_steps=800, group_size=group))
        workers.append(RoboDojoEnv(cfg, n, seed_offset=rank, total_num_processes=ranks, backend=backend))
    seen = []
    for _ in range(epochs):
        batch = []
        for worker in workers:
            worker.reset()
            ids = np.array(worker.backend.layout_ids).reshape(-1, group)
            assert (ids == ids[:, :1]).all()
            batch.extend(ids[:, 0])
            worker.update_reset_state_ids()  # RLinf calls this between resets.
        counts = np.bincount(batch, minlength=layouts)
        assert counts.max() - counts.min() <= 1
        seen.extend(batch)
    counts = np.bincount(seen, minlength=layouts)
    assert counts.max() - counts.min() <= 1


def test_layout_directories_keep_native_file_order_and_distinct_paths():
    class Manager:
        task_name = "stack_bowls"

        def __init__(self, config):
            self.config = config

        def init_eval(self):
            seed = self.config["seed"]
            self.seed_list = [0, 1]
            self.seed_info = {i: {"scene_layout": f"{seed}/stack_bowls_{i}.json"} for i in self.seed_list}

    manager = Manager({"seed": 0})
    manager.init_eval()
    assert runtime._combine_layouts(manager, [0, 2]) == 4
    assert [manager.seed_info[i]["scene_layout"] for i in manager.seed_list] == [
        "0/stack_bowls_0.json", "0/stack_bowls_1.json", "2/stack_bowls_0.json", "2/stack_bowls_1.json",
    ]
    assert runtime._layout_seeds(1) == [1]
    with pytest.raises(ValueError, match="duplicate"):
        runtime._layout_seeds([0, 0])


@pytest.mark.parametrize(
    "actions", [torch.zeros(2, 3, 13), torch.full((2, 3, 14), float("nan"))]
)
def test_bad_actions_do_not_reach_simulator(actions):
    task, backend = env()
    task.reset()
    with pytest.raises(ValueError):
        task.chunk_step(actions)
    assert not backend.calls


def test_native_instructions_are_preserved_per_slot_without_optional_score():
    task, backend = env()
    snapshot = backend.snapshot

    def native_snapshot(row=None):
        result = snapshot(row)
        result.pop("official_score")
        for obs, text in zip(result["observations"], ["Align the cubes.", "Push the red cube."]):
            obs["instruction"] = np.str_(text)
        return result

    backend.snapshot = native_snapshot
    obs, _ = task.reset()
    assert obs["task_descriptions"] == ["Align the cubes.", "Push the red cube."]
    next_obs, _, _, _, infos = task.chunk_step(torch.zeros(2, 1, 14))
    assert next_obs[-1]["task_descriptions"] == obs["task_descriptions"]
    assert "official_score" not in infos[-1]["episode"]


def test_rgb_and_joint_roundtrip():
    packed = pack_observations([observation(2)], DIMS)
    assert packed["states"].shape == (1, 14)
    assert packed["main_images"][0, 0, 0].tolist() == [255, 0, 0]
    assert packed["wrist_images"].shape == (1, 2, 8, 8, 3)


def test_ee_observation_and_action_reach_native_pose_control(monkeypatch):
    task, backend = env(action_type="ee")
    native = observation()
    poses = np.array([0.3, 0.1, 0.2, 1, 0, 0, 0, 0.5] * 2, dtype=np.float32)
    native["state"].update(unpack_robot_state(poses, "ee", DIMS))
    snapshot = backend.snapshot

    def ee_snapshot(row=None):
        result = snapshot(row)
        result["observations"] = [native, native]
        return result

    monkeypatch.setattr(backend, "snapshot", ee_snapshot)
    obs, _ = task.reset()
    np.testing.assert_array_equal(obs["states"][0], poses)
    actions = obs["states"].unsqueeze(1).clone()
    actions[0, 0, 0] += 0.01
    task.chunk_step(actions)
    sent = backend.calls[0][0][0]
    assert "left_arm_joint_state" not in sent
    np.testing.assert_array_equal(sent["left_ee_pose"], actions[0, 0, :7])
    np.testing.assert_array_equal(sent["right_ee_pose"], poses[8:15])


def test_registry_and_native_action_passthrough():
    from XPolicyLab.infra.RLinf.register import register
    from rlinf.envs import get_env_cls
    from rlinf.envs.action_utils import prepare_actions

    register()
    cfg = OmegaConf.create({"init_params": {"id": ENV_ID}})
    assert get_env_cls("isaaclab", cfg) is RoboDojoEnv
    actions = torch.rand(2, 3, 14)
    result = prepare_actions(actions, "isaaclab", "xpolicylab", 3, 14)
    torch.testing.assert_close(result, actions)


def test_native_timeout_bootstrap_distinguishes_success():
    from rlinf.data.schema.embodied_types import EnvPart, EnvTransition
    from XPolicyLab.infra.RLinf.envs.robodojo.rollout import bootstrap_robodojo_timeout

    trunc = torch.tensor([[False, True], [False, False]])
    term = torch.tensor([[False, False], [True, True]])
    transition = EnvTransition(
        rewards=torch.tensor([[0.0, 0.2], [1.0, 0.0]]),
        dones=trunc | term,
        truncations=trunc,
        terminations=term,
    )
    part = EnvPart(
        sources=[], transition=transition, bootstrap_values=torch.tensor([[2.0], [9.0]])
    )
    result = bootstrap_robodojo_timeout(part, 0.9)
    torch.testing.assert_close(
        result.transition.rewards, torch.tensor([[0.0, 2.0], [1.0, 0.0]])
    )
    assert part.transition.rewards[0, 1] == 0.2  # transport data not mutated
    # Native v0.4 does not add the bonus a second time when auto_reset=false.
    native = result.compute_rewards(
        env_reward_weight=1.0,
        reward_weight=0.0,
        auto_reset=False,
        bootstrap_type="standard",
        gamma=0.9,
    )
    torch.testing.assert_close(native, result.transition.rewards)


def test_simulator_import_does_not_require_policy_runtime():
    root = Path(__file__).resolve().parents[4]
    code = """
import sys
from XPolicyLab.infra.RLinf.envs.robodojo.process import SimulatorProcess
assert 'rlinf' not in sys.modules
assert 'transformers' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, PYTHONPATH=str(root)),
        check=True,
    )


@pytest.fixture
def simulator_factory(tmp_path):
    (tmp_path / "env").mkdir()
    (tmp_path / "env" / "global_configs.py").write_text("BENCHMARK = 'RoboDojo'\n")
    (tmp_path / "fake_sim.py").write_text("""
import os
import sys
import time
from env.global_configs import BENCHMARK
from infra.RLinf.envs.robodojo.runtime import SimulatorBackend
class Backend:
    def __init__(self, cfg): self.cfg = cfg
    def metadata(self):
        return {"benchmark": BENCHMARK, "torch_loaded": "torch" in sys.modules, "args": sys.argv[3:]}
    def reset(self, ids):
        if self.cfg.get("crash"): os._exit(3)
        if self.cfg.get("raise"): raise ValueError("bad layout")
        if self.cfg.get("sleep"): time.sleep(10)
        print("Simulator stdout does not corrupt RPC", flush=True)
        return ids
    def close(self): pass
""")
    return tmp_path


@pytest.mark.parametrize("render_gpu,kit_args", [(None, []), (7, ["--/renderer/activeGpu=7"])])
def test_rpc_separate_process_and_clean_shutdown(simulator_factory, render_gpu, kit_args):
    sim = SimulatorProcess(
        sys.executable, simulator_factory, {}, timeout=2, factory="fake_sim:Backend", render_gpu=render_gpu
    )
    assert not sim.metadata["torch_loaded"]
    assert sim.metadata["benchmark"] == "RoboDojo"
    assert sim.metadata["args"] == kit_args
    assert sim.reset([1, 2]) == [1, 2]
    sim.close()
    assert sim.process.poll() == 0
    sim.close()


@pytest.mark.parametrize(
    "failure, error",
    [("raise", RuntimeError), ("crash", RuntimeError), ("sleep", TimeoutError)],
)
def test_rpc_errors_and_timeouts_reap_child(simulator_factory, failure, error):
    sim = SimulatorProcess(
        sys.executable,
        simulator_factory,
        {failure: True},
        timeout=0.5,
        factory="fake_sim:Backend",
    )
    with pytest.raises(error):
        sim.reset([1])
    assert sim.process.poll() is not None


def test_rollout_publishes_native_terminal_value_with_timeout_bonus():
    from rlinf.data.schema.embodied_types import EnvPart, EnvTransition
    from XPolicyLab.infra.RLinf.envs.robodojo.rollout import RoboDojoRolloutWorker
    from types import SimpleNamespace

    worker = RoboDojoRolloutWorker.__new__(RoboDojoRolloutWorker)
    worker.cfg = OmegaConf.create({"env": {"train": {"init_params": {"id": ENV_ID}}}})
    worker.model_cfg = OmegaConf.create({"model_type": "openpi", "add_value_head": True})
    worker.algorithm_cfg = OmegaConf.create({"gamma": 0.9})
    worker.collect_transitions = False
    worker.enable_rlt = False
    received = []
    worker._predict_rollout_actions = lambda obs: (None, {"prev_values": obs["value"]})
    part = EnvPart(
        sources=[],
        requires_inference=True,
        next_obs={"value": torch.tensor([[2.0]])},
        transition=EnvTransition(
            rewards=torch.zeros(1, 2), truncations=torch.tensor([[False, True]])
        ),
    )
    worker._publish_env_part(
        part, None, None, SimpleNamespace(put=lambda p, **kwargs: received.append(p))
    )
    assert received[0].transition.rewards[0, 1] == pytest.approx(1.8)
    assert received[0].final_prev_values.item() == 2.0


def test_env_worker_closes_train_and_eval_even_after_one_error():
    from XPolicyLab.infra.RLinf.workers.env import XPolicyLabEnvWorker
    from types import SimpleNamespace

    closed = []

    def fail():
        closed.append("train")
        raise ValueError("cleanup error")

    train = SimpleNamespace(close=fail)
    worker = SimpleNamespace(
        env_list=[train],
        eval_env_list=[train, SimpleNamespace(close=lambda: closed.append("eval"))],
    )
    with pytest.raises(RuntimeError, match="cleanup error"):
        XPolicyLabEnvWorker.close_environments(worker)
    assert closed == ["train", "eval"]


def test_native_loss_mask_keeps_terminal_action_and_masks_padding():
    from rlinf.utils.metric_utils import compute_loss_mask

    task, _ = env([{"reward": [1, 0], "success": [True, False]}, {}, {}, {"truncated": [False, True]}])
    task.reset()
    _, _, term1, trunc1, _ = task.chunk_step(torch.zeros(2, 3, 14))
    _, _, term2, trunc2, _ = task.chunk_step(torch.zeros(2, 3, 14))
    dones = torch.stack([torch.zeros_like(term1), term1 | trunc1, term2 | trunc2])
    mask, _ = compute_loss_mask(dones)
    assert mask[:, 0].tolist() == [[True, False, False], [False, False, False]]
    assert mask[:, 1].tolist() == [[True, True, True], [True, False, False]]
    assert mask.any(dim=-1)[:, 0].tolist() == [True, False]


def test_launcher_receives_camera_fence_before_environment_init(monkeypatch):
    order = []
    settings = {}
    app = SimpleNamespace(close=lambda: order.append("closed"))

    def launch(**kwargs):
        order.append("launch")
        settings.update(kwargs)
        return SimpleNamespace(app=app)

    sync = ModuleType("env.camera_manager.capture.render_sync")
    sync.zero_delay_kit_args = lambda: "--camera-fence"
    monkeypatch.setitem(sys.modules, sync.__name__, sync)
    module = ModuleType("isaaclab.app")
    module.AppLauncher = launch
    monkeypatch.setitem(sys.modules, "isaaclab", ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.app", module)
    monkeypatch.setattr(runtime.SimulatorBackend, "_initialize", lambda self, cfg: order.append("initialize"))
    backend = runtime.SimulatorBackend({})
    backend.close()
    assert order == ["launch", "initialize", "closed"]
    assert settings["multi_gpu"] is False
    assert settings["device"] == "cuda:0"
    assert settings["kit_args"].split() == ["--/renderer/multiGpu/enabled=false", "--camera-fence"]


def test_bridge_forwards_native_reward_without_a_second_final_check():
    backend = object.__new__(runtime.SimulatorBackend)
    calls = []
    rewards = [1.0, 0.0, 0.0, 0.0]
    backend.env = SimpleNamespace(
        num_envs=4,
        end_flag=[False] * 4,
        success=[True] * 4,
        step_lim=2,
        take_action_cnt=[0] * 4,
        reset=lambda seed: calls.append(("reset", seed)),
        run_reward=lambda: calls.append("run_reward"),
        get_score=lambda: calls.append("get_score"),
        get_obs_batch=lambda last_frame: [{}] * 4,
        reward_manager=SimpleNamespace(get_score=lambda: [100, 15, 15, 0]),
    )

    def get_reward(*, final_check):
        calls.append(("get_reward", final_check))
        return rewards

    backend.env.reward_manager.get_reward = get_reward
    backend._observe_rewards()

    def take_action_batch(actions, env_idx_list):
        calls.append((actions, env_idx_list))
        backend.env.end_flag = [True, False, True, True]
        backend.env.success = [True, True, False, False]
        backend.env.take_action_cnt = [1, 1, 1, 2]
        backend.env.reward_manager.get_reward(final_check=True)

    backend.env.take_action_batch = take_action_batch
    ids = list(range(4))
    assert backend.reset(ids)["reward"] == [0.0] * 4
    result = backend.step(["action"] * 4, ids)
    assert calls == [("reset", ids), "run_reward", "get_score", (["action"] * 4, ids), ("get_reward", True)]
    assert result["reward"] == rewards
    assert result["success"].tolist() == [True, False, False, False]
    assert result["failed"].tolist() == [False, False, True, False]
    assert result["truncated"].tolist() == [False, False, False, True]
    assert result["official_score"] == [100, 15, 15, 0]

    del backend.env.get_score
    backend.env.end_flag = [False] * 4
    assert "official_score" not in backend.reset(ids)

    backend.env.take_action_batch = lambda *args, **kwargs: None
    with pytest.raises(RuntimeError, match="without evaluating its reward"):
        backend.step(["action"] * 4, ids)
