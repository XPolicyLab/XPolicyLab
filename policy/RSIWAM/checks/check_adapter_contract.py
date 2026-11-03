"""CPU-only boundary checks in a copied XPolicyLab fixture; never loads VPP2.

The real shared checkpoint resolver, robot metadata reader, vendored wrapper,
and bundle hash validator are used. The neural runtime and torch are explicit
test doubles. This is not a server/debug-loop or CUDA parity test.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace

sys.dont_write_bytecode = True
POLICY_DIR = Path(__file__).resolve().parents[1]


def expect_error(kind, callback):
    try:
        callback()
    except kind:
        return
    raise AssertionError(f"Expected {kind.__name__}")


def fixture_checks(workspace):
    import numpy as np

    # Some small CPU audit environments omit h5py. The actual shared JSON/YAML
    # readers do not use it; reject any accidental HDF5 operation in this test.
    hdf5_import_stub = importlib.util.find_spec("h5py") is None
    if hdf5_import_stub:
        unused_hdf5 = ModuleType("h5py")

        def unexpected_hdf5(name):
            raise AssertionError(f"This boundary check must not use h5py.{name}")

        unused_hdf5.__getattr__ = unexpected_hdf5
        sys.modules["h5py"] = unused_hdf5
    adapter = importlib.import_module("XPolicyLab.policy.RSIWAM.model")
    deploy = importlib.import_module("XPolicyLab.policy.RSIWAM.deploy")
    assert adapter.POLICY_DIR == workspace / "XPolicyLab/policy/RSIWAM"
    assert not any(part.is_symlink() for part in adapter.POLICY_DIR.rglob("*"))
    from XPolicyLab.model_template import ModelTemplate
    assert issubclass(adapter.Model, ModelTemplate)
    hashes = adapter.rsiwam_loader.check_package()
    assert hashes["commit"]["step"] == 400
    cfg = dict(bench_name="RoboDojo", env_cfg_type="arx_x5", action_type="ee",
               ckpt_name="rsiwam-original32-lora33-step400", seed=7)
    prepared, dimensions, evaluation_seed = adapter.prepare_configuration(cfg)
    assert evaluation_seed == 7 and prepared["model_seed"] == 1
    assert dimensions == {"arm_dim": [6, 6], "ee_dim": [1, 1]}
    assert prepared["action_dim"] == 16
    for override in [dict(model_seed=7), dict(action_type="joint"), dict(eval_batch=True),
                     dict(env_cfg_type="other"), dict(action_dim=14), dict(seed=-1)]:
        expect_error(ValueError, lambda override=override: adapter.prepare_configuration(dict(cfg, **override)))
    robot_file = workspace / "env_cfg/robot/_robot_info.json"
    original = robot_file.read_text()
    robot_file.write_text(json.dumps({"arx_x5": {"arm_dim": [6], "ee_dim": [1]}}))
    expect_error(ValueError, lambda: adapter.prepare_configuration(cfg))
    robot_file.write_text(original)

    # All path selection uses the actual XPolicyLab resolver. A fixed fallback
    # root works across evaluation seeds; explicit relative paths are portable.
    explicit = adapter.POLICY_DIR / "explicit assets"
    explicit.mkdir()
    fallback = adapter.POLICY_DIR / "checkpoints" / cfg["ckpt_name"]
    fallback.mkdir(parents=True)
    os.environ.pop("RSIWAM_BASE_WEIGHTS", None)
    assert adapter.resolve_base_root(dict(cfg, rsiwam_weights_root="explicit assets")) == explicit
    assert adapter.resolve_base_root(dict(cfg, ckpt_name="./explicit assets")) == explicit
    for seed in [1, 7, 42]:
        assert adapter.resolve_base_root(dict(cfg, seed=seed)) == fallback
    os.environ["RSIWAM_BASE_WEIGHTS"] = "explicit assets"
    assert adapter.resolve_base_root(cfg) == explicit
    os.environ.pop("RSIWAM_BASE_WEIGHTS")
    from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name
    run_root = adapter.POLICY_DIR / "checkpoints" / build_run_dir_name(cfg)
    run_root.mkdir()
    assert adapter.resolve_base_root(cfg) == run_root

    received_configs = []

    class FakeRuntime:
        action_horizon, replan_steps, history_condition_frames = 32, 24, 8

        def __init__(self):
            self.model = SimpleNamespace(requires_grad_=lambda value: None, eval=lambda: None)
            self._history_step_by_env = {}
            self.seen = []
            self.inferences = []

        def observe(self, obs):
            env = obs["env_idx"]
            self._history_step_by_env[env] = self._history_step_by_env.get(env, -1) + 1
            self.seen.append(obs)

        def _infer_action_chunk(self, obs, instruction):
            self.inferences.append((obs["env_idx"], self._history_step_by_env[obs["env_idx"]], instruction))
            return np.arange(32 * 16, dtype=np.float32).reshape(32, 16)

        def reset_env(self, env):
            self._history_step_by_env.pop(env, None)

        def reset(self):
            self._history_step_by_env.clear()

    def get_model(config):
        received_configs.append(dict(config))
        return FakeRuntime()

    # Test doubles are scoped to this subprocess; no full neural import occurs.
    recoverwam = ModuleType("recoverwam")
    recoverwam.__file__ = str(adapter.BUNDLE_DIR / "src/recoverwam/__init__.py")
    policy = ModuleType("recoverwam.policy")
    policy.get_model = get_model
    sys.modules.update({"recoverwam": recoverwam, "recoverwam.policy": policy})
    torch = ModuleType("torch")
    seeds = []
    torch.set_num_threads = lambda value: None
    torch.manual_seed = seeds.append
    torch.backends = SimpleNamespace(cudnn=SimpleNamespace())
    torch.use_deterministic_algorithms = lambda value: None
    torch.utils = SimpleNamespace(deterministic=SimpleNamespace())
    torch.inference_mode = nullcontext
    sys.modules["torch"] = torch
    compose_calls = []
    adapter.rsiwam_loader.check_base = lambda path: Path(path)
    adapter.rsiwam_loader.compose = lambda model, artifacts: compose_calls.append(artifacts["commit"]["identity"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    for seed in [1, 7, 42]:
        model = adapter.Model(dict(cfg, seed=seed))
        assert model.evaluation_seed == seed
        assert received_configs[-1]["seed"] == 1
        assert received_configs[-1]["action_dim"] == 16
    assert seeds == [1, 1, 1]
    assert len(compose_calls) == 3
    expect_error(NotImplementedError, lambda: model.update_obs_batch([]))
    expect_error(NotImplementedError, model.get_action_batch)

    def observation(step, client=None):
        image = np.zeros((8, 11, 3), dtype=np.uint8)
        image[:, :, 0] = 237
        image[:, :, 1] = 51
        image[:, :, 2] = 7
        image.flags.writeable = False
        obs = {
            "vision": {camera: {"color": image} for camera in ["cam_head", "cam_left_wrist", "cam_right_wrist"]},
            "state": {"left_ee_pose": np.arange(7, dtype=np.float32),
                      "left_ee_joint_state": np.array([0.3], dtype=np.float32),
                      "right_ee_pose": np.arange(7, 14, dtype=np.float32),
                      "right_ee_joint_state": np.array([0.9], dtype=np.float32)},
            "instruction": f"step {step}",
            "env_idx": 0,
        }
        if client is not None:
            obs["_recoverwam_client_id"] = client
        return obs

    # State/action slicing must be byte-identical to the unchanged native EE16
    # implementation for the supported registered robot.
    raw = observation(0, "client-A")
    from XPolicyLab.policy.RSIWAM.bundle.runtime.rsiwam_native_wrapper import Model as Native
    np.testing.assert_array_equal(model._native._pack_robodojo_ee16_state(raw), Native._pack_robodojo_ee16_state(raw))
    action_rows = np.arange(48, dtype=np.float32).reshape(3, 16)
    old_actions = Native._unpack_robodojo_ee16_actions(action_rows)
    new_actions = model._native._unpack_robodojo_ee16_actions(action_rows)
    for old, new in zip(old_actions, new_actions):
        assert old.keys() == new.keys()
        for key in old:
            np.testing.assert_array_equal(old[key], new[key])

    model.update_obs(raw)
    encoded = model.model.seen[-1]
    for view in encoded["observation"].values():
        np.testing.assert_array_equal(view["rgb"], raw["vision"]["cam_head"]["color"])
        assert view["rgb"].shape == (8, 11, 3)
    first_env = encoded["env_idx"]
    model.get_action({"_recoverwam_client_id": "client-A"})
    model.update_obs(observation(0, "client-B"))
    second_env = model.model.seen[-1]["env_idx"]
    assert first_env != second_env
    model.reset_client({"_recoverwam_client_id": "client-A"})
    assert first_env not in model.model._history_step_by_env
    assert second_env in model.model._history_step_by_env
    model.get_action({"_recoverwam_client_id": "client-B"})
    malformed = observation(0, "client-C")
    malformed["vision"]["cam_head"]["color"] = b"encoded-image-is-server-responsibility"
    expect_error(ValueError, lambda: model.update_obs(malformed))

    class Client:
        def __init__(self, instance):
            self.instance = instance
            self.calls = []

        def call(self, func_name, **kwargs):
            self.calls.append(func_name)
            return getattr(self.instance, func_name)(**kwargs)

    class Environment:
        def __init__(self, length):
            self.step, self.length = 0, length

        def is_episode_end(self):
            return self.step >= self.length

        def get_obs(self):
            assert not self.is_episode_end(), "Read beyond the episode terminal state"
            return observation(self.step)

        def take_action(self, action):
            assert not self.is_episode_end(), "Action beyond the episode terminal state"
            assert list(action) == ["left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state"]
            assert [len(value) for value in action.values()] == [7, 1, 7, 1]
            self.step += 1

    # Reusing one client across episodes verifies reset and in-chunk termination.
    client = Client(model)
    model.model.inferences.clear()
    before = len(model.model.seen)
    episode = Environment(50)
    deploy.eval_one_episode(episode, client)
    assert episode.step == 50 and len(model.model.seen) - before == 50
    assert [entry[1] for entry in model.model.inferences] == [0, 24, 48]
    assert client.calls[:2] == ["reset", "reset_client"]
    before = len(model.model.seen)
    deploy.eval_one_episode(Environment(3), client)
    assert len(model.model.seen) - before == 3
    assert model.model.inferences[-1][1] == 0
    assert model._native.history_guard_mismatches == 0

    class EmptyClient:
        def call(self, func_name, **kwargs):
            return [] if func_name == "get_action" else None

    expect_error(ValueError, lambda: deploy.eval_one_episode(Environment(1), EmptyClient()))

    # Capture the real shell entrypoint through fake conda/python. No server or
    # environment client is launched, and no activated environment is changed.
    fake_conda = workspace / "fake-conda"
    (fake_conda / "bin").mkdir(parents=True)
    (fake_conda / "etc/profile.d").mkdir(parents=True)
    (fake_conda / "bin/conda").write_text('#!/bin/bash\nprintf "%s\\n" "$FAKE_CONDA_ROOT"\n')
    (fake_conda / "bin/conda").chmod(0o755)
    (fake_conda / "etc/profile.d/conda.sh").write_text('conda() { export CONDA_PREFIX="$FAKE_CONDA_ROOT"; }\n')
    (fake_conda / "bin/python").write_text(
        f'#!{sys.executable}\nimport json, os, sys\n'
        'print(json.dumps({"args":sys.argv[1:], "cublas":os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "gpu":os.environ.get("CUDA_VISIBLE_DEVICES")}))\n')
    (fake_conda / "bin/python").chmod(0o755)
    env = dict(os.environ, PATH=str(fake_conda / "bin") + os.pathsep + os.environ["PATH"], FAKE_CONDA_ROOT=str(fake_conda))
    server_script = adapter.POLICY_DIR / "setup_eval_policy_server.sh"
    for seed in [1, 7, 42]:
        command = ["bash", str(server_script), "RoboDojo", "stack_bowls", cfg["ckpt_name"],
                   "arx_x5", "ee", str(seed), "2", "FAKE_ENV", "19960", "localhost"]
        call = json.loads(subprocess.check_output(command, env=env, text=True))
        assert f"seed={seed}" in call["args"]
        assert not any(arg.startswith("model_seed=") for arg in call["args"])
        assert call["cublas"] == ":4096:8" and call["gpu"] == "2"
        assert call["args"][1] == str(workspace / "XPolicyLab/setup_policy_server.py")
    command[6] = "joint"
    assert subprocess.run(command, env=env, capture_output=True).returncode != 0
    return {
        "status": "CPU_ADAPTER_BOUNDARY_CHECKS_PASSED",
        "normal_xpolicylab_import": True,
        "copied_directory_no_symlink": True,
        "bundle_hash_and_checkpoint_identity": "PASS",
        "shared_checkpoint_resolver": "PASS",
        "shared_robot_dimensions_and_native_pack_parity": "PASS",
        "readonly_rgb_no_swap_or_resize": "PASS",
        "per_action_history_client_isolation_and_episode_reset": "PASS",
        "mid_chunk_terminal_stop": "PASS",
        "environment_seeds": [1, 7, 42],
        "fixed_model_seed": 1,
        "shell_seed_and_cuda_environment": "PASS",
        "neural_runtime_and_torch": "EXPLICIT_TEST_DOUBLES",
        "unused_hdf5_import_stub": hdf5_import_stub,
        "full_base_assets_loaded": False,
        "gpu_used": False,
        "server_started": False,
        "debug_closed_loop_run": False,
        "clean_environment_install_tested": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xpl-root", type=Path, default=POLICY_DIR.parents[1])
    parser.add_argument("--fixture", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.fixture:
        print(json.dumps(fixture_checks(args.fixture.resolve()), indent=2))
        return
    xpl = args.xpl_root.resolve()
    required = ["model_template.py", "utils/checkpoint_resolver.py", "utils/process_data.py", "utils/load_file.py"]
    for name in required:
        if not (xpl / name).is_file():
            parser.error(f"Missing XPolicyLab reference file: {xpl / name}")
    for path in POLICY_DIR.glob("*.py"):
        compile(path.read_text(), str(path), "exec")
    for path in POLICY_DIR.glob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)
    with tempfile.TemporaryDirectory(prefix="rsiwam-xpl-cpu-") as temporary:
        workspace = Path(temporary)
        fixture = workspace / "XPolicyLab"
        (fixture / "policy").mkdir(parents=True)
        (fixture / "utils").mkdir()
        for relative in ["__init__.py", "policy/__init__.py", "utils/__init__.py"]:
            (fixture / relative).write_text("")
        for relative in required:
            shutil.copy2(xpl / relative, fixture / relative)
        (fixture / "setup_policy_server.py").write_text("raise RuntimeError('CPU fixture must not start a server')\n")
        shutil.copytree(POLICY_DIR, fixture / "policy/RSIWAM", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
        (workspace / "env_cfg/robot").mkdir(parents=True)
        (workspace / "env_cfg/arx_x5.yml").write_text("config:\n  robot: arx_x5\n")
        (workspace / "env_cfg/robot/_robot_info.json").write_text(json.dumps({"arx_x5": {"arm_dim": [6, 6], "ee_dim": [1, 1]}}))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(workspace), CUDA_VISIBLE_DEVICES="")
        child = fixture / "policy/RSIWAM/checks/check_adapter_contract.py"
        subprocess.run([sys.executable, str(child), "--fixture", str(workspace)], cwd=workspace, env=env, check=True)


if __name__ == "__main__":
    main()
