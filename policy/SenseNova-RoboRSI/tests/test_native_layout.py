"""CPU-only checks that explicit resume identity reaches actual native reset."""

from __future__ import annotations

from importlib import import_module

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

main = import_module("XPolicyLab.policy.SenseNova-RoboRSI.native_entry").main
wrap_eval_module = import_module("XPolicyLab.policy.SenseNova-RoboRSI.native_entry").wrap_eval_module
NativeLayoutError = import_module("XPolicyLab.policy.SenseNova-RoboRSI.native_layout").NativeLayoutError
NativeSelection = import_module("XPolicyLab.policy.SenseNova-RoboRSI.native_layout").NativeSelection
bind_native_environment = import_module("XPolicyLab.policy.SenseNova-RoboRSI.native_layout").bind_native_environment


class NativeFixture:
    def __init__(self, tmp_path):
        self.task_name, self.eval_seed, self.run_id = "align_blocks", 0, "fixture"
        self.num_envs, self.step_lim = 1, 600
        self.take_action_cnt = [0]
        self.calls = []
        info = {}
        for layout in range(3):
            path = tmp_path / f"align_blocks_{layout}.json"
            path.write_text(json.dumps({"object_pose": layout}))
            info[layout] = {"scene_layout": str(path)}
        self.seed_manager = SimpleNamespace(
            seed_list=[0, 1, 2], seed_info=info,
            st_idx=0, idx=0, ed_idx=3, _current_batch_seeds=None,
        )

    def reset(self, seed=None, options=None):
        self.calls.append((list(seed), options))
        self.env_seeds = list(seed)
        self.current_env_seed_map = {i: value for i, value in enumerate(seed) if value is not None}
        self.take_action_cnt = [0]

    def get_obs_batch(self, env_idx_list=None, last_frame=False):
        return [{"env_idx": 0, "vision": {}}]


def test_layout_physically_selected_with_observed_scene_and_steps(tmp_path):
    env = NativeFixture(tmp_path)
    expected = hashlib.sha256(Path(env.seed_manager.seed_info[2]["scene_layout"]).read_bytes()).hexdigest()
    bind_native_environment(env, NativeSelection("align_blocks", 0, 2, "fixture", expected))
    assert env.seed_manager.seed_list == [2]
    env.reset(seed=env.seed_manager.seed_list)
    env.take_action_cnt[0] = 7
    receipt = env.get_obs_batch()[0]["roborsi_native_identity"]
    assert env.calls == [([2], None)]
    assert receipt["task"] == "align_blocks"
    assert receipt["eval_seed"] == 0 and receipt["layout"] == 2
    assert receipt["scene_sha256"] == expected
    assert receipt["control_step"] == 7 and receipt["step_limit"] == 600
    with pytest.raises(SystemExit, match="unrelated reset"):
        env.reset(seed=[0])


def test_default_queue_and_cross_layout_resets_are_unchanged(tmp_path):
    env = NativeFixture(tmp_path)
    bind_native_environment(env, NativeSelection("align_blocks", 0))
    assert env.seed_manager.seed_list == [0, 1, 2]
    for layout in (0, 1, 2):
        env.reset(seed=[layout])
        assert env.get_obs_batch()[0]["roborsi_native_identity"]["layout"] == layout


@pytest.mark.parametrize("field", ["completed_layout_ids", "abandoned_layout_ids", "details"])
def test_already_accounted_layout_is_never_scored_twice(tmp_path, field):
    env = NativeFixture(tmp_path)
    state = {"task_name": "align_blocks", "eval_seed": 0, "run_id": "fixture",
             "completed_layout_ids": [], "abandoned_layout_ids": [], "details": {}}
    state[field] = {"0": {"layout_id": 2, "score": 0}} if field == "details" else [2]
    original = json.loads(json.dumps(state))
    with pytest.raises(NativeLayoutError, match="completed or abandoned"):
        bind_native_environment(env, NativeSelection("align_blocks", 0, 2, "fixture"), resume_state=state)
    assert state == original
    assert env.calls == []


@pytest.mark.parametrize("task,seed,run", [("other", 0, "fixture"), ("align_blocks", 1, "fixture"), ("align_blocks", 0, "other")])
def test_wrong_episode_identity_fails_before_reset(tmp_path, task, seed, run):
    env = NativeFixture(tmp_path)
    with pytest.raises(NativeLayoutError):
        bind_native_environment(env, NativeSelection(task, seed, 1, run))
    assert env.calls == []


def test_wrong_scene_or_unverified_reset_does_not_publish_receipt(tmp_path):
    env = NativeFixture(tmp_path)
    with pytest.raises(NativeLayoutError, match="SHA"):
        bind_native_environment(env, NativeSelection("align_blocks", 0, 1, scene_sha256="0" * 64))
    env = NativeFixture(tmp_path)
    bind_native_environment(env, NativeSelection("align_blocks", 0, 1))
    with pytest.raises(SystemExit, match="precedes verified"):
        env.get_obs_batch()
    with pytest.raises(SystemExit, match="unrelated reset"):
        env.reset(seed=[0])
    assert env.calls == []


def test_resume_derives_layout_and_checks_scene_sha(tmp_path):
    request = tmp_path / "request.json"
    data = {"format": "xpolicylab_own_episode_resume_v1",
            "identity": {"task": "align_blocks", "eval_seed": 0, "layout": 2, "run_id": "fixture"},
            "native_identity": {"scene_sha256": "a" * 64}}
    request.write_text(json.dumps(data))
    env = {"ROBORSI_RESUME_REQUEST": str(request)}
    selected = NativeSelection.from_launch("align_blocks", 0, environ=env)
    assert selected == NativeSelection("align_blocks", 0, 2, "fixture", "a" * 64)
    with pytest.raises(NativeLayoutError, match="Explicit layout"):
        NativeSelection.from_launch("align_blocks", 0, 1, environ=env)
    env["ROBORSI_RESUME_REQUEST_SHA256"] = "b" * 64
    with pytest.raises(NativeLayoutError, match="request changed"):
        NativeSelection.from_launch("align_blocks", 0, environ=env)


def test_unverified_legacy_resume_requires_native_scene_sha(tmp_path):
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"format": "xpolicylab_own_episode_resume_v1",
        "identity": {"task": "align_blocks", "eval_seed": 0, "layout": 2, "run_id": "fixture"}}))
    with pytest.raises(NativeLayoutError, match="verified native scene SHA"):
        NativeSelection.from_launch("align_blocks", 0, environ={"ROBORSI_RESUME_REQUEST": str(path)})


def test_module_hook_delegates_original_environment_and_scorer(tmp_path):
    env = NativeFixture(tmp_path)
    calls = []
    def original(config, app, resume_state=None, **kwargs):
        calls.append((config, app, resume_state, kwargs))
        return env
    module = SimpleNamespace(create_eval_env=original)
    wrap_eval_module(module, NativeSelection("align_blocks", 0, 2))
    assert module.create_eval_env("config", "app", extra=True) is env
    assert calls == [("config", "app", None, {"extra": True})]
    assert env.seed_manager.seed_list == [2]


def test_entry_keeps_official_arguments_and_prevents_stale_eval_cap(tmp_path, monkeypatch):
    entry = tmp_path / "main.py"
    entry.write_text("# Official main placeholder\n")
    seen = {}
    def run(path, run_name):
        import os, sys
        seen.update(path=path, run_name=run_name, argv=list(sys.argv), cap=os.environ["EVAL_NUM"])
    monkeypatch.setattr("XPolicyLab.policy.SenseNova-RoboRSI.native_entry.runpy.run_path", run)
    for name in ("ROBORSI_RESUME_REQUEST", "ROBORSI_NATIVE_LAYOUT_ID", "ROBORSI_NATIVE_SCENE_SHA256"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EVAL_NUM", "1")
    main(["--official-entry", str(entry), "--layout-id", "2", "--", "--task_name", "align_blocks", "--seed", "0", "--device_id", "1"])
    assert seen["argv"] == [str(entry), "--task_name", "align_blocks", "--seed", "0", "--device_id", "1"]
    assert seen["cap"] == "native"
    assert seen["run_name"] == "__main__"


def test_entry_import_hook_reaches_real_reset_before_evaluation(tmp_path):
    """Exercise the executable wrapper, deferred module load and native reset."""
    package = tmp_path / "src/eval_client"
    package.mkdir(parents=True)
    scene = tmp_path / "align_blocks_2.json"
    scene.write_text('{"objects": [{"pose": [2, 0, 0]}]}')
    (package / "eval_env.py").write_text(
        "from types import SimpleNamespace\n"
        "import os\n"
        "class Native:\n"
        "    task_name='align_blocks'; eval_seed=0; run_id='fixture'\n"
        "    num_envs=1; step_lim=600; take_action_cnt=[0]\n"
        "    def __init__(self):\n"
        "        self.seed_manager=SimpleNamespace(seed_list=[0,1,2], seed_info={2:{'scene_layout':os.environ['SCENE']}}, idx=0,st_idx=0,ed_idx=3,_current_batch_seeds=None)\n"
        "    def reset(self, seed=None, options=None):\n"
        "        self.env_seeds=list(seed); self.current_env_seed_map={0:seed[0]}\n"
        "    def get_obs_batch(self, env_idx_list=None, last_frame=False):\n"
        "        return [{'vision':{}}]\n"
        "    def official_score(self): return 17\n"
        "def create_eval_env(config, app, resume_state=None, **kwargs): return Native()\n"
    )
    entry = package / "main.py"
    entry.write_text(
        "import json\n"
        "from src.eval_client.eval_env import create_eval_env\n"
        "env=create_eval_env(None,None)\n"
        "assert env.seed_manager.seed_list == [2]\n"
        "env.reset(seed=env.seed_manager.seed_list)\n"
        "assert env.official_score() == 17\n"
        "print(json.dumps(env.get_obs_batch()[0]['roborsi_native_identity']))\n"
    )
    environment = {**os.environ, "PYTHONPATH": str(tmp_path),
                   "ROBODOJO_RUN_ID": "fixture", "SCENE": str(scene)}
    for name in ("ROBORSI_RESUME_REQUEST", "ROBORSI_NATIVE_LAYOUT_ID", "ROBORSI_NATIVE_SCENE_SHA256"):
        environment.pop(name, None)
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "native_entry.py"),
         "--official-entry", str(entry), "--layout-id", "2", "--", "--task_name", "align_blocks", "--seed", "0"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, check=True,
    )
    receipt = json.loads(result.stdout)
    assert receipt["layout"] == 2 and receipt["control_step"] == 0
    assert receipt["scene_sha256"] == hashlib.sha256(scene.read_bytes()).hexdigest()


def test_official_exec_restart_keeps_layout_hook(tmp_path, monkeypatch):
    entry = tmp_path / "main.py"
    entry.write_text("# Official main placeholder\n")
    seen = {}
    def execv(executable, arguments):
        seen.update(executable=executable, arguments=list(arguments))
    def run(path, run_name):
        os.execv(sys.executable, [sys.executable, str(entry), "--task_name", "align_blocks", "--seed", "0"])
    monkeypatch.setattr(os, "execv", execv)
    monkeypatch.setattr("XPolicyLab.policy.SenseNova-RoboRSI.native_entry.runpy.run_path", run)
    for name in ("ROBORSI_RESUME_REQUEST", "ROBORSI_NATIVE_LAYOUT_ID", "ROBORSI_NATIVE_SCENE_SHA256"):
        monkeypatch.delenv(name, raising=False)
    main(["--official-entry", str(entry), "--layout-id", "2", "--", "--task_name", "align_blocks", "--seed", "0"])
    assert Path(seen["arguments"][2]).name == "native_entry.py"
    assert seen["arguments"][3:6] == ["--official-entry", str(entry), "--"]
    assert os.environ["ROBORSI_NATIVE_LAYOUT_ID"] == "2"
