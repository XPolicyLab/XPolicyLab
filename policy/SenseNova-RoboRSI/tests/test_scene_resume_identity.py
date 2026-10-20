"""Simulator-owned scene identity must bind the public Model/replay boundary."""

from __future__ import annotations

from importlib import import_module

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

hybrid_agent = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_agent")
hybrid_backend = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_backend")
hybrid_resume = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_resume")
model_module = import_module("XPolicyLab.policy.SenseNova-RoboRSI.model")

IDENTITY = {"task": "align_blocks", "eval_seed": 0, "layout": 14, "run_id": "fixture"}


def receipt(**changes):
    return {"schema": 1, **IDENTITY, "scene_sha256": "a" * 64,
            "control_step": 0, "step_limit": 200, **changes}


def observation(native):
    obs = {
        "instruction": "Align the visible blocks.", "state": {},
        "vision": {camera: {"color": np.zeros((2, 3, 3), dtype=np.uint8)}
                   for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
    }
    if native is not None:
        obs["roborsi_native_identity"] = native
    return obs


@pytest.fixture
def boundary(tmp_path):
    # The public validation boundary needs no service, CLI or simulator import.
    model = model_module.Model.__new__(model_module.Model)
    model.model_cfg = {"task_name": "align_blocks", "seed": 0}
    model._agent_config = SimpleNamespace(document={"agent": {"model": "fixture-model"}})
    model._temporary, model._ended = object(), False
    model._directory, model._step, model._pending = tmp_path, 0, None
    model._resume, model._obs = None, None
    model._episode_id = "fixture-policy"
    model._trajectory_id = "fixture-policy"
    model._trajectory_directory = tmp_path / "trajectory"
    model._tool_python = Path("/unused/tool/python")
    hybrid_resume.write_json(tmp_path / "episode-owner.json", {"identity": dict(IDENTITY)})
    return model


def test_missing_native_receipt_blocks_required_boundary(boundary, monkeypatch):
    monkeypatch.setenv("ROBORSI_REQUIRE_NATIVE_IDENTITY", "1")
    with pytest.raises(ValueError, match="Missing verified"):
        boundary.update_obs(observation(None))


@pytest.mark.parametrize("change", [
    {"scene_sha256": "b" * 64}, {"layout": 0}, {"control_step": 1}, {"step_limit": 0},
])
def test_resume_rejects_wrong_scene_step_and_invalid_cap(boundary, change):
    boundary._resume = {"native_identity": receipt(control_step=9)}
    with pytest.raises(ValueError):
        boundary.update_obs(observation(receipt(**change)))
    assert boundary._obs is None


def test_native_cap_cannot_change_after_receipt_is_bound(boundary):
    boundary.update_obs(observation(receipt()))
    with pytest.raises(ValueError, match="step limit"):
        boundary.update_obs(observation(receipt(step_limit=201)))


def _prepare_start(monkeypatch, boundary):
    prepared = SimpleNamespace(
        source_identity={"mapping": {"base_task": "align_blocks"}},
        workspace=boundary._directory / "workspace",
    )
    monkeypatch.setattr(model_module, "prepare_icl_context", lambda *_: prepared)
    monkeypatch.setattr(hybrid_backend, "load_reference_source", lambda *_: SimpleNamespace())
    monkeypatch.setattr(boundary, "_ensure_vla", lambda: 19802)
    monkeypatch.setattr(boundary, "_claim_resume", lambda: None)
    captured = {}

    def load_agent_config(*_args, **kwargs):
        captured["config_trajectory_id"] = kwargs["trajectory_id"]
        return SimpleNamespace(document={"agent": {"model": "fixture-model"}}, create_agent=lambda _: object())

    monkeypatch.setattr(model_module, "load_agent_config", load_agent_config)

    class Backend:
        def __init__(self, reference, **kwargs):
            captured["backend"] = kwargs

    class Controller:
        def __init__(self, agent, context):
            captured["context"] = context

        def start(self):
            return self

    monkeypatch.setattr(hybrid_backend, "HybridBackend", Backend)
    monkeypatch.setattr(hybrid_agent, "HybridController", Controller)
    return captured


def test_initial_native_cap_must_equal_original_task_cap(boundary, monkeypatch):
    boundary.update_obs(observation(receipt(step_limit=201)))
    captured = _prepare_start(monkeypatch, boundary)
    with pytest.raises(ValueError, match="step limit"):
        boundary._start()
    assert "context" not in captured and "backend" not in captured


@pytest.mark.parametrize("resumed", [False, True])
def test_actual_layout14_binds_fresh_config_and_original_replay(boundary, monkeypatch, resumed):
    native = receipt()
    if resumed:
        profile = model_module.submission()["hybrid"]
        cfg = json.loads((model_module.POLICY_DIR / profile["profile_path"] /
                         "resources/fresh-configs/align_blocks-s0.json").read_text())
        cfg["model"] = "fixture-model"
        cfg["test_episodes"][0].update(layout_id=14, id="align_blocks-s0-eval0-layout14")
        cfg["continue_from_run"] = "/own/stopped/controller"
        cfg["episode_seconds"] -= 1200
        cfg["options"]["replay_fresh_session"] = True
        cfg["options"]["replay"] = {
            "source_record": "/own/stopped/controller/episodes/evaluation-0001",
            "episode_id": "align_blocks-s0-eval0-layout14", "control_step": 9,
        }
        boundary._resume = {"native_identity": receipt(control_step=9), "config": cfg}
    boundary.update_obs(observation(native))
    captured = _prepare_start(monkeypatch, boundary)
    boundary._start()
    cfg = captured["context"].cfg
    assert cfg["test_episodes"][0]["layout_id"] == 14
    assert cfg["test_episodes"][0]["id"] == "align_blocks-s0-eval0-layout14"
    assert captured["backend"]["initial_observation"]["roborsi_native_identity"] == native
    assert captured["backend"]["step_limit"] == 200
    if resumed:
        assert captured["backend"]["replay"]["control_step"] == 9
        assert cfg["options"]["replay_fresh_session"] is True
        assert cfg["continue_from_run"] == "/own/stopped/controller"
    else:
        assert captured["backend"]["replay"] is None
    owner = hybrid_resume.read_json(boundary._directory / "episode-owner.json")
    assert owner["identity"] == IDENTITY and owner["native_identity"] == native


def test_native_restart_cannot_silently_start_fresh_same_episode(boundary):
    boundary._unfinished_native = receipt(control_step=12)
    with pytest.raises(ValueError, match="prepared resume request"):
        boundary.update_obs(observation(receipt()))
    assert boundary._obs is None


def test_controller_start_uses_logical_trajectory_instead_of_attempt_id(boundary, monkeypatch):
    boundary._trajectory_id = "original-recovered-trajectory"
    boundary.update_obs(observation(receipt()))
    captured = _prepare_start(monkeypatch, boundary)
    boundary._start()
    assert captured["config_trajectory_id"] == "original-recovered-trajectory"
    assert captured["config_trajectory_id"] != boundary._episode_id


def test_native_next_layout_can_start_fresh_without_importing_prior_episode(boundary):
    boundary._unfinished_native = receipt(control_step=12)
    hybrid_resume.write_json(boundary._directory / "episode-owner.json", {
        "identity": {**IDENTITY, "layout": None},
    })
    boundary.update_obs(observation(receipt(layout=15)))
    assert boundary._unfinished_native is None
    assert boundary._obs["roborsi_native_identity"]["layout"] == 15


def test_shared_vla_is_reused_without_taking_service_ownership(boundary, monkeypatch):
    import contextlib
    calls = []
    monkeypatch.setenv("ROBORSI_SHARED_VLA_PORT", "19802")
    monkeypatch.setattr(model_module.socket, "create_connection",
                        lambda address, timeout: (calls.append((address, timeout)) or contextlib.nullcontext()))
    boundary._vla = None
    assert boundary._ensure_vla() == 19802
    service = boundary._vla
    service.close()
    assert boundary._ensure_vla() == 19802
    assert calls == [(("127.0.0.1", 19802), 2)]


def test_model_change_between_validation_and_launch_is_rejected(boundary, monkeypatch):
    boundary.update_obs(observation(receipt()))
    captured = _prepare_start(monkeypatch, boundary)
    monkeypatch.setattr(model_module, "load_agent_config", lambda *_args, **_kwargs:
                        SimpleNamespace(document={"agent": {"model": "changed-model"}}))
    with pytest.raises(ValueError, match="ROBORSI_CODEX_MODEL changed"):
        boundary._start()
    assert "backend" not in captured and "context" not in captured
