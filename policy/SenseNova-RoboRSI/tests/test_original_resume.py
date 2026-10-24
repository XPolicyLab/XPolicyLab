"""Exercise original recovery/replay through the official deferred-action bridge."""

from importlib import import_module

import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

hybrid_agent = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_agent")
resume = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_resume")
HybridBackend = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_backend").HybridBackend
cli_config = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.test_hybrid_agent").cli_config
observation = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.test_hybrid_backend").observation
sha = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.test_hybrid_backend").sha


@pytest.fixture
def stopped(tmp_path, reference):
    root = tmp_path / "source"
    run = root / "agent/controller"
    cfg = json.loads(
        (
            reference.source_dir.parent / "resources/fresh-configs/align_blocks-s0.json"
        ).read_text()
    )
    cfg["model"] = "fixture-model"
    identity = {
        "task": "align_blocks",
        "eval_seed": 0,
        "layout": 14,
        "run_id": "fixture-align_blocks-s0-layout14",
    }
    episode = cfg["test_episodes"][0]
    episode.update(layout_id=14, id="align_blocks-s0-eval0-layout14")
    record = run / "episodes/evaluation-0001"
    obs = observation()
    # The original validator compares measured poses and the current head PNG.
    state = {k: v.tolist() for k, v in obs["state"].items()}
    command = {k: v.tolist() for k, v in obs["state"].items() if "joint_state" in k}
    resume.write_json(
        root / "episode-owner.json", {
            "identity": identity, "closed": True,
            "trajectory_id": "fixture-trajectory",
            "native_identity": {"schema": 1, **identity, "scene_sha256": "a" * 64,
                                "control_step": 1, "step_limit": 200},
        }
    )
    resume.write_json(run / "config.json", cfg)
    resume.write_json(
        run / "observations/latest.json",
        {"done": False, "observation": {"control_step": 1, "step_limit": 200}},
    )
    resume.write_json(record / "episode.json", episode)
    resume.write_json(record / "result.json", {"native_done": False})
    (record / "actions.jsonl").write_text(
        json.dumps(
            {
                "status": "completed",
                "control_step": 1,
                "applied_action": {"joint_command": command},
            }
        )
        + "\n"
    )
    (record / "transitions.jsonl").write_text(
        json.dumps({"observation": {"control_step": 1, "state": state}}) + "\n"
    )
    resume.write_json(
        run / "segments/evaluation-0001/result.json", {"wall_seconds": 7200}
    )
    image = run / "observations/evaluation-0001/00001_cam_head.png"
    image.parent.mkdir(parents=True)
    Image.fromarray(obs["vision"]["cam_head"]["color"]).save(image)
    workspace = root / "agent/workspace"
    workspace.mkdir()
    (workspace / "note.md").write_text("Only this interrupted episode's notes")
    return root, cfg, identity, command


def test_original_recovery_budget_workspace_and_ownership(stopped, reference, tmp_path):
    root, cfg, identity, _ = stopped
    output = tmp_path / "request.json"
    request = resume.prepare_resume(root, output, reference)
    assert request["step_offset"] == 1
    assert request["config"]["episode_seconds"] == cfg["episode_seconds"] - 7200
    assert request["config"]["options"]["replay_fresh_session"] is True
    assert request["config"]["continue_from_run"] == str(root / "agent/controller")
    assert resume.load_resume(output, identity) == request
    assert resume.read_json(root / "agent/controller/config.json") == cfg
    assert request["recovery_count"] == 1
    with pytest.raises(ValueError, match="already has a prepared"):
        resume.prepare_resume(root, tmp_path / "duplicate.json", reference)


def native_progress(tmp_path, identity):
    path = tmp_path / "native-progress.json"
    resume.write_json(path, {
        "task_name": identity["task"], "eval_seed": identity["eval_seed"],
        "run_id": identity["run_id"], "policy_name": "SenseNova-RoboRSI",
        "config_name": "arx_x5", "completed_layout_ids": [0],
        "abandoned_layout_ids": [], "success_nums": 1, "fail_nums": 0,
        "details": {"0": {"layout_id": 0, "success": True, "score": 1.0}},
    })
    return path


def test_native_resume_preserves_actual_scene_without_relabeling(stopped, reference, tmp_path):
    root, cfg, old_identity, _ = stopped
    native = native_progress(tmp_path, old_identity)
    output = tmp_path / "request.json"
    request = resume.prepare_resume(root, output, reference, native_progress=native)
    assert request["identity"]["layout"] == 14
    assert request["identity"]["run_id"] == old_identity["run_id"]
    assert request["native_layout_binding"]["actual_layout"] == 14
    assert request["source_files"][str(native)] == resume.digest(native)
    assert resume.read_json(root / "episode-owner.json")["identity"] == old_identity
    assert request["config"]["test_episodes"] == cfg["test_episodes"]
    resume.load_resume(output, request["identity"])
    native.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        resume.load_resume(output, request["identity"])


@pytest.mark.parametrize("field,value", [
    ("task_name", "other"), ("eval_seed", 99), ("run_id", "other"),
    ("completed_layout_ids", [0, 1, 2]), ("success_nums", 2),
])
def test_native_resume_rejects_unbound_or_unscored_progress(stopped, tmp_path, field, value):
    _, _, identity, _ = stopped
    path = native_progress(tmp_path, identity)
    data = resume.read_json(path)
    data[field] = value
    resume.write_json(path, data)
    with pytest.raises(ValueError):
        resume.bind_native_progress(identity, path)


@pytest.mark.parametrize("change", ["task", "layout", "eval_seed", "run_id"])
def test_wrong_deployment_rejected(stopped, reference, tmp_path, change):
    root, _, identity, _ = stopped
    output = tmp_path / "request.json"
    resume.prepare_resume(root, output, reference)
    identity[change] = "other"
    with pytest.raises(ValueError, match="deployment"):
        resume.load_resume(output, identity)


@pytest.mark.parametrize("change", ["source", "notes"])
def test_changed_stopped_records_rejected(stopped, reference, tmp_path, change):
    root, _, identity, _ = stopped
    output = tmp_path / "request.json"
    resume.prepare_resume(root, output, reference)
    path = root / (
        "agent/workspace/note.md"
        if change == "notes"
        else "agent/controller/config.json"
    )
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        resume.load_resume(output, identity)


def test_live_source_rejected(stopped, reference, tmp_path):
    root, _, identity, _ = stopped
    resume.write_json(
        root / "episode-owner.json",
        {
            "identity": identity,
            "closed": False,
            "pid": os.getpid(),
            "process_start": resume.process_identity(os.getpid()),
        },
    )
    with pytest.raises(ValueError, match="writer"):
        resume.prepare_resume(root, tmp_path / "request.json", reference)


def test_action_gap_rejected_before_replay(stopped, reference, tmp_path):
    root, _, _, _ = stopped
    path = root / "agent/controller/episodes/evaluation-0001/actions.jsonl"
    row = json.loads(path.read_text())
    row["control_step"] = 2
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Incomplete"):
        resume.prepare_resume(root, tmp_path / "request.json", reference)


@pytest.mark.parametrize("value", [4, 8])
def test_original_four_recovery_cap(stopped, reference, tmp_path, value):
    root, _, _, _ = stopped
    resume.write_json(root / "adapter-recovery/state.json", {"recovery_count": value})
    with pytest.raises(ValueError, match="Four infrastructure"):
        resume.prepare_resume(root, tmp_path / "request.json", reference)


def test_original_recovery_prompt_only_on_validated_resume(
    stopped, reference, tmp_path
):
    root, cfg, _, _ = stopped
    fresh = hybrid_agent.build_evaluation_prompt(reference, cfg)
    request = resume.prepare_resume(root, tmp_path / "request.json", reference)
    recovered = request["config"]
    with pytest.raises(ValueError, match="author evaluation state"):
        hybrid_agent.build_evaluation_prompt(reference, recovered)
    prompt = hybrid_agent.build_evaluation_prompt(reference, recovered, own_resume=True)
    assert (
        "SAME interrupted episode restored by private recorded-action replay" in prompt
    )
    assert "Discard any old pending VLA proposal" in prompt
    assert "79200 wall seconds" in prompt
    assert "SAME interrupted episode" not in fresh
    resume.validate_policy(recovered, cfg)
    changed = copy.deepcopy(recovered)
    changed["options"]["approved_steps_max"] = 10
    with pytest.raises(ValueError, match="options"):
        resume.validate_policy(changed, cfg)


@pytest.mark.parametrize("diverge", [False, True])
def test_unchanged_native_replay_validator_on_real_bridge(
    stopped, reference, robot, tmp_path, diverge
):
    root, cfg, _, command = stopped
    episode = cfg["test_episodes"][0]
    directory = tmp_path / "attempt"
    directory.mkdir()
    backend = HybridBackend(
        reference,
        task_name="align_blocks",
        episode_dir=directory,
        initial_observation=observation(),
        step_limit=200,
        metadata_path=robot.metadata,
        expected_metadata_sha256=sha(robot.metadata),
        resource_root=robot.root,
        vla_port=19802,
        replay={
            "source_record": str(root / "agent/controller/episodes/evaluation-0001"),
            "episode_id": episode["id"],
            "control_step": 1,
        },
    )
    record, images = directory / "native", directory / "observations"
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(
            backend.request,
            op="reset",
            episode=episode,
            record_dir=record,
            observation_dir=images,
        )
        try:
            pending = backend.next_action(timeout=10)
            assert pending.step_index == 0 and pending.action == command
            obs = observation()
            if diverge:
                obs["state"]["left_ee_pose"][0] += 0.1
            backend.acknowledge(pending.request_id, obs)
            if diverge:
                with pytest.raises(RuntimeError, match="does not match"):
                    future.result(timeout=10)
            else:
                result = future.result(timeout=10)
                assert result["observation"]["control_step"] == 1
                assert not result["done"]
            validation = resume.read_json(record / "replay_validation.json")
            assert validation["matched"] is not diverge
            assert validation["replayed_steps"] == 1
        finally:
            backend.cancel()


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("diverge", [False, True])
def test_resume_controller_launches_only_after_native_validation(
    stopped, reference, robot, tmp_path, monkeypatch, diverge, vla, persistent
):
    root, cfg, identity, _ = stopped
    request = resume.prepare_resume(root, tmp_path / "resume.json", reference)
    directory = tmp_path / "controller-attempt"
    trajectory = tmp_path / "trajectory" if persistent else None
    if trajectory:
        trajectory.mkdir()
    directory.mkdir()
    recovered = request["config"]
    backend = HybridBackend(
        reference,
        task_name="align_blocks",
        episode_dir=trajectory or directory,
        initial_observation=observation(),
        step_limit=200,
        metadata_path=robot.metadata,
        expected_metadata_sha256=sha(robot.metadata),
        resource_root=robot.root,
        vla_port=19802,
        replay=recovered["options"]["replay"],
    )
    launches = []

    class OfflinePlanner:
        process = None

        def __init__(self, **kwargs):
            pass

        def launch_segment(self, cfg, run, home, segment, prompt, session=None):
            assert resume.read_json(run / "resume-validation.json")["matched"]
            assert (
                (run.parent / "workspace/note.md").read_text().startswith("Only this")
            )
            assert "SAME interrupted episode restored" in prompt and session is None
            assert cfg["episode_seconds"] < recovered["episode_seconds"]
            launches.append(prompt)
            program = "\n".join(
                [
                    "import sys,json",
                    "sys.path.insert(0,sys.argv[1])",
                    "from robot import observe,vla_propose,vla_review",
                    "o=observe()['observation']; assert o['control_step']==1",
                    "p=vla_propose('continue own episode')",
                    "r=vla_review(p['proposal_id'],'accept','observed fixture',{'current_subgoal':'continue'},approved_steps=5); assert r['done'] and r['executed_steps']==1",
                ]
            )
            segment.mkdir(exist_ok=True)
            with (
                (segment / "model.jsonl").open("w") as out,
                (segment / "model.stderr").open("w") as err,
            ):
                self.process = subprocess.Popen(
                    [sys.executable, "-c", program, str(run / "tools")],
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                    env={
                        "PATH": os.defpath,
                        "POLICYLAB_SOCKET": str((run / "bridge/robot.sock").resolve()),
                    },
                )
            return self.process

        def cancel(self):
            if self.process and self.process.poll() is None:
                self.process.terminate()
                self.process.wait(2)

    monkeypatch.setattr(hybrid_agent, "HybridCodexCliPlanner", OfflinePlanner)
    agent = hybrid_agent.HybridEmbodiedAgent(
        mcp_servers=(),
        output_dir=directory,
        planner="codex_cli",
        model=cfg["model"],
        reasoning_effort="max",
        max_turns=None,
        max_tokens=None,
        codex_cli=cli_config(Path("/not-used")),
    )
    interface = reference.source_dir.parent / "resources/interfaces/BENCHMARK.md"
    context = hybrid_agent.HybridRunContext(
        reference,
        backend,
        recovered,
        directory,
        root / "agent/workspace",
        root / "agent/workspace",
        interface,
        Path(sys.executable),
        Path("/not-used"),
        "0" * 64,
        request,
        trajectory,
    )
    controller = hybrid_agent.HybridController(agent, context).start()
    try:
        replay = controller.next_action(timeout=10)
        assert replay.step_index == 0 and not launches
        obs = observation()
        if diverge:
            obs["state"]["left_ee_pose"][0] += 0.1
        controller.acknowledge(replay.request_id, obs)
        if diverge:
            controller._thread.join(10)
            assert controller.done and controller.error and not launches
        else:
            action = controller.next_action(timeout=10)
            assert action.step_index == 1 and len(launches) == 1
            controller.acknowledge(action.request_id, observation(), terminated=True)
            controller._thread.join(10)
            assert controller.done and controller.error is None
            assert controller.result.stats["native_success"] is None
            result = resume.read_json(
                directory / "controller/episodes/evaluation-0001/result.json"
            )
            assert result["native_done"] and result["success"] is None
        assert (root / "agent/workspace/note.md").is_file()
    finally:
        controller.cancel()
    if persistent:
        assert (directory / "controller/episodes").is_symlink()
        assert (trajectory / "controller/config.json").is_file()
        assert (trajectory / "controller/provenance.json").is_file()
        assert (trajectory / "controller/episodes/evaluation-0001/episode.json").is_file()
        if not diverge:
            assert (trajectory / "controller/observations/latest.json").is_file()
            assert (trajectory / "controller/provenance.json").read_bytes() == (directory / "controller/provenance.json").read_bytes()
        else:
            validation = resume.read_json(trajectory / "controller/episodes/evaluation-0001/native/replay_validation.json")
            assert validation["matched"] is False


def bare_model(monkeypatch, trajectory_dir):
    """Exercise Model lifecycle without installing simulator/weight dependencies."""
    monkeypatch.setitem(
        sys.modules,
        "XPolicyLab.utils.process_data",
        SimpleNamespace(get_robot_action_dim_info=lambda _: None),
    )
    module = import_module("XPolicyLab.policy.SenseNova-RoboRSI.model")

    monkeypatch.setattr(module, "load_agent_config", lambda *a, **k: None)
    value = module.Model.__new__(module.Model)
    value.model_cfg = {
        "task_name": "align_blocks", "seed": 0,
        "trajectory_dir": str(trajectory_dir),
    }
    value._connector = value._temporary = value._icl = value._obs = value._pending = (
        None
    )
    value._step = 0
    value._resume = None
    value._awaiting_feedback = False
    return value


@pytest.mark.parametrize("task_name", ["align_blocks", "build_tower", "stack_blocks"])
def test_model_start_uses_shared_interface_for_each_task(tmp_path, monkeypatch, task_name):
    value = bare_model(monkeypatch, tmp_path / "trajectory")
    module = import_module("XPolicyLab.policy.SenseNova-RoboRSI.model")
    backend_module = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_backend")
    policy_dir = Path(__file__).resolve().parents[1]
    pins = json.loads((policy_dir / "submission.json").read_text())
    value.model_cfg["task_name"] = task_name
    value._directory = tmp_path / "episode"
    value._directory.mkdir()
    value._trajectory_directory = tmp_path / "trajectory"
    value._trajectory_id = "shared-interface-fixture"
    value._tool_python = Path(sys.executable)
    value._obs = {}
    value._agent_config = SimpleNamespace(
        document={"agent": {"model": "fixture-model"}},
        create_agent=lambda _case: object(),
    )
    workspace = value._directory / "workspace"
    monkeypatch.setattr(module, "POLICY_DIR", policy_dir)
    monkeypatch.setattr(module, "submission", lambda: pins)
    monkeypatch.setattr(module, "load_agent_config", lambda *a, **k: value._agent_config)
    monkeypatch.setattr(
        module, "prepare_icl_context",
        lambda *_args: SimpleNamespace(
            source_identity={"mapping": {"base_task": task_name}}, workspace=workspace,
        ),
    )
    monkeypatch.setattr(value, "_ensure_vla", lambda: 19803)
    monkeypatch.setattr(value, "_claim_resume", lambda: None)
    monkeypatch.setattr(backend_module, "load_reference_source", lambda *_args: object())
    monkeypatch.setattr(backend_module, "HybridBackend", lambda *a, **k: object())
    contexts = []

    def controller(_agent, context):
        contexts.append(context)
        return SimpleNamespace(start=lambda: object())

    monkeypatch.setattr(hybrid_agent, "HybridController", controller)
    value._start()

    assert len(contexts) == 1
    context = contexts[0]
    assert context.public_instructions == (
        policy_dir / pins["hybrid"]["profile_path"] / "resources/interfaces/BENCHMARK.md"
    )
    assert context.cfg["options"]["task"] == task_name
    assert context.cfg["test_episodes"][0]["task"] == task_name


def test_model_request_consumed_once_and_source_preserved(
    stopped, reference, tmp_path, monkeypatch
):
    root, _, identity, _ = stopped
    output = tmp_path / "request.json"
    resume.prepare_resume(root, output, reference)
    monkeypatch.setenv("ROBODOJO_RUN_ID", identity["run_id"])
    monkeypatch.setenv("ROBORSI_RESUME_REQUEST", str(output))
    monkeypatch.setenv("ROBORSI_RESUME_REQUEST_SHA256", resume.digest(output))
    value = bare_model(monkeypatch, tmp_path / "attempts")
    value.reset()
    assert not output.with_suffix(".json.claimed").exists()
    # The official client may reset during initialization before the episode.
    value.reset()
    assert not output.with_suffix(".json.claimed").exists()
    value._claim_resume()
    path = value._directory
    assert value._resume["source"] == str(root)
    assert resume.read_json(path / "episode-owner.json")["identity"] == identity
    value.close()
    assert path.exists() and resume.read_json(path / "episode-owner.json")["closed"]
    with pytest.raises(FileExistsError):
        value.reset()
    value.close()


def test_native_terminal_during_unvalidated_replay_is_technical_error(
    tmp_path, monkeypatch
):
    value = bare_model(monkeypatch, tmp_path / "attempts")
    value._directory = tmp_path
    value._resume = {"step_offset": 1}
    value._connector = SimpleNamespace(
        done=True,
        error=RuntimeError("replay diverged"),
        acknowledge=lambda *a, **k: None,
        close=lambda: None,
    )
    value._pending = SimpleNamespace(request_id=1)
    value.update_obs = lambda obs: setattr(value, "_obs", obs)
    with pytest.raises(RuntimeError, match="validated continuation"):
        value.end_episode({"public": "terminal fixture"})
    assert value._connector is None


def test_resume_rejects_foreign_replay_and_nonfinite_actions(
    stopped, reference, tmp_path
):
    root, _, identity, _ = stopped
    output = tmp_path / "request.json"
    request = resume.prepare_resume(root, output, reference)
    request["config"]["options"]["replay"]["source_record"] = "/other/episode"
    resume.write_json(output, request)
    with pytest.raises(ValueError, match="own-episode checkpoint"):
        resume.load_resume(output, identity)


def test_original_budget_exhaustion_rejects_resume(stopped, reference, tmp_path):
    root, cfg, _, _ = stopped
    resume.write_json(
        root / "agent/controller/segments/evaluation-0001/result.json",
        {"wall_seconds": cfg["episode_seconds"] - 100},
    )
    with pytest.raises(ValueError, match="24h wall budget exhausted"):
        resume.prepare_resume(root, tmp_path / "request.json", reference)


def test_own_workspace_preserves_original_symlink_semantics(
    stopped, reference, tmp_path
):
    root, _, identity, _ = stopped
    link = root / "agent/workspace/note-link"
    link.symlink_to("note.md")
    output = tmp_path / "request.json"
    resume.prepare_resume(root, output, reference)
    resume.load_resume(output, identity)
    link.unlink()
    link.symlink_to("other.md")
    with pytest.raises(ValueError, match="workspace changed"):
        resume.load_resume(output, identity)


def test_archive_preserves_links_without_reading_external_targets(
    tmp_path, monkeypatch
):
    archive_episode = import_module("XPolicyLab.policy.SenseNova-RoboRSI.trajectory_archive").archive_episode

    source, dest = tmp_path / "source", tmp_path / "archive"
    source.mkdir()
    secret = tmp_path / "private-fixture.txt"
    secret.write_text("private fixture must never be copied")
    (source / "link").symlink_to(secret)
    (source / "missing").symlink_to("/not-present-cli-helper")
    (source / "trace.jsonl").write_text('{"fixture":true}\n')
    monkeypatch.setenv("ROBORSI_TRACE_ROOT", str(dest))
    monkeypatch.setenv("ROBODOJO_RUN_ID", "fixture-run")
    result = Path(archive_episode(source, task="align_blocks", episode_id="fixture"))
    assert (result / "runtime/link").is_symlink()
    assert (result / "runtime/missing").is_symlink()
    manifest = resume.read_json(result / "manifest.json")
    links = {r["path"]: r for r in manifest["files"] if "symlink" in r}
    assert set(links) == {"runtime/link", "runtime/missing"}
    assert all("sha256" not in r and "bytes" not in r for r in links.values())
    assert (source / "trace.jsonl").exists()


def test_two_models_cannot_start_same_resume(stopped, reference, tmp_path, monkeypatch):
    root, _, identity, _ = stopped
    output = tmp_path / "request.json"
    resume.prepare_resume(root, output, reference)
    monkeypatch.setenv("ROBODOJO_RUN_ID", identity["run_id"])
    monkeypatch.setenv("ROBORSI_RESUME_REQUEST", str(output))
    monkeypatch.setenv("ROBORSI_RESUME_REQUEST_SHA256", resume.digest(output))
    first = bare_model(monkeypatch, tmp_path / "first-attempts")
    second = bare_model(monkeypatch, tmp_path / "second-attempts")
    try:
        first.reset()
        second.reset()
        first._claim_resume()
        with pytest.raises(FileExistsError):
            second._claim_resume()
        assert resume.read_json(output.with_suffix(".json.claimed"))["attempt"] == str(first._directory)
    finally:
        first.close()
        second.close()


def test_standard_resume_launcher_binds_scene_and_refuses_changed_arguments(stopped, reference, tmp_path):
    launch_environment = import_module("XPolicyLab.policy.SenseNova-RoboRSI.resume_eval").launch_environment
    root, _, identity, _ = stopped
    output = tmp_path / "request.json"
    resume.prepare_resume(root, output, reference)
    arguments = ["RoboDojo", "align_blocks", "roborsi-codex-v1", "arx_x5", "joint", "0", "0", "0", "uv", "RoboDojo"]
    env = launch_environment(output, arguments, {})
    assert env["ROBORSI_NATIVE_LAYOUT_ID"] == "14"
    assert env["ROBODOJO_RUN_ID"] == identity["run_id"]
    assert env["ROBORSI_NATIVE_SCENE_SHA256"] == "a" * 64
    assert env["ROBORSI_RESUME_REQUEST_SHA256"] == resume.digest(output)
    assert env["ROBORSI_REQUIRE_NATIVE_IDENTITY"] == "1"
    with pytest.raises(ValueError, match="Conflicting"):
        launch_environment(output, arguments, {"ROBORSI_NATIVE_LAYOUT_ID": "0"})
    arguments[1] = "stack_bowls"
    with pytest.raises(ValueError, match="arguments"):
        launch_environment(output, arguments, {})


def test_missing_actual_scene_receipt_cannot_prepare_resume(stopped, reference, tmp_path):
    root, _, _, _ = stopped
    owner_path = root / "episode-owner.json"
    owner = resume.read_json(owner_path)
    owner.pop("native_identity")
    resume.write_json(owner_path, owner)
    with pytest.raises(ValueError, match="Missing verified"):
        resume.prepare_resume(root, tmp_path / "request.json", reference)
