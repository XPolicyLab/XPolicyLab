"""A lost replay ACK retains the verified parent checkpoint and spent budget."""

from importlib import import_module

import sys
from pathlib import Path

hybrid_agent = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_agent")
resume = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_resume")
HybridBackend = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_backend").HybridBackend
cli_config = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.test_hybrid_agent").cli_config
observation = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.test_hybrid_backend").observation
reference = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.test_hybrid_backend").reference
robot = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.test_hybrid_backend").robot
sha = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.test_hybrid_backend").sha
stopped = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.test_original_resume").stopped


def test_replay_ack_loss_retries_verified_parent_without_new_budget(
    stopped, reference, robot, tmp_path, monkeypatch
):
    source, cfg, identity, command = stopped
    request = resume.prepare_resume(source, tmp_path / "first-request.json", reference)
    failed = tmp_path / "failed-attempt"
    agent_dir = failed / "agent"
    agent_dir.mkdir(parents=True)
    backend = HybridBackend(
        reference, task_name="align_blocks", episode_dir=failed,
        initial_observation=observation(), step_limit=200,
        metadata_path=robot.metadata, expected_metadata_sha256=sha(robot.metadata),
        resource_root=robot.root, vla_port=19802,
        replay=request["config"]["options"]["replay"],
    )
    launches = []

    class NoLaunchPlanner:
        process = None

        def __init__(self, **kwargs):
            pass

        def launch_segment(self, *args, **kwargs):
            launches.append(True)
            raise AssertionError("Model must never launch before completed replay validation")

        def cancel(self):
            pass

    monkeypatch.setattr(hybrid_agent, "HybridCodexCliPlanner", NoLaunchPlanner)
    agent = hybrid_agent.HybridEmbodiedAgent(
        mcp_servers=(), output_dir=agent_dir, planner="codex_cli", model=cfg["model"],
        reasoning_effort="max", max_turns=None, max_tokens=None,
        codex_cli=cli_config(Path("/not-used")),
    )
    context = hybrid_agent.HybridRunContext(
        reference, backend, request["config"], agent_dir,
        source / "agent/workspace", source / "agent/workspace",
        reference.source_dir.parent / "resources/interfaces/align_blocks-s0.md",
        Path(sys.executable), Path("/not-used"), "0" * 64, request, None,
    )
    controller = hybrid_agent.HybridController(agent, context).start()
    try:
        pending = controller.next_action(timeout=10)
        assert pending.step_index == 0 and pending.action == command
        # The public simulator never acknowledged execution of this replay step.
        controller.cancel(timeout=10)
        assert controller.done and not launches
        run = agent_dir / "controller"
        assert not (run / "resume-validation.json").exists()
        assert not (run / "observations/latest.json").exists()
        summary = resume.read_json(run / "summary.json")
        assert "Simulator exited during native replay" in summary["error"]
        segment = resume.read_json(run / "segments/evaluation-0001/result.json")
        assert segment["wall_seconds"] > 0
        resume.write_json(failed / "episode-owner.json", {
            "identity": identity, "closed": True,
            "trajectory_id": request["trajectory_id"],
            "native_identity": {"schema": 1, **identity, "scene_sha256": "a" * 64,
                                "control_step": 0, "step_limit": 200},
            "recovery_state": request["recovery_state"],
            "recovery_count": request["recovery_count"],
        })
        next_path = tmp_path / "second-request.json"
        recovered = resume.prepare_resume(failed, next_path, reference)
        assert recovered["step_offset"] == 1
        assert recovered["source"] == str(source)
        assert recovered["workspace_source"] == str(failed)
        assert recovered["config"]["options"]["replay"]["control_step"] == 1
        assert recovered["config"]["continue_from_run"] == str(source / "agent/controller")
        assert recovered["recovery_count"] == request["recovery_count"] + 1
        spent = request["config"]["episode_seconds"] - recovered["config"]["episode_seconds"]
        assert spent > 0 and abs(spent - segment["wall_seconds"]) < 0.01
        assert resume.load_resume(next_path, identity) == recovered
    finally:
        controller.cancel(timeout=10)
