"""Check original prompts, CLI transport, recovery and deferred tool feedback."""

from __future__ import annotations

from importlib import import_module

import copy
import hashlib
import io
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from roborsi.agents.config import AgentConfig
from roborsi.embodied_agent import EmbodiedAgent
from roborsi.planner.codex_cli import CodexCliPlanner
from roborsi.planner.codex_cli_config import CodexCliConfig

hybrid_agent = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_agent")
hybrid_launcher = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_launcher")
BackendClosed = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_backend").BackendClosed
load_reference_source = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_backend").load_reference_source

P = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reference():
    pins = json.loads((P / "submission.json").read_text())["hybrid"]
    return load_reference_source(
        P / pins["source_path"], P / pins["manifest_path"], pins["manifest_sha256"]
    )


@pytest.fixture
def cfg():
    config = json.loads(
        (
            P / "vendor/robodojo_adaptive/resources/fresh-configs/align_blocks-s0.json"
        ).read_text()
    )
    config["model"] = "fixture-model"
    return config


def cli_config(executable):
    return CodexCliConfig.from_dict(
        {
            "executable": str(executable),
            "expected_version": "codex-cli 0.160.1",
            "expected_sha256": "f34a4d2301892ae96c90097786bfe5dc269f187b6f69faf42a7b357b8c081e35",
            "tools": "external",
            "inherit_env": False,
            "env_vars": ["CODEX_API_KEY", "NO_PROXY", "no_proxy", "HTTPS_PROXY"],
            "config_overrides": {
                "model_provider": "runtime",
                "model_providers.runtime.base_url": "https://provider.example.invalid/v1",
                "model_providers.runtime.wire_api": "responses",
                "model_providers.runtime.env_key": "CODEX_API_KEY",
                "model_auto_compact_token_limit": 60000,
            },
        }
    )


def test_prompt_is_byte_exact_against_reference_align_blocks(reference, cfg):
    prompt = hybrid_agent.build_evaluation_prompt(reference, cfg)
    assert len(prompt.encode()) == 8498
    assert (
        hashlib.sha256(prompt.encode()).hexdigest()
        == "20b1fbf3327208c9e7143a4c6574e730ad59186844f51745579fb60d0e005132"
    )
    assert "roborsi_action" not in prompt and "MCP" not in prompt


def test_agent_config_custom_factory_reaches_embodied_agent(tmp_path):
    runtime = P / "runtime"
    runtime.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="hybrid-config-test-", dir=runtime
    ) as temporary:
        preset = Path(temporary) / "agent.yaml"
        raw = {
            "schema_version": 1,
            "source_root": "../..",
            "factory": "sense_nova_roborsi_factory:make_agent",
            "code": [
                "../../sense_nova_roborsi_factory.py",
                "../../hybrid_agent.py",
                "../../hybrid_launcher.py",
            ],
            "agent": {
                "planner": "codex_cli",
                "model": "fixture-model",
                "reasoning_effort": "max",
                "max_turns": None,
                "max_tokens": None,
                "planner_timeout_s": 0,
                "action_timeout_s": None,
                "codex_cli": {
                    "executable": "/not-launched",
                    "expected_version": "codex-cli 0.160.1",
                    "tools": "external",
                },
            },
        }
        preset.write_text(yaml.safe_dump(raw))
        case = tmp_path / "episode"
        case.mkdir()
        agent = AgentConfig.from_file(preset).create_agent(case)
        assert isinstance(agent, hybrid_agent.HybridEmbodiedAgent)
        assert isinstance(agent, EmbodiedAgent)
        assert agent.planner == "codex_cli" and agent.codex_cli.tools == "external"
        assert (case / "agent-config.json").is_file()
        agent.close()


def test_launcher_uses_original_mounts_and_hides_secrets_in_memfd(
    reference, cfg, tmp_path, monkeypatch
):
    binary, bwrap = tmp_path / "codex", tmp_path / "bwrap"
    binary.write_bytes(b"\x7fELFfixture")
    bwrap.write_bytes(b"fixture-bubblewrap")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    python = tmp_path / "agent-env/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("fixture")
    run = tmp_path / "run"
    for name in [
        "bridge",
        "tools",
        "observations/evaluation-0001",
        "workspace",
        "segment",
    ]:
        (run / name).mkdir(parents=True)
    training = tmp_path / "training"
    training.mkdir()
    cfg.update(
        agent_python=str(python),
        training_set=str(training),
        _observation_episode="evaluation-0001",
    )
    planner = hybrid_launcher.HybridCodexCliPlanner(
        reference=reference,
        bubblewrap=bwrap,
        bubblewrap_sha256=hashlib.sha256(bwrap.read_bytes()).hexdigest(),
        output_dir=run,
        repo_root=workspace,
        model=cfg["model"],
        timeout_s=None,
        dashboard_events=None,
        reasoning_effort="max",
        config=cli_config(binary),
    )
    assert isinstance(planner, CodexCliPlanner)
    identity = {
        "executable": str(binary),
        "native_binary": {"path": str(binary)},
        "version": "codex-cli 0.160.1",
    }
    monkeypatch.setattr(planner, "preflight", lambda: identity)
    monkeypatch.setenv("CODEX_API_KEY", "fixture-api-secret")
    monkeypatch.setenv("NO_PROXY", "localhost,provider.example.invalid")
    monkeypatch.setenv("no_proxy", "lowercase.example.invalid")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.invalid:8080")
    seen = {}

    class Input(io.BytesIO):
        def close(self):
            seen["prompt"] = self.getvalue()
            super().close()

    process = SimpleNamespace(stdin=Input(), pid=999999)

    def spawn(argv, **kwargs):
        seen["argv"], seen["env"] = argv, kwargs["env"]
        seen["options"] = os.pread(kwargs["pass_fds"][-1], 1024 * 1024, 0).split(b"\0")
        return process

    planner._spawn = spawn
    personal_home = os.environ.get("HOME")
    # A stale template must never override the resolved model on the planner.
    cfg["model"] = "stale-template-model"
    returned = planner.launch_segment(
        cfg,
        run,
        tmp_path / "private-home",
        run / "segment",
        "original prompt",
        "same-session",
    )
    assert returned is process
    assert seen["argv"][:2] == [str(bwrap), "--args"]
    assert "fixture-api-secret" not in repr(seen["argv"])
    assert "fixture-api-secret" not in repr(seen["env"])
    assert b"fixture-api-secret" in seen["options"]
    for name, value in ((b"NO_PROXY", b"localhost,provider.example.invalid"),
                        (b"no_proxy", b"lowercase.example.invalid"),
                        (b"HTTPS_PROXY", b"http://proxy.example.invalid:8080")):
        index = seen["options"].index(name)
        assert seen["options"][index + 1] == value
    assert not any(b"auth.json" in part for part in seen["options"])
    for mount in [
        b"/training",
        b"/workspace",
        b"/tools",
        b"/bridge",
        b"/observations/evaluation-0001",
        b"--unshare-pid",
        b"--unshare-user",
        b"--clearenv",
        b"/codex-resources",
        b"/codex-code-mode-host",
        b"/proc/self/exe",
        b"--symlink",
    ]:
        assert mount in seen["options"]
    boundary = seen["argv"].index("--")
    command = seen["argv"][boundary + 1 :]
    assert command[:4] == ["/codex", "exec", "resume", "same-session"]
    assert 'model_providers.runtime.env_key="CODEX_API_KEY"' in command
    assert command[command.index("--model") + 1] == "fixture-model"
    assert 'model_providers.runtime.base_url="https://provider.example.invalid/v1"' in command
    assert "stale-template-model" not in command
    assert seen["prompt"] == b"original prompt"
    assert not (tmp_path / "private-home/auth.json").exists()
    assert os.environ.get("HOME") == personal_home
    assert "fixture-api-secret" not in (run / "segment/cli-identity.json").read_text()


@pytest.mark.parametrize(
    "failure,new_session",
    [("maximum number of images exceeded", True), ("HTTP 429 capacity", False)],
)
def test_original_live_context_recovery_keeps_the_same_physical_episode(
    reference, cfg, tmp_path, monkeypatch, failure, new_session
):
    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    clock = Clock()
    monkeypatch.setattr(reference.runner, "time", clock)
    monkeypatch.setattr(reference.runner, "stop_process", lambda _process: None)
    experiment = SimpleNamespace(
        lock=threading.RLock(),
        phase="evaluation",
        latest={"done": False, "observation": {"control_step": 7}},
        segment_done=False,
        fault=None,
        review_status=lambda _reply: {},
        finalize=lambda _reason: None,
        checkpoint=lambda _name: "checkpoint",
    )
    calls = []

    def launch(_cfg, _run, _home, segment, prompt, session):
        calls.append((session, prompt))
        rows = [{"type": "thread.started", "thread_id": "source-session"}]
        if len(calls) == 1:
            rows.append({"type": "turn.failed", "error": {"message": failure}})
        else:
            experiment.segment_done = True
            experiment.latest["done"] = True
        (segment / "model.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        return SimpleNamespace(poll=lambda: 0, returncode=0)

    result = reference.runner.run_segment(
        experiment,
        cfg,
        tmp_path,
        tmp_path / "home",
        "evaluation",
        "original",
        120,
        "initial-session",
        launch=launch,
    )
    assert len(calls) == 2
    assert calls[1][0] == (None if new_session else "source-session")
    assert "SAME live episode" in calls[1][1]
    assert result["live_context_recoveries"][0]["native_reset"] is False
    assert result["live_context_recoveries"][0]["control_step"] == 7
    assert result["status"] == "finished"


class DeferredBackend:
    """No simulator: each original step waits for the test's explicit ACK."""

    def __init__(self):
        self.actions = queue.Queue()
        self.step = 0
        self.closed = False
        self.pending = None

    def reply(self):
        return {
            "observation": {
                "instruction": "Align blocks",
                "control_step": self.step,
                "step_limit": 200,
            },
            "done": self.step == 5,
            "metrics": {"success": None} if self.step == 5 else {},
        }

    def request(self, *, op, **kwargs):
        if op == "reset":
            Path(kwargs["record_dir"]).mkdir(parents=True)
            return self.reply()
        if op == "vla_propose":
            target = {
                arm: {
                    "position": [0.0, 0.0, 1.0],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "gripper_closed": False,
                }
                for arm in ["left", "right"]
            }
            return {
                "proposal_id": "fixture-proposal",
                "control_step": self.step,
                "targets": [copy.deepcopy(target) for _ in range(50)],
                "out_of_bounds": [[] for _ in range(50)],
            }
        if op == "step":
            self.pending = SimpleNamespace(
                action=kwargs["action"], event=threading.Event()
            )
            self.actions.put(self.pending)
            self.pending.event.wait()
            if self.closed:
                raise BackendClosed("Cancelled")
            return self.reply()
        raise AssertionError(op)

    def next_action(self, timeout):
        item = self.actions.get(timeout=timeout)
        if isinstance(item, Exception):
            raise item
        return item

    def acknowledge(self):
        self.step += 1
        self.pending.event.set()

    def cancel(self):
        self.closed = True
        if self.pending is not None:
            self.pending.event.set()
        self.actions.put(BackendClosed("Cancelled"))


def test_original_tools_review_and_five_acknowledged_steps_close_without_self_scoring(
    reference, cfg, tmp_path, monkeypatch
):
    backend = DeferredBackend()
    prepared = tmp_path / "prepared"
    (prepared / "prep/skills").mkdir(parents=True)
    (prepared / "prep/skills/original.md").write_text("Original complete experience")
    training = tmp_path / "training"
    training.mkdir()
    (training / "public.txt").write_text("public fixture")
    episode = tmp_path / "episode"
    episode.mkdir()
    original_request = reference.runner.Experiment.request

    class OfflinePlanner:
        def __init__(self, **_kwargs):
            self.process = None

        def launch_segment(self, _cfg, run, _home, segment, prompt, session=None):
            (segment / "prompt.txt").write_text(prompt)
            program = "\n".join(
                [
                    "import sys,json,os",
                    "sys.path.insert(0,sys.argv[1])",
                    "from robot import observe,vla_propose,vla_review",
                    "assert observe()['observation']['control_step']==0",
                    "p=vla_propose('align blocks')",
                    "r=vla_review(p['proposal_id'],'accept','observed fixture',{'current_subgoal':'align'},approved_steps=5)",
                    "assert r['executed_steps']==5 and r['done']",
                    "print(json.dumps({'type':'thread.started','thread_id':'offline-session'}))",
                ]
            )
            with (
                (segment / "model.jsonl").open("w") as output,
                (segment / "model.stderr").open("w") as error,
            ):
                self.process = subprocess.Popen(
                    [sys.executable, "-c", program, str(run / "tools")],
                    stdout=output,
                    stderr=error,
                    env={
                        "PATH": os.defpath,
                        "POLICYLAB_SOCKET": str((run / "bridge/robot.sock").resolve()),
                    },
                    start_new_session=True,
                )
            return self.process

        def cancel(self):
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                self.process.wait(2)

    monkeypatch.setattr(hybrid_agent, "HybridCodexCliPlanner", OfflinePlanner)
    agent = hybrid_agent.HybridEmbodiedAgent(
        mcp_servers=(),
        output_dir=episode,
        planner="codex_cli",
        model=cfg["model"],
        reasoning_effort="max",
        max_turns=None,
        max_tokens=None,
        codex_cli=cli_config(P / "runtime/codex/bin/codex"),
    )
    context = hybrid_agent.HybridRunContext(
        reference,
        backend,
        cfg,
        episode,
        prepared,
        training,
        P / "vendor/robodojo_adaptive/resources/interfaces/align_blocks-s0.md",
        Path(sys.executable),
        Path("/not-used"),
        "0" * 64,
    )
    controller = hybrid_agent.HybridController(agent, context).start()
    try:
        for step in range(5):
            pending = controller.next_action(timeout=10)
            assert pending.action["mode"] == "eef" and backend.step == step
            controller.acknowledge()
        controller._thread.join(10)
        assert controller.done and controller.error is None
        assert controller.result.stats["native_success"] is None
        assert controller.result.finish_result is None
        result = json.loads(
            (episode / "controller/episodes/evaluation-0001/result.json").read_text()
        )
        assert result["native_done"] is True and result["success"] is None
        assert (
            episode / "workspace/prep/skills/original.md"
        ).read_text() == "Original complete experience"
        assert reference.runner.Experiment.request is original_request
        assert not (episode / "controller/bridge").exists()
    finally:
        controller.cancel()


def test_cancel_releases_backend_before_agent_close_without_experiment_lock():
    calls = []
    backend = SimpleNamespace(cancel=lambda: calls.append("backend"))

    class Agent:
        def bind_reference(self, _context):
            return self

        def close(self):
            calls.append("agent")

    controller = hybrid_agent.HybridController(
        Agent(), SimpleNamespace(backend=backend)
    )
    controller.cancel()
    assert calls == ["backend", "agent"]
    assert controller.next_action() is None


def test_controller_cleanup_allows_3600_seconds_without_sleeping():
    waits = []
    backend = SimpleNamespace(cancel=lambda: None)

    class Agent:
        def bind_reference(self, _context):
            return self

        def close(self):
            return None

    controller = hybrid_agent.HybridController(Agent(), SimpleNamespace(backend=backend))
    controller._thread = SimpleNamespace(
        ident=1,
        join=lambda timeout: waits.append(timeout),
        is_alive=lambda: False,
    )
    controller.close()
    assert waits == [3600.0]


def test_terminal_rpc_timeout_includes_cleanup_and_ack_margin():
    import yaml

    config = yaml.safe_load((P / "deploy.yml").read_text())
    assert config["request_timeout_s"] == 3660.0
    assert config["request_timeout_s"] >= hybrid_agent.CONTROLLER_CLEANUP_TIMEOUT_SECONDS + 60
