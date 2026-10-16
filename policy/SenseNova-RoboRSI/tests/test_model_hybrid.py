"""Formal Model boundary with original hybrid tools, controller and robot FK.

Only external CLI/VLA transports are fake. No model connectivity, weights,
physics, hidden task state or official task success is established here.
"""

from __future__ import annotations

from importlib import import_module

import hashlib
import itertools
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from roborsi.agents.config import AgentConfig
from roborsi.embodied_agent import EmbodiedAgent
from roborsi.planner.codex_cli import CodexCliPlanner

model_module = import_module("XPolicyLab.policy.SenseNova-RoboRSI.model")
pi05_runtime = import_module("XPolicyLab.policy.SenseNova-RoboRSI.pi05_runtime")
runtime_config = import_module("XPolicyLab.policy.SenseNova-RoboRSI.runtime_config")
HybridBackend = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_backend").HybridBackend
write_icl_fixture = import_module("XPolicyLab.policy.SenseNova-RoboRSI.tests.icl_fixture").write_icl_fixture

P = Path(__file__).resolve().parents[1]
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")

# Selected public robot state and one applied action from the user-authorized
# align_blocks-s0 reference, actions.jsonl first completed record SHA256
# 8806115b9c347abf867287a3b4c9e77f2c708a52026bdab89bdb6870b2c6daa6.
# These are test oracles only; production never imports evaluation history.
PUBLIC_INITIAL_STATE = {
    "left_arm_joint_state": [
        5.115578100012286e-13,
        -1.156115904982105e-15,
        1.767066072491508e-17,
        7.853062682560905e-16,
        -9.987136985467604e-13,
        2.5006576670483227e-12,
    ],
    "right_arm_joint_state": [
        5.115569426394906e-13,
        -1.1561103992679478e-15,
        1.7667572032507807e-17,
        7.853009213606109e-16,
        -9.98713048025457e-13,
        2.5006572333674537e-12,
    ],
    "left_ee_pose": [
        -0.2995290756225586,
        -0.35229986906051636,
        0.921500027179718,
        0.7069999575614929,
        -4.327419134142474e-08,
        -6.000607299938565e-08,
        0.7072135210037231,
    ],
    "right_ee_pose": [
        0.3004709482192993,
        -0.35229986906051636,
        0.921500027179718,
        0.7069999575614929,
        -4.327419134142474e-08,
        -6.000607299938565e-08,
        0.7072135210037231,
    ],
    "left_ee_joint_state": [1.0],
    "right_ee_joint_state": [1.0],
}
REFERENCE_FIRST_TARGET = {
    "left": {
        "position": [-0.29895338905424945, -0.35204281248979735, 0.9227395144080791],
        "quaternion_wxyz": [
            0.7092693510078885,
            -0.004814355491152279,
            0.004558958755034487,
            0.7049063949185901,
        ],
        "gripper_closed": False,
    },
    "right": {
        "position": [0.3002658248007931, -0.35239608570848935, 0.9216652205751124],
        "quaternion_wxyz": [
            0.7060236086908873,
            -0.00010513804534916596,
            -0.0006332427563468402,
            0.7081880060553835,
        ],
        "gripper_closed": False,
    },
}
REFERENCE_FIRST_JOINTS = {
    "left_arm_joint_state": [
        -0.004405992571264505,
        0.012208224274218082,
        0.008526327088475227,
        -0.009531314484775066,
        0.0020634757820516825,
        -0.0003582918143365532,
    ],
    "right_arm_joint_state": [
        0.0015923457685858011,
        -0.00060113804647699,
        0.0004326393536757678,
        -0.00028652846231125295,
        -0.0011641853488981724,
        -0.0010417369194328785,
    ],
}


def observation(step=0):
    """Measured-state fixture; RGB is distinctive, immutable, and never BGR."""
    colors = ([240, 25, 10], [15, 225, 30], [35, 45, 210])
    state = {
        name: np.asarray(values, dtype=np.float64).copy()
        for name, values in PUBLIC_INITIAL_STATE.items()
    }
    # Simulate a new public gripper measurement without inventing task state or
    # asserting perfect tracking of the issued arm commands.
    state["left_ee_joint_state"] = np.array([1.0 - 0.01 * step])
    vision = {}
    for camera, color in zip(CAMERAS, colors, strict=True):
        rgb = np.full((8, 9, 3), color, dtype=np.uint8)
        rgb.setflags(write=False)
        vision[camera] = {"color": rgb}
    return {
        "instruction": "Align the blocks using the visible set square.",
        "state": state,
        "vision": vision,
        "additional_info": {"frequency": 25},
        "data_format_version": "v1.0",
    }


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def adapter_factory(monkeypatch):
    """An isolated adapter with real verified sources and tiny sealed ICL."""
    # Each temporary policy is independently authenticated. Isolate Python's
    # import cache too, rather than allowing a previous test's client source to
    # make this policy's codec verification accidentally pass.
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in tuple(sys.modules):
        if name == "openpi_client" or name.startswith("openpi_client."):
            monkeypatch.delitem(sys.modules, name)
    pinned_source = P / "runtime/RoboRSI"
    if not pinned_source.is_dir():
        pytest.fail("Install pinned RoboRSI before hybrid Model verification")
    monkeypatch.syspath_prepend(str(pinned_source))
    hybrid_agent = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_agent")

    # The development machine has no sibling official env_cfg installation.
    # Supply its public six-arm/one-gripper dimensions from the pinned metadata;
    # the official state/action implementation itself remains unchanged.
    robot = json.loads((P / "hybrid/robot-metadata.json").read_text())
    monkeypatch.setattr(
        model_module,
        "get_robot_action_dim_info",
        lambda _name: {
            "arm_dim": [
                len(robot["arms"][arm]["joint_names"]) for arm in ("left", "right")
            ],
            "ee_dim": [1, 1],
        },
    )
    monkeypatch.setenv("CODEX_API_KEY", "synthetic-hybrid-offline-key")
    monkeypatch.delenv("ROBORSI_CODEX_BASE_URL", raising=False)
    models, services, planners, connections = [], [], [], []
    identifiers = itertools.count()
    temporary = tempfile.TemporaryDirectory(
        prefix="hybrid-model-test-", dir=P / "runtime"
    )

    class FakeServer:
        def __init__(self, **kwargs):
            self.options, self.closed, self.started, self.port = kwargs, False, 0, 19802
            services.append(self)

        def start(self):
            self.started += 1
            return self

        def close(self):
            self.closed = True

    class Connection:
        def __init__(self):
            self.inputs, self.first, self.closed = [], True, False
            self.actions = np.tile(
                [0.002, 0.001, 0, 0, 0, 0, 1, -0.001, 0, 0, 0, 0, 0, 1],
                (50, 1),
            )

        def send(self, payload):
            from openpi_client import msgpack_numpy

            assert isinstance(payload, bytes)
            self.inputs.append(msgpack_numpy.unpackb(payload))

        def recv(self, timeout):
            from openpi_client import msgpack_numpy

            if self.first:
                self.first = False
                return msgpack_numpy.packb({"synthetic_vla_transport": True})
            return msgpack_numpy.packb({"actions": self.actions.copy()})

        def close(self):
            self.closed = True

    import websockets.sync.client

    def connect(uri, **kwargs):
        assert uri == "ws://127.0.0.1:19802"
        assert kwargs["compression"] is None
        connection = Connection()
        connections.append(connection)
        return connection

    monkeypatch.setattr(websockets.sync.client, "connect", connect)
    monkeypatch.setattr(pi05_runtime, "Pi05Server", FakeServer)

    def create(*, behavior="review", **changes):
        directory = Path(temporary.name) / f"adapter-{next(identifiers)}"
        directory.mkdir()
        for name in (
            "agent.yaml",
            "submission.json",
        ):
            shutil.copy2(P / name, directory / name)
        # AgentConfig verifies that the imported custom factory is one of the
        # exact selected source paths, even though its preset is episode-local.
        preset = yaml.safe_load((directory / "agent.yaml").read_text())
        preset["source_root"] = str(P)
        preset["code"] = [str(P / name) for name in preset["code"]]
        (directory / "agent.yaml").write_text(yaml.safe_dump(preset))
        shutil.copytree(P / "hybrid", directory / "hybrid")
        (directory / "vendor").mkdir()
        shutil.copytree(
            P / "vendor/pi05",
            directory / "vendor/pi05",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy2(
            P / "vendor/pi05-source.json", directory / "vendor/pi05-source.json"
        )
        pins = json.loads((directory / "submission.json").read_text())
        # Loose original modules must retain their stable, verified source path
        # after each temporary adapter is cleaned; never import through a link
        # which disappears while the process-wide module cache remains alive.
        for key in ("source_path", "profile_path", "manifest_path"):
            pins["hybrid"][key] = str(P / pins["hybrid"][key])
        (directory / "submission.json").write_text(json.dumps(pins))
        runtime = directory / "runtime"
        runtime.mkdir()
        (runtime / "RoboRSI").symlink_to(
            pinned_source.resolve(), target_is_directory=True
        )
        (directory / "checkpoints/pi05-arx-x5-seed0-step59999").mkdir(parents=True)
        write_icl_fixture(directory, "align_blocks")
        monkeypatch.setattr(model_module, "POLICY_DIR", directory)
        # The external tool subprocess is already replaced below. Keep its
        # declared interpreter separate from the host in this offline fixture.
        monkeypatch.setattr(
            model_module,
            "resolve_tool_python",
            lambda policy: policy / "runtime/agent-venv/bin/python",
        )
        monkeypatch.setattr(runtime_config, "POLICY_DIR", directory)
        traces, processes = [], []

        def launch(planner, cfg, run, _home, segment, prompt, session=None):
            """Only replace the model's external subprocess transport."""
            planners.append(planner)
            trace = directory / f"tool-events-{len(traces)}.jsonl"
            traces.append(trace)
            program = "\n".join(
                [
                    "import json,sys",
                    "from pathlib import Path",
                    "sys.path.insert(0,sys.argv[1])",
                    "from robot import observe,act,vla_propose,vla_review",
                    "trace=Path(sys.argv[2])",
                    "def record(event,**values):",
                    "    with trace.open('a') as f: f.write(json.dumps(dict(event=event,**values))+'\\n')",
                    "print(json.dumps({'type':'thread.started','thread_id':'synthetic-hybrid-session'}),flush=True)",
                    "initial=observe()",
                    "record('observe',result=initial)",
                    f"behavior={behavior!r}",
                    "if behavior=='exit': sys.exit(9)",
                    "try:",
                    "    act({'mode':'eef'})",
                    "    raise AssertionError('raw action bypassed review')",
                    "except RuntimeError as error:",
                    "    assert 'vla_propose' in str(error)",
                    "for round_index in range(20):",
                    "    proposal=vla_propose('Align the visible blocks with the set square.')",
                    "    record('propose',horizon=proposal['horizon'],control_step=proposal['control_step'])",
                    "    decision='accept'; targets=None",
                    f"    reference_target={REFERENCE_FIRST_TARGET!r}",
                    "    if behavior=='reference_first':",
                    "        decision='modify'; targets=[reference_target for _ in range(5)]",
                    "    result=vla_review(proposal['proposal_id'],decision,'Synthetic public-observation review.',{'current_subgoal':'Align the blocks.'},targets=targets)",
                    "    record('review',result=result)",
                    "    if result['done']: break",
                    "else: raise AssertionError('fixture exceeded bounded rounds')",
                ]
            )
            (segment / "prompt.txt").write_text(prompt)
            with (
                (segment / "model.jsonl").open("w") as output,
                (segment / "model.stderr").open("w") as error,
            ):
                process = subprocess.Popen(
                    [sys.executable, "-c", program, str(run / "tools"), str(trace)],
                    stdout=output,
                    stderr=error,
                    env={
                        "PATH": os.defpath,
                        "POLICYLAB_SOCKET": str((run / "bridge/robot.sock").resolve()),
                    },
                    start_new_session=True,
                )
            with planner._process_lock:
                planner._process = process
            processes.append(process)
            return process

        monkeypatch.setattr(
            hybrid_agent.HybridCodexCliPlanner, "launch_segment", launch
        )
        pins = json.loads((directory / "submission.json").read_text())
        model = model_module.Model(
            {
                "bench_name": "RoboDojo",
                "task_name": "align_blocks",
                "ckpt_name": pins["agent_label"],
                "env_cfg_type": "arx_x5",
                "action_type": "joint",
                "seed": 0,
                "eval_batch": False,
                **changes,
            }
        )
        models.append(model)
        return SimpleNamespace(
            model=model,
            directory=directory,
            traces=traces,
            processes=processes,
            services=services,
            planners=planners,
            connections=connections,
        )

    yield create
    for model in models:
        model.shutdown()
    temporary.cleanup()
    for name in tuple(sys.modules):
        if name == "openpi_client" or name.startswith("openpi_client."):
            sys.modules.pop(name)


def assert_joint_chunk(chunk):
    assert len(chunk) == 1
    assert {name: value.shape for name, value in chunk[0].items()} == {
        "left_arm_joint_state": (6,),
        "left_ee_joint_state": (1,),
        "right_arm_joint_state": (6,),
        "right_ee_joint_state": (1,),
    }
    assert all(value.dtype == np.float32 for value in chunk[0].values())
    assert all(np.isfinite(value).all() for value in chunk[0].values())


def test_formal_model_original_review_feedback_and_terminal_unknown(adapter_factory):
    adapter = adapter_factory()
    model = adapter.model
    assert isinstance(model._agent_config, AgentConfig)
    model.reset()
    directory = model._directory
    initial = observation()
    for step in range(8):
        model.update_obs(initial if step == 0 else observation(step))
        assert_joint_chunk(model.get_action())
        assert model._connector.context.backend.control_step == step
        assert (
            model._connector.context.agent_python
            == adapter.directory / "runtime/agent-venv/bin/python"
        )
        with pytest.raises(ValueError, match="fresh update_obs"):
            model.get_action()
    controller = model._connector
    backend = controller.context.backend
    assert isinstance(backend, HybridBackend)
    assert isinstance(controller.agent, EmbodiedAgent)
    assert isinstance(adapter.planners[0], CodexCliPlanner)
    assert len(adapter.planners) == len(adapter.processes) == 1
    assert (
        backend.metadata_identity["sha256"]
        == hashlib.sha256((P / "hybrid/robot-metadata.json").read_bytes()).hexdigest()
    )
    model.end_episode(observation(8))
    assert controller.done and controller.error is None
    assert controller.result.stats["native_success"] is None
    assert controller.result.stop_reason == "finished"
    assert backend.control_step == 8
    assert backend._latest["state"]["left_ee_joint_state"].tolist() == [0.92]
    assert backend._metrics == {
        "success": None,
        "source": "not_exposed_to_policy",
    }
    events = records(adapter.traces[0])
    assert [item["control_step"] for item in events if item["event"] == "propose"] == [
        0,
        5,
    ]
    reviews = [item["result"] for item in events if item["event"] == "review"]
    assert [item["executed_steps"] for item in reviews] == [5, 3]
    assert reviews[-1]["done"] is True and "success" not in reviews[-1]
    assert len(adapter.connections) == 1
    payloads = adapter.connections[0].inputs
    assert len(payloads) == 2 and payloads[1]["state"][6] == pytest.approx(0.95)
    for output_name, camera in zip(
        ("cam_high", "cam_left_wrist", "cam_right_wrist"), CAMERAS, strict=True
    ):
        np.testing.assert_array_equal(
            payloads[0]["images"][output_name],
            initial["vision"][camera]["color"].transpose(2, 0, 1),
        )
        assert not initial["vision"][camera]["color"].flags.writeable
    assert adapter.connections[0].closed
    assert adapter.processes[0].poll() == 0
    assert not directory.exists() and model._connector is None
    assert len(adapter.services) == 1 and not adapter.services[0].closed
    model.shutdown()
    assert adapter.services[0].closed


def test_formal_model_first_dls_action_matches_reference_public_oracle(adapter_factory):
    adapter = adapter_factory(behavior="reference_first")
    model = adapter.model
    model.reset()
    model.update_obs(observation())
    action = model.get_action()
    assert_joint_chunk(action)
    for channel, expected in REFERENCE_FIRST_JOINTS.items():
        np.testing.assert_allclose(action[0][channel], expected, rtol=0, atol=1e-6)
    for arm in ("left", "right"):
        np.testing.assert_array_equal(action[0][arm + "_ee_joint_state"], [1.0])
    model.end_episode(observation(1))
    review = next(
        item["result"]
        for item in records(adapter.traces[0])
        if item["event"] == "review"
    )
    assert review["executed_steps"] == 1 and review["done"] is True


def test_reset_cancels_pending_review_and_isolates_episode_workspace(adapter_factory):
    adapter = adapter_factory()
    model = adapter.model
    model.reset()
    model.update_obs(observation())
    model.get_action()
    old_controller = model._connector
    old_directory = model._directory
    (old_directory / "agent/workspace/previous-episode.txt").write_text(
        "private episode"
    )
    model.reset()
    assert old_controller.done and not old_controller._thread.is_alive()
    assert adapter.processes[0].poll() is not None
    assert adapter.connections[0].closed
    assert not old_directory.exists()
    assert model._directory != old_directory
    model.update_obs(observation())
    model.get_action()
    assert not (model._directory / "agent/workspace/previous-episode.txt").exists()
    assert len(adapter.services) == 1 and adapter.services[0].started == 1
    assert len(adapter.processes) == 2
    model.end_episode(observation(1))
    assert all(process.poll() is not None for process in adapter.processes)


def test_empty_polls_preserve_conversation_without_repeating_action_ack(
    adapter_factory, monkeypatch
):
    adapter = adapter_factory()
    HybridController = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_agent").HybridController

    original_next = HybridController.next_action
    original_ack = HybridController.acknowledge
    polls, acknowledgements = [], []

    def next_action(controller, timeout=None):
        polls.append(timeout)
        if len(polls) in {1, 3, 4}:
            raise queue.Empty
        return original_next(controller, timeout=timeout)

    def acknowledge(controller, request_id, *args, **kwargs):
        acknowledgements.append((request_id, kwargs.get("terminated", False)))
        return original_ack(controller, request_id, *args, **kwargs)

    monkeypatch.setattr(HybridController, "next_action", next_action)
    monkeypatch.setattr(HybridController, "acknowledge", acknowledge)
    model = adapter.model
    model.reset()
    model.update_obs(observation())
    assert model.get_action() == []
    controller = model._connector
    assert model._obs is not None and model._pending is None
    assert acknowledgements == []
    assert_joint_chunk(model.get_action())
    model.update_obs(observation(1))
    assert model.get_action() == []
    assert controller.context.backend.control_step == model._step == 1
    assert len(acknowledgements) == 1
    assert model.get_action() == []
    assert len(acknowledgements) == 1
    assert_joint_chunk(model.get_action())
    assert model._connector is controller
    assert len(adapter.processes) == len(adapter.planners) == 1
    assert len(adapter.connections) == 1
    with pytest.raises(ValueError, match="fresh update_obs"):
        model.get_action()
    model.end_episode(observation(2))
    assert len(acknowledgements) == 2
    assert len({request for request, _ in acknowledgements}) == 2
    assert [terminated for _, terminated in acknowledgements] == [False, True]
    assert polls == [30.0] * 5
    assert controller.done and controller.error is None
    assert adapter.processes[0].poll() == 0


def test_bad_terminal_observation_still_closes_active_review(adapter_factory):
    adapter = adapter_factory()
    model = adapter.model
    model.reset()
    model.update_obs(observation())
    model.get_action()
    controller, directory = model._connector, model._directory
    invalid = observation(1)
    del invalid["vision"]["cam_right_wrist"]
    with pytest.raises(ValueError, match="three decoded"):
        model.end_episode(invalid)
    assert controller.done and not controller._thread.is_alive()
    assert adapter.processes[0].poll() is not None
    assert adapter.connections[0].closed
    assert model._connector is None and not directory.exists()


def test_server_shutdown_registry_releases_pending_cli_and_owned_vla(adapter_factory):
    adapter = adapter_factory()
    model = adapter.model
    model.reset()
    model.update_obs(observation())
    model.get_action()
    controller, directory = model._connector, model._directory
    model_module.close_models()
    assert controller.done and not controller._thread.is_alive()
    assert adapter.processes[0].poll() is not None
    assert adapter.connections[0].closed and adapter.services[0].closed
    assert model._vla is None and model._connector is None
    assert not directory.exists()


def test_cli_exit_before_action_is_an_interface_failure_with_cleanup(
    adapter_factory, monkeypatch
):
    adapter = adapter_factory(behavior="exit")
    HybridController = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_agent").HybridController

    original = HybridController.next_action
    polls = []

    def next_action(controller, timeout=None):
        polls.append(timeout)
        if len(polls) == 1:
            raise queue.Empty
        return original(controller, timeout=timeout)

    monkeypatch.setattr(HybridController, "next_action", next_action)
    model = adapter.model
    model.reset()
    model.update_obs(observation())
    directory = model._directory
    assert model.get_action() == []
    with pytest.raises(RuntimeError, match="hybrid Agent failed"):
        model.get_action()
    assert adapter.processes[0].poll() == 9
    assert model._connector is None and not directory.exists()


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"action_type": "ee"}, "action_type=joint"),
        ({"env_cfg_type": "other_robot"}, "env_cfg_type=arx_x5"),
        ({"bench_name": "other_benchmark"}, "bench_name=RoboDojo"),
        ({"eval_batch": True}, "eval_batch=false"),
    ],
)
def test_hybrid_rejects_unsupported_official_configuration(
    adapter_factory, changes, message
):
    with pytest.raises(ValueError, match=message):
        adapter_factory(**changes)


def test_hybrid_requires_runtime_key_before_creating_episode(
    adapter_factory, monkeypatch
):
    adapter = adapter_factory()
    monkeypatch.delenv("CODEX_API_KEY")
    with pytest.raises(ValueError, match="CODEX_API_KEY is required"):
        model_module.Model(adapter.model.model_cfg)


def test_instruction_list_and_bad_public_state_cleanup(adapter_factory):
    adapter = adapter_factory()
    model = adapter.model
    model.reset()
    obs = observation()
    del obs["instruction"]
    obs["instructions"] = ["Align the blocks.", "Use the visible set square."]
    model.update_obs(obs)
    assert model._obs["instruction"] == "Align the blocks.\nUse the visible set square."
    assert_joint_chunk(model.get_action())
    controller, directory = model._connector, model._directory
    invalid = observation(1)
    invalid["state"]["left_arm_joint_state"] = np.zeros(5)
    model.update_obs(invalid)
    with pytest.raises(ValueError, match="6 finite values"):
        model.get_action()
    assert controller.done and not controller._thread.is_alive()
    assert adapter.processes[0].poll() is not None
    assert model._connector is None and not directory.exists()
