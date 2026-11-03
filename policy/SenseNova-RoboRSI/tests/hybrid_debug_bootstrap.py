"""Test-only Python launcher; production entry points remain byte-identical.

This injects a synthetic local VLA/CLI transport and robot-FK-consistent public
observations. It never contacts a provider or constructs a simulator.
"""

from __future__ import annotations

from importlib import import_module

import builtins
import copy
import hashlib
import importlib
import json
import os
import queue
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np


def install_transports(policy: Path, events: Path):
    hybrid_agent = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_agent")
    pi05_runtime = import_module("XPolicyLab.policy.SenseNova-RoboRSI.pi05_runtime")

    class Server:
        port = 19802

        def __init__(self, **_kwargs):
            pass

        def start(self):
            return self

        def close(self):
            pass

    # Preflight still checks the real pinned native CLI and all source/ICL
    # manifests. Only unavailable VLA installation/service checks are replaced,
    # explicitly reported as synthetic rather than claimed as weight validation.
    pi05_runtime.Pi05Server = Server
    pi05_runtime.check_pi05_runtime = lambda *_args, **_kwargs: {
        "scope": "synthetic VLA service transport; weights were not tested"
    }
    sys.modules["pi05_runtime"] = pi05_runtime

    class Connection:
        def __init__(self):
            self.first = True
            self.actions = np.tile(
                [0.002, 0.001, 0, 0, 0, 0, 1, -0.001, 0, 0, 0, 0, 0, 1],
                (50, 1),
            )

        def send(self, value):
            from openpi_client import msgpack_numpy

            assert isinstance(value, bytes)
            value = msgpack_numpy.unpackb(value)
            assert value["state"].shape == (14,)
            assert set(value["images"]) == {
                "cam_high",
                "cam_left_wrist",
                "cam_right_wrist",
            }

        def recv(self, timeout):
            from openpi_client import msgpack_numpy

            if self.first:
                self.first = False
                return msgpack_numpy.packb({"synthetic_transport": True})
            return msgpack_numpy.packb({"actions": self.actions.copy()})

        def close(self):
            pass

    import websockets.sync.client

    websockets.sync.client.connect = lambda *_args, **_kwargs: Connection()

    def launch(planner, cfg, run, _home, segment, prompt, session=None):
        program = "\n".join(
            [
                "import json,sys,os",
                "from pathlib import Path",
                "sys.path.insert(0,sys.argv[1])",
                "from robot import observe,vla_propose,vla_review",
                "events=Path(sys.argv[2])",
                "def record(event,**fields):",
                "    with events.open('a') as f: f.write(json.dumps(dict(event=event,**fields))+'\\n')",
                "record('start',pid=os.getpid())",
                "print(json.dumps({'type':'thread.started','thread_id':'synthetic-hybrid-debug'}),flush=True)",
                "record('observe',result=observe())",
                "for index in range(5):",
                "    p=vla_propose('Align the visible blocks using the set square.')",
                "    record('proposal',horizon=p['horizon'],control_step=p['control_step'],fk=p['fk_validation'])",
                "    r=vla_review(p['proposal_id'],'accept','Synthetic public measured-state review.',{'current_subgoal':'Align the blocks.'})",
                "    record('review',result=r)",
                "    if r['done']: break",
                "else: raise AssertionError('debug fixture exceeded twenty actions')",
            ]
        )
        (segment / "prompt.txt").write_text(prompt)
        (segment / "fixture-transport.json").write_text(
            json.dumps(
                {
                    "scope": "fake CLI process; original tools and Controller are unchanged",
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "session": session,
                }
            )
        )
        with (
            (segment / "model.jsonl").open("w") as output,
            (segment / "model.stderr").open("w") as error,
        ):
            process = subprocess.Popen(
                [sys.executable, "-c", program, str(run / "tools"), str(events)],
                stdout=output,
                stderr=error,
                start_new_session=True,
                env={
                    "PATH": os.defpath,
                    "POLICYLAB_SOCKET": str((run / "bridge/robot.sock").resolve()),
                },
            )
        with planner._process_lock:
            planner._process = process
        return process

    hybrid_agent.HybridCodexCliPlanner.launch_segment = launch
    original_next = hybrid_agent.HybridController.next_action

    def next_action(controller, timeout=None):
        count = getattr(controller, "_synthetic_poll_count", 0) + 1
        controller._synthetic_poll_count = count
        if count in {1, 7}:
            # Exercise actual [] serialization across the unmodified official
            # WS client/server, before the first action and after its fifth ACK.
            with events.with_name("empty-polls.jsonl").open("a") as stream:
                stream.write(json.dumps({"poll": count, "timeout_s": timeout}) + "\n")
            raise queue.Empty
        return original_next(controller, timeout=timeout)

    hybrid_agent.HybridController.next_action = next_action


def run_debug_client(script: Path, policy: Path, events: Path):
    """Patch only the official test environment's public observation supplier."""
    from scipy.spatial.transform import Rotation

    load_reference_source = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_backend").load_reference_source

    pins = json.loads((policy / "submission.json").read_text())["hybrid"]
    load_reference_source(
        policy / pins["source_path"],
        policy / pins["manifest_path"],
        pins["manifest_sha256"],
    )
    kinematics = importlib.import_module("eef_kinematics")
    metadata = json.loads((policy / pins["robot_metadata_path"]).read_text())
    arms = {}
    for arm, spec in metadata["arms"].items():
        root = np.asarray(spec["root_pose"])
        arms[arm] = (
            kinematics.transform(root[:3], root[3:]),
            kinematics.ArmFK(
                policy / "hybrid" / spec["urdf_path"],
                spec["joint_names"],
                spec["base_link"],
                spec["ee_link"],
            ),
        )
    original_builder = builtins.__build_class__

    def build_class(function, name, *args, **kwargs):
        cls = original_builder(function, name, *args, **kwargs)
        if name != "TestEnv" or Path(function.__code__.co_filename) != script:
            return cls
        original_obs, original_action = cls.get_obs, cls.take_action

        def get_obs(self, env_idx=0):
            # Preserve the official image creation and DEBUG_OBS_ENCODED path.
            obs = original_obs(self, env_idx)
            with events.with_name("observation-calls.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "control_step": self.episode_step,
                            "encoded": bool(self.obs_encoded),
                        }
                    )
                    + "\n"
                )
            previous = getattr(self, "_synthetic_joint_feedback", {})
            for arm, (root, fk) in arms.items():
                joints = np.asarray(previous.get(arm + "_arm_joint_state", np.zeros(6)))
                transform = root @ fk.matrix(joints)
                quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
                obs["state"][arm + "_arm_joint_state"] = joints.copy()
                obs["state"][arm + "_ee_joint_state"] = np.asarray(
                    previous.get(arm + "_ee_joint_state", [1.0])
                ).copy()
                obs["state"][arm + "_ee_pose"] = np.r_[
                    transform[:3, 3], quaternion[[3, 0, 1, 2]]
                ]
            return obs

        def take_action(self, action):
            original_action(self, action)
            # Perfect tracking is explicitly synthetic; native physics and task
            # scoring are never inferred from this debug environment.
            self._synthetic_joint_feedback = copy.deepcopy(action)

        cls.get_obs, cls.take_action = get_obs, take_action
        return cls

    builtins.__build_class__ = build_class
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        builtins.__build_class__ = original_builder


def main():
    fixture = json.loads(Path(sys.argv[1]).read_text())
    script = Path(sys.argv[2]).resolve()
    sys.argv = [str(script), *sys.argv[3:]]
    policy = Path(fixture["policy"])
    # Match direct Python script execution for the script's fallback imports.
    sys.path.insert(0, str(script.parent))
    sys.path.insert(0, str(policy / "runtime/RoboRSI"))
    install_transports(policy, Path(fixture["events"]))
    if script.name == "debug_env_client.py":
        run_debug_client(script, policy, Path(fixture["events"]))
    else:
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
