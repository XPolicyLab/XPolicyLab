"""Original hybrid review/FK/IK with synthetic public observations and VLA I/O.

No CLI/provider, simulator, native task or VLA weights are exercised here.
"""

from __future__ import annotations

from importlib import import_module

import copy
import hashlib
import json
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

BackendClosed = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_backend").BackendClosed
HybridBackend = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_backend").HybridBackend
load_reference_source = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_backend").load_reference_source

POLICY_DIR = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="session")
def reference():
    pins = json.loads((POLICY_DIR / "submission.json").read_text())["hybrid"]
    manifest = POLICY_DIR / pins["manifest_path"]
    if not manifest.is_file():
        pytest.fail("Missing adaptive vendor snapshot; no checkout fallback")
    modules = load_reference_source(
        POLICY_DIR / pins["source_path"], manifest, pins["manifest_sha256"]
    )
    modules.hybrid_host.install()
    return modules


@pytest.fixture
def robot(tmp_path):
    """Clearly synthetic six-joint URDF, never a claim about real X5 geometry."""
    root = tmp_path / "public-robot"
    root.mkdir()
    urdf = root / "synthetic-six-joint.urdf"
    joints = []
    for i, axis in enumerate(("0 0 1", "0 1 0", "1 0 0", "0 0 1", "0 1 0", "1 0 0"), 1):
        parent = "base_link" if i == 1 else f"link{i - 1}"
        joints.append(
            f'<joint name="joint{i}" type="revolute"><parent link="{parent}"/><child link="link{i}"/><origin xyz="0.1 0 0" rpy="0 0 0"/><axis xyz="{axis}"/><limit lower="-3" upper="3" effort="100" velocity="5"/></joint>'
        )
    urdf.write_text('<robot name="synthetic">' + "".join(joints) + "</robot>")
    arm = {
        "urdf_path": urdf.name,
        "urdf_sha256": sha(urdf),
        "joint_names": [f"joint{i}" for i in range(1, 7)],
        "base_link": "base_link",
        "ee_link": "link6",
        "soft_joint_pos_limits": [[-3.0, 3.0]] * 6,
        "root_pose": [0, 0, 0, 1, 0, 0, 0],
    }
    right = {**copy.deepcopy(arm), "root_pose": [0, 1, 0, 1, 0, 0, 0]}
    metadata = root / "robot.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "robot": "arx_x5",
                "sources": {"synthetic_urdf_sha256": sha(urdf)},
                "arms": {"left": arm, "right": right},
            }
        )
    )
    return SimpleNamespace(root=root, metadata=metadata, urdf=urdf)


def observation(joints=None):
    joints = np.zeros(6) if joints is None else np.asarray(joints)
    color = np.full((3, 4, 3), [241, 18, 9], dtype=np.uint8)
    color.flags.writeable = False
    return {
        "instruction": "Synthetic fixture: align the blocks.",
        "state": {
            "left_arm_joint_state": joints,
            "right_arm_joint_state": joints,
            "left_ee_joint_state": np.array([1.0]),
            "right_ee_joint_state": np.array([1.0]),
            "left_ee_pose": np.array([0.6, 0, 0, 1, 0, 0, 0]),
            "right_ee_pose": np.array([0.6, 1, 0, 1, 0, 0, 0]),
        },
        "vision": {
            camera: {"color": color}
            for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist")
        },
        "additional_info": {"frequency": 25},
    }


@pytest.fixture
def backend(reference, robot, tmp_path):
    directory = tmp_path / "episode"
    directory.mkdir()
    value = HybridBackend(
        reference,
        task_name="align_blocks",
        episode_dir=directory,
        initial_observation=observation(),
        step_limit=200,
        metadata_path=robot.metadata,
        expected_metadata_sha256=sha(robot.metadata),
        resource_root=robot.root,
        vla_port=19802,
    )
    yield value
    value.close()


@pytest.fixture
def experiment(backend, reference):
    root = backend.directory
    (root / "workspace").mkdir()
    (root / "episodes").mkdir()
    (root / "observations").mkdir()
    config = {
        "terminal_feedback": False,
        "options": {
            "review_window_steps": 200,
            "review_max_windows": 1,
            "review_allow_reset": False,
            "vla_horizon": 50,
            "max_repeat": 1,
            "stop_min_steps": 200,
        },
    }
    exp = reference.runner.Experiment(config, root, backend)
    exp.phase = "evaluation"
    exp.reset_episode(
        {
            "id": "synthetic-align-s0",
            "task": "align_blocks",
            "robot": "arx_x5",
            "seed": 0,
            "layout_id": 0,
        }
    )
    return exp


@pytest.fixture
def vla(monkeypatch):
    load_openpi_client = import_module("XPolicyLab.policy.SenseNova-RoboRSI.pi05_source").load_openpi_client

    codec = load_openpi_client(POLICY_DIR)

    class Connection:
        def __init__(self):
            self.actions = np.zeros((50, 14))
            self.actions[:, [6, 13]] = [-0.5, 1.5]
            self.inputs = []
            self.first = True
            self.closed = False

        def send(self, value):
            assert isinstance(value, bytes)
            self.inputs.append(codec.unpackb(value))

        def recv(self, timeout):
            if self.first:
                self.first = False
                return codec.packb({"synthetic_metadata": True})
            return codec.packb({"actions": self.actions})

        def close(self):
            self.closed = True

    connection = Connection()
    import websockets.sync.client

    def connect(uri, **kwargs):
        assert uri == "ws://127.0.0.1:19802"
        assert kwargs["compression"] is None
        return connection

    monkeypatch.setattr(websockets.sync.client, "connect", connect)
    return connection


def propose(exp):
    return exp.request(
        {"op": "vla_propose", "subgoal": "Place the visible block beside its neighbor."}
    )


def review(proposal, **kwargs):
    return {
        "op": "vla_review",
        "proposal_id": proposal["proposal_id"],
        "decision": "accept",
        "reason": "Synthetic measured-state review.",
        "assessment": {"current_subgoal": "Synthetic current subgoal."},
        "targets": None,
        **kwargs,
    }


def test_original_proposal_rgb_joint_order_and_gripper_only_clip(
    experiment, backend, vla
):
    source_rgb = observation()["vision"]["cam_head"]["color"]
    proposal = propose(experiment)
    assert proposal["horizon"] == 50 and proposal["selected_waypoint_indices"] == list(
        range(50)
    )
    assert proposal["robot_moved"] is False and backend.control_step == 0
    assert proposal["fk_validation"]["passed"] is True
    assert proposal["gripper_clipped_values"] == 100
    assert proposal["targets"][0]["left"]["gripper_closed"] is True
    assert proposal["targets"][0]["right"]["gripper_closed"] is False
    payload = vla.inputs[0]
    assert payload["state"].shape == (14,) and payload["state"].dtype == np.float32
    np.testing.assert_array_equal(payload["state"], [0] * 6 + [1] + [0] * 6 + [1])
    assert set(payload["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    np.testing.assert_array_equal(
        payload["images"]["cam_high"], source_rgb.transpose(2, 0, 1)
    )
    assert not source_rgb.flags.writeable
    assert (source_rgb[0, 0] == [241, 18, 9]).all()


def test_gpt_observations_preserve_rgb_pixels_in_png(experiment):
    from PIL import Image

    cameras = experiment.latest["observation"]["cameras"]
    assert len(cameras) == 3
    for camera, uri in cameras.items():
        assert uri.startswith("obs://") and uri.endswith(".png")
        path = experiment.active["observation_dir"] / uri.removeprefix("obs://")
        encoded = path.read_bytes()
        assert encoded.startswith(b"\x89PNG\r\n\x1a\n")
        with Image.open(path) as image:
            assert image.format == "PNG" and image.mode == "RGB"
            rgb = np.asarray(image)
        assert rgb.shape == (3, 4, 3) and rgb.dtype == np.uint8
        np.testing.assert_array_equal(rgb, observation()["vision"][camera]["color"])


def test_original_default_five_review_yields_one_joint_per_real_ack(
    experiment, backend, vla
):
    proposal = propose(experiment)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(experiment.request, review(proposal))
        for step in range(5):
            action = backend.next_action(timeout=2)
            assert action.step_index == step
            assert set(action.action) == {
                name for name in observation()["state"] if "joint" in name
            }
            assert not result.done()
            assert action.action["left_ee_joint_state"] == [0.0]
            assert action.action["right_ee_joint_state"] == [1.0]
            backend.acknowledge(action.request_id, observation())
        reply = result.result(timeout=2)
    assert reply["executed_steps"] == 5 and backend.control_step == 5
    assert reply["observation"]["control_step"] == 5
    assert experiment.vla_pending is None
    with pytest.raises(queue.Empty):
        backend.next_action(timeout=0.01)


def test_terminal_unknown_success_truncates_review_without_fabricating_score(
    experiment, backend, vla
):
    proposal = propose(experiment)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(
            experiment.request, review(proposal, approved_steps=50)
        )
        for step in range(3):
            action = backend.next_action(timeout=2)
            backend.acknowledge(action.request_id, observation(), terminated=step == 2)
        reply = result.result(timeout=2)
    assert reply["done"] is True and reply["executed_steps"] == 3
    assert "success" not in reply
    assert experiment.latest["metrics"] == {
        "success": None,
        "source": "not_exposed_to_policy",
    }
    assert experiment.results[0]["success"] is None
    assert experiment.vla_pending is None
    assert backend.control_step == 3


def test_each_approved_target_recomputes_original_dls_from_new_measured_joints(
    experiment, backend, vla
):
    from scipy.spatial.transform import Rotation

    proposal = propose(experiment)
    targets = copy.deepcopy(proposal["targets"][:5])
    for target in targets:
        target["left"]["position"][1] += 0.01
    issued = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(
            experiment.request,
            review(proposal, decision="modify", targets=targets),
        )
        for _ in range(5):
            pending = backend.next_action(timeout=2)
            issued.append(pending.action["left_arm_joint_state"])
            measured = observation()
            for arm in ("left", "right"):
                joints = np.asarray(pending.action[f"{arm}_arm_joint_state"])
                kinematics = backend._adapter.kinematics
                pose = kinematics.root(arm) @ kinematics.fk[arm].matrix(joints)
                xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
                measured["state"][f"{arm}_arm_joint_state"] = joints
                measured["state"][f"{arm}_ee_pose"] = np.r_[
                    pose[:3, 3], xyzw[[3, 0, 1, 2]]
                ]
            backend.acknowledge(pending.request_id, measured)
        assert result.result(timeout=2)["executed_steps"] == 5
    assert not np.allclose(issued[0], issued[1])


def test_reject_moves_nothing_then_allows_new_proposal(experiment, backend, vla):
    proposal = propose(experiment)
    reply = experiment.request(review(proposal, decision="reject"))
    assert reply["status"] == "rejected" and reply["robot_moved"] is False
    assert backend.control_step == 0
    with pytest.raises(queue.Empty):
        backend.next_action(timeout=0.01)
    assert propose(experiment)["proposal_id"] != proposal["proposal_id"]


@pytest.mark.parametrize("span", [True, 1, 51, 5.0])
def test_original_invalid_review_span_keeps_unmoved_proposal(
    experiment, backend, reference, vla, span
):
    proposal = propose(experiment)
    with pytest.raises(reference.InvalidAction, match="integer from 2 to 50"):
        experiment.request(review(proposal, approved_steps=span))
    assert (
        backend.control_step == 0
        and experiment.vla_pending["proposal_id"] == proposal["proposal_id"]
    )


def test_direct_act_and_unchanged_modify_are_rejected(
    experiment, backend, reference, vla
):
    proposal = propose(experiment)
    with pytest.raises(reference.InvalidAction, match="vla_propose and vla_review"):
        experiment.request({"op": "act", "repeat": 1, "action": {}})
    with pytest.raises(reference.InvalidAction, match="actual change"):
        experiment.request(
            review(proposal, decision="modify", targets=proposal["targets"][:5])
        )
    assert backend.control_step == 0


def test_bounds_checked_against_measured_state_each_step(
    experiment, backend, reference, vla
):
    proposal = propose(experiment)
    targets = copy.deepcopy(proposal["targets"][:5])
    targets[1]["left"]["position"][0] += 0.06
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(
            experiment.request, review(proposal, decision="modify", targets=targets)
        )
        action = backend.next_action(timeout=2)
        backend.acknowledge(action.request_id, observation())
        with pytest.raises(reference.InvalidAction, match="5 cm"):
            result.result(timeout=2)
    assert backend.control_step == 1 and experiment.vla_pending is None
    with pytest.raises(queue.Empty):
        backend.next_action(timeout=0.01)


def test_zero_motion_geometry_rejection_keeps_proposal_repairable(
    experiment, backend, reference, vla
):
    proposal = propose(experiment)
    targets = copy.deepcopy(proposal["targets"][:5])
    targets[0]["left"]["position"][0] += 0.06
    with pytest.raises(reference.InvalidAction, match="5 cm"):
        experiment.request(review(proposal, decision="modify", targets=targets))
    assert (
        backend.control_step == 0
        and experiment.vla_pending["proposal_id"] == proposal["proposal_id"]
    )


def test_cancel_releases_step_even_while_original_review_holds_experiment_lock(
    experiment, backend, vla
):
    proposal = propose(experiment)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(experiment.request, review(proposal))
        action = backend.next_action(timeout=2)
        with pytest.raises(ValueError, match="current delivered"):
            backend.acknowledge(action.request_id + 1, observation())
        assert backend.control_step == 0
        backend.cancel()
        with pytest.raises(BackendClosed):
            result.result(timeout=2)
    assert vla.closed
    with pytest.raises(BackendClosed):
        backend.next_action(timeout=0.01)
    backend.close()


@pytest.mark.parametrize(
    "actions", [np.zeros((49, 14)), np.zeros((50, 13)), np.full((50, 14), np.nan)]
)
def test_invalid_vla_trajectory_rejected_without_motion(
    experiment, backend, vla, actions
):
    vla.actions = actions
    with pytest.raises(RuntimeError, match="Invalid pi0.5 trajectory"):
        backend.request(op="vla_propose", subgoal="Synthetic proposal.")
    assert backend.control_step == 0


def test_metadata_and_urdf_pins_are_required(reference, robot, tmp_path):
    directory = tmp_path / "bad-episode"
    directory.mkdir()
    kwargs = dict(
        task_name="align_blocks",
        episode_dir=directory,
        initial_observation=observation(),
        step_limit=200,
        metadata_path=robot.metadata,
        expected_metadata_sha256=sha(robot.metadata),
        resource_root=robot.root,
        vla_port=19802,
    )
    with pytest.raises(ValueError, match="pinned resource"):
        HybridBackend(reference, **{**kwargs, "expected_metadata_sha256": "0" * 64})
    robot.urdf.write_text(robot.urdf.read_text() + "\n")
    with pytest.raises(ValueError, match="pinned resource"):
        HybridBackend(reference, **kwargs)


def test_fk_mismatch_rejected_before_any_action(backend, reference):
    backend._latest["state"]["left_ee_pose"] = np.array([0.7, 0, 0, 1, 0, 0, 0])
    with pytest.raises(ValueError, match="FK does not match"):
        backend.request(
            op="reset",
            episode={"task": "align_blocks", "robot": "arx_x5"},
            record_dir=backend.directory / "native",
            observation_dir=backend.directory / "observations",
        )
    assert backend.control_step == 0


def test_native_owner_controls_reset_and_termination(experiment, backend, reference):
    with pytest.raises(reference.InvalidAction, match="owns resets"):
        backend.request(
            op="reset",
            episode={"task": "align_blocks", "robot": "arx_x5"},
            record_dir=backend.directory / "native",
            observation_dir=backend.directory / "observations",
        )
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(backend.request, op="finish")
        try:
            pending = backend.next_action(timeout=2)
            assert pending.kind == "finish" and pending.action == {}
            assert pending.step_index == 0 and not result.done()
            with pytest.raises(ValueError, match="terminal acknowledgement"):
                backend.acknowledge(pending.request_id, observation())
            assert backend.control_step == 0 and not result.done()
            backend.acknowledge(pending.request_id, observation(), terminated=True)
            reply = result.result(timeout=2)
        finally:
            if not result.done():
                backend.cancel()
    assert reply["done"] is True
    assert reply["metrics"] == {
        "success": None, "source": "not_exposed_to_policy"
    }
    assert backend.control_step == 0
