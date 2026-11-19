"""Offline checks for the yam32 state bridge, the relative-action anchor and the ee action path.

Run from the benchmark root with the policy env, e.g.

    cd RoboDojo
    PYTHONPATH=$PWD:$PWD/XPolicyLab/policy/FlexPi/FlexPi:$PWD/XPolicyLab/policy/FlexPi/FlexPi/src \
        conda run -n flexpi python XPolicyLab/policy/FlexPi/tests/test_state_bridge.py

No simulator, no weights. Also collectable by pytest.
"""

from types import SimpleNamespace

import numpy as np
import torch

from XPolicyLab.policy.FlexPi.state_bridge import (
    _LINK6_TO_GRASP_POS,
    _LINK6_TO_GRASP_ROT,
    _YAM32_IDS_FOR_RD14,
    EE_POSE_DIM,
    build_yam32_state,
    ee_action_dim_info,
    eef_block_to_pose7,
    env_pose_to_base,
    load_base_transforms,
    mat_to_quat_wxyz,
    pose7_error,
    resolve_bridge,
    rot6d_rows_to_mat,
    yam32_to_rd14,
    yam32_to_rd_ee16,
)

RNG = np.random.default_rng(0)


def _random_quat_wxyz():
    q = RNG.normal(size=4)
    return q / np.linalg.norm(q)


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def test_resolve_bridge():
    assert resolve_bridge("auto", 14, 14) == "none"
    assert resolve_bridge("auto", 32, 14) == "yam32"
    assert resolve_bridge("auto", None, 14) == "none"
    assert resolve_bridge("yam32", None, 14) == "yam32"
    # Mixed yam/mm2 + AgiBot checkpoints list admissible raw widths.
    assert resolve_bridge("auto", (32, 34), 14) == "yam32"
    try:
        resolve_bridge("auto", 20, 14)
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported dims must raise")


def test_index_map_matches_yam_layout():
    from flexpi.datasets.lerobot.utils.yam_eef import STATE_LAYOUT

    expected = np.r_[
        np.arange(32)[STATE_LAYOUT["left_joint"]],
        np.arange(32)[STATE_LAYOUT["left_grip"]],
        np.arange(32)[STATE_LAYOUT["right_joint"]],
        np.arange(32)[STATE_LAYOUT["right_grip"]],
    ]
    np.testing.assert_array_equal(_YAM32_IDS_FOR_RD14, expected)


def test_base_transforms_from_env_cfg():
    tf = load_base_transforms("arx_x5")
    np.testing.assert_allclose(tf["left"][0], [-0.3, -0.45, 0.765])
    np.testing.assert_allclose(tf["right"][0], [0.3, -0.45, 0.765])
    for side in ("left", "right"):
        assert abs(np.linalg.norm(tf[side][1]) - 1.0) < 1e-9


def test_base_pose_maps_to_identity_and_matches_quaternion_math():
    base_pos, base_q = np.array([0.3, -0.45, 0.765]), np.array([0.707, 0.0, 0.0, 0.707])
    base_q = base_q / np.linalg.norm(base_q)
    pos, R = env_pose_to_base(np.r_[base_pos, base_q], base_pos, base_q)
    np.testing.assert_allclose(pos, 0.0, atol=1e-9)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-9)

    # A pose defined in the base frame, pushed to the env frame, must come back.
    from flexpi.datasets.lerobot.utils.yam_eef import rot6d_to_rot_mat

    rel_pos, rel_q = RNG.normal(size=3), _random_quat_wxyz()
    base_q_inv = base_q * np.array([1.0, -1.0, -1.0, -1.0])
    rot_q = lambda q, v: _quat_mul(_quat_mul(q, np.r_[0.0, v]), q * np.array([1.0, -1.0, -1.0, -1.0]))[1:]
    env_pose = np.r_[base_pos + rot_q(base_q, rel_pos), _quat_mul(base_q, rel_q)]
    pos, R = env_pose_to_base(env_pose, base_pos, base_q)
    np.testing.assert_allclose(pos, rel_pos, atol=1e-9)
    _, R_rel = env_pose_to_base(np.r_[0, 0, 0, rel_q], np.zeros(3), np.array([1.0, 0, 0, 0]))
    np.testing.assert_allclose(R, R_rel, atol=1e-9)
    assert np.allclose(_quat_mul(base_q_inv, base_q), [1, 0, 0, 0])
    np.testing.assert_allclose(rot6d_to_rot_mat(R[:2, :].reshape(6)), R, atol=1e-9)


def test_round_trip_and_rot6d_rows():
    from flexpi.datasets.lerobot.utils.yam_eef import rot9_to_rot6d

    tf = load_base_transforms("arx_x5")
    packed14 = RNG.uniform(-1, 1, size=14).astype(np.float32)
    packed14[[6, 13]] = [0.2, 0.9]
    obs_state = {
        "left_ee_pose": np.r_[RNG.normal(size=3), _random_quat_wxyz()],
        "right_ee_pose": np.ones(7),  # debug client value: unnormalized quaternion
    }
    s32 = build_yam32_state(obs_state, packed14, tf, eef_frame="arm_base_grasp")
    assert s32.shape == (32,) and s32.dtype == np.float32
    np.testing.assert_allclose(yam32_to_rd14(s32[None])[0], packed14, atol=1e-6)

    p_left, R_left = env_pose_to_base(obs_state["left_ee_pose"], *tf["left"])
    R_grasp = R_left @ _LINK6_TO_GRASP_ROT
    np.testing.assert_allclose(s32[3:9], rot9_to_rot6d(R_grasp.reshape(9)), atol=1e-6)
    np.testing.assert_allclose(s32[0:3], p_left + R_left @ _LINK6_TO_GRASP_POS, atol=1e-6)


def test_sim_zero_pose_matches_yam_fk_zero():
    """RoboDojo reset obs (all joints 0) must land on YAM FK(q=0) at grasp_site.

    Sim values: env_cfg/robot/dual_x5.yml bases and the link6 ee_pose recorded at frame 0 of
    RoboDojo expert data. YAM reference: the YAM rig's FK (i2rt YAM + linear_4310, grasp_site) at q=0,
    L pos [0.248, 0.004, 0.172], rot6d rows [0, 0, 1, 0, 1, 0]; R arm is the same up to FK noise.
    """
    tf = load_base_transforms("arx_x5")
    obs_state = {
        "left_ee_pose": np.array([-0.3, -0.352, 0.922, 0.707, 0.0, 0.0, 0.707]),
        "right_ee_pose": np.array([0.3, -0.352, 0.922, 0.707, 0.0, 0.0, 0.707]),
    }
    s32 = build_yam32_state(obs_state, np.zeros(14), tf, eef_frame="arm_base_grasp")
    for pos_sl, rot_sl in ((slice(0, 3), slice(3, 9)), (slice(9, 12), slice(12, 18))):
        assert np.linalg.norm(s32[pos_sl] - [0.248, 0.004, 0.172]) < 0.015
        np.testing.assert_allclose(s32[rot_sl], [0, 0, 1, 0, 1, 0], atol=1e-3)


def test_world_link6_reproduces_robodojo_dataset():
    """The default EEF frame must reproduce what the RoboDojo 34D dataset stores.

    Values are episode 1200 (insert_key) frame 0 of robodojo_34d: the ee_pose the
    simulator hands the adapter, and the 32D EEF block the dataset holds for that
    same frame. They are the SAME numbers because RoboDojo writes
    ``get_link_pose(is_relative=True)`` (world orientation, env-origin-relative
    position) and the converter passes it through untouched -- so `world_link6`
    is a pass-through and any transform here is a training/eval mismatch.

    Tolerance is 1e-6, not 0: the dataset is stored float32 (its own rot6d rows
    are orthonormal only to 4.5e-08), so the rot6d -> R -> quat round trip used
    to build the input carries ~2e-08. The bug this guards against was 0.64 m.
    """
    tf = load_base_transforms("arx_x5")
    obs_state = {
        "left_ee_pose": np.array(
            [-0.299529, -0.352300, 0.921500, 0.707000, -0.0, -0.0, 0.707214]
        ),
        "right_ee_pose": np.array(
            [0.300471, -0.352300, 0.921500, 0.707000, -0.0, -0.0, 0.707214]
        ),
    }
    expected = {
        "left": [-0.299529, -0.352300, 0.921500,
                 -0.000302, -1.0, -0.0, 1.0, -0.000302, -0.0],
        "right": [0.300471, -0.352300, 0.921500,
                  -0.000302, -1.0, -0.0, 1.0, -0.000302, -0.0],
    }
    s32 = build_yam32_state(obs_state, np.zeros(14), tf)  # default eef_frame
    np.testing.assert_allclose(s32[0:9], expected["left"], atol=1e-6)
    np.testing.assert_allclose(s32[9:18], expected["right"], atol=1e-6)

    # The two arms must stay distinguishable: in the common frame the dataset
    # uses they are 0.6 m apart, while the per-arm frame collapses them onto
    # each other at this (home) pose, which is what scored 0.0 on all 40 tasks.
    assert abs(s32[9] - s32[0]) > 0.5
    collapsed = build_yam32_state(
        obs_state, np.zeros(14), tf, eef_frame="arm_base_grasp"
    )
    np.testing.assert_allclose(collapsed[0:3], collapsed[9:12], atol=1e-6)


def test_eef_frame_rejects_unknown_value():
    tf = load_base_transforms("arx_x5")
    obs_state = {"left_ee_pose": np.r_[np.zeros(3), 1, 0, 0, 0],
                 "right_ee_pose": np.r_[np.ones(3), 1, 0, 0, 0]}
    try:
        build_yam32_state(obs_state, np.zeros(14), tf, eef_frame="base")
    except ValueError as exc:
        assert "eef_frame" in str(exc)
    else:
        raise AssertionError("unknown eef_frame must raise")


def test_gripper_clip():
    chunk = np.zeros((2, 32), dtype=np.float32)
    chunk[:, 18] = [-0.3, 0.5]
    chunk[:, 19] = [1.4, 1.0]
    out = yam32_to_rd14(chunk)
    np.testing.assert_allclose(out[:, 6], [0.0, 0.5])
    np.testing.assert_allclose(out[:, 13], [1.0, 1.0])


def _fake_deploy(transform, total_dim=32):
    """Minimal stand-in carrying just what _denormalize_action reads."""
    identity = SimpleNamespace(backward=lambda batch: batch)
    transforms = list(transform) if isinstance(transform, (list, tuple)) else [transform]

    def action_state_transform(batch):
        for t in transforms:
            batch = t.forward(batch)
        return batch

    class Merger:

        @staticmethod
        def backward(batch):
            batch["action"] = {"default": batch["action"]}
            if "state" in batch:
                batch["state"] = {"default": batch["state"]}
            return batch

    processor = SimpleNamespace(
        shape_meta={"action": [{"key": "default"}], "state": [{"key": "default"}]},
        action_state_merger=Merger(),
        normalizer=identity,
        action_state_transforms=transforms,
        action_state_transform=action_state_transform,
    )
    Merger.total_dim = total_dim
    return SimpleNamespace(processor=processor, _zero_anchor_warned=False)


def test_anchor_decodes_relative_joints():
    from experiments.robotwin.flexpi_policy.deploy_policy import WorldActionRobotWinPolicy as Policy
    from flexpi.datasets.lerobot.transforms.yam_relative_action import Yam32DRelativeAction

    tf = load_base_transforms("arx_x5")
    state14 = RNG.uniform(-1, 1, size=14).astype(np.float32)
    obs_state = {
        "left_ee_pose": np.r_[RNG.normal(size=3), _random_quat_wxyz()],
        "right_ee_pose": np.r_[RNG.normal(size=3), _random_quat_wxyz()],
    }
    state32 = build_yam32_state(obs_state, state14, tf)

    # Absolute target chunk near the state, encoded the way training does.
    T = 4
    abs32 = np.repeat(state32[None], T, axis=0)
    abs32[:, 20:32] += RNG.normal(scale=0.05, size=(T, 12))
    abs32[:, 18:20] = [0.3, 0.7]
    transform = Yam32DRelativeAction(anchor="first")
    encoded = transform.forward({
        "action": {"default": torch.as_tensor(abs32)[None]},
        "state": {"default": torch.as_tensor(state32)[None, None]},
    })["action"]["default"][0]

    fake = _fake_deploy(transform)
    decoded = Policy._denormalize_action(fake, encoded.clone(), anchor_state=state32)[0]
    np.testing.assert_allclose(decoded[:, 20:32], abs32[:, 20:32], atol=1e-5)
    np.testing.assert_allclose(decoded[:, 18:20], abs32[:, 18:20], atol=1e-6)
    np.testing.assert_allclose(yam32_to_rd14(decoded)[:, [0, 6, 7, 13]], abs32[:, [20, 18, 26, 19]], atol=1e-5)

    # Regression: without the anchor the joints come back as bare deltas.
    fake = _fake_deploy(transform)
    zero_decoded = Policy._denormalize_action(fake, encoded.clone())[0]
    assert fake._zero_anchor_warned
    assert np.abs(zero_decoded[:, 20:32] - abs32[:, 20:32]).max() > 0.1


def test_mixed_34d_checkpoint_decodes_to_yam32():
    """mm2+yam+AgiBot checkpoints: AlignTo34DByWidth -> Yam32DRelativeAction(34)."""
    from experiments.robotwin.flexpi_policy.deploy_policy import WorldActionRobotWinPolicy as Policy
    from flexpi.datasets.lerobot.transforms.mixed_embodiment_align import (
        _PAD_SLOTS,
        AlignTo34DByWidth,
    )
    from flexpi.datasets.lerobot.transforms.yam_relative_action import Yam32DRelativeAction

    tf = load_base_transforms("arx_x5")
    state14 = RNG.uniform(-1, 1, size=14).astype(np.float32)
    obs_state = {
        "left_ee_pose": np.r_[RNG.normal(size=3), _random_quat_wxyz()],
        "right_ee_pose": np.r_[RNG.normal(size=3), _random_quat_wxyz()],
    }
    state32 = build_yam32_state(obs_state, state14, tf)

    T = 4
    abs32 = np.repeat(state32[None], T, axis=0)
    abs32[:, 20:32] += RNG.normal(scale=0.05, size=(T, 12))
    abs32[:, 18:20] = [0.3, 0.7]
    chain = [AlignTo34DByWidth(), Yam32DRelativeAction(anchor="first", state_dim=34)]
    batch = {
        "action": {"default": torch.as_tensor(abs32, dtype=torch.float32)[None]},
        "state": {"default": torch.as_tensor(state32, dtype=torch.float32)[None, None]},
    }
    for t in chain:
        batch = t.forward(batch)
    encoded = batch["action"]["default"][0]
    assert encoded.shape == (T, 34)
    assert batch["action_dim_is_pad"].nonzero().flatten().tolist() == list(_PAD_SLOTS)
    assert torch.all(encoded[:, list(_PAD_SLOTS)] == 0)

    fake = _fake_deploy(chain, total_dim=34)
    decoded = Policy._denormalize_action(fake, encoded.clone(), anchor_state=state32)[0]
    assert decoded.shape == (T, 32)
    np.testing.assert_allclose(decoded[:, 20:32], abs32[:, 20:32], atol=1e-5)
    np.testing.assert_allclose(decoded[:, 18:20], abs32[:, 18:20], atol=1e-6)
    np.testing.assert_allclose(decoded[:, :18], abs32[:, :18], atol=1e-4)


# --- ee action path ------------------------------------------------------------

# Episode 1200 frame 0, as RoboDojo reports it (see test_world_link6_reproduces_robodojo_dataset).
_EP1200_EE = {
    "left_ee_pose": np.array([-0.299529, -0.352300, 0.921500, 0.707000, -0.0, -0.0, 0.707214]),
    "right_ee_pose": np.array([0.300471, -0.352300, 0.921500, 0.707000, -0.0, -0.0, 0.707214]),
}
_SIDES = (("left", slice(0, 9)), ("right", slice(9, 18)))


def _random_ee_state():
    return {
        "left_ee_pose": np.r_[RNG.normal([-0.25, -0.3, 1.0], 0.1), _random_quat_wxyz()],
        "right_ee_pose": np.r_[RNG.normal([0.25, -0.3, 1.0], 0.1), _random_quat_wxyz()],
    }


def _small_rotation_wxyz(angle):
    axis = RNG.normal(size=3)
    axis /= np.linalg.norm(axis)
    return np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis]


def _assert_pose_close(a, b, pos_mm=1e-3, rot_deg=1e-3):
    pos, rot = pose7_error(a, b)
    assert pos < pos_mm and rot < rot_deg, f"{a} vs {b}: {pos:.2e} mm, {rot:.2e} deg"


def test_ee_pose_round_trips_through_the_state_block():
    """obs pose7 -> 32D EEF block -> pose7 is the identity, for either eef_frame.

    This is the contract that makes an EEF-block target mean what RoboDojo will
    read: the action goes back through exactly the frame the state came in by.
    """
    tf = load_base_transforms("arx_x5")
    for obs_state in [_EP1200_EE] + [_random_ee_state() for _ in range(20)]:
        for frame in ("world_link6", "arm_base_grasp"):
            s32 = build_yam32_state(obs_state, np.zeros(14), tf, eef_frame=frame)
            for side, sl in _SIDES:
                pose7 = eef_block_to_pose7(s32[sl], side, frame, tf)
                _assert_pose_close(pose7, obs_state[f"{side}_ee_pose"])
                assert pose7[3] >= 0.0


def test_ee16_matches_the_simulator_contract():
    """yam32_to_rd_ee16 + ee unpack gives exactly RoboDojo's ee action dicts.

    The key/dim table is the ee half of validate_action_dict for a dual-arm robot
    (src/eval_client/eval_env.py): a 7D pose and the robot's gripper width per arm.
    """
    from XPolicyLab.utils.process_data import get_robot_action_dim_info, unpack_robot_state

    info = get_robot_action_dim_info("arx_x5")
    tf = load_base_transforms("arx_x5")
    T = 5
    chunk = np.stack([
        build_yam32_state(_random_ee_state(), RNG.uniform(-1, 1, 14), tf) for _ in range(T)
    ])
    chunk[:, 18] = [-0.5, 0.0, 0.3, 1.0, 1.7]
    chunk[:, 19] = [2.0, 1.0, 0.5, 0.0, -1.0]
    ee16, held = yam32_to_rd_ee16(chunk)
    assert ee16.shape == (T, 16) and ee16.dtype == np.float32 and held == 0
    np.testing.assert_allclose(ee16[:, 7], [0.0, 0.0, 0.3, 1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(ee16[:, 15], [1.0, 1.0, 0.5, 0.0, 0.0], atol=1e-6)

    contract = {
        "left_ee_pose": EE_POSE_DIM,
        "left_ee_joint_state": info["ee_dim"][0],
        "right_ee_pose": EE_POSE_DIM,
        "right_ee_joint_state": info["ee_dim"][1],
    }
    steps = unpack_robot_state(ee16, "ee", ee_action_dim_info(info), source_type="obs")
    assert len(steps) == T
    for t, step in enumerate(steps):
        assert set(step) == set(contract)
        for key, dim in contract.items():
            assert step[key].shape == (dim,), (key, step[key].shape)
        for side, sl in _SIDES:
            _assert_pose_close(step[f"{side}_ee_pose"], eef_block_to_pose7(chunk[t, sl], side))
            assert abs(np.linalg.norm(step[f"{side}_ee_pose"][3:]) - 1.0) < 1e-6

    # XPolicyLab sizes an ee pose as 7 itself since bb87d19, whatever arm_dim says,
    # so the stock dim info now gives the same steps.
    stock = unpack_robot_state(ee16, "ee", info, source_type="obs")
    assert len(stock) == T
    for got, want in zip(stock, steps):
        assert set(got) == set(want) and all(np.array_equal(got[k], want[k]) for k in want)


def test_ee16_holds_a_pose_that_has_no_rotation():
    tf = load_base_transforms("arx_x5")
    chunk = np.repeat(build_yam32_state(_random_ee_state(), np.zeros(14), tf)[None], 3, axis=0)
    chunk[0, 3:9] = 0.0                                  # left: zero rot6d rows
    chunk[1, 9] = np.nan                                 # right: non-finite position
    chunk[2, 3:9] += RNG.normal(scale=0.05, size=6)      # left: off-orthonormal, still a rotation
    hold = {"left": np.r_[1.0, 2.0, 3.0, 1.0, 0, 0, 0], "right": np.r_[4.0, 5.0, 6.0, 1.0, 0, 0, 0]}
    ee16, held = yam32_to_rd_ee16(chunk, hold_pose7=hold)
    assert held == 2 and np.isfinite(ee16).all()
    np.testing.assert_allclose(ee16[0, 0:7], hold["left"])
    np.testing.assert_allclose(ee16[1, 8:15], hold["right"])
    # A merely non-orthonormal rot6d is Gram-Schmidted, not held.
    R = rot6d_rows_to_mat(chunk[2, 3:9])
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(ee16[2, 3:7], mat_to_quat_wxyz(R), atol=1e-6)
    try:
        yam32_to_rd_ee16(chunk)
    except ValueError:
        pass
    else:
        raise AssertionError("an invalid pose without hold_pose7 must raise")


def test_decoded_eef_block_is_the_ee_target():
    """A 34D-relative chunk decoded against the anchor sends the true absolute EE targets.

    Same chain as test_mixed_34d_checkpoint_decodes_to_yam32, but with the EEF
    moved (3 cm, 0.2 rad) so the SE(3) relative decode is not the identity.
    """
    from experiments.robotwin.flexpi_policy.deploy_policy import WorldActionRobotWinPolicy as Policy
    from flexpi.datasets.lerobot.transforms.mixed_embodiment_align import AlignTo34DByWidth
    from flexpi.datasets.lerobot.transforms.yam_relative_action import Yam32DRelativeAction

    tf = load_base_transforms("arx_x5")
    anchor_obs = _random_ee_state()
    state32 = build_yam32_state(anchor_obs, RNG.uniform(-1, 1, 14).astype(np.float32), tf)
    T = 4
    abs32 = np.repeat(state32[None], T, axis=0)
    targets = []
    for t in range(T):
        target = {}
        for side in ("left", "right"):
            pose = anchor_obs[f"{side}_ee_pose"].astype(np.float64).copy()
            pose[:3] += RNG.normal(scale=0.03, size=3)
            pose[3:] = _quat_mul(pose[3:], _small_rotation_wxyz(0.2))
            target[f"{side}_ee_pose"] = pose
        abs32[t, :18] = build_yam32_state(target, np.zeros(14), tf)[:18]
        targets.append(target)
    abs32[:, 20:32] += RNG.normal(scale=0.05, size=(T, 12))

    chain = [AlignTo34DByWidth(), Yam32DRelativeAction(anchor="first", state_dim=34)]
    batch = {
        "action": {"default": torch.as_tensor(abs32, dtype=torch.float32)[None]},
        "state": {"default": torch.as_tensor(state32, dtype=torch.float32)[None, None]},
    }
    for t in chain:
        batch = t.forward(batch)
    fake = _fake_deploy(chain, total_dim=34)
    decoded = Policy._denormalize_action(fake, batch["action"]["default"][0].clone(), anchor_state=state32)[0]

    ee16, held = yam32_to_rd_ee16(np.asarray(decoded))
    assert held == 0
    for t in range(T):
        for i, side in enumerate(("left", "right")):
            _assert_pose_close(ee16[t, 8 * i:8 * i + 7], targets[t][f"{side}_ee_pose"], pos_mm=0.1, rot_deg=0.02)


def _robodojo_obs(obs_state, env_idx=0):
    """What obs_manager hands the server, minus depth (under dummy nothing reads it)."""
    vision = {cam: {"color": np.zeros((48, 64, 3), np.uint8)} for cam in ("cam_head", "cam_left_wrist", "cam_right_wrist")}
    return {"vision": vision, "state": obs_state, "instruction": "test", "env_idx": env_idx}


def _obs_state():
    return {
        "left_arm_joint_state": RNG.uniform(-1, 1, 6),
        "right_arm_joint_state": RNG.uniform(-1, 1, 6),
        "left_ee_joint_state": np.array([0.25]),
        "right_ee_joint_state": np.array([0.75]),
        **_random_ee_state(),
    }


def _dummy_model(action_type, **cfg):
    from XPolicyLab.policy.FlexPi.model import Model

    return Model({
        "action_type": action_type,
        "env_cfg_type": "arx_x5",
        "allow_dummy_policy": True,
        "state_bridge": "yam32",
        "replan_steps": 3,
        "log_action_chunk": False,
        **cfg,
    })


def test_model_ee_dummy_holds_the_current_pose_and_traces_it():
    import os
    import tempfile

    obs_state = _obs_state()
    with tempfile.TemporaryDirectory() as dump:
        os.environ["FLEXPI_DUMP_ACTIONS"] = dump
        try:
            model = _dummy_model("ee")
        finally:
            del os.environ["FLEXPI_DUMP_ACTIONS"]
        model.reset()
        model.update_obs_batch([_robodojo_obs(obs_state)])
        steps = model.get_action_batch([0])[0]
        model.reset()

        assert len(steps) == 3
        for step in steps:
            assert set(step) == {"left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state"}
            for side, grip in (("left", 0.25), ("right", 0.75)):
                _assert_pose_close(step[f"{side}_ee_pose"], obs_state[f"{side}_ee_pose"])
                np.testing.assert_allclose(step[f"{side}_ee_joint_state"], [grip], atol=1e-6)

        with np.load(os.path.join(dump, "ep000_env0.npz")) as z:
            assert str(z["action_type"]) == "ee" and "rd14" not in z.files
            assert z["sent16"].shape == (1, 3, 16) and z["joint_head14"].shape == (1, 3, 14)


def test_model_joint_dummy_is_unchanged():
    model = _dummy_model("joint")
    model.update_obs_batch([_robodojo_obs(_obs_state())])
    steps = model.get_action_batch([0])[0]
    assert set(steps[0]) == {"left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state", "right_ee_joint_state"}
    assert all(np.all(v == 0) for step in steps for v in step.values())


def test_model_ee_needs_the_eef_block_and_a_known_type():
    for action_type, cfg, needle in (("ee", {"state_bridge": "auto"}, "state_bridge"), ("delta_ee", {}, "action_type")):
        try:
            _dummy_model(action_type, **cfg)
        except ValueError as exc:
            assert needle in str(exc), exc
        else:
            raise AssertionError(f"action_type={action_type} {cfg} must raise")


def test_sent_arrays_keep_rd14_for_commanded_joints_only():
    from XPolicyLab.policy.FlexPi.model import _sent_arrays

    a = np.zeros((2, 14))
    assert set(_sent_arrays("joint", a, a)) == {"rd14"}
    assert set(_sent_arrays("ee", np.zeros((2, 16)), a)) == {"sent16", "joint_head14"}


def test_deploy_label_check_exits_on_a_mislabelled_run():
    import os

    import XPolicyLab.policy.FlexPi.deploy as deploy

    exits, real_exit = [], os._exit
    env = SimpleNamespace(additional_info="ckpt_name=x,action_type=joint")
    ee_action = {"left_ee_pose": np.zeros(7), "left_ee_joint_state": np.zeros(1),
                 "right_ee_pose": np.zeros(7), "right_ee_joint_state": np.zeros(1)}
    os._exit = exits.append
    try:
        deploy._LABEL["checked"] = False
        deploy._check_action_label(env, {"left_arm_joint_state": np.zeros(6), "left_ee_joint_state": np.zeros(1)})
        assert exits == []
        deploy._LABEL["checked"] = False
        deploy._check_action_label(env, ee_action)
        assert exits == [2]
    finally:
        os._exit = real_exit
        deploy._LABEL["checked"] = False


def test_deploy_ik_probe_counts_fails_and_jumps():
    import XPolicyLab.policy.FlexPi.deploy as deploy

    q = np.zeros(6)
    results = iter([
        {"status": "Success", "joint_value": q + 0.1},
        {"status": "Fail"},
        {"status": "Fail"},
        {"status": "Success", "joint_value": q + np.r_[0, 0, 0, 0, 0, 2 * np.pi]},
    ])

    class RobotManager:
        def solve_ik(self, target_pose, env_idx, robot, trans="world"):
            return next(results)

        def get_joint(self, robot, env_idx_list=None):
            return {env_idx_list[0]: q}

    env = SimpleNamespace(robot_manager=RobotManager())
    deploy._IK["stats"].clear()
    deploy._IK["off"] = False
    deploy._install_ik_probe(env)
    deploy._install_ik_probe(env)  # once per robot_manager: no double counting
    robot = SimpleNamespace(arm_name="left_arm")
    for _ in range(4):
        env.robot_manager.solve_ik(np.zeros(7), 0, robot)
    assert deploy._ik_episode(0) == {
        "left_arm": {"calls": 4, "fails": 2, "max_fail_run": 2, "max_dq": round(2 * np.pi, 4), "jumps": 1}
    }
    assert deploy._ik_episode(1) is None
    deploy._IK["stats"].clear()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
