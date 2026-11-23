"""Offline replay of the RoboDojo_real corpus through the yam32 bridge, one eef_frame per corpus variant.

    raw        robodojo_realworld_34d            each robot's EE pose as recorded   eef_frame world_link6
    relabeled  robodojo_realworld_34d_relabeled  R_relabeled = R_raw @ T[robot]     eef_frame relabeled

For 12 episodes of each robot (arx_x5, piper, piper_x; 2 from each of its 6 tasks),
every frame is replayed against the upstream lowdim the corpus was built from
(float64, rebuilt into the live ``obs['state']``: ``{side}_arm_joint_state`` (6),
``{side}_ee_joint_state`` (1), ``{side}_ee_pose`` (7, wxyz)):

    state  build_yam32_state(obs['state'])           == the corpus state, 32 real slots
    ee     eef_block_to_pose7(corpus action EEF)     == the raw action ee_poses, quaternion up to sign
    joint  yam32_to_rd14(corpus action)              == the raw action joints, grippers clipped to [0, 1]

and a few chunks per robot through ``Model`` itself (dummy policy and GT replay,
both action types), which is where env_cfg_type reaches the bridge. Corpus row t
is raw frame t (the corpus drops the last raw frame), and corpus episode i is raw
episode i % 100 of source dir i // 100 (meta/info.json ``source_dirs``).

The data roots come from ROBODOJO_REAL_LOWDIM_ROOT (one .npz per upstream RoboDojo_real HDF5,
``<source_dir>/episode_<k:07d>.npz``, holding ``state__<key>`` / ``action__<key>`` for every key of the HDF5
``state`` / ``action`` groups), ROBODOJO_REAL_RAW_ROOT (the raw corpus, whose meta/info.json carries
``source_dirs`` and ``dropped_episodes``) and ROBODOJO_REAL_RELABELED_ROOT (the relabeled corpus); the tests
that read them skip when one is unset or absent. Run like test_state_bridge.py, from the benchmark root:

    PYTHONPATH=$PWD:$PWD/XPolicyLab/policy/FlexPi/FlexPi:$PWD/XPolicyLab/policy/FlexPi/FlexPi/src \
        python XPolicyLab/policy/FlexPi/tests/test_robodojo_real_frames.py
"""

import contextlib
import functools
import json
import os

import numpy as np

from XPolicyLab.policy.FlexPi.state_bridge import (
    _LINK6_TO_GRASP_ROT,
    _RELABEL_ROT,
    EEF_FRAMES,
    build_yam32_state,
    eef_block_to_pose7,
    pose7_error,
    yam32_to_rd14,
)
from XPolicyLab.utils.process_data import pack_robot_state

LOWDIM_ROOT = os.environ.get("ROBODOJO_REAL_LOWDIM_ROOT", "")
RAW_ROOT = os.environ.get("ROBODOJO_REAL_RAW_ROOT", "")
RELABELED_ROOT = os.environ.get("ROBODOJO_REAL_RELABELED_ROOT", "")
CORPUS = {"world_link6": RAW_ROOT, "relabeled": RELABELED_ROOT}

ROBOTS = ("arx_x5", "piper", "piper_x")
UPSTREAM_EPISODES_PER_TASK = 100
EPISODES_PER_TASK = 2
# All three robots are dual-arm 6 + 1 (XPolicyLab utils/robot/_robot_info.json).
ROBOT_INFO = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
KEEP32 = list(range(26)) + list(range(27, 33))  # 34D -> 32D: drop the pad joints 26/33
SIDES = (("left", slice(0, 9)), ("right", slice(9, 18)))
VISION = {cam: {"color": np.zeros((48, 64, 3), np.uint8)} for cam in ("cam_head", "cam_left_wrist", "cam_right_wrist")}
# The corpus is float32: 1 ulp is 1.2e-7 at 1.0, 2.4e-7 at the largest joint angles.
F32_TOL = 1e-6


# --- data ------------------------------------------------------------------------

def _require_data():
    missing = [p for p in (LOWDIM_ROOT, RAW_ROOT, RELABELED_ROOT) if not os.path.isdir(p)]
    if missing:
        import pytest

        pytest.skip(f"RoboDojo_real data not found: {missing}")


@functools.lru_cache(maxsize=None)
def _info(root):
    with open(os.path.join(root, "meta", "info.json")) as f:
        return json.load(f)


@functools.lru_cache(maxsize=None)
def _corpus_episode(root, ep):
    """``(state34, action34, robot)`` of corpus episode ``ep``, float32 as stored."""
    import pyarrow.parquet as pq

    info = _info(root)
    path = info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep)
    table = pq.read_table(os.path.join(root, path))
    robots = set(table["embodiment_index"].to_numpy().tolist())
    assert len(robots) == 1, (root, ep, robots)
    state = np.stack(table["observation.state"].to_numpy(zero_copy_only=False))
    action = np.stack(table["action"].to_numpy(zero_copy_only=False))
    return state, action, info["embodiments"][robots.pop()]


@functools.lru_cache(maxsize=None)
def _raw_episode(ep):
    source = _info(RAW_ROOT)["source_dirs"][ep // UPSTREAM_EPISODES_PER_TASK]
    path = os.path.join(LOWDIM_ROOT, source, f"episode_{ep % UPSTREAM_EPISODES_PER_TASK:07d}.npz")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


@functools.lru_cache(maxsize=None)
def _episodes_by_robot():
    """``{robot: [corpus episode]}``: EPISODES_PER_TASK seeded picks from every task, dropped ones excluded."""
    info = _info(RAW_ROOT)
    dropped = {int(ep) for ep in info["dropped_episodes"]}
    rng = np.random.default_rng(0)
    out = {}
    for task in range(len(info["source_dirs"])):
        first = task * UPSTREAM_EPISODES_PER_TASK
        kept = [ep for ep in range(first, first + UPSTREAM_EPISODES_PER_TASK) if ep not in dropped]
        picks = sorted(int(ep) for ep in rng.choice(kept, size=EPISODES_PER_TASK, replace=False))
        out.setdefault(_corpus_episode(RAW_ROOT, picks[0])[2], []).extend(picks)
    assert sorted(out) == sorted(ROBOTS) and all(len(eps) >= 10 for eps in out.values()), out
    return out


def _obs_state(raw, t, kind="state"):
    """RoboDojo's live ``obs['state']`` at raw frame ``t``; ``kind="action"`` gives the commanded one."""
    return {
        f"{side}_{key}": raw[f"{kind}__{side}_{key}s"][t]
        for side in ("left", "right")
        for key in ("arm_joint_state", "ee_joint_state", "ee_pose")
    }


def _pack14(obs_state):
    """The adapter's 14D pack of ``obs['state']`` (model.py _encode_obs)."""
    return pack_robot_state(
        {"state": obs_state}, "joint", ROBOT_INFO, source_type="obs", state_type="state"
    ).astype(np.float32)


def _quat_error(got, want):
    """Worst elementwise |got - want| over unit wxyz quaternions, ``want`` normalized and sign-matched to ``got``."""
    want = want / np.linalg.norm(want, axis=-1, keepdims=True)
    want = want * np.sign(np.sum(got * want, axis=-1, keepdims=True))
    return float(np.abs(got - want).max())


def _corpus_errors(eef_frame):
    """Per robot, the worst |build_yam32_state(raw obs) - corpus state| against each corpus variant."""
    errors = {}
    for robot, eps in _episodes_by_robot().items():
        for ep in eps:
            raw = _raw_episode(ep)
            n = len(_corpus_episode(RAW_ROOT, ep)[0])
            assert len(raw["state__left_ee_poses"]) == n + 1, ep
            built = np.stack([
                build_yam32_state(obs, _pack14(obs), None, eef_frame=eef_frame, env_cfg_type=robot)
                for obs in (_obs_state(raw, t) for t in range(n))
            ])
            for frame, root in CORPUS.items():
                state34, _, ep_robot = _corpus_episode(root, ep)
                assert ep_robot == robot, (root, ep, ep_robot, robot)
                err = float(np.abs(built - state34[:, KEEP32]).max())
                errors[robot, frame] = max(errors.get((robot, frame), 0.0), err)
    return errors


@contextlib.contextmanager
def _patched(module, name, value):
    real = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, real)


def _model_module():
    import XPolicyLab.policy.FlexPi.model as model

    return model


def _dummy_model(action_type, env_cfg_type, eef_frame, **cfg):
    """A dummy-policy Model with the yam32 bridge. RoboDojo's sim env_cfg has no
    piper/piper_x, so the arm/gripper widths come from ROBOT_INFO instead."""
    model = _model_module()
    with _patched(model, "get_robot_action_dim_info", lambda _: dict(ROBOT_INFO)):
        return model.Model({
            "action_type": action_type,
            "env_cfg_type": env_cfg_type,
            "eef_frame": eef_frame,
            "allow_dummy_policy": True,
            "state_bridge": "yam32",
            "log_action_chunk": False,
            **cfg,
        })


def _random_obs_state(rng):
    obs_state = {}
    for side in ("left", "right"):
        q = rng.normal(size=4)
        obs_state[f"{side}_ee_pose"] = np.r_[rng.normal(scale=0.3, size=3), q / np.linalg.norm(q)]
    return obs_state


@contextlib.contextmanager
def _replay_env(episodes, root):
    """FLEXPI_REPLAY_GT / FLEXPI_REPLAY_GT_ROOT, read by the Models built inside."""
    os.environ["FLEXPI_REPLAY_GT"], os.environ["FLEXPI_REPLAY_GT_ROOT"] = episodes, root
    try:
        yield
    finally:
        del os.environ["FLEXPI_REPLAY_GT"], os.environ["FLEXPI_REPLAY_GT_ROOT"]


def _assert_sends_the_raw_action(steps, raw, t, action_type, tag):
    """``steps`` (Model's per-step action dicts from raw frame ``t``) are the raw commanded actions."""
    for k, step in enumerate(steps):
        want = _obs_state(raw, t + k, "action")
        for side in ("left", "right"):
            grip = np.clip(want[f"{side}_ee_joint_state"], 0.0, 1.0)
            np.testing.assert_allclose(step[f"{side}_ee_joint_state"], grip, rtol=0, atol=F32_TOL)
            if action_type == "joint":
                got, ref = step[f"{side}_arm_joint_state"], want[f"{side}_arm_joint_state"]
                np.testing.assert_allclose(got, ref, rtol=0, atol=F32_TOL)
            else:
                got, ref = step[f"{side}_ee_pose"], want[f"{side}_ee_pose"]
                np.testing.assert_allclose(got[:3], ref[:3], rtol=0, atol=F32_TOL)
                assert _quat_error(got[3:], ref[3:]) <= F32_TOL, (tag, t, k, side)


# --- no data needed ------------------------------------------------------------------

def test_relabeled_maps_rot6d_rows_by_the_robot_constant():
    """Positions, grippers and joints are world_link6's; each rot6d row is ``row @ T``.

    T is a signed permutation, so this holds bit for bit on float32 -- the rule the
    corpus relabel used (``raw_row @ P``).
    """
    np.testing.assert_array_equal(_RELABEL_ROT["arx_x5"], _LINK6_TO_GRASP_ROT)
    rng = np.random.default_rng(0)
    for robot in ROBOTS:
        T = _RELABEL_ROT[robot]
        np.testing.assert_array_equal(T @ T.T, np.eye(3))
        assert abs(np.linalg.det(T) - 1.0) < 1e-12
        for _ in range(20):
            obs_state, packed14 = _random_obs_state(rng), rng.uniform(-1, 1, 14)
            ref = build_yam32_state(obs_state, packed14, None, eef_frame="world_link6")
            s32 = build_yam32_state(obs_state, packed14, None, eef_frame="relabeled", env_cfg_type=robot)
            np.testing.assert_array_equal(s32[18:], ref[18:])
            for side, sl in SIDES:
                np.testing.assert_array_equal(s32[sl][:3], ref[sl][:3])
                np.testing.assert_array_equal(s32[sl][3:9].reshape(2, 3), ref[sl][3:9].reshape(2, 3) @ T)
                pos_mm, rot_deg = pose7_error(
                    eef_block_to_pose7(s32[sl], side, "relabeled", None, robot), obs_state[f"{side}_ee_pose"]
                )
                assert pos_mm < 1e-3 and rot_deg < 1e-3, (robot, side, pos_mm, rot_deg)


def test_relabeled_rejects_an_unknown_robot():
    """No tool-frame constant -> ValueError: in the bridge, and in Model at init."""
    obs_state = {"left_ee_pose": np.r_[np.zeros(3), 1, 0, 0, 0], "right_ee_pose": np.r_[np.ones(3), 1, 0, 0, 0]}
    block9 = np.r_[np.zeros(3), 1, 0, 0, 0, 1, 0]
    for robot in (None, "dual_x5", "franka"):
        for call in (
            lambda: build_yam32_state(obs_state, np.zeros(14), None, eef_frame="relabeled", env_cfg_type=robot),
            lambda: eef_block_to_pose7(block9, "left", "relabeled", None, robot),
        ):
            try:
                call()
            except ValueError as exc:
                assert "env_cfg_type" in str(exc), exc
            else:
                raise AssertionError(f"relabeled with env_cfg_type={robot!r} must raise")
    for action_type in ("joint", "ee"):
        try:
            _dummy_model(action_type, "dual_x5", "relabeled")
        except ValueError as exc:
            assert "relabeled" in str(exc), exc
        else:
            raise AssertionError("Model must refuse relabeled for an unknown env_cfg_type at init")


def test_mount_poses_are_read_by_arm_base_grasp_only():
    model = _model_module()
    calls = []

    def load_base_transforms(env_cfg_type):
        calls.append(env_cfg_type)
        return {side: (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])) for side in ("left", "right")}

    with _patched(model, "load_base_transforms", load_base_transforms):
        for robot in ROBOTS:
            for frame in ("world_link6", "relabeled"):
                for action_type in ("joint", "ee"):
                    m = _dummy_model(action_type, robot, frame)
                    assert m.state_bridge == "yam32" and m._base_tf is None, (robot, frame, action_type)
        assert calls == []
        m = _dummy_model("joint", "arx_x5", "arm_base_grasp")
        assert calls == ["arx_x5"] and set(m._base_tf) == {"left", "right"}


def test_replay_root_sets_the_ee_frame():
    """FLEXPI_REPLAY_GT under ee admits only the frame of the root's EEF block, at init.

    A root whose meta/info.json carries ``eef_frame`` with state_bridge's table (the
    relabeled corpus) needs ``relabeled``; one whose info.json does not (the sim and raw
    corpora), or that has none, needs ``world_link6``; one recording any other (or an
    empty) table is refused. Under joint the EEF block is not sent: every frame passes.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        table = {r: p.tolist() for r, p in _RELABEL_ROT.items()}
        other = {"arx_x5": (_RELABEL_ROT["arx_x5"] @ np.diag([-1.0, -1.0, 1.0])).tolist()}   # XPolicyLab's arx_x5 entry
        needs = {os.path.join(tmp, "absent"): "world_link6"}
        for name, info, frame in (("relabeled", {"eef_frame": {"P": table}}, "relabeled"),
                                  ("plain", {"task_name": "robodojo_34d"}, "world_link6"),
                                  ("other_table", {"eef_frame": {"P": other}}, None),
                                  ("empty_table", {"eef_frame": {"P": {}}}, None)):
            os.makedirs(os.path.join(tmp, name, "meta"))
            with open(os.path.join(tmp, name, "meta", "info.json"), "w") as f:
                json.dump(info, f)
            if frame is None:   # a root relabeled with another (or no) table is refused outright
                with _replay_env("0", os.path.join(tmp, name)):
                    for action_type in ("joint", "ee"):
                        try:
                            _dummy_model(action_type, "arx_x5", "relabeled")._replay.eef_frame()
                        except ValueError as exc:
                            assert "differs from state_bridge's relabeled table" in str(exc), exc
                        else:
                            raise AssertionError(f"root {name} must be refused under {action_type}")
                continue
            needs[os.path.join(tmp, name)] = frame
        for root, needed in needs.items():
            with _replay_env("0", root):
                for frame in EEF_FRAMES:
                    assert _dummy_model("joint", "arx_x5", frame)._replay.eef_frame() == needed, (root, frame)
                    if frame == needed:
                        _dummy_model("ee", "arx_x5", frame)
                        continue
                    try:
                        _dummy_model("ee", "arx_x5", frame)
                    except ValueError as exc:
                        assert f"needs eef_frame={needed}," in str(exc), exc
                    else:
                        raise AssertionError(f"FLEXPI_REPLAY_GT_ROOT={root} under ee must refuse eef_frame={frame}")


def test_ee_logs_measure_the_current_pose_in_eef_frame():
    """_log_bridge_once and _log_ee_chunk take the current pose back through eef_frame.

    Under the dummy ee policy every target is a hold of the current pose, so every
    "target - measured", "chunk reach" and "tracking" figure reads 0 when the pose
    is measured in the frame the targets are sent in. Measured in another frame it
    reads that frame's offset: 90 deg for relabeled arx_x5 and piper_x, the mount
    pose and tool offset for arm_base_grasp. world_link6 is the control.
    """
    import io
    import re

    rng = np.random.default_rng(1)
    for robot, frame in (("arx_x5", "relabeled"), ("piper_x", "relabeled"),
                         ("arx_x5", "arm_base_grasp"), ("arx_x5", "world_link6")):
        obs_state = _random_obs_state(rng)
        for side in ("left", "right"):
            obs_state[f"{side}_arm_joint_state"] = rng.uniform(-1, 1, 6)
            obs_state[f"{side}_ee_joint_state"] = np.array([0.5])
        model = _dummy_model("ee", robot, frame, replan_steps=4, log_action_chunk=True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for _ in range(2):
                model.update_obs_batch([{"vision": VISION, "state": obs_state, "instruction": "x", "env_idx": 0}])
                model.get_action_batch([0])
        log = out.getvalue()
        # Per arm: _log_bridge_once once, step0 on both calls, tracking from the second call on.
        figures = re.findall(r"(\S+) mm (\S+) deg", log)
        reach = re.findall(r"chunk reach (\S+) mm", log)
        assert len(figures) == 8 and len(reach) == 4, (robot, frame, log)
        assert all(float(mm) == 0.0 and float(deg) == 0.0 for mm, deg in figures), (robot, frame, log)
        assert all(float(mm) == 0.0 for mm in reach), (robot, frame, log)


# --- replay against the corpus ------------------------------------------------------------

def test_world_link6_state_is_the_raw_corpus_state():
    _require_data()
    errors = _corpus_errors("world_link6")
    for robot in ROBOTS:
        assert errors[robot, "world_link6"] <= F32_TOL, errors
    # The other variant differs wherever the robot's constant is not the identity.
    assert errors["piper", "relabeled"] <= F32_TOL, errors
    assert errors["arx_x5", "relabeled"] > 0.5 and errors["piper_x", "relabeled"] > 0.5, errors


def test_relabeled_state_is_the_relabeled_corpus_state():
    _require_data()
    errors = _corpus_errors("relabeled")
    for robot in ROBOTS:
        assert errors[robot, "relabeled"] <= F32_TOL, errors
    assert errors["arx_x5", "world_link6"] > 0.5 and errors["piper_x", "world_link6"] > 0.5, errors


def test_relabeled_table_is_the_corpus_relabel_table():
    _require_data()
    table = _info(RELABELED_ROOT)["eef_frame"]["P"]
    assert sorted(table) == sorted(_RELABEL_ROT)
    for robot, P in table.items():
        np.testing.assert_array_equal(np.asarray(P), _RELABEL_ROT[robot])


def test_ee_output_is_the_raw_action_pose():
    """Each corpus action EEF block back through its eef_frame is the raw commanded ee_pose."""
    _require_data()
    for frame, root in CORPUS.items():
        for robot, eps in _episodes_by_robot().items():
            pos_err = quat_err = 0.0
            for ep in eps:
                _, action34, _ = _corpus_episode(root, ep)
                raw = _raw_episode(ep)
                for side, sl in SIDES:
                    got = np.stack([
                        eef_block_to_pose7(block9, side, frame, None, robot) for block9 in action34[:, KEEP32][:, sl]
                    ])
                    want = raw[f"action__{side}_ee_poses"][: len(got)]
                    pos_err = max(pos_err, float(np.abs(got[:, :3] - want[:, :3]).max()))
                    quat_err = max(quat_err, _quat_error(got[:, 3:], want[:, 3:]))
            assert pos_err <= F32_TOL and quat_err <= F32_TOL, (frame, robot, pos_err, quat_err)


def test_joint_output_is_the_raw_action_joints():
    """The joint half of either corpus is the raw commanded joints, grippers clipped to [0, 1]."""
    _require_data()
    for robot, eps in _episodes_by_robot().items():
        for ep in eps:
            raw = _raw_episode(ep)
            for root in CORPUS.values():
                _, action34, _ = _corpus_episode(root, ep)
                n = len(action34)
                want = np.concatenate([
                    raw["action__left_arm_joint_states"][:n],
                    np.clip(raw["action__left_ee_joint_states"][:n], 0.0, 1.0),
                    raw["action__right_arm_joint_states"][:n],
                    np.clip(raw["action__right_ee_joint_states"][:n], 0.0, 1.0),
                ], axis=1)
                err = float(np.abs(yam32_to_rd14(action34[:, KEEP32]) - want).max())
                assert err <= F32_TOL, (robot, ep, root, err)


def test_model_serves_both_corpora_in_both_action_types():
    """The adapter end to end (dummy policy): raw obs -> 32D state, corpus action chunk -> what RoboDojo gets."""
    _require_data()
    horizon = 32
    for robot, eps in _episodes_by_robot().items():
        ep = eps[0]
        raw = _raw_episode(ep)
        for frame, root in CORPUS.items():
            state34, action34, _ = _corpus_episode(root, ep)
            for action_type in ("joint", "ee"):
                model = _dummy_model(action_type, robot, frame, replan_steps=horizon)
                for t in range(0, len(state34) - horizon, 101):
                    model.update_obs_batch([{"vision": VISION, "state": _obs_state(raw, t), "instruction": "x", "env_idx": 0}])
                    anchor = model._batch_obs[0]["joint_action"]["vector"]
                    np.testing.assert_allclose(anchor, state34[t, KEEP32], rtol=0, atol=F32_TOL)
                    steps = model._emit(0, anchor, action34[t:t + horizon, KEEP32])
                    assert len(steps) == horizon
                    _assert_sends_the_raw_action(steps, raw, t, action_type, (robot, frame))


def test_replay_sends_the_raw_action_from_either_corpus():
    """FLEXPI_REPLAY_GT through Model: each corpus under its own eef_frame sends the raw commanded actions.

    The crossed pairing is refused under ee: taking the relabeled corpus back through
    world_link6 sends rotations off by the robot's constant (90 deg for arx_x5 and
    piper_x), and the raw corpus through relabeled is off the same way.
    """
    _require_data()
    horizon = 32
    for robot, eps in _episodes_by_robot().items():
        ep = eps[0]
        raw = _raw_episode(ep)
        for frame, root in CORPUS.items():
            n = len(_corpus_episode(root, ep)[0])
            with _replay_env(str(ep), root):
                for crossed in CORPUS:
                    if crossed == frame:
                        continue
                    try:
                        _dummy_model("ee", robot, crossed)
                    except ValueError as exc:
                        assert f"needs eef_frame={frame}," in str(exc), exc
                    else:
                        raise AssertionError(f"replay of {root} under ee must refuse eef_frame={crossed}")
                for action_type in ("joint", "ee"):
                    model = _dummy_model(action_type, robot, frame, replan_steps=horizon)
                    for t in range(n - horizon + 1):
                        # Every frame is an observation: the replay serves action[t:] at the t-th one.
                        model.update_obs_batch([{"vision": VISION, "state": _obs_state(raw, t), "instruction": "x", "env_idx": 0}])
                        if t % horizon == 0:
                            steps = model.get_action_batch([0])[0]
                            assert len(steps) == horizon
                            _assert_sends_the_raw_action(steps, raw, t, action_type, (robot, frame, ep))


if __name__ == "__main__":
    import pytest

    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except pytest.skip.Exception as exc:
                print(f"skip {name}: {exc}")
                continue
            print(f"ok  {name}")
