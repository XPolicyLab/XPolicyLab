"""CPU contracts for the OpenWAM ``tianji_wuji`` (58-D WUJI hybrid) path.

Runs without torch: the adapter's conversion helpers and the torch-free reader
module ``openwam.dataloader.robodojo_wuji58`` are loaded directly, bypassing the
``openwam.dataloader`` package ``__init__`` (which imports the torch-backed
readers). The network is never built.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import h5py
import numpy as np

POLICY_DIR = Path(__file__).resolve().parents[1]
OPENWAM_ROOT = POLICY_DIR / "OpenWAM"
REPO_ROOT = POLICY_DIR.parents[1]
for entry in (str(REPO_ROOT.parent), str(OPENWAM_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from XPolicyLab.utils import wuji58 as W  # noqa: E402


def _stub_package(name: str, path: Path) -> None:
    """Register a namespace stand-in for a package whose __init__ needs torch."""
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module


def _load_torch_free_modules():
    importlib.import_module("openwam")  # light __init__
    _stub_package("openwam.dataloader", OPENWAM_ROOT / "openwam" / "dataloader")
    contract = importlib.import_module("openwam.dataloader.robodojo_contract")
    normalization = importlib.import_module("openwam.dataloader.utils.normalization")
    reader = importlib.import_module("openwam.dataloader.robodojo_wuji58")
    return contract, normalization, reader


contract, normalization, reader = _load_torch_free_modules()

# The adapter imports torch-free; Model.__init__ is exercised only through
# allow_dummy_policy, with the robot registry stubbed (env_cfg lives outside).
adapter_spec = importlib.util.spec_from_file_location("openwam_adapter_under_test", POLICY_DIR / "model.py")
adapter = importlib.util.module_from_spec(adapter_spec)
adapter_spec.loader.exec_module(adapter)
if importlib.util.find_spec("PIL") is None:
    adapter._to_pil = lambda value, ctx: np.asarray(value)  # image wrapping is not under test


def _rand_quat(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def _rand_pose(rng):
    return np.concatenate([rng.uniform(-0.5, 0.5, 3), _rand_quat(rng)])


def _obs_state(rng):
    return {
        "left_ee_pose": _rand_pose(rng),
        "left_ee_joint_state": rng.uniform(-0.4, 1.5, 20),
        "right_ee_pose": _rand_pose(rng),
        "right_ee_joint_state": rng.uniform(-0.4, 1.5, 20),
    }


def _write_episode(path: Path, rng, T: int, *, with_actions: bool) -> dict:
    """Synthetic tianji_wuji episode (poses + hands only; cameras are not read here)."""
    states = [_obs_state(rng) for _ in range(T)]
    targets = [_obs_state(rng) for _ in range(T)]
    with h5py.File(path, "w") as handle:
        for key in W.STATE_KEYS:
            handle.create_dataset(f"state/{key}s", data=np.stack([s[key] for s in states]).astype(np.float32))
            if with_actions:
                handle.create_dataset(f"action/{key}s", data=np.stack([t[key] for t in targets]).astype(np.float32))
        handle.create_dataset("instruction", data=np.bytes_(b"wave"))
    return {"states": states, "targets": targets}


class ContractTest(unittest.TestCase):
    def test_constants_and_rot6d_dims(self):
        self.assertEqual(contract.WUJI58_DIM, 58)
        self.assertEqual(contract.ROBODOJO_WUJI58_EMBODIMENT, "tianji_wuji")
        self.assertEqual(contract.ROBODOJO_WUJI58_ACTION_MODE, "wuji58")
        self.assertEqual(normalization.ROT6D_DIMS_WUJI58, tuple(range(23, 29)) + tuple(range(52, 58)))
        self.assertEqual(normalization._ROT6D_DIMS_BY_WIDTH[58], normalization.ROT6D_DIMS_WUJI58)
        # EEF20 dims untouched
        self.assertEqual(normalization.ROT6D_DIMS_EEF20, (3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18))

    def test_validate_embodiment_eef_path_unchanged(self):
        contract.validate_embodiment("arx_x5")
        with self.assertRaisesRegex(ValueError, "only supported OpenWAM embodiment is 'arx_x5'"):
            contract.validate_embodiment("tianji_wuji")
        contract.validate_embodiment("tianji_wuji", action_mode="wuji58")
        with self.assertRaisesRegex(ValueError, "supports only embodiment 'tianji_wuji'"):
            contract.validate_embodiment("arx_x5", action_mode="wuji58")
        with self.assertRaisesRegex(ValueError, "simulation-only"):
            contract.validate_embodiment("tianji_wuji", variant="real", action_mode="wuji58")
        with self.assertRaises(ValueError):
            contract.validate_action_mode("joint")

    def test_frame_contract_fingerprint_tracks_wrist_frame(self):
        te2 = reader.wuji58_frame_contract_fingerprint("te2")
        self.assertEqual(te2, reader.wuji58_frame_contract_fingerprint(None))
        self.assertNotEqual(te2, reader.wuji58_frame_contract_fingerprint("world"))
        payload = contract.wuji58_frame_contract("te2")
        self.assertEqual(payload["repr_mode"], W.REPR_MODE)
        self.assertEqual(payload["te2_eye"], list(W.TE2_EYE))
        with self.assertRaises(ValueError):
            reader.resolve_wrist_frame("base")


class ReaderTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.rng = np.random.default_rng(0)
        self.frame = W.WristFrame.te2()

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_with_recorded_actions_matches_trajectory58(self):
        path = self.dir / "episode_0.hdf5"
        data = _write_episode(path, self.rng, 6, with_actions=True)
        with h5py.File(path, "r") as handle:
            self.assertEqual(reader.validate_wuji58_state_action_datasets(handle, source="ep"), 6)
            state58, action58 = reader.read_wuji58(handle)
        self.assertEqual(state58.shape, (6, 58))
        self.assertEqual(action58.shape, (6, 58))
        self.assertEqual(state58.dtype, np.float32)
        for t in range(6):
            np.testing.assert_allclose(state58[t], W.state58_from_obs(data["states"][t], self.frame), rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(
                action58[t], W.action58_from_target(data["targets"][t], data["states"][t], self.frame), rtol=1e-5, atol=1e-5
            )
            # decoding each row against its own frame's state recovers the recorded target
            decoded = W.decode_action58(action58[t], data["states"][t], self.frame)
            for key in W.STATE_KEYS:
                np.testing.assert_allclose(decoded[key], data["targets"][t][key], atol=1e-4)
        # window read
        with h5py.File(path, "r") as handle:
            s, a = reader.read_wuji58(handle, 2, 5)
        np.testing.assert_allclose(s, state58[2:5])
        np.testing.assert_allclose(a, action58[2:5])

    def test_read_next_state_targets_when_actions_absent(self):
        path = self.dir / "episode_1.hdf5"
        data = _write_episode(path, self.rng, 5, with_actions=False)
        with h5py.File(path, "r") as handle:
            self.assertFalse(reader.episode_has_recorded_actions(handle))
            state58, action58 = reader.read_wuji58(handle)
            # window inside the episode reads one extra row so every row has a target
            s_win, a_win = reader.read_wuji58(handle, 1, 3)
            with self.assertRaises(IndexError):
                reader.read_wuji58(handle, 4, 5)
        self.assertEqual(state58.shape, (4, 58))
        self.assertEqual(action58.shape, (4, 58))
        for t in range(4):
            decoded = W.decode_action58(action58[t], data["states"][t], self.frame)
            for key in W.STATE_KEYS:
                np.testing.assert_allclose(decoded[key], data["states"][t + 1][key], atol=1e-4)
        self.assertEqual(s_win.shape, (2, 58))
        np.testing.assert_allclose(a_win, action58[1:3])

    def test_validation_rejects_gripper_width_and_partial_actions(self):
        path = self.dir / "bad.hdf5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("state/left_ee_poses", data=np.tile([0, 0, 0, 1, 0, 0, 0], (3, 1)).astype(np.float32))
            handle.create_dataset("state/right_ee_poses", data=np.tile([0, 0, 0, 1, 0, 0, 0], (3, 1)).astype(np.float32))
            handle.create_dataset("state/left_ee_joint_states", data=np.zeros((3, 1), np.float32))  # a gripper, not a hand
            handle.create_dataset("state/right_ee_joint_states", data=np.zeros((3, 20), np.float32))
        with h5py.File(path, "r") as handle:
            with self.assertRaisesRegex(ValueError, r"\(T, 20\)"):
                reader.validate_wuji58_state_action_datasets(handle, source="bad")
        with h5py.File(path, "a") as handle:
            del handle["state/left_ee_joint_states"]
            handle.create_dataset("state/left_ee_joint_states", data=np.zeros((3, 20), np.float32))
            handle.create_dataset("action/left_ee_poses", data=np.tile([0, 0, 0, 1, 0, 0, 0], (3, 1)).astype(np.float32))
        with h5py.File(path, "r") as handle:
            with self.assertRaisesRegex(KeyError, "incomplete"):
                reader.validate_wuji58_state_action_datasets(handle, source="bad")

    def test_stats_payload_validation_and_rot6d_pin(self):
        def block(scale):
            rng = np.random.default_rng(1)
            b = {
                "mean": rng.normal(size=58).astype(np.float32),
                "std": np.full(58, scale, np.float32),
                "min": np.full(58, -scale, np.float32),
                "max": np.full(58, scale, np.float32),
                "q01": np.full(58, -scale / 2, np.float32),
                "q99": np.full(58, scale / 2, np.float32),
            }
            return b

        action, state = block(2.0), block(3.0)
        payload = {
            reader.WUJI58_STATS_KEY: action,
            reader.WUJI58_STATE_STATS_KEY: state,
            "metadata": {
                "variant": "sim",
                "embodiment": "tianji_wuji",
                "action_mode": "wuji58",
                "repr_mode": W.REPR_MODE,
                "source_frame": contract.ROBODOJO_WUJI58_SOURCE_FRAME,
                "contract_id": contract.ROBODOJO_WUJI58_CONTRACT_ID,
                "frame_contract_fingerprint": reader.wuji58_frame_contract_fingerprint("te2"),
                "tasks": ["wave"],
            },
        }
        with self.assertRaisesRegex(ValueError, "non-identity rot6d"):
            reader.validate_wuji58_stats_payload(payload, source="stats")
        reader.pin_wuji58_rot6d_identity(action)
        reader.pin_wuji58_rot6d_identity(state)
        for key, value in {"min": -1.0, "max": 1.0, "q01": -1.0, "q99": 1.0, "mean": 0.0, "std": 1.0}.items():
            np.testing.assert_array_equal(action[key][list(normalization.ROT6D_DIMS_WUJI58)], value)
            self.assertEqual(float(action[key][0]), float(block(2.0)[key][0]))  # non-rot6d dims untouched
        a, s = reader.validate_wuji58_stats_payload(payload, source="stats", expected_tasks=["wave"])
        self.assertEqual(float(a["std"][0]), 2.0)
        self.assertEqual(float(s["std"][0]), 3.0)
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            reader.validate_wuji58_stats_payload(payload, source="stats", wrist_frame="world")
        with self.assertRaisesRegex(ValueError, "metadata.tasks"):
            reader.validate_wuji58_stats_payload(payload, source="stats", expected_tasks=["other"])
        del payload[reader.WUJI58_STATE_STATS_KEY]
        with self.assertRaises(KeyError):
            reader.validate_wuji58_stats_payload(payload, source="stats")


class _DummyPrompt:
    @staticmethod
    def format(text):
        return text


def _dummy_model(env_cfg_type: str, action_type: str = "ee", **extra):
    """Model in allow_dummy_policy mode with the robot registry and openwam import stubbed."""
    registry = {
        "tianji_wuji": W.TIANJI_WUJI_ROBOT_ACTION_DIM_INFO,
        "arx_x5": {"arm_dim": [6, 6], "ee_dim": [1, 1]},
        "marvin": {"arm_dim": [7, 7], "ee_dim": [12, 12]},  # neither contract
    }
    original_dim_info = adapter.get_robot_action_dim_info
    original_root = adapter._resolve_openwam_root
    adapter.get_robot_action_dim_info = lambda name: registry[name]
    adapter._resolve_openwam_root = lambda value: OPENWAM_ROOT
    multiview = types.ModuleType("openwam.dataloader.transforms.multiview")
    multiview.format_prompt_for_inference = lambda text: f"<{text}>"
    transforms = types.ModuleType("openwam.dataloader.transforms")
    transforms.__path__ = []
    saved = {k: sys.modules.get(k) for k in ("openwam.dataloader.transforms", "openwam.dataloader.transforms.multiview")}
    sys.modules["openwam.dataloader.transforms"] = transforms
    sys.modules["openwam.dataloader.transforms.multiview"] = multiview
    try:
        cfg = {"env_cfg_type": env_cfg_type, "action_type": action_type, "allow_dummy_policy": True, "replan_steps": 4}
        cfg.update(extra)
        return adapter.Model(cfg)
    finally:
        adapter.get_robot_action_dim_info = original_dim_info
        adapter._resolve_openwam_root = original_root
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _obs(rng, env_idx=None):
    img = np.zeros((8, 8, 3), np.uint8)
    obs = {
        "vision": {name: {"color": img} for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
        "state": _obs_state(rng),
        "instruction": "wave",
    }
    if env_idx is not None:
        obs["env_idx"] = env_idx
    return obs


class AdapterTest(unittest.TestCase):
    def test_mode_selection_from_registry(self):
        model = _dummy_model("tianji_wuji")
        self.assertTrue(model.wuji58)
        self.assertEqual(model.action_mode, "wuji58")
        self.assertEqual(model.raw_action_dim, 58)
        self.assertEqual(model.wrist_frame_name, "te2")
        eef = _dummy_model("arx_x5")
        self.assertFalse(eef.wuji58)
        self.assertEqual(eef.raw_action_dim, 20)
        with self.assertRaisesRegex(ValueError, "action_type must be 'ee'"):
            _dummy_model("tianji_wuji", action_type="joint")
        with self.assertRaisesRegex(ValueError, "ee_dim \\[20, 20\\]"):
            _dummy_model("marvin")  # 20-D hands but driven in joint space on the registry side
        with self.assertRaises(ValueError):
            _dummy_model("tianji_wuji", wrist_frame="base")

    def test_state_is_58d_and_matches_codec(self):
        model = _dummy_model("tianji_wuji")
        rng = np.random.default_rng(2)
        obs = _obs(rng, env_idx=3)
        model.update_obs_batch([obs])
        state = np.asarray(model._batch[3]["state"])
        self.assertEqual(state.shape, (58,))
        np.testing.assert_allclose(state, W.state58_from_obs(obs["state"], W.WristFrame.te2()), rtol=1e-5, atol=1e-5)
        self.assertEqual(model._batch[3]["prompt"], "<wave>")
        self.assertIn(3, model._decode_state)

    def test_dummy_chunk_holds_pose(self):
        model = _dummy_model("tianji_wuji")
        rng = np.random.default_rng(3)
        obs = _obs(rng, env_idx=0)
        model.update_obs_batch([obs])
        chunk = model.get_action()
        self.assertEqual(len(chunk), 4)
        for action in chunk:
            self.assertEqual(set(action), set(W.STATE_KEYS))
            for key in W.STATE_KEYS:
                self.assertEqual(action[key].dtype, np.float32)
                np.testing.assert_allclose(action[key], obs["state"][key], atol=1e-5)
            self.assertEqual(action["left_ee_joint_state"].shape, (20,))

    def test_chunk_decode_is_iterative_per_frame(self):
        """Per-frame deltas (each vs. the previous target) decode back to the targets."""
        model = _dummy_model("tianji_wuji")
        rng = np.random.default_rng(4)
        obs = _obs(rng, env_idx=7)
        model.update_obs_batch([obs])
        targets = [_obs_state(rng) for _ in range(5)]
        refs = [obs["state"]] + targets[:-1]
        chunk = np.stack([W.action58_from_target(t, r, W.WristFrame.te2()) for t, r in zip(targets, refs)])
        decoded = model._chunk_to_native(chunk, 7)
        self.assertEqual(len(decoded), 5)
        for got, want in zip(decoded, targets):
            for key in W.STATE_KEYS:
                np.testing.assert_allclose(got[key], want[key], atol=1e-4)
        with self.assertRaises(RuntimeError):
            model._chunk_to_native(np.zeros((2, 20)), 7)

    def test_per_env_decode_state_and_reset(self):
        model = _dummy_model("tianji_wuji")
        rng = np.random.default_rng(5)
        obs_a, obs_b = _obs(rng, env_idx=1), _obs(rng, env_idx=2)
        model.update_obs_batch([obs_a, obs_b])
        chunks = model.get_action_batch([2, 1])
        np.testing.assert_allclose(chunks[0][0]["left_ee_pose"], obs_b["state"]["left_ee_pose"], atol=1e-5)
        np.testing.assert_allclose(chunks[1][0]["left_ee_pose"], obs_a["state"]["left_ee_pose"], atol=1e-5)
        model.reset()
        self.assertEqual(model._decode_state, {})
        with self.assertRaises(ValueError):
            model.get_action_batch([1])

    def test_rejects_gripper_width_state(self):
        model = _dummy_model("tianji_wuji")
        rng = np.random.default_rng(6)
        obs = _obs(rng)
        obs["state"]["left_ee_joint_state"] = np.zeros(1)
        with self.assertRaisesRegex(ValueError, "Wuji hand"):
            model.update_obs(obs)

    def test_checkpoint_contract_mismatch(self):
        model = _dummy_model("tianji_wuji")
        cfg = {"dataloader": {"action_mode": "eef"}}
        with self.assertRaisesRegex(RuntimeError, "action_mode='eef'"):
            model._check_checkpoint_contract(_OmegaLike(cfg), "ckpt")
        with self.assertRaisesRegex(RuntimeError, "wrist_frame='world'"):
            model._check_checkpoint_contract(_OmegaLike({"dataloader": {"action_mode": "wuji58", "wrist_frame": "world"}}), "ckpt")
        with self.assertRaisesRegex(RuntimeError, "unify_action=false"):
            model._check_checkpoint_contract(_OmegaLike({"dataloader": {"action_mode": "wuji58", "unify_action": True}}), "ckpt")
        with self.assertRaisesRegex(RuntimeError, "action_dim=80"):
            model._check_checkpoint_contract(
                _OmegaLike({"dataloader": {"action_mode": "wuji58"}, "model": {"architecture": {"action_dim": 80}}}), "ckpt"
            )
        model._check_checkpoint_contract(
            _OmegaLike({"dataloader": {"action_mode": "wuji58", "unify_action": False}, "model": {"architecture": {"action_dim": 58, "state_dim": 58}}}),
            "ckpt",
        )
        eef = _dummy_model("arx_x5")
        with self.assertRaisesRegex(RuntimeError, "action_mode='wuji58'"):
            eef._check_checkpoint_contract(_OmegaLike({"dataloader": {"action_mode": "wuji58"}}), "ckpt")
        eef._check_checkpoint_contract(_OmegaLike({"dataloader": {}}), "ckpt")  # legacy eef ckpt: default


def _OmegaLike(cfg: dict):
    """A config OmegaConf.select can read: the real DictConfig when omegaconf exists, else a dict."""
    try:
        from omegaconf import OmegaConf

        if hasattr(OmegaConf, "create"):
            return OmegaConf.create(cfg)
    except ImportError:
        pass
    return cfg


def _install_omegaconf_stub():
    if importlib.util.find_spec("omegaconf") is not None:
        return
    module = types.ModuleType("omegaconf")

    class OmegaConf:
        @staticmethod
        def select(cfg, key, default=None):
            node = cfg
            for part in key.split("."):
                if not isinstance(node, dict) or part not in node:
                    return default
                node = node[part]
            return node

    module.OmegaConf = OmegaConf
    sys.modules["omegaconf"] = module


_install_omegaconf_stub()


if __name__ == "__main__":
    unittest.main()
