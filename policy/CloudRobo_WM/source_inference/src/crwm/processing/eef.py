"""Frozen Joint14 + EEF input/output contract for the Clean EEF checkpoint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from .normalization import Joint14Normalizer, STATE_INDICES, ACTION_INDICES, STATE_DIM, ACTION_DIM
from .eef_geometry import quaternion_to_matrix, matrix_to_quaternion, rotation_6d, rotvec_to_matrix

PROFILE = "robotwin_joint14_eef_cleanonly_zscore_v1"
STATS_KEY = "robotwin_aloha_joint14_eef_cleanonly_zscore_v1"
EEF_FRAME = "raw_fl_link6_fr_link6_world"
STATE_POSITION = np.asarray([*range(0, 3), *range(34, 37)])
STATE_ROTATION = np.asarray([*range(3, 9), *range(37, 43)])
STATE_EEF = np.asarray([*range(0, 9), *range(34, 43)])
ACTION_EEF = np.asarray([*range(0, 6), *range(31, 37)])
URDF_SHA256 = "097c59fb19a7b482249c6097df8319586ea7cfd268c015516f103b289a7e761a"
EMBODIMENT_SHA256 = "a4f043d2d4b44cca5ec47fb498a3194da3f87ff9530dcbfdcee1539a44423d57"
CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"
DEFINITION = dict(state_position="raw_fl_link6_fr_link6_world_xyz_m",
    state_rotation="first_two_matrix_columns_identity_unclipped",
    action_position="future_world_xyz_minus_fixed_observed_anchor_xyz_m",
    action_rotation="body_R_anchor_T_R_future_window_continuous_rotvec_rad",
    action_target="anchor_plus_horizon_offset_plus_one_original_frame",
    horizon=32, terminal_repeated_pose_supervised=False,
    state_statistics="each_selected_raw_frame_once",
    action_statistics="each_valid_anchor_and_32_horizon_pair_once",
    normalization="FP32_(x-mean)/(std+1e-8)_clip_[-5,5]", ddof=0)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_environment(robot):
    """Reject a live robot whose endpoints, URDF, joint order or base moved."""
    from .kinematics import sha256_file
    for side, prefix in (("left", "fl"), ("right", "fr")):
        urdf = Path(getattr(robot, side + "_urdf_path")).resolve()
        if (sha256_file(urdf) != URDF_SHA256
                or sha256_file(urdf.parent.parent / "config.yml") != EMBODIMENT_SHA256
                or list(getattr(robot, side + "_arm_joints_name")) != [prefix + "_joint" + str(i) for i in range(1, 7)]
                or getattr(robot, side + "_move_group") != prefix + "_link6"):
            raise ValueError("Live RoboTwin robot differs from the pinned EEF training assets")
        pose = getattr(robot, side + "_entity_origion_pose")
        if (not np.allclose(pose.p, [0, -0.65, 0], atol=1e-6, rtol=0)
                or not np.allclose(quaternion_to_matrix(pose.q), [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-6, rtol=0)):
            raise ValueError("Live RoboTwin base/world transform differs from EEF training")


class JointEefNormalizer:
    """Exactly 32 valid state slots and 26 valid action slots in S96/A90."""

    def __init__(self, config_path: str | Path):
        self.config = json.loads(Path(config_path).read_text())
        if self.config.get("format") != "crwm.joint14_eef.normalization.v1":
            raise ValueError("Expected EEF normalization; a Joint14-only checkpoint is incompatible")
        self.stats_key = self.config["stats_key"]
        self.group = self.config["group"]
        if (self.stats_key != STATS_KEY or self.group.get("profile") != PROFILE
                or self.group.get("mode") != "z-score" or self.group.get("eef_contract") != DEFINITION
                or self.group.get("native_stats_sha256") != fingerprint({
                    k: v for k, v in self.group.items() if k != "native_stats_sha256"})):
            raise ValueError("Sealed EEF statistics/coordinate contract differs from training")
        self.joint = Joint14Normalizer(Path(config_path).parent / "joint14_normalization.json")
        self._valid = {}
        for kind, dim, joints, eef in (
                ("state", STATE_DIM, STATE_INDICES, STATE_EEF),
                ("action", ACTION_DIM, ACTION_INDICES, ACTION_EEF)):
            expected = np.zeros(dim, bool)
            expected[joints] = expected[eef] = True
            fields = self.group[kind]
            for name in ("count", "mean", "std", "q01", "q99"):
                values = np.asarray(fields[name])
                if values.shape != (dim,) or not np.isfinite(values).all():
                    raise ValueError("Malformed sealed EEF statistics")
                base = np.asarray(self.group["base_native_group"][kind][name])
                if not np.array_equal(values[joints], base[joints]):
                    raise ValueError("EEF profile changed the original Joint14 statistics")
            if (not np.array_equal(np.asarray(fields["count"]) > 0, expected)
                    or (np.asarray(fields["std"]) < 0).any()):
                raise ValueError("EEF profile must enable exactly 32 state / 26 action slots")
            self._valid[kind] = expected
        self.action_valid = self._valid["action"].copy()
        self._fk = None

    def validate_checkpoint(self, contract: dict) -> int:
        groups = contract["statistics"]["groups"]
        if set(groups) != {self.stats_key} or groups[self.stats_key] != self.group:
            raise ValueError("Checkpoint EEF statistics differ from the bundled sealed group")
        conditions = contract["conditions"]
        choices = [int(i) for i, c in conditions.items()
                   if c.get("stats_key") == self.stats_key and c.get("embodiment") == "aloha_agilex"
                   and c.get("protocol") == "absolute_joint_relative_eef"
                   and c.get("normalization") == "z-score"]
        if len(choices) != 1:
            raise ValueError("Expected one unambiguous EEF RoboTwin condition")
        # Reuse the baseline's strict native normalization checks.
        native = dict(statistics=dict(groups={self.joint.stats_key: self.group["base_native_group"]}),
                      conditions={str(choices[0]): dict(embodiment="aloha_agilex",
                                                       stats_key=self.joint.stats_key)})
        self.joint.validate_checkpoint(native)
        if contract.get("data", {}).get("horizon") != 32:
            raise ValueError("EEF checkpoint must preserve the original 32-frame horizon")
        sources = [x for x in contract.get("sources", [])
                   if x["source"].get("stats_key") == self.stats_key]
        if len(sources) != 1:
            raise ValueError("Expected exactly one matching EEF source")
        source = sources[0]["source"]
        provenance = sources[0].get("sidecar_provenance", {})
        if (source.get("eef_contract") != DEFINITION or source.get("coordinate_frame") != "world"
                or source.get("endpoint") != "fl_link6/fr_link6"
                or source.get("action_target_offset_frames") != 1
                or source.get("native_stats_sha256") != self.group["native_stats_sha256"]
                or provenance.get("urdf_sha256") != URDF_SHA256
                or provenance.get("embodiment_config_sha256") != EMBODIMENT_SHA256
                or provenance.get("eef_source") != "urdf_fk"
                or provenance.get("joint_semantics") != "raw_absolute_drive_target_radians_gripper_absolute"
                or provenance.get("eef_frame") != EEF_FRAME):
            raise ValueError("Checkpoint EEF source/FK provenance differs")
        return choices[0]

    def _load_fk(self):
        if self._fk is None:
            import torch
            from .kinematics import RobotWinUrdfFK, sha256_file
            assets = CONFIG_ROOT / "kinematics"
            if sha256_file(assets / "config.yml") != EMBODIMENT_SHA256:
                raise ValueError("EEF embodiment asset changed")
            self._fk = RobotWinUrdfFK(assets / "arx5_description_isaac.urdf",
                                      assets / "config.yml", expected_urdf_sha256=URDF_SHA256)
            self._fk.to(device="cpu", dtype=torch.float64).eval()
        return self._fk

    def encode_observation(self, observation: dict):
        """Reproduce stored float32 FK sidecars from raw drive targets, on CPU."""
        import torch
        from .kinematics import infer_fk_pose7
        joint14 = np.asarray(observation["joint_action"]["vector"], np.float32)
        if joint14.shape != (14,) or not np.isfinite(joint14).all():
            raise ValueError("Expected finite raw absolute Joint14 drive targets")
        with torch.inference_mode():
            # The sidecar uses float64 FK followed by float32 pose storage.
            pose = infer_fk_pose7(torch.from_numpy(joint14.copy()).to(torch.float64),
                                 fk=self._load_fk()).float().numpy()
        state, valid = self.encode_state(joint14, pose)
        return state, valid, pose.copy()

    def encode_state(self, joint14, eef_pose):
        """Pack raw world xyz+wxyz EEF poses; retain column-based rotation6d."""
        poses = np.asarray(eef_pose, np.float64)
        if poses.shape != (2, 7) or not np.isfinite(poses).all():
            raise ValueError("Expected two finite raw-link world xyz+wxyz poses")
        result, _ = self.joint.encode_state(joint14)
        rotations = quaternion_to_matrix(poses[:, 3:])
        positions = poses[:, :3].reshape(6).astype(np.float32)
        mean = np.asarray(self.group["state"]["mean"], np.float32)[STATE_POSITION]
        denom = np.asarray(self.group["state"]["std"], np.float32)[STATE_POSITION] + np.float32(1e-8)
        result[STATE_POSITION] = np.clip((positions - mean) / denom, -5, 5)
        result[STATE_ROTATION] = rotation_6d(rotations).reshape(12).astype(np.float32)
        return result, self._valid["state"].copy()

    def denormalize_action(self, prediction):
        x = np.asarray(prediction, np.float32)
        if x.ndim != 2 or x.shape[-1] != ACTION_DIM or not np.isfinite(x[:, self.action_valid]).all():
            raise ValueError("Expected finite [T,90] prediction on all 26 supervised action slots")
        result = np.zeros_like(x)
        result[:, ACTION_INDICES] = self.joint.decode_action(x)
        mean = np.asarray(self.group["action"]["mean"], np.float32)[ACTION_EEF]
        denom = np.asarray(self.group["action"]["std"], np.float32)[ACTION_EEF] + np.float32(1e-8)
        result[:, ACTION_EEF] = x[:, ACTION_EEF] * denom + mean
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite decoded Joint14/EEF action")
        return result

    def decode_action(self, prediction):
        """The executable command stays absolute Joint14, with native grippers."""
        return self.denormalize_action(prediction)[:, ACTION_INDICES].copy()

    def decode_with_eef(self, prediction, anchor_pose):
        """Decode all absolute poses once using the immutable chunk-start anchor."""
        raw = self.denormalize_action(prediction)
        anchor = np.asarray(anchor_pose, np.float32)
        if anchor.shape != (2, 7) or not np.isfinite(anchor).all():
            raise ValueError("Expected a finite fixed EEF anchor [2,7]")
        rotation = quaternion_to_matrix(anchor[:, 3:])
        relative = raw[:, ACTION_EEF].reshape(len(raw), 2, 6).copy()
        absolute = np.empty((len(raw), 2, 7), np.float32)
        for arm in range(2):
            absolute[:, arm, :3] = anchor[arm, :3] + relative[:, arm, :3]
            R = rotation[arm] @ rotvec_to_matrix(relative[:, arm, 3:])
            absolute[:, arm, 3:] = matrix_to_quaternion(R).astype(np.float32)
        return dict(joint14=raw[:, ACTION_INDICES].copy(), eef_relative=relative,
                    eef_pose=absolute, anchor_pose=anchor.copy(), eef_frame=EEF_FRAME)
