"""RoboDojo worker for the three-layer runtime, one task per process.

Extends the general worker with the calls the runtime kernel needs:

    profile                 static facts: action layout, limits, cameras, rates, arm bases
    state                   robot state, without rendering cameras
    tick                    one policy step of absolute joint targets, checked by the shield
    observe                 camera images, intrinsics, camera-to-world extrinsics
    fk / ik / plan          cuRobo kinematics and motion planning, env frame
    action_spec             per arm: mount pose, joint limits and velocity limits; control rates
    policy_obs              RoboDojo's own observation, as its eval client hands a policy (run_policy)
    claim_outcome / outcome evaluator result, readable only by the episode driver

The shield runs here, in the simulator process, so model-written programs in
the kernel cannot bypass or change it. The per-action ``execute`` path is
disabled for the same reason. Motion video is off by default: rendering three
cameras on every physics tick dominated the step time of the general worker.

Importing this module does not import Isaac, torch, or RoboDojo.
"""

import json
import math
import os
import secrets
import sys
import time

from harness_policy.core.contracts import Action, Transition
from harness_policy.runtime.shield import Shield, ShieldProfile

from .adapter import _normalized_opening, _rgb_array, _vector
from .codec import CAMERAS, action_dict, data_url, episode_outcome, policy_observation
from . import kinematics
from .general import RoboDojoTaskEnvironment
from .tasks import DEFAULT_TASK


# The shield catches commands that are bugs (harness_policy.runtime.shield); it never
# changes a command. One 25 Hz tick of the ARX X5 at its URDF velocity limit
# (3.14 rad/s) moves a joint 0.13 rad, and a fast motion moves the hand a few
# centimeters: a jump of 0.3 rad or 0.25 m from the previous accepted command is
# not a motion. The workspace box is the reachable space above the table.
DEFAULT_SHIELD = {
    "max_joint_jump_rad": 0.3,
    "workspace": {"x": [-1.1, 1.1], "y": [-1.25, 0.35], "z": [0.70, 1.565]},
    "max_cartesian_jump_m": 0.25,
}
# How the skills time their own motions: the planner's joint velocity limit (the URDF's,
# in arx_x5_description; the simulator's copy has placeholder limits) and acceleration
# (its configuration). This is how RoboDojo's expert moves, not a safety bound.
PLANNER_LIMITS = {"joint_velocity": 3.14, "joint_acceleration": 10.0}
METHODS = ("profile", "state", "tick", "observe", "fk", "ik", "plan", "claim_outcome", "outcome", "action_spec",
           "policy_obs")


def create_environment(**options):
    if os.environ.get("HARNESS_ROBODOJO_RUNTIME") != "1":
        raise RuntimeError(
            "RoboDojo adapter must run in the harness-policy environment with "
            "HARNESS_ROBODOJO_RUNTIME=1. See benches/robodojo/README.md."
        )
    allowed = {"device", "headless", "checkout", "task", "layout_pack_seed", "layout_source", "dev_layout_dir",
               "record_video", "shield", "ground_truth_log"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unknown RoboDojo runtime options: {sorted(unknown)}")
    device = options.get("device", "cuda:0")
    if device != "cuda:0":
        raise ValueError("RoboDojo worker uses cuda:0 inside CUDA_VISIBLE_DEVICES")
    shield = options.get("shield") or {}
    if not isinstance(shield, dict) or set(shield) - set(DEFAULT_SHIELD):
        raise ValueError(f"shield options must be a table with keys from {sorted(DEFAULT_SHIELD)}")
    return RoboDojoRuntimeEnvironment(
        task=options.get("task", DEFAULT_TASK),
        layout_pack_seed=options.get("layout_pack_seed", 0),
        device=device,
        headless=options.get("headless", True),
        checkout=options.get("checkout"),
        layout_source=options.get("layout_source", "eval"),
        dev_layout_dir=options.get("dev_layout_dir"),
        record_video=bool(options.get("record_video", False)),
        shield=shield,
        ground_truth_log=options.get("ground_truth_log"),
    )


class RoboDojoRuntimeEnvironment(RoboDojoTaskEnvironment):
    name = "robodojo-isaac51-runtime"
    methods = METHODS

    def __init__(self, *, record_video=False, shield=None, ground_truth_log=None, **kwargs):
        # _boot (inside the parent constructor) installs the motion hook.
        self._record_video = bool(record_video)
        # Development only: object poses from the simulator, appended to this file at
        # reset and when the outcome is read, to measure perception and grasp errors
        # offline. Nothing in the kernel or the model can read it.
        self._ground_truth_log = ground_truth_log
        self._shield_options = {**DEFAULT_SHIELD, **(shield or {})}
        self._outcome_token = None
        # (episode id, reason) once RoboDojo marks the running episode unstable mid-way.
        self._unscorable = None
        super().__init__(**kwargs)
        self._groups = self._arm_groups()
        profile = {"groups": {group["name"]: {"joints": group["joints"], "gripper": group["gripper"]}
                              for group in self._groups},
                   "control_hz": float(self._env.obs_manager.collect_freq), **self._shield_options}
        self._shield = Shield(ShieldProfile.from_dict(
            profile, [low for low, _high in self._bounds], [high for _low, high in self._bounds]))
        self._fk_error = self._check_forward_kinematics()
        print(f"RoboDojo runtime worker ready fk_error_m={self._fk_error:.5f}", file=sys.stderr, flush=True)

    # -- Worker protocol ---------------------------------------------------------

    def reset(self, task: str, seed: int):
        observation = super().reset(task, seed)
        # Drop reset frames and anything left from an episode that never read its outcome.
        self._reset_clip()
        self._outcome_token = None
        self._unscorable = None
        self._shield.reset()
        self._log_ground_truth("reset")
        return observation

    def execute(self, action: Action, episode_id: str, step: int) -> Transition:
        raise RuntimeError("The runtime worker accepts motion only through call('tick'), which the shield checks")

    def call(self, name: str, arguments: dict):
        if name not in METHODS:
            raise ValueError(f"Unknown runtime method: {name}")
        return getattr(self, f"_call_{name}")(**arguments)

    # -- Calls -------------------------------------------------------------------

    def _call_profile(self):
        interval = int(self._env.obs_manager.collect_interval)
        return {
            "task": self._task,
            "step_limit": int(self._env.step_lim),
            "control_hz": float(self._env.obs_manager.collect_freq),
            "physics_steps_per_tick": interval,
            "action_terms": self._layout,
            "action_low": [low for low, _high in self._bounds],
            "action_high": [high for _low, high in self._bounds],
            "groups": [{key: group[key] for key in ("name", "joints", "gripper")} for group in self._groups],
            "arm_bases": {group["name"]: [float(v) for v in group["robot"].entity_origin_pose]
                          for group in self._groups},
            "cameras": list(CAMERAS),
            "shield": self._shield_options,
            "planner_limits": PLANNER_LIMITS,
            "fk_check_error_m": self._fk_error,
            "planner_world": "self collision and table only; scene objects and the other arm are not modeled",
            # Measured opening of an empty closed gripper (normalized 0 closed .. 1 open).
            "gripper_empty_opening": 0.185,
            # Distance between the fingers fully open (2 x 0.044 m travel, X5A.urdf).
            "gripper_open_width_m": 0.088,
            # Most RoboDojo tasks count success only with both arms back at their start
            # pose (reward_manager.all_robot_back_to_origin: 0.15 m, 20 degrees).
            "return_to_start": True,
        }

    def _call_state(self, episode_id):
        self._check_episode(episode_id)
        return self._robot_state()

    def _call_tick(self, episode_id, step, command, cameras=None):
        """One policy step; with `cameras`, the result also carries the camera views
        after the step (as observe returns them), for controllers that servo on images."""
        self._raise_if_unscorable(episode_id)
        self._cursor.check(episode_id, step)
        if self._ended():
            raise ValueError("The episode has ended; the command was not executed")
        current = self._joint_state()
        decision = self._shield.check(command, current, fk=self._fk_positions)
        if decision.command is None:
            return {"executed": False, "shield": decision.to_dict(), "state": self._robot_state()}
        payload = action_dict(self._layout, decision.command)
        try:
            self._env.take_action(payload)
        except Exception:
            self._cursor.invalidate()
            raise
        self._cursor.advance()
        if 0 in getattr(self._env, "unstable_envs", ()):
            # RoboDojo drops such an episode from its count (a scripted support arm, e.g. in
            # make_kong, destabilized the scene). Later calls for it repeat this reason, and
            # the outcome carries it, so the run is recorded as unscorable, not as stale.
            self._unscorable = (episode_id, f"RoboDojo marked task={self._task} layout={self._seed} unstable "
                                            f"after step {step}; the episode is not scorable")
            self._cursor.invalidate()
            raise RuntimeError(self._unscorable[1] + " (the command was executed)")
        result = {"executed": True, "shield": decision.to_dict(), "state": self._robot_state()}
        if cameras:
            result["observation"] = self._call_observe(episode_id, cameras=cameras)
        return result

    def _call_observe(self, episode_id, cameras=None, image_format="jpeg", quality=90, raw_state=False):
        """Camera views and calibration. raw_state adds RoboDojo's own observation state
        (what its evaluator hands a policy)."""
        import cv2
        import numpy as np

        self._check_episode(episode_id)
        cameras = list(cameras or CAMERAS)
        unknown = set(cameras) - set(CAMERAS)
        if unknown:
            raise ValueError(f"Unknown cameras: {sorted(unknown)}")
        if image_format not in ("jpeg", "png"):
            raise ValueError("image_format must be 'jpeg' or 'png'")
        raw = policy_observation(self._env.obs_manager)[0]
        vision = raw.get("vision") or {}
        manager = self._env.obs_manager.camera_manager
        names = list(manager.camera_names[0])
        images, intrinsics, extrinsics = {}, {}, {}
        for camera in cameras:
            frame = (vision.get(camera) or {}).get("color")
            if frame is None:
                raise RuntimeError(f"RoboDojo observation is missing {camera}")
            array = _rgb_array(np, frame)
            bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            if image_format == "jpeg":
                ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
                mime = "image/jpeg"
            else:
                ok, encoded = cv2.imencode(".png", bgr)
                mime = "image/png"
            if not ok:
                raise RuntimeError(f"Could not encode {camera}")
            images[camera] = data_url(encoded.tobytes(), mime)
            index = names.index(camera)
            intrinsics[camera] = manager.get_camera_intrinsics(index, 0).tolist()
            # RoboDojo returns the camera xform in the USD convention (looks down -Z,
            # +Y up). Flip Y and Z for OpenCV (x right, y down, z forward). Verified by
            # projecting link6 positions into cam_head. env 0 sits at the world origin.
            transform = manager.get_camera_extrinsics(index, 0)
            transform[:3, :3] = transform[:3, :3] @ np.diag([1.0, -1.0, -1.0])
            extrinsics[camera] = transform.tolist()
        result = {"state": self._robot_state(), "images": images, "intrinsics": intrinsics,
                  "camera_to_world": extrinsics, "camera_convention": "opencv",
                  "image_size": list(array.shape[1::-1])}
        if raw_state:
            result["raw_state"] = {key: np.asarray(value, dtype=float).reshape(-1).tolist()
                                   for key, value in (raw.get("state") or {}).items()}
        return result

    def _call_policy_obs(self, episode_id):
        """RoboDojo's own observation, as its evaluation client hands it to a policy
        (TASK_ENV.get_obs()): the camera colors (lossless, PNG), the state, the
        instruction and additional_info, plus the robot state for the kernel."""
        import cv2
        import numpy as np

        self._check_episode(episode_id)
        raw = policy_observation(self._env.obs_manager)[0]
        vision = {}
        for camera, views in (raw.get("vision") or {}).items():
            frame = (views or {}).get("color")
            if frame is None:
                continue
            ok, encoded = cv2.imencode(".png", cv2.cvtColor(_rgb_array(np, frame), cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError(f"Could not encode {camera}")
            vision[camera] = data_url(encoded.tobytes(), "image/png")
        state = {key: np.asarray(value.detach().cpu() if hasattr(value, "detach") else value, dtype=float).tolist()
                 for key, value in (raw.get("state") or {}).items()}
        return {"vision": vision, "state": state, "instruction": raw.get("instruction"),
                "additional_info": raw.get("additional_info") or {},
                "data_format_version": raw.get("data_format_version"), "robot_state": self._robot_state()}

    def _call_fk(self, command=None):
        if command is None:
            command = self._joint_state()
        return self._forward_kinematics(command)

    def _call_ik(self, episode_id, arm, pose):
        self._check_episode(episode_id)
        group = self._group(arm)
        current = [self._joint_state()[i] for i in group["joints"]]
        return kinematics.inverse(self._planner(group), self._described(group), current, pose)

    def _call_plan(self, episode_id, arm, pose=None, joints=None, raw=False):
        """Policy-rate waypoints to a pose or joints. raw returns the planner's own
        physics-rate path ({"status", "position"}) for a client that resamples it itself."""
        self._check_episode(episode_id)
        group = self._group(arm)
        current = [self._joint_state()[i] for i in group["joints"]]
        return kinematics.plan(self._planner(group), self._described(group), current,
                               int(self._env.obs_manager.collect_interval), pose=pose, joints=joints, raw=raw)

    def _call_action_spec(self):
        """Per target arm its mount pose, soft joint limits and the simulator's joint
        velocity limits, and the control rates (what RoboProbe's action spec reads from
        RoboDojo's robot manager)."""
        import numpy as np

        manager = self._env.robot_manager
        arms = []
        for index, robot in enumerate(manager.robot_list):
            if robot.type != "target":
                continue
            asset = manager.robot_key[index]
            limits = asset.data.soft_joint_pos_limits[0, robot.arm_joint_indices]
            velocities = asset.data.joint_vel_limits[0, robot.arm_joint_indices]
            arms.append({
                "side": str(robot.arm_name).removesuffix("_arm"),
                "mount_pose": [float(v) for v in robot.entity_origin_pose],
                "joint_names": [str(name) for name in robot.arm_joints_name],
                "limits": np.asarray(limits.detach().cpu().numpy() if hasattr(limits, "detach") else limits,
                                     dtype=float).tolist(),
                "velocities": np.asarray(velocities.detach().cpu().numpy() if hasattr(velocities, "detach")
                                         else velocities, dtype=float).tolist(),
            })
        return {"arms": arms, "control_hz": float(self._env.obs_manager.collect_freq),
                "collect_interval": float(self._env.obs_manager.collect_interval),
                "step_limit": int(self._env.step_lim), "robodojo_root": str(self.checkout)}

    def _call_claim_outcome(self, episode_id):
        self._check_episode(episode_id)
        if self._outcome_token is not None:
            raise PermissionError("The outcome token for this episode was already claimed")
        self._outcome_token = secrets.token_hex(16)
        return {"token": self._outcome_token}

    def _call_outcome(self, token):
        if self._outcome_token is None or not secrets.compare_digest(str(token), self._outcome_token):
            raise PermissionError("Invalid outcome token")
        scores = getattr(self._env.reward_manager, "_gated_score_lst", None)
        process_score = 0.0 if not scores else float(scores[0])
        steps = int(self._env.take_action_cnt[0])
        reward, success, terminated, truncated = episode_outcome(
            bool(self._env.end_flag[0]), bool(self._env.success[0]), process_score, steps, int(self._env.step_lim))
        self._log_ground_truth("outcome", success=success, process_score=process_score, steps=steps,
                               unscorable=self._unscorable[1] if self._unscorable else None)
        result = {"reward": reward, "success": success, "terminated": terminated, "truncated": truncated,
                  "process_score": process_score, "steps": steps}
        if self._unscorable is not None:
            result["unscorable"] = self._unscorable[1]
        if self._record_video:
            result.update(self._outcome_videos())
        return result

    # -- Helpers -----------------------------------------------------------------

    def _outcome_videos(self):
        """The episode's motion clip, one MP4 data URL per camera; the driver writes the files.

        A failed encode (e.g. no ffmpeg) is reported instead of raised, so the outcome still arrives.
        """
        try:
            return {"videos": self._encode_clip()}
        except Exception as error:
            self._reset_clip()
            return {"video_error": f"{type(error).__name__}: {error}"}

    def _log_ground_truth(self, event, **extra):
        if not self._ground_truth_log:
            return

        def _flat(value):
            if hasattr(value, "detach"):
                value = value.detach().cpu()
            if hasattr(value, "tolist"):
                value = value.tolist()
            return value
        try:
            layout = self._env.scene_manager.layout_manager
            objects = []
            for kind in ("Rigid", "Articulation"):
                for record in layout.get_layout_records(0, kind):
                    name = record["inst_name"]
                    position, rotation = layout.get_instance_pose(0, inst_name=name)
                    vertices = layout.get_instance_bbox_vertices(name, 0)
                    objects.append({
                        "label": record.get("label"), "inst_name": name, "type": kind,
                        "position": _flat(position), "rotation_wxyz": _flat(rotation),
                        "bbox_vertices": None if vertices is None else _flat(vertices),
                        "description": layout.get_label_descriptions(0, inst_name=name)})
            entry = {"event": event, "time": time.time(), "task": self._task, "seed": self._seed,
                     "episode_id": self._cursor.episode_id, "objects": objects, **extra}
        except Exception as error:  # diagnostics must never break an episode
            entry = {"event": event, "task": self._task, "error": repr(error)}
        with open(self._ground_truth_log, "a") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")

    def _install_motion_video(self):
        if self._record_video:
            super()._install_motion_video()
        else:
            self._reset_clip()  # never fed, so the outcome has no videos

    def _raise_if_unscorable(self, episode_id):
        if self._unscorable is not None and episode_id == self._unscorable[0]:
            raise RuntimeError(self._unscorable[1])

    def _check_episode(self, episode_id):
        self._raise_if_unscorable(episode_id)
        if self._cursor.episode_id is None or episode_id != self._cursor.episode_id:
            raise ValueError("Stale or missing episode")

    def _ended(self):
        return bool(self._env.end_flag[0]) or int(self._env.take_action_cnt[0]) >= int(self._env.step_lim)

    def _arm_groups(self):
        robots = self._target_robots()
        groups, index, arm = [], 0, 0
        for term in self._layout:
            size = int(term["size"])
            if term["command"] == "absolute_joint":
                groups.append({"name": str(term["term"]), "joints": list(range(index, index + size)),
                               "gripper": None, "robot": robots[arm]})
                arm += 1
            elif term["command"] == "gripper_normalized":
                groups[-1]["gripper"] = index
            index += size
        if [group["name"] for group in groups] != ["left_arm", "right_arm"]:
            raise RuntimeError(f"Expected left_arm and right_arm groups, got {[g['name'] for g in groups]}")
        return groups

    def _group(self, arm):
        for group in self._groups:
            if group["name"] == arm:
                return group
        raise ValueError(f"Unknown arm {arm!r}; expected one of {[g['name'] for g in self._groups]}")

    def _planner(self, group):
        return self._env.robot_manager.planner[group["robot"].robot_name]

    def _joint_state(self):
        manager = self._env.robot_manager
        values = []
        for group in self._groups:
            robot = group["robot"]
            arm = manager.get_joint(robot, env_idx_list=[0])[0]
            values.extend(_vector(arm, len(group["joints"]), robot.arm_name))
            measured = manager.get_end_effector_real_val(robot, env_idx_list=[0])[0]
            values.append(_normalized_opening(robot, _vector(measured, None, robot.gripper_name)[0]))
        return values

    def _robot_state(self):
        manager = self._env.robot_manager
        state = {
            "step": self._cursor.step,
            "step_limit": int(self._env.step_lim),
            "ended": self._ended(),
            "joint_position": self._joint_state(),
        }
        for group in self._groups:
            pose = manager.get_real_endpose(group["robot"], env_idx_list=[0])[0]
            state[f"{group['name'].split('_')[0]}_ee_pose"] = _vector(pose, 7, group["name"])
        return state

    @staticmethod
    def _described(group):
        """A group as the kinematics functions take it: name, joint indices, mount pose."""
        return {"name": group["name"], "joints": group["joints"], "base": list(group["robot"].entity_origin_pose)}

    def _forward_kinematics(self, command):
        planners = {group["name"]: self._planner(group) for group in self._groups}
        return kinematics.forward(lambda group: planners[group["name"]], [self._described(g) for g in self._groups],
                                  command)

    def _fk_positions(self, command):
        return {name: pose[:3] for name, pose in self._forward_kinematics(command).items()}

    def _check_forward_kinematics(self):
        """Distance between FK of the measured joints and the simulator's end-effector pose."""
        predicted = self._forward_kinematics(self._joint_state())
        manager = self._env.robot_manager
        error = 0.0
        for group in self._groups:
            measured = _vector(manager.get_real_endpose(group["robot"], env_idx_list=[0])[0], 7, group["name"])
            error = max(error, math.dist(predicted[group["name"]][:3], measured[:3]))
        return error
