"""Episode-local tool state; observations enter only through the public API."""

from copy import deepcopy
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from orimapi.policy import ARX_FIELDS
from orimapi.skills.code_skills.common.geometry import pose_matrix, rgb
from orimapi.skills.code_skills.play_tic_tac_toe.motion import down_rotation
from orimapi.skills.code_skills.play_tic_tac_toe.sandbox import (
    vector as workspace_point,
)

from .geometry import (
    relative_pose,
    solve_pose,
    triangulate,
    unit,
    upright_feature,
    vector,
    verify_pose,
)
from .tracking import PointTracker
from .regions import crop_bounds, outline_candidates, fit_circle, edge_support
from .rim import fit_rim, rim_edge_support
from .envelope import measure as body_envelope
from .sequence import CheckedSequence
from .motion import Motion
from .collision import RobotClearance
from .stereo import guides as stereo_guides, propose as stereo_candidates
from .grasp_preview import render as render_grasp_preview, validate_resources as validate_finger_resources
from .camera_calibration import describe as describe_camera_calibration, CAMERAS
from .sphere import fit_sphere as reconstruct_sphere, sphere_edge_support, BoundarySupportError
from .segmentation import overlay as segmentation_overlay
from .camera_pointing import point_camera
from .pursuit import PointPursuit
from orimapi.state import RGBImage
import cv2


def fields(value, required, optional=()):
    if (
        not isinstance(value, dict)
        or not set(required) <= value.keys()
        or value.keys() - set(required) - set(optional)
    ):
        raise ValueError("Unexpected tool arguments")


def steps(value, minimum=1):
    if type(value) is not int or not minimum <= value <= 120:
        raise ValueError(f"Steps must be an integer in {minimum}..120")
    return value


def arm(value):
    if value not in ("left", "right"):
        raise ValueError("Arm must be left or right")
    return value


class ToolSession:
    def __init__(self, geometry, emit, *, depth=None, action_budget=None, recipes=None,
                 require_cartesian_completion=False, segmentation=None, camera_pointing=False,
                 point_pursuit=False):
        self.geometry, self.emit = geometry, emit
        self.depth, self.action_budget = depth, action_budget
        self.segmentation = segmentation
        if type(camera_pointing) is not bool:
            raise ValueError("Camera pointing opt-in must be boolean")
        self.camera_pointing = camera_pointing
        if type(point_pursuit) is not bool:
            raise ValueError("Point pursuit opt-in must be boolean")
        self.point_pursuit = point_pursuit
        self.recipes = recipes
        collision = geometry.resources.get("robot_collision_config")
        self.clearance = RobotClearance(geometry, collision) if collision else None
        self.motion = Motion(
            geometry, emit, self.clearance,
            require_cartesian_completion=require_cartesian_completion,
        )
        self.tracker = PointTracker()
        self.reset()

    def reset(self):
        self.tracker.reset()
        self.motion.reset()
        if self.depth is not None:
            self.depth.reset()
        if self.segmentation is not None:
            self.segmentation.reset()
        self.epoch = uuid.uuid4().hex
        self.observation = None
        self.frames, self.bindings, self.plans = {}, {}, {}
        self.attachments = {}
        self.regions, self.measurements, self.previews = {}, {}, {}
        self.segmentations, self.segmentation_calls = {}, 0
        self.sequence = None
        self.pursuit = self.pursuit_target = self.pursuit_signature = None
        self.active = self.last_result = None
        self.last_preflight = None
        self.motion_start_step = self.motion_end_step = None
        self.finished = False

    def observe(self, observation):
        if self.observation is not None and observation.step == self.observation.step:
            if observation.env_id != self.observation.env_id:
                raise ValueError("Tool session cannot mix environments")
            return
        before = {p["id"]: p["status"] for p in self.tracker.describe()}
        self.tracker.update(observation)
        self.regions.clear()
        self.measurements.clear()
        self.previews.clear()
        self.segmentations.clear()
        self.segmentation_calls = 0
        self.observation = observation
        if self.clearance is not None:
            self.clearance.observe(observation)
        for side in ("left", "right"):
            name = f"{side}_ee_pose"
            if name in observation.robot_state:
                measured = pose_matrix(observation.robot_state[name])
                predicted = self.geometry.fk(
                    side, observation.robot_state[f"{side}_arm_joint_state"], full=True
                )[6]
                if np.linalg.norm(measured[:3, 3] - predicted[:3, 3]) > 0.005:
                    raise ValueError(
                        "Public calibration disagrees with observed robot pose"
                    )
        cameras = self.geometry.cameras(observation.robot_state)
        self.frames = {}
        for name, image in self.tracker.frames.items():
            camera = cameras.get(name)
            if camera is None or image.shape != (camera.height, camera.width):
                raise ValueError("RGB shape/camera does not match public calibration")
            self.frames[name] = {
                "frame_id": f"{self.epoch}:{name}:{observation.step}",
                "width": camera.width,
                "height": camera.height,
            }
        for row in self.tracker.describe():
            if before.get(row["id"]) != row["status"]:
                self.emit("point_status", step=observation.step, point=row)
        if self.motion.home is None:
            self.motion.bind([], observation)

    def hold(self):
        action = {name: list(self.observation.robot_state[name]) for name in ARX_FIELDS}
        action.update(deepcopy(self.motion.gripper_targets))
        return action

    @property
    def remaining(self):
        return (
            None
            if self.action_budget is None
            else max(0, self.action_budget - self.observation.step)
        )

    @property
    def motion_budget(self):
        if self.remaining is None:
            return None
        return max(0, self.remaining - (self.sequence.reserve if self.sequence else 0))

    def query_images(self):
        return {
            **self.observation.images,
            **{name: row["image"] for name, row in self.previews.items()},
        }

    def describe(self):
        return {
            "frames": deepcopy(self.frames),
            "tracked_points": self.tracker.describe(),
            "remaining_actions": self.remaining,
            "depth_available": self.depth is not None,
            **(
                {"segmentation_proposals_available": True}
                if self.segmentation is not None else {}
            ),
            "last_action_result": deepcopy(self.last_result),
            "inspection_views": {
                name: {key: value for key, value in row.items() if key != "image"}
                for name, row in self.previews.items()
            },
            "current_measurements": deepcopy(self.measurements),
            "public_robot_geometry": self.robot_geometry(),
            "attachments": {
                key: {"arm": value["arm"], "capture_step": value["step"]}
                for key, value in self.attachments.items()
            },
        }

    def robot_geometry(self):
        state = self.observation.robot_state
        chain = getattr(self.geometry, "chain", None)
        return {
            "table_z_m": float(self.geometry.table),
            "tcp_origin": "grasp center, NOT observed ee_pose wrist origin",
            "tcp_offset_from_wrist_m": getattr(self.geometry, "tip_offset", None),
            "local_point_zero": "[0,0,0] targets the grasp center; do not add the wrist-to-grasp offset again",
            "tcp_world": {
                side: self.geometry.fk(side, state[f"{side}_arm_joint_state"]).tolist()
                for side in ("left", "right")
            },
            "planning_joint_bounds_rad": (
                [list(map(float, link.bounds)) for link in chain.links[1:7]]
                if chain is not None
                else None
            ),
            "bounds_scope": "public URDF intersected with planner bounds, not contact or hidden simulator state",
            "cross_arm_guard": (
                "public collision spheres plus hand envelopes"
                if self.clearance
                else "fixed side boundary"
            ),
        }

    def motion_diagnostics(self):
        state = self.observation.robot_state
        end = (
            self.motion_end_step
            if self.motion_end_step is not None
            else self.observation.step
        )
        return {
            "public_robot_geometry": self.robot_geometry(),
            "current_joints": {
                side: state[f"{side}_arm_joint_state"] for side in ("left", "right")
            },
            "desired_waypoint_joints": deepcopy(self.motion.arm_targets),
            "waypoint_joint_error_rad": {
                side: (
                    np.asarray(
                        self.motion.arm_targets.get(
                            f"{side}_arm_joint_state", state[f"{side}_arm_joint_state"]
                        )
                    )
                    - np.asarray(state[f"{side}_arm_joint_state"])
                ).tolist()
                for side in ("left", "right")
            },
            "operation": deepcopy(self.motion.plan[0]) if self.motion.plan else None,
            "kinematic_preflight": deepcopy(self.last_preflight),
            "cartesian_path_waypoints": len(self.motion.path),
            **(
                {"cartesian_completion": deepcopy(self.motion.cartesian_completion)}
                if getattr(self.motion, "require_cartesian_completion", False)
                else {}
            ),
            "path_cursor": self.motion.cursor,
            "motion_elapsed_actions": (
                None if self.motion_start_step is None else end - self.motion_start_step
            ),
            "failure": self.motion.failure,
            "timing_contract": "operation.steps is a nominal minimum, not a scheduled duration. Actual paths expand for joint distance; tracking has a bounded extra allowance.",
            "scope": "public robot and policy telemetry only; cannot identify hidden contacts or prove grasp",
        }

    def locate(self, name):
        if name in self.measurements:
            return deepcopy(self.measurements[name])
        row = self.tracker.current(name)
        binding = self.bindings.get(name)
        if binding is None:
            raise ValueError(f"Point {name!r} has no localization method")
        cameras = self.geometry.cameras(self.observation.robot_state)
        camera = cameras[row["camera"]]
        method = binding["method"]
        if method == "plane":
            position = camera.on_plane(
                row["pixel"], self.geometry.table + binding["height_m"]
            )
            result = {
                "position": position.tolist(),
                "method": "assumed_plane",
                "assumptions": ["Feature lies on the explicitly supplied height plane"],
                "height_above_table_m": binding["height_m"],
            }
        elif method == "depth":
            if self.depth is None:
                raise ValueError(
                    "No configured depth model; request capability or use another explicit method"
                )
            z = self.depth.sample(
                row["camera"], self.frames[row["camera"]]["frame_id"], row["pixel"]
            )
            optical = np.linalg.solve(camera.k, np.r_[row["pixel"], 1]) * z
            point = camera.optical @ np.r_[optical, 1]
            result = {
                "position": point[:3].tolist(),
                "method": "monocular_metric_model",
                "estimated_optical_depth_m": z,
                "scale_calibration": self.depth.scales.get(row["camera"]),
                "assumptions": [
                    "Monocular metric scale and visible surface prediction may be biased"
                ],
                "sensor_ground_truth": False,
            }
        elif method == "upright":
            bases = [self.tracker.current(name) for name in binding["base_points"]]
            if any(base["camera"] != row["camera"] for base in bases):
                raise ValueError("Upright axis clicks must share one current camera")
            result = upright_feature(
                camera,
                [base["pixel"] for base in bases],
                row["pixel"],
                self.geometry.table,
            )
        else:
            other = self.tracker.current(binding["other_point"])
            if row["camera"] == other["camera"]:
                raise ValueError("Triangulation requires different current cameras")
            result = triangulate(
                camera, row["pixel"], cameras[other["camera"]], other["pixel"]
            )
        return {"id": name, "step": self.observation.step, **result}

    def _plan_pose(self, arguments):
        fields(arguments, ("arm", "constraints"), ("pivot_local", "steps"))
        side = arm(arguments["arm"])
        constraints = arguments["constraints"]
        if not isinstance(constraints, list) or not 1 <= len(constraints) <= 8:
            raise ValueError("Use 1..8 constraints")
        evidence, resolved = {}, []

        def point(name):
            evidence[name] = self.locate(name)
            return np.asarray(evidence[name]["position"])

        for row in constraints:
            if not isinstance(row, dict):
                raise ValueError("Constraint must be an object")
            kind = row.get("kind")
            if kind == "point":
                fields(row, ("kind", "local", "target"), ("offset",))
                resolved.append(
                    {
                        "kind": kind,
                        "local": vector(row["local"]).tolist(),
                        "target": (
                            point(row["target"]) + vector(row.get("offset", [0, 0, 0]))
                        ).tolist(),
                    }
                )
            elif kind == "axis":
                if "direction" in row:
                    fields(row, ("kind", "local", "direction"))
                    direction = unit(row["direction"])
                else:
                    fields(row, ("kind", "local", "from", "to"))
                    direction = unit(point(row["to"]) - point(row["from"]))
                resolved.append(
                    {
                        "kind": kind,
                        "local": unit(row["local"]).tolist(),
                        "target": direction.tolist(),
                    }
                )
            elif kind == "plane":
                fields(row, ("kind", "local", "points"))
                if not isinstance(row["points"], list) or len(row["points"]) != 3:
                    raise ValueError("Plane requires three observed points")
                a, b, c = [point(name) for name in row["points"]]
                resolved.append(
                    {
                        "kind": kind,
                        "local": vector(row["local"]).tolist(),
                        "origin": a.tolist(),
                        "normal": unit(np.cross(b - a, c - a)).tolist(),
                    }
                )
            else:
                raise ValueError("Unknown constraint kind")
        initial = self.geometry.fk(
            side, self.observation.robot_state[f"{side}_arm_joint_state"]
        )
        pose, report = solve_pose(initial, resolved)
        return self._save_pose(
            side,
            pose,
            evidence,
            report,
            pivot_local=arguments.get("pivot_local"),
            minimum_steps=arguments.get("steps", 12),
        )

    def _save_pose(
        self, side, pose, evidence, report, *, pivot_local=None, minimum_steps=12
    ):
        workspace_point(pose[:3, 3].tolist())
        tilt_x, tilt_y, yaw = Rotation.from_matrix(
            pose[:3, :3] @ down_rotation(0).T
        ).as_euler("XYZ")
        operation = {
            "op": "move",
            "label": "geometric target",
            "arm": side,
            "position": pose[:3, 3].tolist(),
            "yaw": float(yaw),
            "tilt_x": float(tilt_x),
            "tilt_y": float(tilt_y),
            "gripper": self.motion.gripper_targets[f"{side}_ee_joint_state"][0],
            "steps": steps(minimum_steps),
        }
        if pivot_local is not None:
            pivot = vector(pivot_local)
            if np.linalg.norm(pivot) > 0.3:
                raise ValueError("Local pivot must be within 0.3 m of the grasp center")
            operation["pivot_local"] = pivot.tolist()
        report["ik"] = self.motion.preflight(
            [operation], self.observation, self.motion_budget
        )
        name = uuid.uuid4().hex
        self.plans = {
            name: {
                "operation": operation,
                "pose": pose,
                "evidence": evidence,
                "revision": self.tracker.revision,
                "step": self.observation.step,
            }
        }
        return {
            "plan_id": name,
            "pose": pose.tolist(),
            "report": report,
            "evidence": evidence,
            "scope": "endpoint IK checked; full path checked at execution, no full scene collision guarantee",
        }

    def _capture_attachment(self, arguments):
        fields(arguments, ("arm", "points"))
        side, names = arm(arguments["arm"]), arguments["points"]
        if (
            not isinstance(names, list)
            or not 3 <= len(names) <= 8
            or not all(isinstance(name, str) for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Use 3..8 distinct features on the held object")
        if len(self.attachments) >= 8:
            raise ValueError("At most 8 attachment captures per episode")
        pose = self.geometry.fk(
            side, self.observation.robot_state[f"{side}_arm_joint_state"]
        )
        evidence = {name: self.locate(name) for name in names}
        points = np.array([row["position"] for row in evidence.values()])
        spread = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
        if spread[1] < 0.008:
            raise ValueError(
                "Attachment features must span a non-collinear object region"
            )
        rows = []
        for name, point in zip(names, points, strict=True):
            tracked = self.tracker.current(name)
            rows.append(
                {
                    "id": name,
                    "selection_revision": tracked["selection_revision"],
                    "camera": tracked["camera"],
                    "local": (pose[:3, :3].T @ (point - pose[:3, 3])).tolist(),
                }
            )
        identifier = uuid.uuid4().hex
        self.attachments[identifier] = {
            "arm": side,
            "step": self.observation.step,
            "pose": pose,
            "points": rows,
            "evidence": evidence,
        }
        return {
            "attachment_id": identifier,
            "points": rows,
            "evidence": evidence,
            "status": "captured_not_verified",
            "official_success": None,
            "scope": "Conditional object-to-TCP feature coordinates, not a grasp measurement",
        }

    def _verify_attachment(self, arguments):
        fields(arguments, ("attachment_id",))
        captured = self.attachments.get(arguments["attachment_id"])
        if captured is None:
            raise ValueError("Unknown attachment ID")
        side = captured["arm"]
        pose = self.geometry.fk(
            side, self.observation.robot_state[f"{side}_arm_joint_state"]
        )
        cameras = self.geometry.cameras(self.observation.robot_state)
        errors, missing = [], []
        for point in captured["points"]:
            try:
                current = self.tracker.current(point["id"])
                if current["selection_revision"] != point["selection_revision"]:
                    raise ValueError(
                        "Feature was re-marked; identity is no longer continuous"
                    )
                world = pose[:3, :3] @ point["local"] + pose[:3, 3]
                predicted = cameras[current["camera"]].project(world)
                errors.append(
                    {
                        "id": point["id"],
                        "camera": current["camera"],
                        "error_px": float(np.linalg.norm(predicted - current["pixel"])),
                    }
                )
            except ValueError as error:
                missing.append({"id": point["id"], "reason": str(error)})
        translation = float(np.linalg.norm(pose[:3, 3] - captured["pose"][:3, 3]))
        rotation = float(
            Rotation.from_matrix(captured["pose"][:3, :3].T @ pose[:3, :3]).magnitude()
        )
        usable = len(errors) >= 3 and (translation >= 0.01 or rotation >= 0.12)
        status = "unknown"
        if usable:
            status = (
                "visually_consistent"
                if max(row["error_px"] for row in errors) <= 4
                else "inconsistent"
            )
        return {
            "status": status,
            "point_errors": errors,
            "missing": missing,
            "translation_m": translation,
            "rotation_rad": rotation,
            "rotation_tested": usable and rotation >= 0.12,
            "geometry_estimated": any(
                row["method"] != "rgb_triangulation"
                for row in captured["evidence"].values()
            ),
            "official_success": None,
            "scope": "RGB reprojection consistency under assumed rigid attachment; not proof of grasp or flow. Inconsistency can also mean wrong geometry or tracking.",
        }

    def _start(self, operation, *, plan=None):
        if self.remaining is not None and self.remaining <= 0:
            raise ValueError("Native action budget exhausted")
        self.last_preflight = self.motion.preflight(
            [operation], self.observation, self.motion_budget
        )
        self.motion.bind([operation], self.observation, preserve_home=True)
        self.motion_start_step, self.motion_end_step = self.observation.step, None
        self.active = {
            "operation": operation,
            "plan": plan,
            "start_step": self.observation.step,
        }
        self.plans.clear()
        return {
            "status": "scheduled",
            "operation": operation,
            "official_success": None,
            "minimum_kinematic_steps": self.last_preflight["kinematic_minimum_steps"],
            "timing_contract": "operation.steps is only a minimum; the actual trajectory expands before execution",
        }

    def call(self, name, arguments):
        if self.active is not None or self.sequence is not None or self.finished:
            raise ValueError("Finish current motion before another tool call")
        return self._call(name, arguments)

    def _current_frame(self, arguments):
        camera = arguments["camera"]
        if (
            camera not in self.frames
            or arguments["frame_id"] != self.frames[camera]["frame_id"]
        ):
            raise ValueError("Stale or unknown RGB frame_id")
        return camera, rgb(self.observation.images[camera])

    def _segmentation(self, arguments):
        self._current_frame(arguments)
        identifier = arguments["segmentation_id"]
        if not isinstance(identifier, str) or not 1 <= len(identifier) <= 128:
            raise ValueError("segmentation_id must be a bounded current proposal ID")
        row = self.segmentations.get(identifier)
        if (
            row is None or row["camera"] != arguments["camera"]
            or row["frame_id"] != arguments["frame_id"]
        ):
            raise ValueError("Segmentation proposal is stale or belongs to another camera")
        return row

    def _pursuit_measurement(self, target, expected=None):
        if not isinstance(target, str) or not target.isidentifier():
            raise ValueError("Pursuit target must be a current material-point ID")
        binding = self.bindings.get(target)
        if not binding or binding.get("method") != "triangulate":
            raise ValueError("Pursuit requires two-view RGB triangulation, not a plane/depth/shape-center substitute")
        other = binding["other_point"]
        points = [self.tracker.current(name) for name in (target, other)]
        signature = {
            "binding": deepcopy(binding),
            "points": [
                {k: row[k] for k in ("id", "camera", "selection_revision")}
                for row in points
            ],
        }
        if expected is not None and signature != expected:
            raise ValueError("Pursuit material selection/binding changed; no automatic re-identification")
        measurement = self.locate(target)
        measurement["source_frames"] = {
            row["camera"]: self.frames[row["camera"]]["frame_id"] for row in points
        }
        measurement["point_signature"] = signature
        return measurement, signature

    def _start_pursuit(self, arguments):
        fields(arguments, ("arm", "target", "offset", "max_steps", "reserve_actions", "close"),
               ("empty_hand_review", "engagement_review"))
        if not self.point_pursuit:
            raise ValueError("Point pursuit is not enabled in this frozen candidate")
        side = arm(arguments["arm"])
        close = arguments["close"]
        if type(close) is not bool:
            raise ValueError("Pursuit close mode must be a boolean")
        if close:
            if arguments.get("engagement_review") is not True:
                raise ValueError("Explicit pursuit closure needs current-RGB engagement review")
        elif arguments.get("empty_hand_review") is not True:
            raise ValueError("Initial pursuit needs current-RGB empty-hand/approach review")
        reserve = arguments["reserve_actions"]
        if type(reserve) is not int or reserve < 1 or self.remaining is None:
            raise ValueError("Pursuit requires a positive explicit cleanup reserve and native action limit")
        available = self.remaining - reserve
        measurement, signature = self._pursuit_measurement(arguments["target"])
        pursuit = PointPursuit(
            self.geometry, self.motion, self.observation, measurement,
            side=side, offset=arguments["offset"], max_steps=arguments["max_steps"],
            available_actions=available, close=close,
        )
        self.pursuit = pursuit
        self.pursuit_target, self.pursuit_signature = arguments["target"], signature
        self.motion_start_step, self.motion_end_step = self.observation.step, None
        self.motion.failure = None
        self.active = {
            "operation": {
                "op": "pursue_point", "arm": side, "target": arguments["target"],
                "offset": list(arguments["offset"]), "close": close,
                "allowed_actions": pursuit.allowance, "reserve_actions": reserve,
            },
            "plan": None,
        }
        self.emit("pursuit_started", step=self.observation.step,
                  operation=self.active["operation"], signature=signature,
                  initial_measurement=measurement, preflight=pursuit.preflight)
        return {
            "status": "scheduled", "operation": deepcopy(self.active["operation"]),
            "official_success": None,
            "scope": "Finite current-point pursuit; no prediction/re-identification. Closure occurs only when explicitly requested after current arrival and RGB review; never lift or release automatically.",
        }

    def _pursuit_action(self):
        operation = deepcopy(self.active["operation"])
        try:
            if self.remaining is None or self.remaining < operation["reserve_actions"]:
                raise ValueError("Pursuit reached the preserved native cleanup reserve")
            measurement, _ = self._pursuit_measurement(
                self.pursuit_target, self.pursuit_signature
            )
            action, result = self.pursuit.step(self.observation, measurement)
        except ValueError as error:
            action = None
            result = {
                "status": "failed", "error": str(error), "official_success": None,
                "diagnostics": deepcopy(self.pursuit.latest),
                "proposed_action_dispatched": False,
            }
        if action is not None:
            # The pursuit engine has already applied all proposal guards. Only
            # committed commands become the hold/recovery grip and joint targets.
            self.motion.gripper_targets = {
                f"{side}_ee_joint_state": list(action[f"{side}_ee_joint_state"])
                for side in ("left", "right")
            }
            self.motion.arm_targets = {
                f"{side}_arm_joint_state": list(action[f"{side}_arm_joint_state"])
                for side in ("left", "right")
            }
            self.emit("pursuit_updated", step=self.observation.step,
                      diagnostics=deepcopy(self.pursuit.latest))
            return action
        self.motion_end_step = self.observation.step
        result["operation"] = operation
        self.last_result, self.active = result, None
        self.pursuit = self.pursuit_target = self.pursuit_signature = None
        self.emit("action_verified", step=self.observation.step, **result)
        return None

    def _plan_camera_pointing(self, arguments):
        fields(arguments, ("camera", "frame_id", "target", "empty_hand_review"),
               ("tcp_offset", "steps"))
        if not self.camera_pointing:
            raise ValueError("Camera pointing is not enabled in this frozen candidate")
        if arguments["camera"] not in ("cam_left_wrist", "cam_right_wrist"):
            raise ValueError("Only a wrist camera can receive a pointing plan")
        camera, _ = self._current_frame(arguments)
        side = "left" if camera == "cam_left_wrist" else "right"
        if (
            arguments["empty_hand_review"] is not True
            or self.motion.gripper_targets[f"{side}_ee_joint_state"][0] < 0.99
        ):
            raise ValueError("Pointing needs an explicitly RGB-reviewed empty, already-open camera hand; never open a held object to enable it")
        target_id = arguments["target"]
        target = self.locate(target_id)
        # Do not silently rely on a point tracker in the very camera whose
        # attitude is changing. Existing multi-view geometric targets retain
        # all of their original per-view RGB checks at execution.
        if target["method"] not in {"rgb_circle_fit", "rgb_oriented_rim_fit", "rgb_sphere_fit"}:
            if self.tracker.current(target_id)["camera"] == camera:
                raise ValueError("Use a current target witnessed by another camera, not the moving camera alone")
        geometry_cameras = self.geometry.cameras(self.observation.robot_state)
        tcp = self.geometry.fk(side, self.observation.robot_state[f"{side}_arm_joint_state"])
        mount = np.linalg.inv(tcp) @ geometry_cameras[camera].optical
        minimum_steps = steps(arguments.get("steps", 12))
        offset = arguments.get("tcp_offset", [0, 0, 0])
        candidates, attempts = [], []
        for twist in (0, 90, -90, 180):
            pose, report = point_camera(tcp, mount, target["position"], offset, twist)
            try:
                result = self._save_pose(
                    side, pose, {target_id: deepcopy(target)},
                    {"camera_pointing": report}, minimum_steps=minimum_steps,
                )
                cost = result["report"]["ik"]["kinematic_minimum_steps"]
                candidates.append((cost, report["total_orientation_change_rad"], pose, report))
                attempts.append({"target_axis_twist_deg": twist, "status": "preflight_pass",
                                 "kinematic_minimum_steps": cost})
            except ValueError as error:
                attempts.append({"target_axis_twist_deg": twist, "status": "preflight_rejected",
                                 "error": str(error)})
        if not candidates:
            return {
                "status": "rejected", "camera": camera,
                "error": "No checked pointing pose among the four bounded variants at this TCP offset",
                "attempts": attempts, "action_scheduled": False, "official_success": None,
                "scope": "Not proof that every viewpoint is infeasible. Reassess the actual rejection/clearance or another current view; do not blindly reduce height.",
            }
        _, _, pose, report = min(candidates, key=lambda row: (row[0], row[1]))
        result = self._save_pose(
            side, pose, {target_id: deepcopy(target)},
            {"camera_pointing": report}, minimum_steps=minimum_steps,
        )
        projected = geometry_cameras[camera].k @ np.array(report["modeled_target_optical_m"])
        result.update(
            status="planned_not_executed", camera=camera,
            source_frame_id=arguments["frame_id"],
            modeled_target_pixel=(projected[:2] / projected[2]).tolist(),
            pointing_attempts=attempts,
            empty_hand_review_scope="Caller current-RGB assertion, not an automatic contact/empty-hand certificate",
            selection_scope="Lowest existing kinematic preflight cost among four variants at the caller's TCP offset, not best visibility or full scene clearance",
            action_scheduled=False, official_success=None,
        )
        return result

    def fit_sphere(self, arguments):
        fields(arguments, ("first", "second"), ("arm",))
        side = arm(arguments["arm"]) if "arm" in arguments else None
        if len(self.measurements) >= 32:
            raise ValueError("At most 32 geometric measurements per observation")
        rows, images = [], []
        for key in ("first", "second"):
            row = arguments[key]
            fields(row, ("camera", "frame_id"), ("pixels", "outline_id"))
            if ("pixels" in row) == ("outline_id" in row):
                raise ValueError("Choose sphere pixels OR one current outline_id per view")
            _, image = self._current_frame(row)
            pixels = row.get("pixels")
            if "outline_id" in row:
                if not isinstance(row["outline_id"], str) or not 1 <= len(row["outline_id"]) <= 128:
                    raise ValueError("Sphere outline_id must be a bounded identifier")
                outline = self.regions.get(row["outline_id"])
                if (
                    outline is None or outline["camera"] != row["camera"]
                    or outline["frame_id"] != row["frame_id"]
                ):
                    raise ValueError("Sphere outline must belong to the selected current view")
                pixels = outline["points"]
            rows.append({
                "camera": row["camera"], "frame_id": row["frame_id"],
                "pixels": deepcopy(pixels),
            })
            images.append(image)
        if rows[0]["camera"] == rows[1]["camera"]:
            raise ValueError("Sphere reconstruction requires different current cameras")
        cameras = self.geometry.cameras(self.observation.robot_state)
        try:
            result = reconstruct_sphere(
                [cameras[row["camera"]] for row in rows],
                images, [row["pixels"] for row in rows], self.geometry.table,
            )
        except BoundarySupportError as error:
            return {
                "status": "rejected", "error": str(error),
                "boundary_diagnostics": [
                    {"input_view": key, "camera": row["camera"],
                     "frame_id": row["frame_id"], **details}
                    for key, row, details in zip(
                        ("first", "second"), rows, error.diagnostics, strict=True
                    )
                ],
                "measurement_created": False,
                "action_scheduled": False,
                "official_success": None,
            }
        result["edge_support"] = dict(zip(
            [row["camera"] for row in rows], result.pop("support_by_view"), strict=True
        ))
        if side is not None:
            tcp = self.geometry.fk(
                side, self.observation.robot_state[f"{side}_arm_joint_state"]
            )
            result["tcp_local_center"] = (
                np.linalg.inv(tcp) @ np.r_[result["position"], 1]
            )[:3].tolist()
            result["arm"] = side
            result["local_center_scope"] = (
                "Derived sphere center relative to the CURRENT public FK TCP; "
                "not measured attachment. Refit after grasp, slip or rotation. "
                "Use center plus radius for a spherical surface hypothesis, not "
                "a guessed fixed material point at the bottom of the head."
            )
        identifier = "sphere_" + uuid.uuid4().hex
        result.update(id=identifier, step=self.observation.step, selections=rows)
        self.measurements[identifier] = result
        return {"target_id": identifier, **deepcopy(result)}

    def public_camera_calibration(self, arguments):
        fields(arguments, ("frames",))
        requested = arguments["frames"]
        if (
            not isinstance(requested, dict) or not 1 <= len(requested) <= 3
            or any(name not in CAMERAS for name in requested)
            or any(not isinstance(frame_id, str) for frame_id in requested.values())
        ):
            raise ValueError("Select 1..3 original cameras with exact current frame IDs")
        if self.observation is None:
            raise ValueError("No current observation")
        frames = {}
        for camera, frame_id in requested.items():
            self._current_frame({"camera": camera, "frame_id": frame_id})
            frames[camera] = deepcopy(self.frames[camera])
        return describe_camera_calibration(
            self.geometry.cameras(self.observation.robot_state),
            frames,
            self.observation.robot_state,
            getattr(self.geometry, "tip_offset", None),
        )

    def preview_grasp_pose(self, arguments):
        fields(arguments, ("plan_id", "camera", "frame_id"))
        name = arguments["plan_id"]
        if not isinstance(name, str) or not 1 <= len(name) <= 128:
            raise ValueError("Preview plan ID must be a bounded identifier")
        plan = self.plans.get(name)
        if (
            plan is None
            or plan["step"] != self.observation.step
            or plan["revision"] != self.tracker.revision
        ):
            raise ValueError("Preview requires the current, unchanged pose plan")
        camera, image = self._current_frame(arguments)
        sources = validate_finger_resources(self.geometry)
        calibrated = self.geometry.cameras(self.observation.robot_state)[camera]
        result, preview = render_grasp_preview(image, calibrated, plan["pose"])
        result.update(
            plan_id=name,
            arm=plan["operation"]["arm"],
            source_camera=camera,
            source_frame_id=arguments["frame_id"],
            public_mesh_sources=sources,
        )
        # Only this optional synthetic preview is replaced. Required original
        # camera observations and other requested inspection views are retained.
        self.previews.pop("nominal_finger_preview", None)
        if preview is not None:
            self.previews["nominal_finger_preview"] = {
                "image": RGBImage(preview.tobytes(), preview.shape[1], preview.shape[0]),
                "source_camera": camera,
                "source_frame_id": arguments["frame_id"],
                "plan_id": name,
                "kind": "synthetic_nominal_finger_projection",
                "coordinate_rule": (
                    "Original current-camera dimensions/coordinates. Synthetic "
                    "lines are not observed points and are not a future camera view."
                ),
            }
            result["preview_view"] = "nominal_finger_preview"
        return result

    def _call(self, name, arguments):
        if name == "fit_sphere":
            return self.fit_sphere(arguments)
        if name == "public_camera_calibration":
            return self.public_camera_calibration(arguments)
        if name == "preview_grasp_pose":
            return self.preview_grasp_pose(arguments)
        if name == "stereo_candidates":
            fields(
                arguments,
                (
                    "first_camera", "first_frame_id", "second_camera",
                    "second_frame_id", "first_roi", "second_roi",
                ),
                ("maximum",),
            )
            first, first_rgb = self._current_frame({
                "camera": arguments["first_camera"],
                "frame_id": arguments["first_frame_id"],
            })
            second, second_rgb = self._current_frame({
                "camera": arguments["second_camera"],
                "frame_id": arguments["second_frame_id"],
            })
            if first == second:
                raise ValueError("Stereo candidates require two different current cameras")
            cameras = self.geometry.cameras(self.observation.robot_state)
            result = stereo_candidates(
                first_rgb, second_rgb, cameras[first], cameras[second],
                arguments["first_roi"], arguments["second_roi"],
                arguments.get("maximum", 8),
            )
            previews = []
            for label in ("stereo_candidates_first", "stereo_candidates_second"):
                self.previews.pop(label, None)
            views = (
                ("first", first, first_rgb, arguments["first_frame_id"], "first_pixel"),
                ("second", second, second_rgb, arguments["second_frame_id"], "second_pixel"),
            ) if result["proposals"] else ()
            for tag, camera, image, frame_id, key in views:
                overlay = image.copy()
                for index, row in enumerate(result["proposals"]):
                    pixel = tuple(int(round(v)) for v in row[key])
                    cv2.circle(overlay, pixel, 3, (255, 0, 255), 1)
                    cv2.putText(
                        overlay, str(index), (pixel[0] + 4, max(12, pixel[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1,
                    )
                label = "stereo_candidates_" + tag
                self.previews[label] = {
                    "camera": camera, "frame_id": frame_id,
                    "method": "annotated_original_rgb_correspondence_proposals",
                    "image": RGBImage(overlay.tobytes(), overlay.shape[1], overlay.shape[0]),
                }
                previews.append(label)
            return {
                **result,
                "first_camera": first, "second_camera": second,
                "first_frame_id": arguments["first_frame_id"],
                "second_frame_id": arguments["second_frame_id"],
                "previews": previews,
                "accepted_correspondences": False,
            }
        if name == "stereo_guides":
            fields(
                arguments,
                (
                    "first_camera",
                    "first_frame_id",
                    "second_camera",
                    "second_frame_id",
                    "pixels",
                    "second_roi",
                ),
                ("other_pixels",),
            )
            first, first_rgb = self._current_frame(
                {
                    "camera": arguments["first_camera"],
                    "frame_id": arguments["first_frame_id"],
                }
            )
            second, second_rgb = self._current_frame(
                {
                    "camera": arguments["second_camera"],
                    "frame_id": arguments["second_frame_id"],
                }
            )
            if first == second:
                raise ValueError("Stereo guides require two different current cameras")
            cameras = self.geometry.cameras(self.observation.robot_state)
            result = stereo_guides(
                first_rgb,
                second_rgb,
                cameras[first],
                cameras[second],
                arguments["pixels"],
                arguments["second_roi"],
                arguments.get("other_pixels"),
            )
            return {
                **result,
                "first_camera": first,
                "second_camera": second,
                "first_frame_id": arguments["first_frame_id"],
                "second_frame_id": arguments["second_frame_id"],
            }
        if name == "run_recipe":
            fields(arguments, ("recipe_id", "arguments"))
            if self.recipes is None:
                raise ValueError("No frozen tool recipes are loaded")
            sequence = self.recipes.build(
                arguments["recipe_id"], arguments["arguments"], self.describe()
            )
            self.emit(
                "recipe_built",
                step=self.observation.step,
                recipe_id=arguments["recipe_id"],
                library_sha256=self.recipes.sha256,
                composition=sequence,
            )
            return self._call("run_sequence", sequence)
        if name == "run_sequence":
            self.sequence = CheckedSequence(arguments)
            return {
                "status": "scheduled",
                "stages": len(self.sequence.stages),
                "official_success": None,
            }
        if name == "pursue_point":
            return self._start_pursuit(arguments)
        if name == "plan_camera_pointing":
            return self._plan_camera_pointing(arguments)
        if name == "segment_candidates":
            fields(arguments, ("camera", "frame_id"), ("box", "points"))
            camera, image = self._current_frame(arguments)
            if self.segmentation is None:
                raise ValueError("No frozen SAM proposal resource configured")
            if self.segmentation_calls >= 4:
                raise ValueError("At most four SAM proposal calls per observation")
            # Failed inference also consumes a bounded attempt; never loop
            # indefinitely on the same scene/model error.
            self.segmentation_calls += 1
            masks, report = self.segmentation.propose(
                image, box=arguments.get("box"), points=arguments.get("points"),
                provenance={"camera": camera, "frame_id": arguments["frame_id"],
                            "step": self.observation.step},
            )
            rows = []
            for index, (mask, metadata) in enumerate(zip(masks, report["candidate_metadata"])):
                identifier = uuid.uuid4().hex
                row = {
                    "segmentation_id": identifier, "proposal_index": index,
                    "camera": camera, "frame_id": arguments["frame_id"],
                    **metadata,
                }
                self.segmentations[identifier] = {**deepcopy(row), "mask": mask.copy()}
                rows.append(row)
            provenance = {k: v for k, v in report.items() if k != "candidate_metadata"}
            self.emit("segmentation_proposals", step=self.observation.step,
                      candidates=rows, provenance=provenance)
            return {
                "status": "proposals", "candidates": rows, "provenance": provenance,
                "selected": None, "official_success": None,
                "scope": "Explicitly inspect a candidate AND original RGB. Scores, components and edge support do not prove complete/unoccluded bodies. No metric target or action.",
            }
        if name == "inspect_segmentation":
            fields(arguments, ("camera", "frame_id", "segmentation_id"))
            camera, image = self._current_frame(arguments)
            row = self._segmentation(arguments)
            preview = segmentation_overlay(image, row["mask"])
            alias = "segmentation_" + camera
            self.previews[alias] = {
                "image": RGBImage(preview.tobytes(), preview.shape[1], preview.shape[0]),
                "source_camera": camera, "source_frame_id": arguments["frame_id"],
                "segmentation_id": row["segmentation_id"],
                "width": preview.shape[1], "height": preview.shape[0],
                "coordinate_rule": "Original camera coordinates; overlay is an unaccepted model proposal",
            }
            return {
                "status": "preview_only", "view": alias,
                "candidate": {k: deepcopy(v) for k, v in row.items() if k != "mask"},
                "official_success": None,
                "scope": "Inspect original RGB for identity, missing/occluded parts and inner/outer boundary confusion; the preview does not accept a complete body.",
            }
        if name in {"inspect_region", "outline_candidates"}:
            fields(arguments, ("camera", "frame_id", "roi"))
            camera, image = self._current_frame(arguments)
            box = crop_bounds(image, arguments["roi"])
            if name == "inspect_region":
                x0, y0, x1, y1 = box
                scale = min(3.0, 768 / max(x1 - x0, y1 - y0))
                crop = cv2.resize(image[y0:y1, x0:x1], None, fx=scale, fy=scale)
                alias = "inspect_" + camera
                self.previews[alias] = {
                    "image": RGBImage(crop.tobytes(), crop.shape[1], crop.shape[0]),
                    "source_camera": camera,
                    "source_frame_id": arguments["frame_id"],
                    "roi": list(box),
                    "width": crop.shape[1],
                    "height": crop.shape[0],
                    "coordinate_rule": "Use original camera coordinates for tools, not crop pixels",
                }
                return {
                    "view": alias,
                    **{k: v for k, v in self.previews[alias].items() if k != "image"},
                }
            if len(self.regions) >= 32:
                raise ValueError("At most 32 outline candidates per observation")
            rows = []
            for candidate in outline_candidates(image, box)[: 32 - len(self.regions)]:
                identifier = uuid.uuid4().hex
                row = {
                    "outline_id": identifier,
                    "camera": camera,
                    "frame_id": arguments["frame_id"],
                    **candidate,
                }
                self.regions[identifier] = row
                rows.append(row)
            return {
                "status": "measured" if rows else "unknown",
                "candidates": rows,
                "scope": "RGB edges only; ellipse centers are not 3D circle centers",
            }
        if name == "estimate_body_envelope":
            fields(arguments, ("first", "second"))
            rows, images = [], []
            for key in ("first", "second"):
                row = arguments[key]
                fields(row, ("camera", "frame_id"), ("outline", "segmentation_id", "complete_visible_body"))
                _, image = self._current_frame(row)
                if ("outline" in row) == ("segmentation_id" in row):
                    raise ValueError("Choose an explicit outline OR a current segmentation_id")
                if "segmentation_id" in row:
                    proposal = self._segmentation(row)
                    if row.get("complete_visible_body") is not True:
                        raise ValueError("Explicit current-original-RGB complete_visible_body review is required")
                    if not proposal["geometry_eligible"]:
                        raise ValueError("SAM proposal is clipped, fragmented or fails unchanged outline/RGB checks")
                    row = {**row, "outline": deepcopy(proposal["outline"]),
                           "mask_sha256": proposal["mask_sha256"],
                           "completeness_scope": "Caller assertion, not automatic occlusion/completeness verification"}
                elif "complete_visible_body" in row:
                    raise ValueError("complete_visible_body is only an explicit segmentation selection assertion")
                rows.append(row)
                images.append(image)
            if rows[0]["camera"] == rows[1]["camera"]:
                raise ValueError("Body outlines require two different current cameras")
            cameras = self.geometry.cameras(self.observation.robot_state)
            result = body_envelope(
                [cameras[row["camera"]] for row in rows],
                images,
                [row["outline"] for row in rows],
                self.geometry.table,
            )
            # Deliberately no target_id, tracker binding or accepted point.
            # This is conditional shape information, not a solid contact model.
            return {
                **result,
                "step": self.observation.step,
                "source_outlines": deepcopy(rows),
                "coordinate_system": "public world frame; metres",
                "action_scheduled": False,
            }
        if name == "fit_rim":
            fields(arguments, ("first", "second"))
            rows, images = [], []
            for key in ("first", "second"):
                row = arguments[key]
                fields(row, ("camera", "frame_id", "pixels"))
                _, image = self._current_frame(row)
                rows.append(row)
                images.append(image)
            if rows[0]["camera"] == rows[1]["camera"]:
                raise ValueError("Select current rim pixels in two different cameras")
            if len(self.measurements) >= 32:
                raise ValueError("At most 32 geometric measurements per observation")
            cameras = self.geometry.cameras(self.observation.robot_state)
            first, second = rows
            result = fit_rim(
                cameras[first["camera"]],
                first["pixels"],
                cameras[second["camera"]],
                second["pixels"],
                self.geometry.table,
            )
            result["edge_support"] = {
                row["camera"]: rim_edge_support(
                    image,
                    cameras[row["camera"]],
                    result["position"],
                    result["radius_m"],
                    result["normal"],
                )
                for row, image in zip(rows, images, strict=True)
            }
            identifier = "rim_" + uuid.uuid4().hex
            result.update(
                id=identifier,
                step=self.observation.step,
                selections=deepcopy(rows),
            )
            self.measurements[identifier] = result
            return {"target_id": identifier, **deepcopy(result)}
        if name == "fit_circle":
            fields(arguments, ("first", "second"))
            rows = [self.regions.get(arguments[key]) for key in ("first", "second")]
            if (
                any(row is None for row in rows)
                or rows[0]["camera"] == rows[1]["camera"]
            ):
                raise ValueError("Select two current outlines from different cameras")
            cameras = self.geometry.cameras(self.observation.robot_state)
            first, second = rows
            result = fit_circle(
                cameras[first["camera"]],
                first["points"],
                cameras[second["camera"]],
                second["points"],
                self.geometry.table,
            )
            result["edge_support"] = {
                row["camera"]: edge_support(
                    rgb(self.observation.images[row["camera"]]),
                    cameras[row["camera"]],
                    result["position"],
                    result["radius_m"],
                )
                for row in rows
            }
            identifier = "circle_" + uuid.uuid4().hex
            result.update(id=identifier, step=self.observation.step)
            self.measurements[identifier] = result
            return {"target_id": identifier, **deepcopy(result)}
        if name == "public_motion_failure_diagnostics":
            fields(arguments, ())
            return self.motion_diagnostics()
        if name == "mark_points":
            fields(arguments, ("points",))
            rows = arguments["points"]
            if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
                raise ValueError("Mark 1..16 points at a time")
            for row in rows:
                fields(row, ("id", "camera", "frame_id", "pixel"))
                if (
                    row["camera"] not in self.frames
                    or row["frame_id"] != self.frames[row["camera"]]["frame_id"]
                ):
                    raise ValueError("Stale or unknown RGB frame_id")
            result = []
            for row in rows:
                result.append(self.tracker.mark(row["id"], row["camera"], row["pixel"]))
                self.bindings.pop(row["id"], None)
            return {"points": result}
        if name == "estimate_depth":
            fields(arguments, ("camera",))
            camera = arguments["camera"]
            if self.depth is None:
                raise ValueError(
                    "Depth model is not configured; supply --code-depth-model-path"
                )
            if camera not in self.frames:
                raise ValueError("Unknown current camera")
            return self.depth.estimate(
                self.frames[camera]["frame_id"],
                camera,
                rgb(self.observation.images[camera]),
            )
        if name == "calibrate_depth":
            fields(arguments, ("camera", "table_points"))
            camera_name = arguments["camera"]
            names = arguments["table_points"]
            if self.depth is None or camera_name not in self.frames:
                raise ValueError(
                    "A configured depth model and current camera are required"
                )
            if (
                not isinstance(names, list)
                or not 3 <= len(names) <= 12
                or len(set(names)) != len(names)
            ):
                raise ValueError("Select 3..12 distinct observed table points")
            camera = self.geometry.cameras(self.observation.robot_state)[camera_name]
            anchors = []
            for name in names:
                row = self.tracker.current(name)
                if row["camera"] != camera_name:
                    raise ValueError(
                        "Depth calibration anchors must share the selected camera"
                    )
                world = camera.on_plane(row["pixel"], self.geometry.table)
                expected_depth = float((camera.view @ np.r_[world, 1])[2])
                anchors.append((row["pixel"], expected_depth))
            result = self.depth.calibrate(
                camera_name, self.frames[camera_name]["frame_id"], anchors
            )
            self.tracker.revision += 1
            return result
        if name == "locate_points":
            fields(arguments, ("points",))
            rows = arguments["points"]
            if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
                raise ValueError("Locate 1..16 current points")
            result = []
            for row in rows:
                method = row.get("method")
                required = ("id", "method") + (
                    ("height_m",)
                    if method == "plane"
                    else (
                        ("other_point",)
                        if method == "triangulate"
                        else ("base_points",) if method == "upright" else ()
                    )
                )
                fields(row, required)
                if method not in {"depth", "plane", "triangulate", "upright"}:
                    raise ValueError("Unknown localization method")
                if method == "plane" and (
                    type(row["height_m"]) not in (int, float)
                    or not 0 <= row["height_m"] <= 0.6
                ):
                    raise ValueError(
                        "Plane height must be 0..0.6 m above the public table"
                    )
                if method == "upright" and (
                    not isinstance(row["base_points"], list)
                    or len(row["base_points"]) != 2
                    or not all(isinstance(name, str) for name in row["base_points"])
                    or len(set(row["base_points"])) != 2
                    or row["id"] in row["base_points"]
                ):
                    raise ValueError(
                        "Upright localization needs two distinct base point IDs"
                    )
                self.tracker.current(row["id"])
                self.bindings[row["id"]] = deepcopy(row)
                result.append(self.locate(row["id"]))
            self.tracker.revision += 1
            return {"points": result}
        if name == "plan_pose":
            return self._plan_pose(arguments)
        if name == "plan_relative_pose":
            fields(
                arguments,
                ("arm", "translation", "rotation_vector", "pivot_local"),
                ("steps",),
            )
            side = arm(arguments["arm"])
            current = self.geometry.fk(
                side, self.observation.robot_state[f"{side}_arm_joint_state"]
            )
            pose = relative_pose(
                current,
                arguments["translation"],
                arguments["rotation_vector"],
                arguments["pivot_local"],
            )
            return self._save_pose(
                side,
                pose,
                {},
                {
                    "scope": "Relative to current measured TCP; local pivot is a hypothesis, not observed object geometry",
                },
                pivot_local=arguments["pivot_local"],
                minimum_steps=arguments.get("steps", 12),
            )
        if name == "capture_attachment":
            return self._capture_attachment(arguments)
        if name == "verify_attachment":
            return self._verify_attachment(arguments)
        if name == "execute_plan":
            fields(arguments, ("plan_id",))
            plan = self.plans.get(arguments["plan_id"])
            if (
                plan is None
                or plan["revision"] != self.tracker.revision
                or plan["step"] != self.observation.step
            ):
                raise ValueError(
                    "Stale or unknown plan; replan using current observations"
                )
            return self._start(plan["operation"], plan=plan)
        if name == "gripper":
            fields(arguments, ("arm", "value", "steps"))
            value = arguments["value"]
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError("Gripper target must be in 0..1")
            return self._start(
                {
                    "op": "grip",
                    "label": "gripper command, not grasp proof",
                    "arm": arm(arguments["arm"]),
                    "gripper": value,
                    "steps": steps(arguments["steps"], 8 if value < 0.8 else 6),
                    "minimum_close_steps": 8,
                }
            )
        if name == "home":
            fields(arguments, ("release", "steps"))
            if arguments["release"] is not True:
                raise ValueError(
                    "Home opens both grippers; explicit release=true is required"
                )
            return self._start(
                {
                    "op": "home",
                    "label": "release and home",
                    "steps": steps(arguments["steps"], 12),
                }
            )
        if name == "wait":
            fields(arguments, ("steps",))
            return self._start(
                {
                    "op": "wait",
                    "label": "wait for fresh visual evidence",
                    "steps": steps(arguments["steps"]),
                }
            )
        if name == "verify":
            fields(arguments, ("first", "second", "tolerance_m"))
            tolerance = arguments["tolerance_m"]
            if type(tolerance) not in (int, float) or not 0.001 <= tolerance <= 0.1:
                raise ValueError("Verification tolerance must be 0.001..0.1 m")
            try:
                first, second = self.locate(arguments["first"]), self.locate(
                    arguments["second"]
                )
            except ValueError as error:
                return {
                    "status": "unknown",
                    "reason": str(error),
                    "official_success": None,
                }
            distance = float(
                np.linalg.norm(np.asarray(first["position"]) - second["position"])
            )
            estimated = any(
                row["method"] != "rgb_triangulation" for row in (first, second)
            )
            return {
                "status": ("estimated_" if estimated else "")
                + ("pass" if distance <= tolerance else "fail"),
                "distance_m": distance,
                "evidence": [first, second],
                "official_success": None,
                "scope": "point distance only; not object identity, stable grasp or full task success",
            }
        if name == "request_capability":
            fields(arguments, ("name", "reason"))
            if any(
                not isinstance(v, str) or not v.strip() or len(v) > 2000
                for v in arguments.values()
            ):
                raise ValueError("Capability name and reason must be bounded text")
            from .prompt import TOOLS, SEGMENTATION_TOOLS

            if arguments["name"] == "pursue_point" and self.point_pursuit:
                return {"status": "available", "tool": arguments["name"]}

            if arguments["name"] == "plan_camera_pointing" and self.camera_pointing:
                return {"status": "available", "tool": arguments["name"]}

            if arguments["name"] in SEGMENTATION_TOOLS and self.segmentation is not None:
                return {"status": "available", "tool": arguments["name"]}

            preview_supported = True
            if arguments["name"] == "preview_grasp_pose":
                try:
                    validate_finger_resources(self.geometry)
                except (ValueError, OSError, KeyError, TypeError):
                    preview_supported = False
            if (
                arguments["name"] in TOOLS
                and preview_supported
                and (
                    arguments["name"] not in {"estimate_depth", "calibrate_depth"}
                    or self.depth is not None
                )
                and (
                    arguments["name"] != "run_recipe"
                    or self.recipes
                    and self.recipes.documents
                )
            ):
                return {"status": "available", "tool": arguments["name"]}
            self.emit("capability_requested", **arguments)
            return {"status": "requires_integration", **arguments}
        if name == "finish":
            fields(arguments, ())
            self.finished = True
            return {"status": "holding_until_native_end", "official_success": None}
        raise ValueError("Unknown tool")

    def action(self):
        if self.sequence is not None:
            return self.sequence.action(self)
        return self._motion_action()

    def _motion_action(self):
        if self.active is None:
            return None
        if self.pursuit is not None:
            return self._pursuit_action()
        operation, plan = self.active["operation"], self.active["plan"]
        if self.motion.failure or self.motion.index >= len(self.motion.plan):
            self.motion_end_step = self.observation.step
            result = {
                "status": "failed" if self.motion.failure else "completed",
                "error": self.motion.failure,
                "operation": operation,
                "official_success": None,
                "diagnostics": self.motion_diagnostics(),
            }
            if "guard_rejection" in self.active:
                result["guard_rejection"] = deepcopy(self.active["guard_rejection"])
            if plan is not None:
                side = operation["arm"]
                actual = self.geometry.fk(
                    side, self.observation.robot_state[f"{side}_arm_joint_state"]
                )
                result["verification"] = verify_pose(actual, plan["pose"])
            else:
                result["verification"] = {
                    "status": "unknown",
                    "scope": "Inspect fresh RGB; joint/gripper command completion is not object success",
                }
            self.last_result, self.active = result, None
            self.emit("action_verified", step=self.observation.step, **result)
            return None
        if self.motion_budget is not None and self.motion_budget <= 0:
            self.motion.failure = (
                "Native action budget or sequence recovery reserve exhausted"
            )
            return self._motion_action()
        if plan is not None:
            try:
                cameras = self.geometry.cameras(self.observation.robot_state)
                for name, evidence in plan["evidence"].items():
                    if evidence["method"] == "rgb_sphere_fit":
                        for camera in evidence["edge_support"]:
                            sphere_edge_support(
                                rgb(self.observation.images[camera]), cameras[camera],
                                evidence["position"], evidence["radius_m"],
                            )
                        continue
                    if evidence["method"] == "rgb_oriented_rim_fit":
                        for camera in evidence["edge_support"]:
                            rim_edge_support(
                                rgb(self.observation.images[camera]),
                                cameras[camera],
                                evidence["position"],
                                evidence["radius_m"],
                                evidence["normal"],
                            )
                        continue
                    if evidence["method"] == "rgb_circle_fit":
                        for camera in evidence["edge_support"]:
                            edge_support(
                                rgb(self.observation.images[camera]),
                                cameras[camera],
                                evidence["position"],
                                evidence["radius_m"],
                            )
                        continue
                    row = self.tracker.current(name)
                    expected = cameras[row["camera"]].project(evidence["position"])
                    if np.linalg.norm(expected - row["pixel"]) > 12:
                        raise ValueError(
                            "Target moved or metric estimate disagrees with current RGB"
                        )
            except ValueError as error:
                self.motion.failure = str(error)
                return self._motion_action()
        # A proposed action is not dispatched until all existing guards pass.
        # step() may already update targets or prepare completion events; keep
        # those transactional so a rejected home cannot silently open a hand.
        grippers_before = deepcopy(self.motion.gripper_targets)
        original_emit = self.motion.emit
        pending_events = []
        rejection = None
        try:
            self.motion.emit = lambda event, **fields: pending_events.append((event, fields))
            action = self.motion.step(self.observation)
            try:
                for side in ("left", "right"):
                    self.motion._guard(side, action[f"{side}_arm_joint_state"])
            except ValueError as error:
                rejection = str(error)
        finally:
            self.motion.emit = original_emit
        if rejection is not None:
            previous_failure = self.motion.failure
            self.motion.gripper_targets = grippers_before
            self.motion.arm_targets = {
                f"{side}_arm_joint_state": list(
                    self.observation.robot_state[f"{side}_arm_joint_state"]
                )
                for side in ("left", "right")
            }
            self.motion.hold = self.hold()
            self.motion.failure = (
                f"Public action guard rejected an unsent proposal: {rejection}"
            )
            self.active["guard_rejection"] = {
                "error": rejection,
                "proposed_action_dispatched": False,
                "gripper_commands_preserved": True,
                "previous_motion_failure": previous_failure,
            }
            self.emit(
                "motion_guard_rejected",
                step=self.observation.step,
                **self.active["guard_rejection"],
                proposed_action=action,
                uncommitted_motion_events=[
                    {"event": event, **fields} for event, fields in pending_events
                ],
            )
            # Return a failed tool result and fresh observation for bounded
            # replanning, not an automatic retry or unchecked recovery motion.
            return self._motion_action()
        for event, fields in pending_events:
            original_emit(event, **fields)
        return action
