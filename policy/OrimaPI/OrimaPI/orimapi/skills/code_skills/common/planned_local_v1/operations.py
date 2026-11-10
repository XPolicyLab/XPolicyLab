"""Complete local operations. GPT never supplies joint paths or metric XYZ.

Route distances below describe motions, NOT rejection margins. Residuals are
diagnostic. Failures come from measured-geometry contracts, original robot
guards, native planning/physics, or the unchanged task action budget.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np

from .geometry import docking_pose, frame, row_slots, unit, vector


class OperationFailure(ValueError):
    pass


def load_tracker():
    directory = Path(__file__).resolve().parents[2] / "match_and_pick_from_conveyor/primitives"
    spec = importlib.util.spec_from_file_location("_planned_local_rgb_track", directory/"rgb_geometry/tool.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Operations:
    def __init__(self, owner):
        self.owner = owner
        self.stages = []
        self.name = None

    @property
    def api(self):
        return self.owner.api

    def move(self, side, target, name, *, free=False):
        self.name = name
        result = {}
        start = self.owner.latest.step
        # This name selects the existing c008 full native cuRobo route.
        mode = "above pen shaft" if free else "contact"
        code = self.api.move_stage(self.api.arm(side), mode, np.asarray(target).copy(), result)
        row = dict(stage=name, start_step=start, end_step=self.owner.latest.step,
                   planner="native_curobo" if free else "cartesian_line", **result)
        self.stages.append(row)
        self.owner.emit("local_motion", **row)
        if code or result.get("plan_ok") is False:
            raise OperationFailure(f"{name}: {result}")

    def grip(self, side, value, name):
        self.name = name
        if not self.api.set_gripper(self.api.arm(side), value):
            raise OperationFailure(f"{name}: episode ended during gripper command")
        self.stages.append({"stage": name, "commanded_opening": value,
                            "step": self.owner.latest.step, "contact_verified": False})

    def object(self, identity, objects):
        if identity not in objects:
            raise ValueError(f"Missing fresh coarse identity: {identity}")
        return self.owner.measure.object(objects[identity])

    def feature(self, identity, name, objects):
        if identity not in objects:
            raise ValueError(f"Missing fresh feature identity: {identity}")
        return self.owner.measure.feature(objects[identity], name)

    def pick(self, args, objects):
        side, identity = args["arm"], args["source"]
        if side in self.owner.held:
            raise ValueError("This arm already has an unresolved object; use place/dock/handover")
        body = self.object(identity, objects)
        target = self.api.arm(side).tcp().copy()
        yaw = np.deg2rad(body["yaw_deg"]+90)
        opening = np.array([np.cos(yaw), np.sin(yaw), 0.])
        approach = unit(args.get("approach", [0, 0, -1]))
        # A side grasp may need a vertical finger-opening axis.
        if abs(np.dot(approach, opening)) > .95:
            opening = np.array([0., 0., 1.])
        opening = unit(opening-approach*np.dot(approach, opening))
        target[:3, :3] = np.column_stack((approach, opening, np.cross(approach, opening)))
        coarse = objects[identity]
        if "grasp" in coarse.get("features", {}):
            point = self.feature(identity, "grasp", objects)
        else:
            point = vector(body["center"])
            point[2] = body["base_z"] + body["height"]*args.get("grasp_fraction", .6)
        clearance = self.owner.profile["travel_lift_m"]
        pre = point-approach*clearance
        # Raise with current orientation BEFORE rotating or crossing objects.
        pose = self.api.arm(side).tcp().copy()
        pose[2, 3] = max(pose[2, 3], body["top_z"]+clearance, pre[2])
        self.move(side, pose, "raise before pickup", free=True)
        target[:3, 3] = [pre[0], pre[1], pose[2, 3]]
        self.move(side, target, "stage above measured grasp", free=True)
        self.grip(side, 1., "open before pickup")
        target[:3, 3] = pre
        self.move(side, target, "lower to pregrasp")
        target[:3, 3] = point
        self.move(side, target, "contact measured grasp")
        self.grip(side, 0., "close on source")
        # Register at ACTUAL closure pose, not the commanded target pose.
        tcp = self.api.arm(side).tcp().copy()
        center = np.r_[body["center"], 1.]
        self.owner.held[side] = {
            "identity": identity, "body": body,
            "center_in_tcp": (np.linalg.inv(tcp) @ center).tolist(),
            "grasp_fraction": args.get("grasp_fraction", .6),
            "retention": "unconfirmed", "evidence": "commanded close, not actual finger/contact sensing",
        }
        target = tcp.copy()
        target[2, 3] += self.owner.profile["pickup_lift_m"]
        self.move(side, target, "lift source after closure")
        return {"close_commanded": True, "held_candidate": deepcopy(self.owner.held[side]),
                "retention_verified": False}

    def yaw_left(self, side, coarse):
        if not {"front", "back"} <= coarse.get("features", {}).keys():
            raise ValueError("Front-left orientation requires fresh measured front/back features")
        front = self.owner.measure.feature(coarse, "front")
        back = self.owner.measure.feature(coarse, "back")
        direction = front-back
        if np.linalg.norm(direction[:2]) < 1e-9:
            raise ValueError("Visible front/back does not define horizontal polarity")
        angle = np.pi-np.arctan2(direction[1], direction[0])
        c, s = np.cos(angle), np.sin(angle)
        delta = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        pose = self.api.arm(side).tcp().copy()
        pose[:3, :3] = delta @ pose[:3, :3]
        self.move(side, pose, "align measured front toward world left", free=True)

    def place(self, args, objects, *, destination_point=None):
        side = args["arm"]
        held = self.owner.held.get(side)
        if held is None:
            raise ValueError("No held candidate to place")
        identity = held["identity"]
        if args.get("front_left"):
            self.yaw_left(side, objects[identity])
        if destination_point is None:
            explicit_support = "support" in objects[args["destination"]].get("features", {})
            if args.get("mode", "support") == "inside" or explicit_support:
                # Container bottom is NOT its rim or the public table.
                support = self.feature(args["destination"], "support", objects)
            else:
                destination = self.object(args["destination"], objects)
                support = np.array([*destination["center"][:2], destination["top_z"]])
            offset = args.get("slot", [0., 0.])
            if np.any(offset):
                destination = self.object(args["destination"], objects)
                yaw = np.deg2rad(destination["yaw_deg"])
                axes = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                # Semantic normalized slot in the currently measured container axes.
                support[:2] += axes @ (np.array(offset) * [
                    destination["length"]/2, destination["width"]/2])
        else:
            support = vector(destination_point)
        desired_center = support+np.array([0., 0., held["body"]["height"]/2])
        pose = self.api.arm(side).tcp().copy()
        # Rigid attachment is a named hypothesis, not verified retention.
        center = (pose @ np.asarray(held["center_in_tcp"]))[:3]
        goal = pose.copy()
        goal[:3, 3] += desired_center-center
        clearance = self.owner.profile["travel_lift_m"]
        travel = pose.copy()
        travel[2, 3] = max(pose[2, 3], goal[2, 3]+clearance)
        self.move(side, travel, "raise held object", free=True)
        travel[:2, 3] = goal[:2, 3]
        self.move(side, travel, "transport above measured destination", free=True)
        self.move(side, goal, "lower onto measured support")
        self.grip(side, 1., "release on support")
        self.owner.held.pop(side)
        self.owner.releases.append({
            "identity": identity, "destination": args.get("destination"),
            "step": self.owner.latest.step, "containment_verified": False,
        })
        goal = self.api.arm(side).tcp().copy()
        goal[2, 3] += clearance
        self.move(side, goal, "withdraw after placement")
        return {"released": True, "placement_verified": False, "attachment_assumption": True}

    def transfer(self, args, objects):
        # Destination/front geometry is measured AFTER pickup, on current RGB.
        # If grasp has moved it, GPT supplies a new current coarse identity on
        # the separate place operation instead of using transfer(front_left).
        if args.get("front_left"):
            raise ValueError("Use pick then current-RGB place(front_left) after object motion")
        self.pick(args, objects)
        return self.place(args, objects)

    def sort_row(self, args, objects):
        identities = self.owner.plan["targets"]
        bodies = {name: self.object(name, objects) for name in identities}
        start = self.feature(args["destination"], "row_start", objects)
        end = self.feature(args["destination"], "row_end", objects)
        # World left->right, never a possibly mirrored camera ordering.
        if start[0] > end[0]:
            start, end = end, start
        ranked, slots = row_slots(bodies, start, end)
        self.owner.ledger["measured_size_order"] = ranked
        self.pick(args, objects)
        return {**self.place(args, objects, destination_point=slots[args["source"]]),
                "measured_size_order": ranked}

    def dock(self, args, objects):
        """Measured 3-feature rigid alignment, axial entry, optional seat."""
        side, identity, destination = args["arm"], args["source"], args["destination"]
        if self.owner.held.get(side, {}).get("identity") != identity:
            raise ValueError("Dock the current held object, using current post-pickup feature pixels")
        source = [self.feature(identity, name, objects) for name in ("mate_a", "mate_b", "mate_c")]
        target = [self.feature(destination, name, objects) for name in ("mate_a", "mate_b", "mate_c")]
        seat = (self.feature(destination, "seat", objects)
                if "seat" in objects[destination].get("features", {}) else None)
        # Feature triangle direction defines axial insertion; no arbitrary yaw.
        axis = unit(np.cross(target[1]-target[0], target[2]-target[0]))
        sign = args.get("entry_sign", -1)
        axis *= sign
        goal = docking_pose(self.api.arm(side).tcp(), source, target)
        pre = goal.copy()
        pre[:3, 3] -= axis*self.owner.profile["travel_lift_m"]
        self.move(side, pre, "align material features outside entry", free=True)
        self.move(side, goal, "enter measured mating features")
        if seat is not None:
            goal[:3, 3] += seat-target[0]
            self.move(side, goal, "advance to visible seating feature")
        if args.get("release", True):
            self.grip(side, 1., "release after mating trajectory")
            self.owner.held.pop(side)
            self.owner.releases.append({"identity": identity, "destination": destination,
                                        "step": self.owner.latest.step, "seating_verified": False})
            goal = self.api.arm(side).tcp().copy()
            goal[:3, 3] -= axis*self.owner.profile["travel_lift_m"]
            self.move(side, goal, "axial withdrawal after mating release")
        return {"released": args.get("release", True), "seating_verified": False,
                "feature_frame_alignment": True, "object_slip_unobserved": True}

    def handover(self, args, objects):
        giver, receiver, identity = args["arm"], args["receiver"], args["source"]
        if giver == receiver or receiver in self.owner.held:
            raise ValueError("Handover needs a distinct empty receiving arm")
        if self.owner.held.get(giver, {}).get("identity") != identity:
            raise ValueError("Giver is not holding this source")
        point = self.feature(identity, "receiver_grasp", objects)
        current = self.api.arm(giver).tcp()
        approach = unit(self.api.arm(receiver).tcp()[:3, 3]-point)
        approach *= -1
        opening = current[:3, 1]
        opening = unit(opening-approach*np.dot(approach, opening))
        pose = self.api.arm(receiver).tcp().copy()
        pose[:3, :3] = np.column_stack((approach, opening, np.cross(approach, opening)))
        pose[:3, 3] = point-approach*self.owner.profile["travel_lift_m"]
        self.move(receiver, pose, "stage receiver for handover", free=True)
        self.grip(receiver, 1., "open receiver")
        pose[:3, 3] = point
        self.move(receiver, pose, "receiver contacts tool handle")
        self.grip(receiver, 0., "close receiver before giver release")
        held = deepcopy(self.owner.held[giver])
        center = current @ np.array(held["center_in_tcp"])
        held["center_in_tcp"] = (np.linalg.inv(self.api.arm(receiver).tcp()) @ center).tolist()
        held.update(retention="unconfirmed", evidence="dual closure hypothesis; RGB review required")
        self.owner.held[receiver] = held
        self.grip(giver, 1., "release giver after receiver closure")
        self.owner.held.pop(giver)
        retreat = self.api.arm(giver).tcp().copy()
        retreat[:3, 3] -= current[:3, 0]*self.owner.profile["travel_lift_m"]
        self.move(giver, retreat, "withdraw giving hand after handover")
        return {"handed_over_commanded": True, "retention_verified": False}

    def stroke(self, args, objects):
        side, destination = args["arm"], args["destination"]
        contact = self.feature(args["source"], "contact", objects)
        end = self.feature(destination, "end", objects)
        start = self.feature(destination, "start", objects)
        pose = self.api.arm(side).tcp().copy()
        offset = pose[:3, 3]-contact
        if args.get("tool_held", True) and side not in self.owner.held:
            raise ValueError("This stroke requires a held tool")
        pre = pose.copy()
        pre[:3, 3] = start+offset+[0., 0., self.owner.profile["travel_lift_m"]]
        self.move(side, pre, "stage above stroke start", free=True)
        pre[:3, 3] = start+offset
        self.move(side, pre, "lower tool onto measured stroke start")
        pre[:3, 3] = end+offset
        self.move(side, pre, "execute measured contact stroke")
        pre = self.api.arm(side).tcp().copy()
        pre[2, 3] += self.owner.profile["travel_lift_m"]
        self.move(side, pre, "withdraw after stroke")
        return {"stroke_executed": True, "object_entry_verified": False}

    def push(self, args, objects):
        # Empty-hand keyboard push: move nominal TCP to locally measured
        # contact and end features; no hidden keyboard dimensions.
        side, identity = args["arm"], args["source"]
        if side in self.owner.held:
            raise ValueError("Empty-hand push cannot discard a held object")
        start = self.feature(identity, "start", objects)
        end = self.feature(args["destination"], "end", objects)
        pose = self.api.arm(side).tcp().copy()
        pose[:3, 3] = start+[0., 0., self.owner.profile["travel_lift_m"]]
        self.move(side, pose, "stage above push", free=True)
        self.grip(side, 0., "close empty hand for push")
        pose[:3, 3] = start
        self.move(side, pose, "contact keyboard edge")
        pose[:3, 3] = end
        self.move(side, pose, "push toward measured frame")
        pose = self.api.arm(side).tcp().copy()
        pose[2, 3] += self.owner.profile["travel_lift_m"]
        self.move(side, pose, "withdraw after push")
        return {"push_executed": True, "alignment_verified": False}

    def intercept(self, args, objects):
        side, identity = args["arm"], args["source"]
        if side in self.owner.held:
            raise ValueError("Interception needs an available arm")
        body = self.object(identity, objects)
        support = self.feature(identity, "support", objects)
        if body["top_z"] <= support[2]:
            raise ValueError("Measured belt support must be below the measured object top")
        body.update(base_z=float(support[2]), height=float(body["top_z"]-support[2]))
        body["center"][2] = (body["top_z"]+support[2])/2
        # Actual local 3-frame tracker, not a GPT velocity guess.
        module = load_tracker()
        box = objects[identity]["box"]
        tracked, code = module.run(self.api, "rgb_track", {
            **dict(zip(("u0", "v0", "u1", "v1"), box)),
            "surface_z": body["top_z"], "support_z": float(support[2]),
            "interval": .4, "center_mode": "extent",
        })
        if code or tracked.get("plan_ok") is False:
            raise OperationFailure(f"Local conveyor tracking: {tracked}")
        point = tracked["center_xyz"]
        velocity = tracked["velocity_xy"]
        # Nothing moves between this timestamped measurement and timed_pick.
        held = {
            "identity": identity, "body": body,
            # At closure, estimate the object-center offset from its measured
            # half-height and the tracker's grasp height. The arm's orientation
            # is taken AFTER the interception, not from its preposition.
            "center_in_tcp": None,
            "retention": "unconfirmed", "evidence": "timed closing/lift; inspect fresh RGB",
        }
        def retain_hypothesis():
            held["center_in_tcp"] = [
                *(self.api.arm(side).tcp()[:3, :3].T @
                  np.array([0., 0., (body["top_z"]+support[2])/2-point[2]])).tolist(), 1.,
            ]
            self.owner.held[side] = held
        try:
            result = self.owner.tool("timed_pick", {
                "arm": side, **dict(zip(("x", "y", "z"), point)),
                "vx": velocity[0], "vy": velocity[1],
                "top_z": body["top_z"], "lift": self.owner.profile["pickup_lift_m"],
            })
        except Exception as error:
            # A failed lift AFTER timed closure must not erase the held
            # hypothesis, otherwise the next interception may open midair.
            if any(s.get("stage") == "close" for s in getattr(error, "result", {}).get("stages", [])):
                retain_hypothesis()
            raise
        retain_hypothesis()
        return {"tracking": tracked, "interception": result, "retention_verified": False}

    def inspect(self, args, objects):
        if args.get("object"):
            return self.object(args["object"], objects)
        return self.owner.measure.segment()

    def measure_feature(self, args, objects):
        return {"point_world": self.feature(args["object"], args["feature"], objects).tolist()}

    def view(self, args, objects):
        # Viewpoint derives from the target's public-table ray. This is NOT
        # passed off as a metric grasp point.
        coarse = objects[args["object"]]
        camera = self.api.robot.cameras(self.owner.latest.robot_state)["cam_high"]
        center = camera.on_plane(coarse["point"], self.api.robot.table)
        side = args["arm"]
        pose = self.api.arm(side).tcp().copy()
        # Existing scene tool documents where a downward wrist sees the table.
        pose[:3, :3] = self.api.operations.core.tool_rotation("down", "x", pose[:3, :3])
        pose[:3, 3] = center+[0., .12, .25]
        self.move(side, pose, "obtain calibrated measurement viewpoint", free=True)
        return {"view_changed": True, "refresh_coarse_pixels": True}

    def home(self, args, objects):
        if self.owner.held:
            raise ValueError("Resolve held objects through supported placement before task-final home")
        sides = ("left", "right") if args["arm"] == "both" else (args["arm"],)
        paths = {s: self.api.motion.time_path(np.stack([
            self.api.arm(s).joints(), self.api.arm(s).home_joints])) for s in sides}
        if not self.api.run(paths):
            raise OperationFailure("Native episode ended during home")
        return {"home_commanded": True, "official_success": None}

    def wait(self, args, objects):
        return {"completed": self.api.hold(round(args["seconds"]*25))}

    def run(self, operation, args, objects):
        self.stages, self.name = [], operation
        return getattr(self, operation)(args, objects)
