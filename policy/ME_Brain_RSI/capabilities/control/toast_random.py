"""Make toast (random variant): demo-style flat handoff between the arms.

Left arm (donor) pinches the top edge of the end slice on the rack, lifts it,
turns it flat and offers it towards the right. The right arm (receiver) takes
the opposite edge with its pads above/below the slice, the donor lets go, the
receiver turns the slice vertical, inserts it into a toaster slot and releases.
After two slices the receiver presses the lever. Scene geometry comes from the
offline oracle bridge (``ctx.oracle_objects``) or RGB perception.
"""
import math
import os

import numpy as np

from ..dual import (CartesianPathError, cartesian_path, down_rotation, grip_steps, rz, wrap, matrix_to_quat,
                    smooth_joint_path)
from ..executor import ZoneExecutor, _log, SAFE_TRAVEL_CLEARANCE
from ..motion import quat_matrix

BREAD_HALF = 0.0585        # public bread slice half width/height
BREAD_THICK = 0.0118
TIP_BELOW_TCP = 0.0126
PICK_EDGE_DEPTH = 0.020        # donor pads 20 mm down the slice: deeper pinch resists the gravity torque when turned flat
HANDOFF_Z = 1.00
HANDOFF_Y = -0.20
HANDOFF_X = -0.045
RECEIVER_OPEN = 0.65   # 6.5 cm aperture: the 8.7 cm one hung the lower finger 2 cm over the toaster roof and stalled the approach (L14/L26 read 0.0 twice)
DONOR_MAX_READING = 0.12       # one crust edge reads ~0.07-0.10; more means two slices or a cocked one
RECEIVER_MAX_READING = 0.125   # pads 1.9 cm apart: anything wider is a cocked slice caught diagonally
RECEIVER_DZ = (0.0, -0.015, 0.015)   # the cantilevered slice sags: try lower first            # receiver aperture at the handoff (+-4 cm around the slice plane)
# Apertures of 0.22/0.30 get blocked on the slice tops (the finger collision shape is wider
# than the pad); 0.45 pinches reliably but nudges the neighbour, so the second slice is
# taken from the other, undisturbed end of the row.
DONOR_OPEN = 0.45
PICK_FREE_SHIFT = 0.0
# The pinched slice hangs 2*BREAD_HALF - PICK_EDGE_DEPTH below the TCP; the remaining row's
# tops are PICK_EDGE_DEPTH above the grasp TCP. Carrying across the row needs this lift.
CARRY_CLEARANCE_LIFT = 2 * BREAD_HALF + 0.018
CARRY_RETREATS = ((0.0, 0.03), (0.0, 0.06), (0.0, 0.09), (0.03, 0.06), (0.03, 0.09), (0.06, 0.06), (0.06, 0.09))   # (away from the row, towards the robot)
RECEIVER_EDGE_DEPTH = 0.0585   # pads at the slice centre: the domed slice is thickest there, so
                               # hanging it does not wedge it out of the pads (edge grips slid out)

# Public toaster geometry (toaster frame; root 80 mm above the table) from the asset tables.
TOASTER_ASSETS = {
    0: dict(footprint=(-.105056, .102198, -.078268, .077172), roof=.080233,
            slots=[(-.079149, .076291, -.039408, -.013502), (-.079149, .076291, .012405, .038312)],
            floor=-.049301,
            lever=dict(kind='knob', x0=.102198, x1=.115152, y0=-.013502, y1=.012405,
                       top_at_rest=.055173, travel=.078, tcp_x=.1205,
                       pitch=math.radians(25), tip_depth=.009971)),
    1: dict(footprint=(-.110779, .115714, -.070570, .069101), roof=.075138,
            slots=[(-.057931, .062866, -.047920, -.014324), (-.057931, .062866, .012855, .046452)],
            floor=-.019234,
            lever=dict(kind='knob', x0=.115735, x1=.138363, y0=-.013569, y1=.012100,
                       top_at_rest=.035208, travel=.075, tcp_x=.1370,
                       pitch=math.pi / 4, tip_depth=.012523)),
    2: dict(footprint=(-0.127079, 0.119925, -0.082755, 0.078748), roof=0.081116,
            slots=[(-0.074713, 0.058699, -0.043804, -0.009604), (-0.074713, 0.058699, 0.005596, 0.039797)],
            floor=-0.061386,
            lever=dict(kind="knob", x0=0.115175, x1=0.142982, y0=-0.016254, y1=0.012247,
                       top_at_rest=0.0293, travel=0.085, tcp_x=0.139)),   # full range 0.0864: 0.078 left the ratio at 0.83-0.86 (>=0.85 required)
    4: dict(footprint=(-0.120400, 0.106340, -0.062489, 0.062445), roof=0.078564,
            slots=[(-0.068653, 0.054593, -0.026707, -0.007348), (-0.068653, 0.054593, 0.007304, 0.026663)],
            floor=-0.030893,
            lever=dict(kind="plate", x0=0.051637, x1=0.126308, y0=-0.034361, y1=0.034316,
                       top_at_rest=0.036785, travel=0.045, tcp_x=0.122)),   # full range 0.0432
}


def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
def scene_from_oracle(ctx):
    breads = []
    toaster = None
    for instance, obj in ctx.oracle_objects.items():
        label = str(obj.get("label"))
        if "position" not in obj:
            continue
        position = np.asarray(obj["position"], float)
        rotation = quat_matrix(obj["quaternion_wxyz"])
        if label.startswith("bread_"):
            # Thin axis is the public model Z; the in-plane axis closest to vertical points up.
            normal = rotation[:, 2]
            candidates = [rotation[:, 0], -rotation[:, 0], rotation[:, 1], -rotation[:, 1]]
            up = max(candidates, key=lambda a: a[2])
            breads.append(dict(name=instance, label=label, centre=position, normal=normal, up=up))
        elif label == "toaster":
            variant = int(instance.split("_")[1])
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            joints = obj.get("joint_positions") or []
            toaster = dict(root=position, rotation=rotation, yaw=yaw, variant=variant, joints=joints)
    return dict(breads=breads, toaster=toaster)


def perceive(ctx):
    cache = getattr(ctx, "_toast_scene_cache", None)
    if cache is not None and cache[0] == ctx.steps:
        return cache[1]
    if getattr(ctx, "oracle_objects", None) and os.environ.get("COAP_ORACLE_MODE", "").startswith("absolute"):
        scene = scene_from_oracle(ctx)
    else:
        from .toast_vision import scene_from_rgb
        scene = scene_from_rgb(ctx)
    ctx._toast_scene_cache = (ctx.steps, scene)
    return scene


def toaster_slots(toaster):
    """World slot centres (two), slot long axis (world unit), roof and floor heights."""
    asset = TOASTER_ASSETS[toaster["variant"]]
    root, rotation = toaster["root"], toaster["rotation"]
    slots = []
    for x0, x1, y0, y1 in asset["slots"]:
        local = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1), asset["roof"]])
        slots.append(root + rotation @ local)
    long_axis = unit(rotation[:, 0][:2])
    across = unit(rotation[:, 1][:2])
    return dict(centres=slots, long_axis=long_axis, across=across, roof=root[2] + asset["roof"],
                floor=root[2] + asset["floor"], half_length=0.5 * (asset["slots"][0][1] - asset["slots"][0][0]),
                half_width=0.5 * (asset["slots"][0][3] - asset["slots"][0][2]))


def lever_target(toaster):
    """World point on top of the lever at rest, and the press depth."""
    asset = TOASTER_ASSETS[toaster["variant"]]
    lever = asset["lever"]
    root, rotation = toaster["root"], toaster["rotation"]
    local = np.array([0.5 * (lever["x0"] + lever["x1"]), 0.5 * (lever["y0"] + lever["y1"]), lever["top_at_rest"]])
    return root + rotation @ local, lever["travel"], lever["kind"]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
class ToastExecutor(ZoneExecutor):
    def __init__(self, ctx, scene):
        toaster = scene["toaster"]
        asset = TOASTER_ASSETS[toaster["variant"]]
        fp = asset["footprint"]
        centre_local = np.array([0.5 * (fp[0] + fp[1]), 0.5 * (fp[2] + fp[3]), 0.0])
        centre = toaster["root"] + toaster["rotation"] @ centre_local
        half = (0.5 * (fp[1] - fp[0]), 0.5 * (fp[3] - fp[2]))
        roof = toaster["root"][2] + asset["roof"]
        super().__init__(ctx, centre, toaster["rotation"][:2, 0], half, roof + 0.125)
        self.scene = scene
        self.toaster = toaster
        self.slots = toaster_slots(toaster)
        self.roof = roof
        self.deadline = 1400 - 80
        self.donor = "left"
        self.receiver = "right"

    # ---- geometry ---------------------------------------------------------------
    def in_toaster(self, point):
        """Inside the toaster footprint (plus margin): slices already inserted are not shelf slices."""
        return self.box_margin_distance(np.asarray(point)[:2]) < 0.03

    def end_slice(self, breads, slot_index=0):
        """An end slice of the rack row: the +X end first, then the other (undisturbed) end."""
        return (max if slot_index % 2 == 0 else min)(breads, key=lambda b: b["centre"][0])

    def pick_geometry(self, bread, breads=None):
        top = bread["centre"] + BREAD_HALF * bread["up"]
        n_h = bread["normal"].copy()
        n_h[2] = 0.0
        n_h = unit(n_h)
        if n_h[0] < 0:
            n_h = -n_h
        gripper_yaw = math.atan2(n_h[1], n_h[0]) - math.pi / 2
        tcp = top.copy()
        tcp[2] = top[2] - PICK_EDGE_DEPTH
        out = n_h.copy()
        if breads:
            # Fingers open 40 mm around a 25 mm pitch: bias the pinch away from the row so the
            # inner finger comes down in the gap beside the neighbour instead of on its crust.
            row_centre = np.mean([b["centre"] for b in breads], axis=0)
            side = float(np.sign((bread["centre"] - row_centre) @ n_h)) or 1.0
            out = side * n_h
            tcp = tcp + PICK_FREE_SHIFT * out
        return dict(top=top, n_h=n_h, yaw=gripper_yaw, tcp=tcp, out=out)

    @staticmethod
    def frame(tool_x, tool_y):
        tool_x = unit(tool_x)
        tool_y = unit(np.asarray(tool_y, float) - np.dot(tool_y, tool_x) * tool_x)
        tool_z = np.cross(tool_x, tool_y)
        return np.column_stack([tool_x, tool_y, tool_z])

    def pick_frame(self, arm, position, candidates, seed):
        """Among equivalent frames pick the reachable one with the smallest wrist roll."""
        best = None
        for rotation in candidates:
            q = self.models[arm].solve(position, rotation, seed, prefer_small_roll=True)
            if q is not None and (best is None or abs(q[5]) < best[0]):
                best = (abs(q[5]), rotation)
        return candidates[0] if best is None else best[1]

    def handoff_frames(self, n_h, pick_rotation=None):
        """Donor offers the slice flat along n_h; the receiver takes the far edge region.

        Along n_h the slice spans [tcp_d - PICK_EDGE_DEPTH, tcp_d - PICK_EDGE_DEPTH + 2*BREAD_HALF].
        The receiver tool points -n_h with its finger tips RECEIVER_EDGE_DEPTH inside the far edge.
        Closing axis up or down are equivalent pinches; the reachable small-roll one is used.

        A pitched pick holds the upright slice rotated by the pitch about the closing axis,
        so the offer frame is rotated back by the same angle: the slice, not the tool x
        axis, must end up along n_h (the receiver closed on empty air 4 cm beside a slice
        offered from a 0.5 rad pick).
        """
        donor_tcp = np.array([HANDOFF_X, HANDOFF_Y, HANDOFF_Z])
        up = np.array([0.0, 0.0, 1.0])
        slice_in_tool = np.array([1.0, 0.0, 0.0])
        if pick_rotation is not None:
            slice_in_tool = -np.asarray(pick_rotation, float).T @ up      # the hanging slice points down
            slice_in_tool[1] = 0.0
            slice_in_tool = unit(slice_in_tool)
        closing = np.array([0.0, 1.0, 0.0])
        tool_basis = np.column_stack([slice_in_tool, closing, np.cross(slice_in_tool, closing)])
        compensated = [self.frame(n_h, up) @ tool_basis.T, self.frame(n_h, -up) @ tool_basis.T]
        plain = [self.frame(n_h, up), self.frame(n_h, -up)]
        donor_rot = None
        for group in (compensated, plain):
            reachable = [rotation for rotation in group
                         if self.models[self.donor].solve(donor_tcp, rotation, self.ctx.joint_targets[self.donor], prefer_small_roll=True) is not None]
            if reachable:
                donor_rot = self.pick_frame(self.donor, donor_tcp, reachable, self.ctx.joint_targets[self.donor])
                if group is plain and pick_rotation is not None:
                    _log(self.ctx, "offer_uncompensated", slice_in_tool=slice_in_tool.round(3).tolist())
                break
        if donor_rot is None:
            donor_rot = compensated[0]
        far_edge = donor_tcp + n_h * (2 * BREAD_HALF - PICK_EDGE_DEPTH)
        receiver_tcp = far_edge - n_h * RECEIVER_EDGE_DEPTH + n_h * TIP_BELOW_TCP
        receiver_rot = self.pick_frame(self.receiver, receiver_tcp, [self.frame(-n_h, up), self.frame(-n_h, -up)],
                                       self.ctx.joint_targets[self.receiver])
        return donor_tcp, donor_rot, receiver_tcp, receiver_rot

    # ---- donor ------------------------------------------------------------------
    def pick_rotation(self, arm, g, lift_height=0.08):
        """Tool-down pinch across the slice; pitched about the closing axis (pads stay
        parallel to the slice faces) when the upright pose is out of reach.

        Both antipodal yaws pinch alike; the one whose lift configuration can also travel
        to the handoff point on one IK branch (no wrist wrap) with the smaller roll wins.
        Shelves right in front of the base put the upright lift near the wrist limit, where
        the carry used to fail at its first step.
        """
        model = self.models[arm]
        home = self.home[arm]
        hover = g["tcp"] + np.array([0.0, 0.0, 0.04])
        lift = g["tcp"] + np.array([0.0, 0.0, lift_height])
        handoff = np.array([HANDOFF_X, HANDOFF_Y, HANDOFF_Z])
        carry_z = max(lift[2], self.table + 0.09 + 2 * BREAD_HALF)
        best = None
        pitches = (0.0, 0.3, -0.3, 0.5, -0.5, 0.65, -0.65)
        balance_signs = bool(self.ctx.memory.get("toast_slices_taken"))
        for index, pitch in enumerate(pitches):
            for yaw in (g["yaw"], g["yaw"] + math.pi):
                rotation = down_rotation(yaw, pitch)
                q = model.solve(g["tcp"], rotation, home)
                if q is None or model.solve(hover, rotation, q) is None:
                    continue
                q_lift = model.solve(lift, rotation, q)
                if q_lift is None:
                    continue
                chain_ok = True
                previous = q_lift
                for point in (np.r_[0.5 * (lift[:2] + handoff[:2]), carry_z], np.r_[handoff[:2], carry_z], handoff):
                    q_next = model.ik(point, rotation, previous)
                    if q_next is None or not model.valid(q_next) or abs(q_next[5]) > 3.0 or abs(q_next[5] - previous[5]) > math.pi:
                        chain_ok = False
                        break
                    previous = q_next
                cost = abs(pitch) * 2.0 + abs(q_lift[5]) * 0.5 + (0.0 if chain_ok else 10.0)
                if best is None or cost < best[0]:
                    best = (cost, rotation, pitch, yaw, chain_ok, float(q_lift[5]))
            tier_complete = not balance_signs or index == len(pitches) - 1 or abs(pitches[index + 1]) != abs(pitch)
            if tier_complete and best is not None and best[4] and (
                    abs(best[2]) == abs(pitch) if balance_signs else best[2] == pitch):
                break
        if best is None:
            return None
        _log(self.ctx, "donor_pick_pitch", pitch=best[2], antipodal=bool(abs(wrap(best[3] - g["yaw"])) > 1.0),
             carry_ok=best[4], lift_roll=round(best[5], 2), balanced_signs=balance_signs)
        return best[1]

    def carry_crossing(self, g, breads, turn_point):
        """Shelf slices the hanging slice would pass over on the straight carry to ``turn_point``."""
        n_h = g["n_h"]
        travel = float((turn_point - g["tcp"]) @ n_h)
        low, high = min(0.0, travel) - BREAD_THICK, max(0.0, travel) + BREAD_THICK
        offsets = [float((b["centre"] - g["tcp"]) @ n_h) for b in breads]
        return [b for b, offset in zip(breads, offsets) if abs(offset) > BREAD_THICK and low < offset < high]

    def clear_row_for_carry(self, arm, g, rotation, crossed):
        """Raise the hanging slice above the remaining row before carrying across it.

        The 8 cm lift leaves the slice bottom ~37 mm below the neighbours' tops; dragged across
        the row it pushed the remaining slices 12-16 mm towards the free end, so the second pick
        later landed on a slice top or pinched two (every qv060 second-pick loss). The far shelf
        end is near the reach limit, where the straight lift stops short: then retreat along the
        row towards the robot (and away from the row), which keeps the pinched slice in its own
        plane beside the neighbours, and lift again. Returns the achieved TCP height.
        """
        ctx = self.ctx
        model = self.models[arm]
        clearance = g["tcp"][2] + CARRY_CLEARANCE_LIFT
        start = model.tcp_matrix(ctx.joint_targets[arm])[:3, 3]
        yield from self.linear_move(arm, np.r_[start[:2], clearance], rotation, step_m=0.015, allow_partial=True)
        achieved = model.tcp_matrix(ctx.joint_targets[arm])[:3, 3]
        retreat = None
        if achieved[2] < clearance - 0.004:
            side = unit(np.cross(np.array([0.0, 0.0, 1.0]), g["n_h"]))
            home_tcp = model.tcp_matrix(self.home[arm])[:3, 3]
            toward = side if float((home_tcp - g["tcp"]) @ side) > 0.0 else -side
            best = None
            for d_out, d_toward in CARRY_RETREATS:
                shifted = achieved + d_out * g["out"] + d_toward * toward
                if self.box_margin_distance(shifted[:2]) < 0.07:
                    continue
                try:
                    move = cartesian_path(model, ctx.joint_targets[arm], shifted, rotation, step_m=0.02, max_joint_step=0.09)
                except CartesianPathError:
                    continue
                try:
                    rise = cartesian_path(model, move[-1], np.r_[shifted[:2], clearance], rotation, step_m=0.015, max_joint_step=0.09)
                except CartesianPathError as error:
                    rise = list(getattr(error, "partial", []))
                if not rise:
                    continue
                height = float(model.tcp_matrix(rise[-1])[2, 3])
                if best is None or height > best[0] + 0.004:
                    best = (height, shifted, d_out, d_toward)
                if height >= clearance - 0.004:
                    break
            if best is not None and best[0] > achieved[2] + 0.01:
                height, shifted, d_out, d_toward = best
                yield from self.linear_move(arm, shifted, rotation, step_m=0.02, allow_partial=True)
                yield from self.linear_move(arm, np.r_[shifted[:2], clearance], rotation, step_m=0.015, allow_partial=True)
                achieved = model.tcp_matrix(ctx.joint_targets[arm])[:3, 3]
                retreat = dict(out=d_out, toward=d_toward)
        _log(ctx, "donor_carry_clearance", crossed=[b["label"] for b in crossed], lift=round(float(achieved[2] - g["tcp"][2]), 3),
             target=round(CARRY_CLEARANCE_LIFT, 3), retreat=retreat)
        return float(achieved[2])

    def donor_pick_and_offer(self, bread, breads=None):
        ctx = self.ctx
        arm = self.donor
        g = self.pick_geometry(bread, breads)
        rotation = self.pick_rotation(arm, g)
        if rotation is None:
            _log(ctx, "donor_unreachable", tcp=g["tcp"].round(3).tolist())
            return False
        hover = g["tcp"].copy()
        hover[2] += 0.04
        self.donor_pick = dict(geometry=g, rotation=rotation, hover=hover)
        ctx.memory["phase"] = "donor_pick"
        # Shelf pitch is 25 mm: a 26 mm aperture keeps the finger pads inside the gaps
        # beside the slice (a wider one pushed the neighbour over, then two came up).
        yield from grip_steps(ctx, arm, DONOR_OPEN, 2)
        yield from self.free_move(arm, hover, rotation, min_tcp_z=self.table + 0.11, holding=False, allow_flip=False)
        yield from self.linear_move(arm, g["tcp"], rotation, step_m=0.015)
        yield from self.settle(arm, 1)
        yield from grip_steps(ctx, arm, 0.0, 6)
        yield from self.settle(arm, 4)
        reading = float(ctx.state[f"{arm}_ee_joint_state"][0])
        held = reading > 0.035
        _log(ctx, "donor_grasp", reading=reading, held=held, bread=bread["label"])
        if held and reading > DONOR_MAX_READING:
            # Too thick for one crust edge: the inner pad caught the neighbour as well. Let go
            # and pinch again with the pads shifted away from the row.
            yield from grip_steps(ctx, arm, DONOR_OPEN, 4)
            yield from self.linear_move(arm, hover, rotation, step_m=0.02, allow_partial=True)
            shifted = g["tcp"] + 0.008 * g["out"]
            yield from self.linear_move(arm, hover + 0.008 * g["out"], rotation, step_m=0.02, allow_partial=True)
            yield from self.linear_move(arm, shifted, rotation, step_m=0.015, allow_partial=True)
            yield from self.settle(arm, 1)
            yield from grip_steps(ctx, arm, 0.0, 6)
            yield from self.settle(arm, 4)
            reading = float(ctx.state[f"{arm}_ee_joint_state"][0])
            held = reading > 0.035
            _log(ctx, "donor_regrasp", reading=reading, held=held, bread=bread["label"])
        if not held:
            yield from grip_steps(ctx, arm, DONOR_OPEN, 3)
            yield from self.linear_move(arm, hover, rotation, step_m=0.03, allow_partial=True)
            return False
        lift = g["tcp"].copy()
        lift[2] += 0.08
        ctx.memory["phase"] = "donor_lift"
        fraction = yield from self.linear_move(arm, lift, rotation, step_m=0.015, allow_partial=True)
        if fraction < 0.6:
            _log(ctx, "donor_lift_short", fraction=fraction)
        reading = float(ctx.state[f"{arm}_ee_joint_state"][0])
        if reading < 0.035:
            _log(ctx, "donor_lost_on_lift", reading=reading)
            yield from grip_steps(ctx, arm, DONOR_OPEN, 3)
            return False
        donor_tcp, donor_rot, _, _ = self.handoff_frames(g["n_h"], rotation)
        slice_in_tool = -np.asarray(rotation, float).T @ np.array([0.0, 0.0, 1.0])
        slice_in_tool[1] = 0.0
        self.offer_slice_in_tool = unit(slice_in_tool)
        ctx.memory["phase"] = "donor_offer"
        # Carry (slice hanging) to a turning point short of the handoff, turn flat there, then
        # slide to the handoff point. A turn whose joint blend wanders wandered towards the
        # toaster (6 cm away at most) and knocked the hanging slice out (L0/L35).
        turn_point = donor_tcp + np.array([-0.05, -0.05, 0.0])
        model = self.models[arm]
        if (model.solve(turn_point, rotation, ctx.joint_targets[arm]) is None
                or model.solve(turn_point, donor_rot, ctx.joint_targets[arm], prefer_small_roll=True) is None):
            turn_point = donor_tcp
        carry_floor = self.table + 0.09 + 2 * BREAD_HALF
        crossed = self.carry_crossing(g, breads or [], turn_point)
        if crossed:
            carry_z = yield from self.clear_row_for_carry(arm, g, rotation, crossed)
            carry_floor = max(carry_floor, carry_z - 0.005)
            raised = np.r_[turn_point[:2], max(turn_point[2], carry_z)]
            if (model.solve(raised, rotation, ctx.joint_targets[arm]) is not None
                    and model.solve(raised, donor_rot, ctx.joint_targets[arm], prefer_small_roll=True) is not None):
                turn_point = raised
        yield from self.free_move(arm, turn_point, rotation, min_tcp_z=carry_floor,
                                  max_step=0.06, step_m=0.02, holding=True)
        reading = float(ctx.state[f"{arm}_ee_joint_state"][0])
        if reading < 0.035:
            _log(ctx, "donor_lost_on_carry", reading=reading)
            yield from grip_steps(ctx, arm, DONOR_OPEN, 3)
            return False
        yield from self.rotate_in_place(arm, donor_rot, max_step=0.07, wander_lift=0.08)
        yield from self.settle(arm, 2)
        reading = float(ctx.state[f"{arm}_ee_joint_state"][0])
        if reading < 0.035:
            _log(ctx, "donor_lost_on_turn", reading=reading)
            yield from grip_steps(ctx, arm, DONOR_OPEN, 3)
            return False
        if np.linalg.norm(turn_point - donor_tcp) > 1e-6:
            turned = model.tcp_matrix(ctx.joint_targets[arm])[:3, :3]
            fraction = yield from self.linear_move(arm, donor_tcp, turned, step_m=0.015, allow_partial=True)
            if fraction < 0.9:
                _log(ctx, "offer_slide_short", fraction=round(fraction, 2))
            yield from self.settle(arm, 2)
        actual = self.models[arm].tcp_matrix(ctx.joint_targets[arm])
        _log(ctx, "donor_offer", tcp=actual[:3, 3].round(4).tolist(), tool_x=actual[:3, 0].round(3).tolist(),
             tool_y=actual[:3, 1].round(3).tolist(), slice_dir=(actual[:3, :3] @ self.offer_slice_in_tool).round(3).tolist())
        return True

    def donor_return(self):
        """Bring the still-held slice back to its shelf seat (upright) and let go."""
        ctx = self.ctx
        arm = self.donor
        info = getattr(self, "donor_pick", None)
        if info is None:
            yield from grip_steps(ctx, arm, DONOR_OPEN, 3)
            yield from self.go_home(arm, 14)
            return
        g, rotation, hover = info["geometry"], info["rotation"], info["hover"]
        ctx.memory["phase"] = "donor_return"
        try:
            yield from self.rotate_in_place(arm, rotation, max_step=0.05, wander_lift=0.08)
            yield from self.free_move(arm, g["tcp"] + np.array([0.0, 0.0, 0.08]), rotation,
                                      min_tcp_z=self.table + 0.12 + 2 * BREAD_HALF, max_step=0.06, step_m=0.02, holding=True)
            yield from self.linear_move(arm, g["tcp"] + np.array([0.0, 0.0, 0.004]), rotation, step_m=0.015, allow_partial=True)
        except Exception as error:
            _log(ctx, "donor_return_failed", error=str(error)[:100])
        yield from grip_steps(ctx, arm, DONOR_OPEN, 4)
        yield from self.settle(arm, 2)
        yield from self.linear_move(arm, hover, rotation, step_m=0.03, allow_partial=True)
        yield from self.go_home(arm, 14)

    def donor_release_and_retreat(self, n_h):
        ctx = self.ctx
        arm = self.donor
        yield from grip_steps(ctx, arm, 0.6, 5)
        yield from self.settle(arm, 2)
        current = self.models[arm].tcp_matrix(ctx.joint_targets[arm])
        slice_dir = unit(current[:3, :3] @ getattr(self, "offer_slice_in_tool", np.array([1.0, 0.0, 0.0])))
        back = current[:3, 3] - slice_dir * 0.07 + np.array([0.0, 0.0, 0.04])
        yield from self.linear_move(arm, back, current[:3, :3], step_m=0.02, allow_partial=True)
        yield from self.go_home(arm, 14)

    # ---- receiver --------------------------------------------------------------
    def receiver_take(self, n_h):
        """Take the slice from the donor's ACTUAL pose (slice direction = offer tool frame x slice_in_tool)."""
        ctx = self.ctx
        arm = self.receiver
        donor = self.models[self.donor].tcp_matrix(ctx.joint_targets[self.donor])
        slice_in_tool = getattr(self, "offer_slice_in_tool", np.array([1.0, 0.0, 0.0]))
        tool_x = unit(donor[:3, :3] @ slice_in_tool)
        tool_y = donor[:3, 1]
        far_edge = donor[:3, 3] + tool_x * (2 * BREAD_HALF - PICK_EDGE_DEPTH)
        receiver_tcp = far_edge - tool_x * RECEIVER_EDGE_DEPTH + tool_x * TIP_BELOW_TCP
        # The two fingers are not alike: with the closing axis pointing down the pinch read a
        # clean 0.084 in 12/12 diagnostic handoffs, pointing up it stalled at 0.17 in 7/7 and
        # those slices went in cocked (blocked inserts, slices flung by the tamp). Take the
        # down-pointing frame whenever it is reachable, and never let the approach flip it.
        up = np.array([0.0, 0.0, 1.0])
        receiver_rot = None
        for candidate in (self.frame(-tool_x, -up), self.frame(-tool_x, up)):
            if self.models[arm].solve(receiver_tcp, candidate, ctx.joint_targets[arm], prefer_small_roll=True) is not None:
                receiver_rot = candidate
                break
        if receiver_rot is None:
            receiver_rot = self.frame(-tool_x, -up)
        self.receiver_reading = None
        n_h = tool_x
        # Reorient well beyond the far edge first: swinging the open pads from the tool-down
        # pose at the toaster straight into the approach point swept them through the slice
        # and pried it out of the donor's pinch (second handoff of L5). The low horizontal
        # staging leg is skipped when it would run over the toaster (the arm stalled on the
        # toaster top 15 cm short of the slice in L16).
        staging = receiver_tcp + n_h * 0.16
        over_toaster = any(self.box_margin_distance(receiver_tcp[:2] + n_h[:2] * s) < 0.10 for s in (0.07, 0.11, 0.16))
        ctx.memory["phase"] = "receiver_approach"
        yield from grip_steps(ctx, arm, RECEIVER_OPEN, 2)
        if not over_toaster:
            yield from self.free_move(arm, staging, receiver_rot, min_tcp_z=self.table + 0.11, max_step=0.08, holding=False, allow_flip=False)
            receiver_rot = self.last_rotation[arm]
        else:
            _log(ctx, "receiver_staging_skipped")
        # The offered slice sags or cocks a few degrees in the donor's edge pinch: a wide
        # aperture straddles it anyway, and a second try shifts the pads vertically.
        for attempt, dz in enumerate(RECEIVER_DZ):
            target = receiver_tcp + np.array([0.0, 0.0, dz])
            pre = target + n_h * 0.07
            ctx.memory["phase"] = "receiver_approach"
            yield from grip_steps(ctx, arm, RECEIVER_OPEN, 2)
            fraction = 0.0
            if not over_toaster:
                fraction = yield from self.linear_move(arm, pre, receiver_rot, step_m=0.02, allow_partial=True)
            if fraction < 0.9:
                yield from self.free_move(arm, pre, receiver_rot, min_tcp_z=self.table + 0.11, max_step=0.08, holding=False, allow_flip=False)
                receiver_rot = self.last_rotation[arm]
            # The short straight approach occasionally starts on an IK branch edge (a wrist
            # discontinuity at its first step): fall back to a checked joint interpolation.
            fraction = yield from self.linear_move(arm, target, receiver_rot, step_m=0.012, max_joint_step=0.05, allow_partial=True)
            if fraction < 0.9:
                goal = self.models[arm].solve(target, receiver_rot, ctx.joint_targets[arm])
                if goal is None:
                    _log(ctx, "receiver_approach_failed", attempt=attempt, fraction=round(fraction, 2))
                    continue
                for q in smooth_joint_path(ctx.joint_targets[arm].copy(), goal, 0.05):
                    yield q, None
            yield from self.settle(arm, 2)
            ctx.memory["phase"] = "receiver_grasp"
            yield from grip_steps(ctx, arm, 0.0, 6)
            yield from self.settle(arm, 4)
            reading = float(ctx.state[f"{arm}_ee_joint_state"][0])
            held = 0.035 < reading < RECEIVER_MAX_READING
            # A wider reading is the cocked slice caught diagonally; it flattens into a proper
            # pinch as soon as the donor lets go, and shifted retries never read better.
            diagonal = RECEIVER_MAX_READING <= reading < 0.25
            _log(ctx, "receiver_grasp", reading=reading, held=held or diagonal, diagonal=diagonal, attempt=attempt)
            if held or diagonal:
                self.receiver_reading = reading
                return True
            yield from grip_steps(ctx, arm, RECEIVER_OPEN, 3)
            yield from self.linear_move(arm, pre, receiver_rot, step_m=0.02, allow_partial=True)
        return False

    def guarded_move(self, arm, position, rotation, step_m=0.012, lag_limit=0.015):
        """Straight TCP move that stops once the measured TCP lags the command by ``lag_limit``.

        Returns ("done", 1.0), ("blocked", fraction) or ("partial", fraction) when the path
        itself could not be completed.
        """
        ctx = self.ctx
        model = self.models[arm]
        start = ctx.joint_targets[arm].copy()
        try:
            path = cartesian_path(model, start, position, rotation, step_m=step_m, max_joint_step=0.09)
            complete = True
        except CartesianPathError as error:
            _log(ctx, "partial_path", arm=arm, fraction=getattr(error, "fraction", 0.0), error=str(error))
            path = getattr(error, "partial", [])
            complete = False
        for index, q in enumerate(path):
            yield q, None
            if index % 2 == 1:
                measured = model.tcp_matrix(np.asarray(ctx.state[f"{arm}_arm_joint_state"], float))[:3, 3]
                commanded = model.tcp_matrix(q)[:3, 3]
                if measured[2] - commanded[2] > lag_limit:
                    return "blocked", (index + 1) / max(1, len(path))
        return ("done" if complete else "partial"), 1.0

    def receiver_insert(self, slot_index):
        ctx = self.ctx
        arm = self.receiver
        slots = self.slots
        centre = slots["centres"][slot_index]
        long_axis = slots["long_axis"]
        across = slots["across"]
        # Gripper yaw: closing axis along the slot's across direction (slice plane along the slot).
        # Both antipodal yaws pinch equally; pick the one with the wrist roll away from its limit.
        base_yaw = math.atan2(across[1], across[0]) - math.pi / 2
        hang = 2 * BREAD_HALF - RECEIVER_EDGE_DEPTH
        tilt = math.radians(30.0)
        axis = np.r_[across, 0.0]
        current_q = ctx.joint_targets[arm].copy()
        position_now = self.models[arm].tcp_matrix(current_q)[:3, 3]
        best = None
        for candidate in (base_yaw, base_yaw + math.pi):
            q = self.models[arm].solve(position_now, down_rotation(candidate, tilt, axis), current_q, prefer_small_roll=True)
            if q is not None and (best is None or abs(q[5]) < best[0]):
                best = (abs(q[5]), candidate)
        gripper_yaw = base_yaw if best is None else best[1]
        _log(ctx, "insert_yaw", slot=slot_index, yaw=gripper_yaw, roll=None if best is None else best[0])
        # Tilt about the slot normal: the tool tips towards -long_axis and the lower
        # leading corner moves (hang*sin t + half*cos t) towards -long_axis of the TCP.
        tilted = down_rotation(gripper_yaw, tilt, axis)
        # A small residual tilt keeps the wrist away from the straight-down singularity.
        vertical = down_rotation(gripper_yaw, math.radians(4.0), axis)
        corner_shift = hang * math.sin(tilt) + BREAD_HALF * math.cos(tilt)
        leading_corner = centre[:2] - long_axis * (slots["half_length"] - 0.012)
        approach_xy = leading_corner + long_axis * corner_shift
        corner_drop = hang * math.cos(tilt) + BREAD_HALF * math.sin(tilt)
        approach = np.r_[approach_xy, self.roof + corner_drop + 0.03]
        ctx.memory["phase"] = "receiver_carry"
        # Reorient at the handoff point first (slice swings down to hang), then translate.
        yield from self.rotate_in_place(arm, tilted, max_step=0.07, wander_lift=0.06)
        yield from self.settle(arm, 2)
        yield from self.free_move(arm, approach, tilted, min_tcp_z=self.table + 0.12 + hang,
                                  box_z=approach[2] - 0.01, max_step=0.07, step_m=0.025, holding=True)
        yield from self.settle(arm, 2)
        ctx.memory["phase"] = "receiver_insert"
        # Lower until the leading corner is 2 cm inside, then straighten while descending.
        # Every descent is guarded: a corner catching the slot edge shows up as the measured
        # TCP lagging above the command, and pressing on with the stiff arm has thrown the
        # slice (and the arm) across the table.
        entry = approach.copy()
        entry[2] = self.roof + corner_drop - 0.02
        final = np.r_[centre[:2], self.roof + TIP_BELOW_TCP + 0.006]
        status, fraction = yield from self.guarded_move(arm, entry, tilted, step_m=0.012)
        if status == "done":
            status, fraction = yield from self.guarded_move(arm, final, vertical, step_m=0.012)
        if status == "blocked":
            _log(ctx, "insert_blocked", slot=slot_index, fraction=round(fraction, 2))
            lifted = self.models[arm].tcp_matrix(ctx.joint_targets[arm])[:3, 3] + np.array([0.0, 0.0, 0.03])
            yield from self.linear_move(arm, lifted, vertical, step_m=0.015, allow_partial=True)
            yield from self.settle(arm, 2)
            # Second try: straight down over the slot centre, shifted across if needed.
            for shift in (0.0, 0.006, -0.006, 0.012, -0.012):
                above = np.r_[centre[:2] + across * shift, lifted[2]]
                yield from self.linear_move(arm, above, vertical, step_m=0.015, allow_partial=True)
                target = np.r_[centre[:2] + across * shift, final[2]]
                status, fraction = yield from self.guarded_move(arm, target, vertical, step_m=0.010)
                _log(ctx, "insert_retry", slot=slot_index, shift=shift, status=status, fraction=round(fraction, 2))
                if status == "done" or fraction > 0.7:
                    break
                yield from self.linear_move(arm, above, vertical, step_m=0.015, allow_partial=True)
        yield from self.settle(arm, 3)
        _log(ctx, "insert_release", slot=slot_index, fraction=fraction, status=status,
             tcp=self.models[arm].tcp_matrix(ctx.joint_targets[arm])[:3, 3].round(4).tolist())
        yield from grip_steps(ctx, arm, 0.6, 5)
        yield from self.settle(arm, 3)
        # Clear the roof with the empty gripper.
        up = final.copy()
        up[2] += 0.035
        yield from self.linear_move(arm, up, vertical, step_m=0.02, allow_partial=True)
        # Tamp: a released slice often stays wedged at the slot entrance (top edge ~7 cm above
        # the roof, centre too high for the rule). Push its top edge down to roof level with the
        # closed finger tips; a slice that already dropped is simply not there. A slice released
        # high after a blocked insert is cocked: pounding it has flung it off the table.
        # A slice taken with the wide (off-centre) pinch goes in cocked; tamping it has
        # flung slices off the toaster (both slices on the table in official L28).
        cocked = (getattr(self, "receiver_reading", None) or 0.0) >= RECEIVER_MAX_READING
        if status != "done" or fraction < 0.99 or cocked:
            _log(ctx, "insert_tamp", slot=slot_index, status="skipped", cocked=cocked)
            yield from grip_steps(ctx, arm, RECEIVER_OPEN, 3)
            # Rise as far as the arm allows before retreating: the open pads straddled a slice
            # that had not dropped and dragged it off the roof when they left sideways (L6).
            yield from self.linear_move(arm, up + np.array([0.0, 0.0, 0.05]), vertical, step_m=0.02, allow_partial=True)
            current = self.models[arm].tcp_matrix(ctx.joint_targets[arm])[:3, 3]
            yield from self.linear_move(arm, current + np.array([0.0, -0.12, 0.0]), vertical, step_m=0.03, allow_partial=True)
            return
        tamp_rotation = vertical   # the pure tool-down frame sits on the wrist singularity and its lift failed
        tamp_xy = centre[:2]
        tamp_start = np.r_[tamp_xy, self.roof + 0.10 + TIP_BELOW_TCP]
        tamp_end = np.r_[tamp_xy, self.roof + TIP_BELOW_TCP + 0.003]
        # Close only once the pads are clear above the slice: 3.5 cm up they still straddled a
        # slice that had not dropped, pinched its top edge, lifted it out and let it fall
        # (variant-4 slots in official L22/L28/L33, both slices ending on the table).
        lifted = yield from self.linear_move(arm, tamp_start, tamp_rotation, step_m=0.02, allow_partial=True)
        if lifted < 0.9:
            _log(ctx, "insert_tamp", slot=slot_index, status="skipped", lift=round(lifted, 2))
            yield from grip_steps(ctx, arm, RECEIVER_OPEN, 3)
            current = self.models[arm].tcp_matrix(ctx.joint_targets[arm])[:3, 3]
            yield from self.linear_move(arm, current + np.array([0.0, -0.12, 0.0]), vertical, step_m=0.03, allow_partial=True)
            return
        yield from grip_steps(ctx, arm, 0.0, 4)
        status, fraction = yield from self.guarded_move(arm, tamp_end, tamp_rotation, step_m=0.010, lag_limit=0.012)
        _log(ctx, "insert_tamp", slot=slot_index, status=status, fraction=round(fraction, 2))
        yield from self.settle(arm, 2)
        yield from self.linear_move(arm, tamp_start, tamp_rotation, step_m=0.02, allow_partial=True)
        yield from grip_steps(ctx, arm, 0.6, 3)
        back = tamp_start + np.array([0.0, -0.12, 0.0])
        yield from self.linear_move(arm, back, tamp_rotation, step_m=0.03, allow_partial=True)

    def press_lever(self):
        """Press the lever lip from outside the front wall with the closed finger tips.

        The lever sits flush inside the body outline except for its front lip, so the
        TCP goes to the lip (public ``tcp_x``) with the tool pitched 45 degrees down
        towards the toaster; a vertical press over the lever centre lands on the body.
        """
        ctx = self.ctx
        arm = self.receiver
        top, travel, kind = lever_target(self.toaster)
        asset = TOASTER_ASSETS[self.toaster["variant"]]["lever"]
        root, rotation_t = self.toaster["root"], self.toaster["rotation"]
        axis = unit(np.r_[rotation_t[:2, 0], 0.0])            # toaster local x (front direction)
        lateral = np.r_[self.slots["across"], 0.0]
        xy = root[:2] + axis[:2] * asset["tcp_x"]
        pitch = asset.get('pitch', math.pi / 4)
        tip_depth = asset.get('tip_depth', TIP_BELOW_TCP)
        tool_x = unit(-axis * math.cos(pitch) + np.array([0.0, 0.0, -math.sin(pitch)]))
        rotation = self.pick_frame(arm, np.r_[xy, top[2] + 0.06], [self.frame(tool_x, lateral), self.frame(tool_x, -lateral)],
                                   ctx.joint_targets[arm])
        ctx.memory["phase"] = "lever"
        yield from grip_steps(ctx, arm, 0.0, 4)
        above = np.r_[xy, top[2] + tip_depth + 0.06]
        yield from self.free_move(arm, above, rotation, min_tcp_z=self.table + 0.14, holding=False, allow_flip=False)
        # The knob top is only ~2 cm deep and the pitched pads jam against the body when the
        # tips land a few millimetres too far in: a blocked press (measured TCP stalls while
        # the command keeps descending) is retried with the tips shifted outwards.
        for attempt, shift in enumerate((0.005, 0.010, 0.0, 0.015)):
            xy_k = root[:2] + axis[:2] * (asset["tcp_x"] + shift)
            above_k = np.r_[xy_k, top[2] + tip_depth + 0.06]
            if attempt:
                yield from self.linear_move(arm, above_k, rotation, step_m=0.02, allow_partial=True)
            contact = np.r_[xy_k, top[2] + tip_depth + 0.004]
            yield from self.linear_move(arm, contact, rotation, step_m=0.015, allow_partial=True)
            pressed = np.r_[xy_k, top[2] + tip_depth - travel]
            status, fraction = yield from self.guarded_move(arm, pressed, rotation, step_m=0.01, lag_limit=0.03)
            yield from self.settle(arm, 6)
            _log(ctx, "lever_pressed", kind=kind, fraction=fraction, travel=travel, status=status, attempt=attempt, shift=shift,
                 tcp=self.models[arm].tcp_matrix(ctx.joint_targets[arm])[:3, 3].round(4).tolist())
            if status != "blocked" or fraction > 0.85:
                break
            yield from self.linear_move(arm, above_k, rotation, step_m=0.02, allow_partial=True)
        yield from self.linear_move(arm, above, rotation, step_m=0.02, allow_partial=True)

    def run_pair(self, donor_generator, receiver_generator=None):
        """Run the donor generator (returning its value) with an optional receiver generator alongside."""
        result = {}

        def wrapped():
            result["value"] = yield from donor_generator

        yield from self.scheduler.run(**{self.donor: wrapped(), self.receiver: receiver_generator})
        return result.get("value")

    # ---- program ---------------------------------------------------------------
    def run(self):
        ctx = self.ctx
        placed = 0
        for slot_index in (0, 1):
            if ctx.steps > self.deadline:
                break
            scene = perceive(ctx)
            # Shelf slices: upright, at rack height, outside the toaster (fallen or inserted ones are not).
            breads = [b for b in scene["breads"] if 0.04 < b["centre"][2] - self.table < 0.11 and b["up"][2] > 0.85
                      and not self.in_toaster(b["centre"])]
            if len(breads) < 2:
                _log(ctx, "no_rack_slice")
                break
            bread = self.end_slice(breads, slot_index)
            g = self.pick_geometry(bread, breads)
            try:
                # The receiver returns home while the donor picks: approaching the offered slice
                # from the tool-down pose left at the toaster took a wild joint blend that swept
                # the open pads through the slice (second handoffs of L5/L16).
                parked = self.go_home(self.receiver, 14) if slot_index else None
                ok = yield from self.run_pair(self.donor_pick_and_offer(bread, breads), parked)
            except Exception as error:
                _log(ctx, "donor_error", slice=bread["label"], error=f"{type(error).__name__}: {str(error)[:80]}")
                yield from self.run_arm(self.donor, self.donor_return())
                continue
            if not ok:
                _log(ctx, "donor_failed", slice=bread["label"])
                yield from self.run_arm(self.donor, self.go_home(self.donor, 14))
                continue
            # The slice has left the shelf whatever happens next (RGB scene bookkeeping).
            ctx.memory.setdefault("toast_slices_taken", []).append(bread["label"])
            try:
                took = yield from self.run_arm(self.receiver, self.receiver_take(g["n_h"]))
            except Exception as error:
                _log(ctx, "receiver_error", error=f"{type(error).__name__}: {str(error)[:80]}")
                took = False
            if not took:
                _log(ctx, "receiver_failed")
                yield from self.run_arm(self.receiver, grip_steps(ctx, self.receiver, RECEIVER_OPEN, 3))
                yield from self.run_arm(self.receiver, self.go_home(self.receiver, 14))
                yield from self.run_arm(self.donor, self.donor_return())
                continue
            yield from self.run_arm(self.donor, self.donor_release_and_retreat(g["n_h"]))
            try:
                yield from self.run_arm(self.receiver, self.receiver_insert(slot_index))
            except Exception as error:
                _log(ctx, "insert_error", error=f"{type(error).__name__}: {str(error)[:80]}")
                yield from self.run_arm(self.receiver, grip_steps(ctx, self.receiver, RECEIVER_OPEN, 3))
                yield from self.run_arm(self.receiver, self.go_home(self.receiver, 14))
                continue
            placed += 1
        ctx.memory["slices_inserted"] = placed
        if placed >= 1:
            yield from self.run_arm(self.receiver, self.press_lever())
        ctx.stage = "home"
        ctx.memory["phase"] = "home"
        yield from self.scheduler.run(left=self.go_home("left", 20), right=self.go_home("right", 20))
        for _ in range(6):
            yield ctx.hold_action()


def make_toast_random(ctx):
    if not (getattr(ctx, "oracle_objects", None) and os.environ.get("COAP_ORACLE_MODE", "").startswith("absolute")):
        for _ in range(3):   # the first rendered frames are not reliable
            yield ctx.hold_action()
    scene = perceive(ctx)
    if scene["toaster"] is None or len(scene["breads"]) < 2:
        raise RuntimeError("Toast scene measurement missing")
    _log(ctx, "scene", toaster=dict(root=scene["toaster"]["root"].tolist(), yaw=scene["toaster"]["yaw"],
                                     variant=scene["toaster"]["variant"]),
         breads=[dict(label=b["label"], centre=b["centre"].tolist()) for b in scene["breads"]])
    executor = ToastExecutor(ctx, scene)
    yield from executor.run()
