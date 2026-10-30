"""The grasp depth of low objects (switch OURS_GRASP_DEPTH, opt-in), the run's half.

Rule: a contact descent onto a low object goes down until the
fingers' lowest point is C_M = 4 mm above the support under the fingertips. Invisible to the model: no prompt text, no
note in any result; the adjusted action carries the evidence key `automatic_depth` (MARK ...) for the journal, and every
decision goes to the session's grasp_depth.jsonl. With the switch off nothing of this module is imported by the run.
Official conditions only (OURS_RGB_ONLY: the measurement code is agp_core/rgb_measure.py).

Which arm actions (a `run` step's runtime action, before the runtime gets it; the `step` tool is never changed):
  1. command move_ee, intent contact, a target relative to a measurement (position_offset with a measurement_id: the
     DSL's `label+dx,dy,dz`), without max_samples (a checkpoint keeps its target);
  2. the measurement is valid, of this arm, RGB-only, and its point lies at most LOW_TOP_M = 25 mm above the table
     (the base frame's z; any height source: match, plane, fallback, uncertain);
  3. the gripper is open: no grasp attempt active in the runtime and the observation before the step shows the reading
     n >= 0.5 (the observer then places the fingers at the open limit, which this rule uses);
  4. the approach is within MAX_TILT_DEG = 45 deg of vertical (the TCP's z axis, towards the fingertips, against
     straight down, at the target rotation);
  5. the agent's own contact target (the action as written, before any lowering) puts the fingers' lowest point at most
     LOW_FINGER_M = 25 mm above the table: the TCP target's height minus `depth` below (the hull geometry at the target
     rotation). Condition 2 alone would also admit a tall target referenced to a table-plane point, e.g. a handle
     grasped 17 cm above a plane measurement of the object's base centre on the table (lowered by 10 mm in an offline
     replay without this condition); such a target is not a low grasp and stays as written.
The rule: the fingers' lowest point at the target is computed from agp_core/x5_link_hulls.json (rgb_measure.link_hulls:
link7 and link8 placed as the observation's capture places them, at the open limit) at the target rotation: depth = how
far the lowest finger vertex lies below the TCP. Each finger's landing point is the centroid of its vertices within
LANDING_BAND_M of its lowest vertex (base frame xy at the target). The bound for the TCP is support + C_M + depth. The
TCP target only goes down, never up: nothing changes when the agent's TCP height is at most TOLERANCE_M = 0.5 mm above
the bound (or below it).
Safeguard (a), the support: the higher of the two fingertip landing surfaces, measured in the current observation with
the measurement code of `measure` (agp_core/rgb_measure.py: a top-image pick at the landing point's pixel, projected at the
table height and, when the matched height differs by more than zmatch.LR_TOLERANCE_M, once more at that height; the
pick is matched into the wrist images exactly as `measure` matches a top pick, earlier top captures included where the
robot covers the pixel) -- no ground truth, no depth, no journal entry, no review image, no text for the model. A
landing point is measured when its result's height source is match or plane (SUPPORT_SOURCES; without a plane_z only
match occurs). The support is never below the table (z = 0). When a landing point is not measured, the TCP goes down by
at most UNMEASURED_CAP_M = 3 mm (the support then counts the measured landing points and the table).
Safeguard (b): never down by more than MAX_LOWER_M = 10 mm.
Records: grasp_depth.jsonl (one line per run step with a candidate: per arm the decision, the reason or the branch,
the geometry, the landing measurements, the lowering), and the adjusted action's evidence `automatic_depth`.
Branches: "measured" (both landing points measured, the full lowering), "measured, capped at 10 mm", "unmeasured"
(a landing point without a match/plane result, the full lowering of at most 3 mm), "unmeasured, capped at 3 mm".
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time

import numpy as np

# ---- the rule and its safeguards
C_M = 0.004                     # the fingers' lowest point this far above the support (c = 4 mm)
LOW_TOP_M = 0.025               # condition 2: measured top at most 25 mm above the table (any measurement source)
LOW_FINGER_M = 0.025            # condition 5: the agent's own target puts the fingers' lowest point at most 25 mm
                                # above the table
MAX_TILT_DEG = 45.0             # the approach within 45 deg of vertical
TOLERANCE_M = 0.0005            # nothing changes within 0.5 mm of the bound
UNMEASURED_CAP_M = 0.003        # safeguard (a): at most 3 mm when a landing surface has no match/plane result
                                # (engineering value: the median adjustment in offline replays is 2.2 mm)
MAX_LOWER_M = 0.010             # safeguard (b): never more than 10 mm (the observed cases needed 1.3 to 9.8 mm)
SUPPORT_SOURCES = ("match", "plane")
# ---- engineering values
LANDING_BAND_M = 0.001          # a finger's landing patch: its vertices within 1 mm of its lowest vertex
OPEN_READING = 0.5              # the observer's open/closed split (agp_core/rgb_observe.py capture_geometry)
TABLE_Z_M = 0.0                 # the table top in the arm's base frame (agp_core/rgb_measure.py, public camera model)
FINGER_LINKS = ("link7", "link8")
RECORD_FILE = "grasp_depth.jsonl"
MARK = "OURS_GRASP_DEPTH"       # the start of the adjusted action's evidence `automatic_depth` (the ledger's marker)


def quat_matrix(q):
    """Rotation matrix of a quaternion dict {w, x, y, z} (normalised)."""
    w, x, y, z = (float(q[k]) for k in "wxyz")
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def tilt_deg(R):
    """Angle (deg) between the TCP's z axis (the approach, towards the fingertips) and straight down."""
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(np.asarray(R, float)[2, 2])))))


def finger_points_tcp(capture, arm):
    """{link: (N, 3)} the finger hull vertices (agp_core/x5_link_hulls.json) in the TCP frame, with the fingers where the
    capture's robot_state places them (the open limit while the reading is >= 0.5), from the capture's own public
    geometry: the base from the top camera's world and base poses, the TCP from T_base_tcp_by_arm."""
    from agp_core import rgb_measure
    top = capture.calibration["top"]
    T_world_base = np.asarray(top["T_world_optical"], float) @ np.linalg.inv(
        np.asarray(top["T_base_optical_by_arm"][arm], float))
    inv = np.linalg.inv(T_world_base @ capture.tcp(arm))
    links = capture.robot["arms"][arm]["T_world_links"]
    hulls, _ = rgb_measure.link_hulls()
    out = {}
    for link in FINGER_LINKS:
        T = np.asarray(links[link], float)
        V = np.vstack(hulls[link])
        world = V @ T[:3, :3].T + T[:3, 3]
        out[link] = world @ inv[:3, :3].T + inv[:3, 3]
    return out


def finger_geometry(points_tcp, R, target_xy):
    """The fingers at the target rotation R (base frame): depth (m) of the lowest finger vertex below the TCP, and per
    finger its landing point (base xy: the centroid of its vertices within LANDING_BAND_M of its lowest vertex)."""
    R = np.asarray(R, float)
    depth, landing = 0.0, {}
    for link, V in points_tcp.items():
        Vb = np.asarray(V, float) @ R.T
        low = float(Vb[:, 2].min())
        patch = Vb[Vb[:, 2] <= low + LANDING_BAND_M]
        landing[link] = [float(target_xy[0] + patch[:, 0].mean()), float(target_xy[1] + patch[:, 1].mean())]
        depth = max(depth, -low)
    return depth, landing


def landing_height(session, capture, arm, xy):
    """The surface under base point xy (see the module docstring, safeguard (a)): {measured, z, source, tries}."""
    from agp_core import rgb_measure, zmatch
    current = rgb_measure.load_capture(session, capture)
    K, T = current.K("top"), current.T("top", arm)
    h, w = current.size("top")
    tries, z = [], TABLE_Z_M
    for _ in range(2):
        uv, depth = rgb_measure.project(K, T, np.array([[xy[0], xy[1], z]]))
        if not float(depth[0]) > 0:
            tries.append({"projected_at_m": round(z, 6), "source": "none", "reason": "behind the top camera"})
            break
        u, v = int(math.floor(float(uv[0][0]))), int(math.floor(float(uv[0][1])))
        if not (0 <= u < w and 0 <= v < h):
            tries.append({"projected_at_m": round(z, 6), "pixel": [u, v], "source": "none",
                          "reason": "outside the top image"})
            break
        height, _ = rgb_measure._match(session, current, {"arm": arm, "cam": "top", "u": u, "v": v}, review=False)
        tries.append({"projected_at_m": round(z, 6), "pixel": [u, v], "source": height.get("source"),
                      "z": None if height.get("z") is None else round(float(height["z"]), 6),
                      "older_view": height.get("older_view"), "partner": height.get("partner")})
        if height.get("source") not in SUPPORT_SOURCES or height.get("z") is None:
            break
        if abs(float(height["z"]) - z) <= zmatch.LR_TOLERANCE_M:
            break
        if len(tries) == 2:                                  # the second pass disagrees with the first: not measured
            tries[-1]["reason"] = "the second pass disagrees with the first"
            break
        z = float(height["z"])
    last = tries[-1] if tries else {"source": "none"}
    measured = (last.get("source") in SUPPORT_SOURCES and last.get("z") is not None and "reason" not in last
                and (len(tries) == 1 or abs(float(last["z"]) - float(tries[0]["z"])) <= zmatch.LR_TOLERANCE_M))
    return {"measured": bool(measured), "z": last.get("z") if measured else None, "source": last.get("source"),
            "tries": tries}


def _measurement(runtime, measurement_id):
    try:
        row = runtime.db.execute("SELECT value, valid FROM measurements WHERE id=?",
                                 (int(str(measurement_id)[1:]),)).fetchone()
    except Exception:                       # noqa: BLE001 - no journal: no decision
        return None, False
    if not row:
        return None, False
    return json.loads(row[0]), bool(row[1])


def _arm_state(runtime, arm):
    try:
        return (runtime.state.get("arms") or {}).get(arm) or {}
    except Exception:                       # noqa: BLE001
        return {}


def arm_decision(runtime, arm, action, obs_full, landing=None):
    """(decision, adjusted action or None) for one runtime arm action (module docstring). decision: {candidate, ...};
    candidate False: not a contact descent onto a measurement-relative target at all (nothing recorded)."""
    args = action.get("args") or {}
    if (action.get("intent") != "contact" or action.get("command") != "move_ee" or "position_offset" not in args
            or not action.get("measurement_id")):
        return {"candidate": False}, None
    out = {"candidate": True, "measurement_id": action["measurement_id"], "adjusted": False}
    if "max_samples" in args:
        return dict(out, reason="a checkpoint (max_samples)"), None
    entry, valid = _measurement(runtime, action["measurement_id"])
    if entry is None or not valid or not entry.get("ok"):
        return dict(out, reason="measurement absent or invalid"), None
    if entry.get("arm") != arm:
        return dict(out, reason="measurement of the other arm"), None
    result = entry.get("result") or {}
    source = (result.get("height") or {}).get("source")
    out["source"] = source
    if not result.get("rgb_only"):
        return dict(out, reason="not an RGB-only measurement"), None
    point = result.get("point_base") or {}
    offset = args["position_offset"]
    try:
        top = float(point["z"])
        target = [float(point[k]) + float(offset[k]) for k in "xyz"]
    except (KeyError, TypeError, ValueError):
        return dict(out, reason="measurement without a point"), None
    out.update(top_m=round(top, 6), target_tcp_z_m=round(target[2], 6))
    if top - TABLE_Z_M > LOW_TOP_M:
        return dict(out, reason=f"measured top {top * 1000:.1f} mm above the table (R1a: at most "
                                f"{LOW_TOP_M * 1000:.0f} mm)"), None
    if _arm_state(runtime, arm).get("grasp_active"):
        return dict(out, reason="the gripper is closed: a grasp attempt is active"), None
    state = (((obs_full or {}).get("state") or {}).get("arms") or {}).get(arm) or {}
    try:
        n = float(state.get("gripper_command"))
    except (TypeError, ValueError):
        n = None
    if n is None or not math.isfinite(n) or n < OPEN_READING:
        return dict(out, reason=f"the gripper reading {n} is not open (>= {OPEN_READING})"), None
    try:
        R = quat_matrix(args["rotation"])
    except (KeyError, TypeError, ValueError):
        return dict(out, reason="no target rotation"), None
    tilt = tilt_deg(R)
    out["tilt_deg"] = round(tilt, 3)
    if tilt > MAX_TILT_DEG:
        return dict(out, reason=f"approach {tilt:.1f} deg from vertical (at most {MAX_TILT_DEG:.0f} deg)"), None
    capture_id = (obs_full or {}).get("capture")
    session = getattr(runtime, "session", None)
    try:
        from agp_core import rgb_measure
        capture = rgb_measure.load_capture(session, capture_id)
        robot = (capture.robot or {}).get("arms", {}).get(arm) or {}
        upper = float((capture.robot or {}).get("finger_joint_range_m", [0.0, 0.044])[1])
        if abs(float(robot.get("finger_q_assumed_m", -1.0)) - upper) > 1e-9:
            return dict(out, reason="the capture does not place the fingers at the open limit"), None
        depth, points = finger_geometry(finger_points_tcp(capture, arm), R, target[:2])
    except Exception as exc:                  # noqa: BLE001 - no capture geometry: no decision
        return dict(out, reason=f"no finger geometry: {type(exc).__name__}: {exc}"[:200]), None
    out.update(finger_depth_m=round(depth, 6), landing_xy={k: [round(v, 6) for v in p] for k, p in points.items()})
    finger_low = target[2] - depth - TABLE_Z_M        # condition 5: the agent's own target, the fingers' lowest point
    out["target_finger_low_m"] = round(finger_low, 6)
    if finger_low > LOW_FINGER_M + 1e-9:             # (floating point: exactly 25 mm still applies)
        return dict(out, reason=(f"the agent's contact target puts the fingers' lowest point {finger_low * 1000:.1f} mm "
                                 f"above the table (R1a: at most {LOW_FINGER_M * 1000:.0f} mm)")), None
    table_bound = TABLE_Z_M + C_M + depth
    if target[2] <= table_bound + TOLERANCE_M:      # not even the table bound lowers it: no landing measurement
        return dict(out, reason=(f"the TCP target is {(target[2] - table_bound) * 1000:.1f} mm from the table bound "
                                 f"(at most {TOLERANCE_M * 1000:.1f} mm above it, or below)"),
                    bound_tcp_z_m=round(table_bound, 6)), None
    landing = landing or landing_height
    heights = {link: landing(session, capture_id, arm, xy) for link, xy in points.items()}
    out["landing"] = heights
    measured = [float(hh["z"]) for hh in heights.values() if hh.get("measured")]
    all_measured = len(measured) == len(heights)
    support = max([TABLE_Z_M] + measured)
    bound = support + C_M + depth
    lower = target[2] - bound
    out.update(support_z_m=round(support, 6), bound_tcp_z_m=round(bound, 6), lower_wanted_m=round(lower, 6),
               all_measured=all_measured)
    if lower <= TOLERANCE_M:
        return dict(out, reason=(f"the TCP target is {lower * 1000:.1f} mm above the bound over the measured support "
                                 f"(at most {TOLERANCE_M * 1000:.1f} mm, or below)")), None
    cap = MAX_LOWER_M if all_measured else UNMEASURED_CAP_M
    applied = min(lower, cap)
    branch = ("measured" if all_measured else "unmeasured") + (
        f", capped at {cap * 1000:.0f} mm" if lower > cap + 1e-12 else "")
    out.update(adjusted=True, branch=branch, lower_m=round(applied, 6), capped=bool(lower > cap + 1e-12),
               cap_m=cap, new_tcp_z_m=round(target[2] - applied, 6))
    adjusted = copy.deepcopy(action)
    adjusted["args"]["position_offset"]["z"] = float(offset["z"]) - applied
    evidence = dict(adjusted.get("evidence") or {})
    evidence["automatic_depth"] = (f"{MARK} ({branch}): the contact target {applied * 1000:.1f} mm lower, the "
                                   f"fingers' lowest point {C_M * 1000:.0f} mm above the support "
                                   f"{support * 1000:.1f} mm (measurement {action['measurement_id']}, {source})")
    adjusted["evidence"] = evidence
    return out, adjusted


def apply_step(runtime, actions, obs_full, run_observation_id, k, landing=None):
    """The grasp depth rule on one run step's runtime actions: (actions with every adjusted contact replaced, record or
    None). The record (one line of grasp_depth.jsonl) lists every candidate arm; None when no arm action is a
    candidate."""
    out, record = dict(actions), {}
    for arm, action in actions.items():
        try:
            decision, adjusted = arm_decision(runtime, arm, action, obs_full, landing)
        except Exception as exc:                 # noqa: BLE001 - the step runs as written
            decision, adjusted = {"candidate": True, "adjusted": False,
                                  "reason": f"error: {type(exc).__name__}: {exc}"[:300]}, None
        if not decision.get("candidate"):
            continue
        record[arm] = decision
        if adjusted is not None:
            out[arm] = adjusted
    if not record:
        return out, None
    entry = {"run_observation_id": run_observation_id, "k": k, "action_id": f"{run_observation_id}.{k}",
             "arms": record}
    write(getattr(runtime, "session", None), entry)
    return out, entry


def write(session, entry):
    """Append one line to the session's grasp_depth.jsonl (never raises)."""
    try:
        if session is None:
            return
        with (Path(session) / RECORD_FILE).open("a") as stream:
            stream.write(json.dumps(dict(entry, wall=time.time()), default=str) + "\n")
    except Exception:                            # noqa: BLE001
        pass
