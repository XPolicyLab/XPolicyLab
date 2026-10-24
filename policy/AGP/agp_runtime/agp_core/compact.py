"""Compact model-facing views of runtime results.

The full result is still archived unchanged in native_tool_results. Only the text
the policy reads is reduced: calibration matrices, static robot geometry and
advisory prose are dropped because the prompt states them once.

RGB-only mode (OURS_RGB_ONLY): an observation marked state.observation == "rgb_only" shows steps_left and
steps_used instead of sim_time_s / samples_used and has no gap_mm (the finger opening is not observed); a measurement
from agp_core/rgb_measure.py adds z_from (plane | match | uncertain), on_robot and older_view (the match used an older
top image because the robot covered the pixel in the latest one); step outcomes leave out the measured
finger joints. Results without these markers, and every result with the switch off, are shown unchanged.

Visual estimates (only results with the RGB-only markers change):
  * an rgb_only arm state shows `gap_mm`, the pad gap estimated from that arm's wrist image (state
    gripper_gap_estimate, agp_core/visual_gap.py), when the estimate's quality is ok; otherwise `gap_mm` is null and
    `gap_q` gives the quality (low | unavailable); a state without an estimate shows no gap;
  * a measurement whose raised plane_z was only the fallback (agp_core/rgb_measure.py, height source 'fallback') shows
    z_from 'fallback' and `match` (uncertain | none | disagrees), with `match_z` when the matcher found a height.

Hold clearance (OURS_HOLD_CLEARANCE): an arm outcome with a `note` (only that switch writes one: the sentence that a
lift or a final descent was added) shows it; outcomes without one are shown unchanged.

Return reserve (OURS_RETURN_RESERVE): an rgb_only observation whose state has `return_reserve` (only
agp_core/return_reserve.py writes it) shows it as `budget` {return, spare, releases, per_release, more_releases}; an
observation without it is shown unchanged.

Grip width (OURS_GRIP_WIDTH): an arm state with `gripper_width_mm` (only agp_core/rgb_observe.py writes it, for a partial
width) shows `gripper: "width"` and `width_mm` (the commanded pad gap); an arm outcome with `ours_grip_width` (only
agp_core/grip_width.py writes it) shows `gap_mm` (the pad gap reached: the wrist-image estimate, else what the official
reading shows, else null) and `fingers` (at the requested gap, stopped on an object, short of the requested gap, not
confirmed). Everything else is shown unchanged.

Plan guard (OURS_PLAN_GUARD): an arm outcome with `ours_plan_guard` and a `reason` (only agp_core/speed_profiles.py writes
them, on a plan it refused as "Planning rejected") shows that `reason` (the guard, the path and straight distance, what
to do instead) next to the error. Outcomes without them are shown unchanged.

Hide end (OURS_HIDE_END): with the switch on, an observation whose state has no `episode_ended`
(agp_core/official_worker.py's HideEndMixin removes it from every reply) shows no `episode_ended`; an observation with it,
and every observation with the switch off, is shown unchanged.
"""
from agp_core import features

GRIPPER_LINKS = {"link6", "link7", "link8", "camera", "camera_base"}


def _r(value, digits=4):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_r(v, digits) for v in value]
    return round(float(value), digits)


def _mm(value):
    return None if value is None else round(float(value) * 1000, 1)


def _clip(text, limit=300):
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "..."


def arm_state(arm, rgb=False):
    geometry = arm.get("gripper_geometry") or {}
    out = {
        "tcp": _r(arm.get("tcp_pose_base")),
        "gap_mm": _mm(geometry.get("nominal_pad_gap_m")),
        "gripper": "open" if float(arm.get("gripper_command", 1.0)) >= 0.5 else "closed",
        "joints": _r(arm.get("joints_rad"), 3),
    }
    if rgb:
        del out["gap_mm"]
        estimate = arm.get("gripper_gap_estimate")
        if isinstance(estimate, dict):           # the visual estimate, shown only when reliable
            ok = estimate.get("quality") == "ok" and estimate.get("pad_gap_m") is not None
            out = {"tcp": out["tcp"], "gap_mm": _mm(estimate["pad_gap_m"]) if ok else None,
                   **({} if ok else {"gap_q": estimate.get("quality")}),
                   "gripper": out["gripper"], "joints": out["joints"]}
    if arm.get("gripper_width_mm") is not None:          # grip width: commanded to a partial width (OURS_GRIP_WIDTH)
        joints = out.pop("joints")
        out.update(gripper="width", width_mm=arm["gripper_width_mm"], joints=joints)
    return out


def observation(obs):
    if not isinstance(obs, dict):
        return None
    if not obs.get("ok"):
        return {"ok": False, "error": _clip(obs.get("error", "observation failed"))}
    state = obs.get("state", {})
    if state.get("observation") == "rgb_only":
        view = {
            "observation_id": obs.get("observation_id"),
            "steps_left": state.get("remaining_actions"),
            "steps_used": state.get("low_level_actions"),
            "episode_ended": state.get("episode_ended"),
            "arms": {name: arm_state(arm, rgb=True) for name, arm in state.get("arms", {}).items()},
        }
        if "episode_ended" not in state and features.enabled("hide_end"):   # hide end: the adapter keeps the flag
            del view["episode_ended"]
        reserve = state.get("return_reserve")
        if isinstance(reserve, dict):            # return reserve: the step accounting
            view["budget"] = {"return": reserve.get("return_steps"), "spare": reserve.get("spare_steps"),
                              "releases": reserve.get("releases"), "per_release": reserve.get("steps_per_release"),
                              "more_releases": reserve.get("more_releases")}
        return view
    view = {
        "observation_id": obs.get("observation_id"),
        "sim_time_s": _r(state.get("sim_time_s"), 2),
        "samples_used": state.get("low_level_actions"),
        "episode_ended": state.get("episode_ended"),
        "arms": {name: arm_state(arm) for name, arm in state.get("arms", {}).items()},
    }
    if "episode_ended" not in state and features.enabled("hide_end"):       # hide end (OURS_HIDE_END)
        del view["episode_ended"]
    return view


def _arm_outcome(outcome):
    item = {
        "outcome": outcome.get("motion_outcome"),
        "reached": outcome.get("target_reached"),
        "pos_err_mm": _mm(outcome.get("position_error_m")),
        "samples": outcome.get("trajectory_samples"),
    }
    if outcome.get("rotation_error_rad") is not None:
        item["rot_err_deg"] = round(float(outcome["rotation_error_rad"]) * 57.29578, 2)
    width = outcome.get("ours_grip_width")
    if isinstance(width, dict):          # grip width (OURS_GRIP_WIDTH): the pad gap reached, where the fingers are
        item["gap_mm"] = None if width.get("gap_mm") is None else round(float(width["gap_mm"]), 1)
        item["fingers"] = width.get("fingers")
    if outcome.get("gripper_measured_m") is not None and not features.enabled("rgb_only"):
        item["finger_joints_mm"] = [_mm(v) for v in outcome["gripper_measured_m"]]
    for key in ("stop_reason", "error"):
        if outcome.get(key):
            item[key] = _clip(outcome[key])
    if outcome.get("note"):              # OURS_HOLD_CLEARANCE: one sentence when the path was changed
        item["note"] = _clip(outcome["note"])
    if outcome.get("ours_plan_guard") and outcome.get("reason"):   # plan guard (OURS_PLAN_GUARD): why the plan was
        item["reason"] = _clip(outcome["reason"], 400)               # refused, with its numbers
    if outcome.get("ok") is False:
        item["ok"] = False
    return {k: v for k, v in item.items() if v is not None}


def step(result):
    out = {"ok": result.get("ok"), "action_id": result.get("action_id"), "status": result.get("status")}
    for key in ("error_code", "error"):
        if result.get(key):
            out[key] = _clip(result[key], 600)
    if result.get("replayed"):
        out["replayed"] = True
    action = result.get("action") or {}
    if action:
        out["arms"] = {arm: _arm_outcome(v) for arm, v in action.get("arms", {}).items()}
        out["samples"] = action.get("low_level_actions")
        out["sim_s"] = _r(action.get("sim_duration_s"), 2)
        if action.get("stop_reason"):
            out["stop_reason"] = _clip(action["stop_reason"])
        if action.get("error"):
            out["action_error"] = _clip(action["error"], 600)
        if action.get("episode_ended"):
            out["episode_ended"] = True
    guidance = (result.get("recovery_guidance") or {}).get("arms", {})
    if guidance:
        out["check"] = {arm: g.get("kind") for arm, g in guidance.items()}
    if "observation" in result:
        out["obs"] = observation(result["observation"])
    return out


def measure(result):
    out = {"ok": result.get("ok"), "observation_id": result.get("observation_id")}
    for key in ("error_code", "error"):
        if result.get(key):
            out[key] = _clip(result[key], 600)
    points = []
    for entry in result.get("results", []):
        query = entry.get("query", {})
        item = {"id": entry.get("measurement_id"), "arm": entry.get("arm", query.get("arm")),
                "target": query.get("target"), "cam": query.get("cam"), "uv": [query.get("u"), query.get("v")]}
        detail = entry.get("result") or {}
        if not entry.get("ok"):
            item["ok"] = False
            item["error"] = _clip(entry.get("error") or detail.get("error") or "invalid")
            points.append(item)
            continue
        point = detail.get("point_base") or {}
        item["xyz"] = [_r(point.get(k)) for k in "xyz"]
        if "plane_z" in query:
            item["plane_z"] = query["plane_z"]
        tcp = detail.get("point_tcp")
        if tcp:
            item["tcp_mm"] = [_mm(tcp[k]) for k in "xyz"]
            pads = (detail.get("pad_projection") or {}).get("pads", {})
            relations = {v.get("projection_relation") for v in pads.values()}
            if len(relations) == 1:
                item["pad_yz"] = relations.pop()
        if detail.get("rgb_only"):
            item["z_from"] = (detail.get("height") or {}).get("source")
            if (detail.get("height") or {}).get("older_view"):
                item["older_view"] = True
            if item["z_from"] == "fallback":      # why the given plane_z was kept
                height = detail.get("height") or {}
                item["match"] = height.get("fallback_reason")
                found = (height.get("match") or {}).get("z")
                if found is not None:
                    item["match_z"] = _r(found)
            robot = detail.get("on_robot")
            if robot:
                item["on_robot"] = f"{robot['arm']} {'gripper' if robot['link'] in GRIPPER_LINKS else 'arm'}"
        if entry.get("reused"):
            item["reused"] = True
        points.append(item)
    out["points"] = points
    return out


def generic(result):
    return {k: v for k, v in result.items() if k not in {"images", "calibration", "state"}}
