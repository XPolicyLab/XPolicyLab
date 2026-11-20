"""`run`: execute a short plan of steps server-side so the model writes only decisions.

Every inner action still goes through ControlRuntime.step / measure unchanged
(evidence, measurement validity, journaling, one-unresolved-action lock).
The runner only fills ids, expands shorthands and stops at the first surprise.
Shorthands that are switched by agp_core.features: default clearance for retreat/home
(ev_defaults), candidate pixels per query (candidates) and command strings (dsl).
RGB-only mode (rgb_only): candidates are ranked by height reliability, a measurement
returns only the chosen candidates' match images, and a run stops at a measure step whose height is uncertain.
Further RGB-only rules (observations and measurements without the RGB-only markers, and everything with the switch
off, are unchanged):
  * `gap>N` / `expect` checks read the pad gap estimated from the arm's wrist image (the observation state's
    gripper_gap_estimate) when the state has no measured gap; an estimate whose quality is not ok cannot confirm the
    bound, so the run stops there ("gap not confirmed").
  * Staging re-measure: when a step moves an arm (any intent but contact) to `label+d` and that label's measurement
    is a top-image pick whose raised plane_z was only the fallback (height source 'fallback'), the same query is
    measured again in the observation after the move, i.e. matched in the wrist image taken from there, and the
    label then names the new measurement. The step's view shows it under `remeasured`, its full result under
    `remeasured` as well. Ranking of candidates: a 'fallback' height ranks after 'plane' and 'match', before
    'uncertain'.
P (OURS_PREC_ZONE, opt-in; with it off nothing here changes and agp_core/prec_zone.py is not imported): a step whose contact
and place descents all qualify (agp_core/prec_zone.py: a match or plane measurement, nearly vertical, long enough) first
runs their fast part as its own runtime action "<run id>.<k>.p" (transit, the intent the runtime requires, evidence with
an automatic clearance), then the step as written; prec_zone.jsonl in the session and `prec_zone` in the full result
record every decision, the step's view gets a short `split` entry.
Place split (OURS_PLACE_SPLIT, opt-in, official conditions only; with it off, or with OURS_PREC_ZONE on, nothing here
changes): P's place half only (agp_core/prec_zone.py plan_place_step): a step whose place descents all qualify (a
measurement-relative or relative-displacement place of a holding arm, nearly vertical, long enough, the split saving
steps) first runs their fast part as "<run id>.<k>.p" (transit, intent observe, evidence with an automatic clearance),
then the precision tail (the action as written, or a move_delta's remaining displacement). Invisible to the model: the
fast part's result is not in the run's result, the view gets no `split` entry, the full result no `prec_zone`; a fast
part that does not reach its target stops the run at that step, shown against the model's own targets
(prec_zone.as_written). Every decision goes to prec_zone.jsonl only.
Grasp depth and the place split's rule (with the switches off nothing here changes):
  * Grasp depth (OURS_GRASP_DEPTH, official conditions only): before the P / place-split hooks, every contact descent of
    the step onto a low measurement-relative target may go lower (agp_core/grasp_depth.py apply_step: the fingers' lowest
    point 4 mm above the support measured under the fingertips; at most 3 mm without a match/plane result there, at most
    10 mm, never up). Invisible: the runtime gets the adjusted action (evidence `automatic_depth`), the result is the
    runtime's own, grasp_depth.jsonl records every decision.
  * OURS_PLACE_SPLIT's rule (agp_core/prec_zone.py plan_place_step: no split over a support of uncertain height, a 0.5 cm
    tail onto the referenced surface, the 2 cm tail for a hanging release; the fast part's estimate at the verified
    fast hold's law for a verified grasp, OURS_HOLD_VERIFIED_FAST). With OURS_PRECISION_2X the estimate's unsplit move
    and tail use the scaled precision rate (the switch is passed only when on).
Intent manipulate (OURS_INTENT_MANIPULATE; with the switch off nothing here changes): step strings accept the intent
word `manipulate` (agp_core/syntax.py); its default speed is precision (not a transit intent); the staging re-measure leaves
out a manipulate move as it leaves out a contact (the move may displace what the label measured, and the runtime has
invalidated every measurement). The rules themselves are the runtime's (agp_core/runtime_ext.py); P, the place split and the
grasp depth never act on a manipulate move (they take contact or place descents only).
Still hold (OURS_WAIT_STILL; with the switch off nothing here changes: the word `still` is unknown in a step string and
a "still" key of a JSON hold is ignored, as before): a hold step whose every arm holds with the option `still`
(`B hold N still`) holds up to N steps (1..STILL_MAX) as a series of runtime hold actions of at most STILL_CHUNK steps
("<run id>.<k>", then "<run id>.<k>.w1", ...), each from the latest observation, and ends as soon as the top image has
stopped changing: two comparisons in a row, each of a chunk's top image with the one before it, in which at most
STILL_FRACTION of the pixels differ by more than STILL_LEVEL in some colour channel (RGB frame difference only). A chunk
never asks for more than steps_left. Every chunk passes the runtime unchanged. The step's view is the last chunk's, plus
`waited` (steps held in all chunks) and `still` (true when the scene came to rest); the run stops at this step when the
image was still changing after N steps, when no steps are left, or when the wall budget ran out while waiting. The
intermediate chunks' full results go into the run's full result before the last one.
Grip width (OURS_GRIP_WIDTH; with the switch off nothing here changes: the verb `width` is unknown in a step string and
a JSON {do: "width"} needs an intent and is then an unknown do, as before): step strings accept `width <mm>`
(agp_core/syntax.py) and JSON arms {do: "width", width_mm}; the action is the runtime's gripper {action "width", width_mm}
(agp_core/runtime_ext.py GripWidthMixin), default intent grasp (WIDTH_INTENT); on a width, as on `open`, the evidence key
support=, empty= or landing= selects the release requirement (the runtime asks for it only when the width is a release).
A width that stopped on an object or whose fingers the image could not confirm is a close's outcome, not a stop; one
short of its gap stops the run ("did not reach the requested gap").
Hold frames (OURS_HOLD_FRAMES; with the switch off nothing here changes and agp_core/hold_frames.py is not imported): after
a `still` hold step that held more than hold_frames.EVERY (50) steps in all, the top images its chunks' own observations
already carry at hold steps 50, 100, ... (the nearest one when a mark has none, never one twice; agp_core/hold_frames.py)
are added to the run's images right after that step, in order and before any image shown after the step, each the
captured 640x480 top.png itself, labelled "top at step N of the hold (steps_used M)"; the hold's end is shown as before.
The step's view gets `frames` (the hold steps of those images) and the run's full result `hold_frames` (one record per
such step). A still hold of 50 steps or fewer and every other step are unchanged; a hold through the `step` tool is at
most 25 samples (the runtime's limit) and never gets them.
"""
import json
import math
import time

from agp_core import compact, features, syntax

WALL_BUDGET_S = 240.0
TRANSIT_INTENTS = {"observe", "transfer", "retreat"}
DEFAULT_INTENT = {"close": "grasp", "open": "place", "home": "retreat", "hold": "observe"}
DEFAULT_CLEARANCE = "server default: retreat or home path assumed clear"
RELEASE_KEYS = (("empty", "empty"), ("landing", "drop"), ("support", "support"))
WIDTH_INTENT = "grasp"  # the default intent of a width action (as for a close)
STILL_CHUNK = 10        # steps per runtime hold action of a `still` hold (the runtime allows 1..25)
STILL_MAX = 1000        # the most steps one `still` hold may ask for
STILL_LEVEL = 16        # a pixel differs when some colour channel changed by more than this (0..255)
STILL_FRACTION = 0.0005 # a comparison is static when at most this fraction of the pixels differ
STILL_STATIC = 2        # static comparisons in a row that end the hold


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2)


def rotation(spec, current):
    """Quaternion dict from {w,x,y,z} | {down: psi} | {tilted: [az, tilt]} | None (keep current)."""
    if spec is None:
        return dict(zip("wxyz", current))
    if "down" in spec:
        h = math.radians(spec["down"]) / 2
        return {"w": 0.0, "x": math.cos(h), "y": math.sin(h), "z": 0.0}
    if "tilted" in spec:
        az, tilt = spec["tilted"]
        th, a = math.radians(az), math.radians(tilt) / 2
        h = th / 2 + math.pi / 4
        return {"w": math.sin(a) * math.sin(math.pi / 4 - th / 2), "x": math.cos(a) * math.cos(h),
                "y": math.cos(a) * math.sin(h), "z": math.sin(a) * math.cos(th / 2 - math.pi / 4)}
    return {k: float(spec[k]) for k in "wxyz"}


def default_evidence(command, intent, evidence, flags):
    """Clearance default for retreat moves and home (ev_defaults); never overrides given evidence."""
    if flags["ev_defaults"] and (intent == "retreat" or command == "home") and "clearance" not in evidence:
        evidence = {**evidence, "clearance": DEFAULT_CLEARANCE}
    return evidence


def expand_arm(spec, arm, labels, tcp_quat, shared_evidence, flags):
    """Compact arm spec -> runtime ArmAction dict."""
    kind = spec["do"]
    width = kind == "width" and bool(flags.get("grip_width"))      # the width action (OURS_GRIP_WIDTH)
    intent = spec.get("intent") or DEFAULT_INTENT.get(kind) or (WIDTH_INTENT if width else None)
    if intent is None:
        raise ValueError(f"{arm}: intent is required for {kind}")
    evidence = {**(shared_evidence or {}), **(spec.get("ev") or {})}
    if (kind == "open" or width) and flags["dsl"] and "release_requirement" not in evidence:
        # Which assertion was given selects the release requirement; the assertion itself stays required.
        for key, requirement in RELEASE_KEYS:
            if key in evidence:
                evidence["release_requirement"] = requirement
                break
    options = {k: spec[k] for k in ("max_samples", "settle_samples", "position_tolerance_m",
                                    "rotation_tolerance_rad") if k in spec}
    speed = spec.get("speed") or ("transit" if intent in TRANSIT_INTENTS else "precision")
    action = {"intent": intent}
    measurement = spec.get("m")
    if kind == "ee":
        args = {"rotation": rotation(spec.get("rot"), tcp_quat), "mode": spec.get("mode", "linear"),
                "speed_profile": speed, **options}
        target = spec["to"]
        if isinstance(target, dict):
            ref = target["at"]
            measurement = labels.get(ref, ref)
            args["position_offset"] = dict(zip("xyz", target.get("d", [0, 0, 0])))
        else:
            args["position"] = dict(zip("xyz", target))
        action.update(command="move_ee", args=args)
    elif kind == "delta":
        args = {"dpos": spec["d"], "speed_profile": speed, **options}
        if "drot_deg" in spec:
            args["drot_deg"] = spec["drot_deg"]
        action.update(command="move_delta", args=args)
    elif kind in ("open", "close"):
        action.update(command="gripper", args={"action": kind})
    elif width:                                                     # width action: the runtime checks the range
        if "width_mm" not in spec:
            raise ValueError(f"{arm}: width needs width_mm (the pad gap in mm)")
        action.update(command="gripper", args={"action": "width", "width_mm": spec["width_mm"]})
    elif kind == "home":
        action.update(command="home", args=options)
    elif kind == "hold":
        action.update(command="hold", args={"samples": int(spec.get("samples", 1))})
    elif kind == "joints":
        action.update(command="move_joints", args={"joints": spec["joints"], **options})
    else:
        raise ValueError(f"{arm}: unknown do={kind!r}")
    if isinstance(measurement, str):
        action["measurement_id"] = labels.get(measurement, measurement)
    evidence = default_evidence(action["command"], intent, evidence, flags)
    if evidence:
        action["evidence"] = evidence
    return action


def _normalize_query(query, flags):
    """One query (object or string) -> (base query without pixels, candidate pixels, pick, zr, label, pose)."""
    label = None
    if isinstance(query, str):
        if not flags["dsl"]:
            raise ValueError("query strings are switched off; use query objects")
        query, label = syntax.parse_query(query, pose=bool(flags.get("pose_query")))
    query = dict(query)
    label = query.pop("label", None) or label
    pose = None
    if flags.get("pose_query"):                          # pose query (OURS_POSE_QUERY); off: the field goes on to
        pose = query.pop("pose", None)                   # the runtime, which refuses it
        if pose not in (None, False, True):
            raise ValueError("pose is true or false")
    if pose and query.get("cam") != "top":
        raise ValueError("pose reads the top camera: use cam top")
    cands, pick, zr = query.pop("cands", None), query.pop("pick", None), query.pop("zr", None)
    if (cands is not None and len(cands) > 1 or pick or zr) and not flags["candidates"]:
        raise ValueError("candidate pixels are switched off; give one u,v per query")
    if cands is None:
        if "u" not in query or "v" not in query:
            raise ValueError("a query needs u and v")
        cands = [[query.pop("u"), query.pop("v")]]
    else:
        query.pop("u", None), query.pop("v", None)
    if not 1 <= len(cands) <= 8 or any(not isinstance(c, (list, tuple)) or len(c) != 2 for c in cands):
        raise ValueError("cands is a list of 1 to 8 [u, v] pixels")
    if pick not in (None, "solid", "high", "low"):
        raise ValueError("pick is solid, high or low")
    if zr is not None and (len(zr) != 2 or zr[0] > zr[1]):
        raise ValueError("zr is [zmin, zmax]")
    if not isinstance(query.get("target"), str) or not query["target"].strip():
        raise ValueError("a query needs target text")
    query.setdefault("visibility", query["target"])
    if "plane_z" in query:
        query.setdefault("height_basis", query["target"])
    return query, [[int(u), int(v)] for u, v in cands], pick or "solid", zr, label, bool(pose)


def _rgb_rank(entry):
    """Reliability of an RGB-only result (lower is better): (uncertain height, on the robot, -match margin)."""
    detail = entry.get("result") or {}
    height = detail.get("height") or {}
    margin = height.get("margin")
    source = height.get("source")
    return (0 if source in ("plane", "match") else 1 if source == "fallback" else 2, 1 if detail.get("on_robot") else 0,
            -(margin if margin is not None else 0.0))


def _select(entries, pick, zr):
    """Keep one candidate: valid, inside zr, then the most reliable height (solid, see _rgb_rank) or highest/lowest z."""
    kept, notes = [], []
    for entry in entries:
        query = entry.get("query") or {}
        where = f"{query.get('u')},{query.get('v')}"
        if not entry.get("ok"):
            notes.append(f"{where} invalid")
            continue
        point = (entry.get("result") or {}).get("point_base") or {}
        z = point.get("z")
        if zr and (z is None or not zr[0] <= z <= zr[1]):
            notes.append(f"{where} z={z:.3f} outside zr" if z is not None else f"{where} no z")
            continue
        kept.append((_rgb_rank(entry), z, entry))
    if not kept:
        return None, notes
    order = {"solid": lambda k: k[0], "high": lambda k: -k[1], "low": lambda k: k[1]}[pick]
    return min(kept, key=order)[2], notes


def measure_queries(runtime, observation_id, queries, names, labels, flags):
    """Measure queries (with candidates) -> (full result, compact points, ok, new labels)."""
    groups, flat = [], []
    for i, raw in enumerate(queries):
        base, cands, pick, zr, label, pose = _normalize_query(raw, flags)
        name = label or (names[i] if names and i < len(names) else None)
        groups.append((len(flat), len(cands), pick, zr, name, pose))
        flat += [dict(base, u=u, v=v) for u, v in cands]
    if len(flat) > 32:
        raise ValueError("at most 32 pixels per measurement call")
    full = runtime.measure(observation_id=observation_id, queries=flat)
    results = full.get("results") or []
    if len(results) != len(flat):
        view = compact.measure(full)
        return full, view.get("points", []), False, {}, view.get("error")
    points, ok, new, selection, poses = [], True, {}, [], []
    for start, count, pick, zr, name, pose in groups:
        entries = results[start:start + count]
        if count == 1:
            chosen, notes = (entries[0] if entries[0].get("ok") else None), []
            view = compact.measure({"results": entries})["points"][0]
        else:
            chosen, notes = _select(entries, pick, zr)
            view = (compact.measure({"results": [chosen]})["points"][0] if chosen else
                    {"ok": False, "target": entries[0].get("query", {}).get("target"),
                     "error": "no candidate kept: " + "; ".join(notes)})
            view["of"] = count
        if chosen is None:
            ok = False
        elif name:
            labels[name] = new[name] = chosen["measurement_id"]
            view["as"] = name
        selection.append({"chosen": chosen and chosen.get("measurement_id"), "pick": pick, "zr": zr,
                          "label": name, "notes": notes})
        if pose:                                  # pose query: the footprint at the kept point
            found = _pose(runtime, observation_id, chosen)
            poses.append(dict(found, measurement_id=chosen and chosen.get("measurement_id")))
            view["pose"] = compact_pose(found)
        points.append(view)
    full = dict(full, selection=selection, selected_ok=ok)
    if flags.get("rgb_only"):     # one review image per kept point, not one per candidate
        kept = {item["chosen"] for item in selection if item["chosen"]}
        full["images"] = [image for entry in results if entry.get("measurement_id") in kept
                          for image in (entry.get("result") or {}).get("images", [])]
    if poses:                     # pose query: the records and one review image per footprint
        full["poses"] = poses
        full["images"] = list(full.get("images") or []) + [p["image"] for p in poses if p.get("image")]
    return full, points, ok, new, None


def _pose(runtime, observation_id, chosen):
    """Pose query (OURS_POSE_QUERY): agp_core/rgb_pose.py's footprint at a kept measurement, at its measured height."""
    if not chosen or not chosen.get("ok"):
        return {"status": "error", "why": ["no valid measurement to start from"]}
    point = (chosen.get("result") or {}).get("point_base") or {}
    if point.get("z") is None:
        return {"status": "error", "why": ["the measurement has no height"]}
    from agp_core import rgb_pose
    try:
        capture = runtime._load_observation(observation_id)["capture"]
        return rgb_pose.pose(runtime.adapter.session, capture, chosen["query"], point["z"])
    except (KeyError, ValueError, OSError) as exc:
        return {"status": "error", "why": [str(exc)]}


def compact_pose(found):
    """The agent's view of a footprint (OURS_POSE_QUERY)."""
    if found.get("status") == "error":
        return {"status": "error", "why": found.get("why")}
    out = {"status": found["status"], "xy": [round(v, 4) for v in found["xy"]],
           "axis_deg": round(found["axis_deg"], 1), "tail_deg": round(found["tail_deg"], 1),
           "length_mm": round(found["length_mm"]), "width_mm": round(found["width_mm"])}
    if found.get("edge_deg") is not None:
        out["edge_deg"] = round(found["edge_deg"], 1)
    if found.get("why"):
        out["why"] = found["why"]
    if found.get("notes"):
        out["notes"] = found["notes"]
    return out


def execute(runtime, observation_id, steps, show=None, evidence=None, labels=None, flags=None):
    started = time.monotonic()
    flags = features.current() if flags is None else flags
    labels = {} if labels is None else labels
    results, shown_images, view_steps, new_labels = [], [], [], {}
    obs_id = observation_id
    try:
        if show is not None and not isinstance(show, dict) and not flags["dsl"]:
            raise ValueError("show strings are switched off; use {at, end, on_stop}")
        show = syntax.parse_show(show)
        try:
            obs_full = runtime._load_observation(observation_id)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{exc} (observation_id must be the latest one)") from exc
    except (ValueError, TypeError) as exc:
        return ({"ok": False, "error": str(exc), "images": []},
                {"ok": False, "done": 0, "stopped": {"at": 0, "reason": str(exc)}, "steps": []})
    stop = None
    zone_records = []                           # P: this run's split decisions (prec_zone on only)
    frame_records = []                          # hold frames: this run's records (hold_frames on only)

    def zone_record(entry):
        zone_records.append(entry)
        prec_zone.record(runtime.session, entry)

    for k, step in enumerate(steps):
        if time.monotonic() - started > WALL_BUDGET_S:
            stop = {"at": k, "reason": "wall budget reached before this step"}
            break
        try:
            if isinstance(step, str):
                if not flags["dsl"]:
                    raise ValueError("step strings are switched off; use step objects")
                pose = bool(flags.get("pose_query"))  # pose query: the query option `pose` too
                if flags.get("grip_width"):           # grip width: the verb `width <mm>` too
                    step = syntax.parse_step(step, bool(flags.get("intent_manipulate")), bool(flags.get("wait_still")),
                                             width=True, pose=pose)
                else:
                    step = (syntax.parse_step(step, bool(flags.get("intent_manipulate")), True, pose=pose)
                            if flags.get("wait_still") else
                            syntax.parse_step(step, bool(flags.get("intent_manipulate")), pose=pose))
            if not isinstance(step, dict):
                raise ValueError("a step is an object or a command string")
        except ValueError as exc:
            stop = {"at": k, "reason": f"invalid step: {exc}"}
            break
        if "measure" in step:
            try:
                full, points, ok, new, error = measure_queries(runtime, obs_id, step["measure"], step.get("as"),
                                                               labels, flags)
            except (KeyError, ValueError, TypeError) as exc:
                stop = {"at": k, "reason": f"invalid step: {exc}"}
                break
            results.append(full)
            new_labels.update(new)
            view_steps.append({"k": k, "measure": points, **({"error": error} if error else {})})
            if not ok:
                stop = {"at": k, "reason": "measurement failed"}
                break
            uncertain = [str(p.get("as") or p.get("id")) for p in points if p.get("z_from") == "uncertain"]
            if flags.get("rgb_only") and uncertain:
                shown_images += [dict(img, label=f"{img['label']} at step {k}") for img in full.get("images", [])]
                stop = {"at": k, "reason": f"height uncertain for {', '.join(uncertain)}: check the match image, then "
                                           "measure again from a wrist view close above the target or give plane_z"}
                break
            continue
        state_arms = (obs_full.get("state") or {}).get("arms", {})
        actions = {}
        try:
            for arm in ("left", "right"):
                if arm in step:
                    quat = state_arms.get(arm, {}).get("tcp_pose_base", [0, 0, 0, 1, 0, 0, 0])[3:]
                    actions[arm] = expand_arm(step[arm], arm, labels, quat, evidence, flags)
            if not actions:
                raise ValueError("a step moves at least one arm")
            still = _still_samples(step) if flags.get("wait_still") else None     # still hold (OURS_WAIT_STILL)
        except (KeyError, ValueError, TypeError) as exc:
            stop = {"at": k, "reason": f"invalid step: {exc}"}
            break
        restage = _restage_plan(runtime, step, actions, labels, flags)     # staging re-measure (rgb_only only)
        if flags.get("grasp_depth") and flags.get("rgb_only"):    # grasp depth (OURS_GRASP_DEPTH): invisible
            from agp_core import grasp_depth
            actions, _ = grasp_depth.apply_step(runtime, actions, obs_full, observation_id, k)
        split = approach = None
        if flags.get("prec_zone"):              # P: a contact / place descent may run as two actions
            from agp_core import prec_zone
            split = prec_zone.plan_step(runtime, actions, obs_full)
            if split is not None and not split["split"]:
                zone_record({"run_observation_id": observation_id, "k": k,
                                                   "decision": "not split", "arms": split["arms"],
                                                   "reasons": split["reasons"]})
            elif split is not None:
                approach = runtime.step(action_id=f"{observation_id}.{k}.p", observation_id=obs_id,
                                        actions=split["fast"])
                verdict = prec_zone.approach_verdict(approach)
                split_record = {"run_observation_id": observation_id, "k": k, "decision": "split",
                                "arms": split["arms"], "fast": split["fast"], "approach": prec_zone.outcome(approach),
                                "verdict": verdict}
                if verdict == "refused":            # nothing moved: the step runs as written
                    zone_record(dict(split_record, decision="split refused by the runtime "
                                                           "before dispatch; the step ran as written"))
                    approach = None
                else:
                    results.append(approach)
                    aobs = approach.get("observation") or {}
                    if aobs.get("ok") and aobs.get("observation_id"):
                        obs_id, obs_full = aobs["observation_id"], aobs
                    if verdict != "reached" or (aobs.get("state") or {}).get("episode_ended"):
                        zone_record(split_record)
                        view = compact.step(approach)
                        view.pop("obs", None)
                        view_steps.append({"k": k, **view, "split": prec_zone.view(split, approach, None)})
                        stop = {"at": k, "reason": ("episode_ended" if (aobs.get("state") or {}).get("episode_ended")
                                                    else "the automatic transit approach of this step (prec_zone) "
                                                         "did not reach its target"
                                                    if approach.get("status") == "done"
                                                    else approach.get("error") or "approach not ok")}
                        labels_here = show.get("on_stop", ["top", "left_wrist", "right_wrist"])
                        shown_images += [dict(img, label=f"{img['label']} after step {k}")
                                         for img in aobs.get("images", []) if img.get("label") in labels_here]
                        break
        place = place_fast = None                # place split (OURS_PLACE_SPLIT): P's place half, invisible
        if not flags.get("prec_zone") and flags.get("place_split") and flags.get("rgb_only"):
            from agp_core import prec_zone
            p2x = {"precision_2x": True} if flags.get("precision_2x") else {}   # OURS_PRECISION_2X: the scaled rate
            place = prec_zone.plan_place_step(runtime, actions, obs_full, bool(flags.get("hold_profile")),
                                              bool(flags.get("hold_profile") and flags.get("hold_verified_fast")),
                                              **p2x)
            if place is not None and not place["split"]:
                prec_zone.record(runtime.session, {"mode": prec_zone.PLACE_SPLIT, "rule": prec_zone.PLACE_RULE,
                                                   "run_observation_id": observation_id,
                                                   "k": k, "decision": "not split", "arms": place["arms"],
                                                   "reasons": place["reasons"],
                                                   **({"estimate": place["estimate"]} if "estimate" in place else {})})
                place = None
            elif place is not None:
                place_fast = runtime.step(action_id=f"{observation_id}.{k}.p", observation_id=obs_id,
                                          actions=place["fast"])
                verdict = prec_zone.approach_verdict(place_fast)
                place_record = {"mode": prec_zone.PLACE_SPLIT, "rule": prec_zone.PLACE_RULE,
                                "run_observation_id": observation_id, "k": k,
                                "decision": "split", "arms": place["arms"], "estimate": place["estimate"],
                                "fast": place["fast"], "approach": prec_zone.outcome(place_fast), "verdict": verdict}
                if verdict == "refused":            # nothing moved: the step runs as written
                    prec_zone.record(runtime.session, dict(place_record, decision="split refused by the runtime "
                                                                                  "before dispatch; the step ran as "
                                                                                  "written"))
                    place = place_fast = None
                else:
                    pobs = place_fast.get("observation") or {}
                    if pobs.get("ok") and pobs.get("observation_id"):
                        obs_id, obs_full = pobs["observation_id"], pobs
                    tails = None
                    if verdict == "reached" and not (pobs.get("state") or {}).get("episode_ended"):
                        tails = prec_zone.place_tail_actions(place, actions, place_fast)
                    if tails is None:               # stop here, the fast part shown as the model's own step
                        shown = prec_zone.as_written(place_fast, place)
                        if callable(getattr(runtime, "_recovery_guidance", None)):
                            prec_zone.restate_guidance(shown, actions, runtime._recovery_guidance)
                        prec_zone.record(runtime.session, dict(place_record, stopped=True, shown={
                            arm: {key: o.get(key) for key in ("target_reached", "motion_outcome", "position_error_m")}
                            for arm, o in ((shown.get("action") or {}).get("arms") or {}).items()}))
                        results.append(shown)
                        view = compact.step(shown)
                        view.pop("obs", None)
                        view_steps.append({"k": k, **view})
                        reason = None
                        if not shown.get("ok") or shown.get("status") != "done":
                            reason = shown.get("error") or "step not ok"
                        elif (pobs.get("state") or {}).get("episode_ended"):
                            reason = "episode_ended"
                        else:
                            for arm in actions:
                                outcome = ((shown.get("action") or {}).get("arms") or {}).get(arm, {})
                                if outcome.get("error") or outcome.get("target_reached") is False:
                                    reason = f"{arm} did not reach its target"
                        stop = {"at": k, "reason": reason or f"{next(iter(actions))} did not reach its target"}
                        labels_here = show.get("on_stop", ["top", "left_wrist", "right_wrist"])
                        shown_images += [dict(img, label=f"{img['label']} after step {k}")
                                         for img in pobs.get("images", []) if img.get("label") in labels_here]
                        break
                    place_record["tail_actions"] = tails
                    actions = tails
        waited = frames = None
        if still is not None:                   # still hold: hold until the top image stops changing
            full, earlier, waited = hold_still(runtime, observation_id, k, obs_id, obs_full, actions, still, started)
            results += earlier
            if flags.get("hold_frames"):        # hold frames: the top images every 50 steps of the hold
                from agp_core import hold_frames
                images, frames = hold_frames.build(earlier + [full], k)
                if frames is not None:
                    shown_images += images
                    frame_records.append(frames)
        else:
            full = runtime.step(action_id=f"{observation_id}.{k}", observation_id=obs_id, actions=actions)
        results.append(full)
        view = compact.step(full)
        view.pop("obs", None)
        view_steps.append({"k": k, **view})
        if waited is not None:
            view_steps[-1].update(waited=waited["waited"], still=waited["still"])
        if frames is not None:                  # hold frames: the hold steps of the attached top images
            view_steps[-1]["frames"] = [frame["step"] for frame in frames["frames"]]
        if approach is not None:                # P: the split's record, with both actions' outcomes
            view_steps[-1]["split"] = prec_zone.view(split, approach, full)
            zone_record(dict(split_record, tail=prec_zone.outcome(full),
                                                   precision_m=prec_zone.precision_left(split, approach)))
        if place_fast is not None:              # place split: the record only (prec_zone.jsonl); nothing in the results
            prec_zone.record(runtime.session, dict(
                place_record, tail=prec_zone.outcome(full), precision_m=prec_zone.precision_left(place, place_fast),
                steps={"fast": (place_fast.get("action") or {}).get("low_level_actions"),
                       "tail": (full.get("action") or {}).get("low_level_actions")}))
        obs = full.get("observation") or {}
        if obs.get("ok") and obs.get("observation_id"):
            obs_id, obs_full = obs["observation_id"], obs
        labels_here = show.get("at", {}).get(str(k)) or show.get("at", {}).get(k)
        reason = None
        if not full.get("ok") or full.get("status") != "done":
            reason = full.get("error") or "step not ok"
        elif (obs.get("state") or {}).get("episode_ended"):
            reason = "episode_ended"
        else:
            for arm, spec in step.items():
                if arm not in ("left", "right"):
                    continue
                outcome = ((full.get("action") or {}).get("arms") or {}).get(arm, {})
                closing = spec["do"] == "close"
                if spec["do"] == "width" and flags.get("grip_width"):     # width action: a stop on an object (or
                    closing = ((outcome.get("ours_grip_width") or {}).get("state")     # unconfirmed fingers) is a
                               in ("stopped_on_object", "unconfirmed"))                # close's outcome, not a stop
                if outcome.get("error") or (not closing and outcome.get("target_reached") is False):
                    reason = f"{arm} did not reach its target"
                    if spec["do"] == "width" and flags.get("grip_width"):
                        reason = f"{arm} did not reach the requested gap"
                gap = _gap(full, arm)
                expect = (step.get("expect") or {}).get(arm) or {}
                if gap is None and ("gap_min_mm" in expect or "gap_max_mm" in expect):
                    gap, quality = _gap_estimate(full, arm)          # rgb_only observations only
                    if gap is None and quality is not None:
                        reason = f"{arm} gap not confirmed: its wrist-image estimate has quality {quality}"
                if gap is not None and "gap_min_mm" in expect and gap < expect["gap_min_mm"]:
                    reason = f"{arm} gap {gap:.1f} mm below {expect['gap_min_mm']}"
                if gap is not None and "gap_max_mm" in expect and gap > expect["gap_max_mm"]:
                    reason = f"{arm} gap {gap:.1f} mm above {expect['gap_max_mm']}"
            if waited is not None and not reason and not waited["still"]:
                reason = waited["reason"]
        if reason:
            stop = {"at": k, "reason": reason}
            labels_here = show.get("on_stop", ["top", "left_wrist", "right_wrist"])
        if labels_here:
            shown_images += [dict(img, label=f"{img['label']} after step {k}") for img in obs.get("images", [])
                             if img.get("label") in labels_here]
        if stop:
            break
        if restage:                       # staging re-measure: the staged view may confirm a fallback height
            done_full, done_points = [], []
            for arm, label, query in restage:
                try:
                    mfull, points, _, new, _ = measure_queries(runtime, obs_id, [query], [label], labels, flags)
                except (KeyError, ValueError, TypeError) as exc:
                    mfull, points, new = {"ok": False, "error": str(exc)}, [{"as": label, "ok": False,
                                                                              "error": str(exc)}], {}
                new_labels.update(new)
                done_full.append(mfull)
                done_points += points
            results[-1] = dict(results[-1], remeasured=done_full)
            view_steps[-1]["remeasured"] = done_points
    if not stop and show.get("end") and obs_full is not None and obs_full.get("images"):
        shown_images += [dict(img, label=f"{img['label']} at end") for img in obs_full["images"]
                         if img.get("label") in show["end"] and img not in shown_images]
    full_result = {"ok": stop is None, "run_observation_id": observation_id,
                   "done": len(view_steps) if stop is None else stop["at"], "stopped": stop, "steps": results,
                   "images": shown_images, "labels": new_labels, "seconds": round(time.monotonic() - started, 1)}
    if zone_records:
        full_result["prec_zone"] = zone_records
    if frame_records:
        full_result["hold_frames"] = frame_records
    view = {"ok": stop is None, "done": len(view_steps) if stop is None else stop["at"], "stopped": stop,
            "steps": view_steps, "labels": new_labels, "obs": compact.observation(obs_full),
            "seconds": round(time.monotonic() - started, 1)}
    return full_result, view


def _still_samples(step):
    """The most steps of a `still` hold step (every arm holds with still and the same samples), or None
    for any other step; ValueError for a step that mixes a still hold with anything else."""
    arms = [step[arm] for arm in ("left", "right") if arm in step]
    if not any(isinstance(spec, dict) and spec.get("still") for spec in arms):
        return None
    if not all(isinstance(spec, dict) and spec.get("do") == "hold" and spec.get("still") for spec in arms):
        raise ValueError("still holds every arm of the step: write B hold N still")
    counts = {spec.get("samples") for spec in arms}
    if len(counts) != 1 or not isinstance(next(iter(counts)), int) or not 1 <= next(iter(counts)) <= STILL_MAX:
        raise ValueError(f"a still hold needs the most steps to hold, one integer 1..{STILL_MAX} for every arm "
                         "(B hold N still)")
    return next(iter(counts))


def _top_pixels(observation):
    """The observation's top image as an int16 array (height, width, 3), or None when it has none or it cannot be read
    (no comparison: the hold goes on). The path is read as the runtime checked it (ControlRuntime._observation:
    Path(path).resolve(), inside the session)."""
    import numpy as np
    from pathlib import Path
    from PIL import Image
    for image in (observation or {}).get("images") or []:
        if image.get("label") == "top" and image.get("path"):
            try:
                with Image.open(Path(image["path"]).resolve()) as decoded:
                    return np.asarray(decoded.convert("RGB"), dtype=np.int16)
            except (OSError, ValueError):
                return None
    return None


def changed_fraction(before, after):
    """Fraction of the pixels in which some colour channel differs by more than STILL_LEVEL (None: not comparable)."""
    if before is None or after is None or before.shape != after.shape:
        return None
    return float(((abs(after - before)).max(axis=2) > STILL_LEVEL).mean())


def hold_still(runtime, observation_id, k, obs_id, obs_full, actions, most, started):
    """Hold the step's arms in chunks of at most STILL_CHUNK steps until the top image is static in
    STILL_STATIC comparisons in a row, `most` steps are held, no steps are left, a chunk does not complete or the run's
    wall budget runs out. Returns (last chunk's full result, earlier chunks' full results, {waited, still, reason})."""
    earlier, full, held, static, chunk_index = [], None, 0, 0, 0
    previous = _top_pixels(obs_full)
    reason = None
    while True:
        left = ((obs_full or {}).get("state") or {}).get("remaining_actions")
        size = min(STILL_CHUNK, most - held, left if isinstance(left, int) else STILL_CHUNK)
        if size < 1:
            reason = ("no steps left to wait" if isinstance(left, int) and left < 1 else None)
            break
        if full is not None:
            if time.monotonic() - started > WALL_BUDGET_S:
                reason = "wall budget reached while waiting for the scene to come to rest"
                break
            earlier.append(full)
        chunk = {arm: dict(action, args=dict(action.get("args") or {}, samples=size)) for arm, action in actions.items()}
        name = f"{observation_id}.{k}" if chunk_index == 0 else f"{observation_id}.{k}.w{chunk_index}"
        full = runtime.step(action_id=name, observation_id=obs_id, actions=chunk)
        chunk_index += 1
        held += int((full.get("action") or {}).get("low_level_actions") or 0)
        obs = full.get("observation") or {}
        if not full.get("ok") or full.get("status") != "done" or not obs.get("ok") or not obs.get("observation_id") \
                or (obs.get("state") or {}).get("episode_ended"):
            break
        obs_id, obs_full = obs["observation_id"], obs
        current = _top_pixels(obs)
        fraction = changed_fraction(previous, current)
        static = static + 1 if fraction is not None and fraction <= STILL_FRACTION else 0
        previous = current
        if static >= STILL_STATIC:
            break
    if full is None:                             # no steps left before the first chunk: nothing is sent
        full = {"ok": False, "status": "rejected", "error": reason or "no steps left to wait"}
    still = static >= STILL_STATIC
    if not still and reason is None:
        reason = f"the top image was still changing after {held} held steps"
    return full, earlier, {"waited": held, "still": still, "reason": reason}


def _gap(result, arm):
    arms = ((result.get("observation") or {}).get("state") or {}).get("arms", {})
    gap = (arms.get(arm, {}).get("gripper_geometry") or {}).get("nominal_pad_gap_m")
    return None if gap is None else gap * 1000


def _gap_estimate(result, arm):
    """(gap in mm, "ok") from an rgb_only observation's wrist-image estimate when its quality is ok,
    (None, quality) when it is not, (None, None) when the observation has no estimate (not rgb_only)."""
    state = (result.get("observation") or {}).get("state") or {}
    if state.get("observation") != "rgb_only":
        return None, None
    estimate = (state.get("arms", {}).get(arm) or {}).get("gripper_gap_estimate")
    if not isinstance(estimate, dict):
        return None, None
    if estimate.get("quality") == "ok" and estimate.get("pad_gap_m") is not None:
        return float(estimate["pad_gap_m"]) * 1000, "ok"
    return None, estimate.get("quality") or "unavailable"


def _measurement(runtime, measurement_id):
    """The journaled measurement entry (read only), or None (unknown id, or a runtime without a journal)."""
    try:
        row = runtime.db.execute("SELECT value FROM measurements WHERE id=?",
                                 (int(str(measurement_id)[1:]),)).fetchone()
    except Exception:            # noqa: BLE001 - no journal, or one that cannot be read: no re-measure
        return None
    return json.loads(row[0]) if row else None


def _restage_plan(runtime, step, actions, labels, flags):
    """Staging re-measure (rgb_only): [(arm, label, query)] to measure again after this motion step: every arm that
    moves with intent other than contact or manipulate to `label+d`, where the label's measurement is that arm's
    top-image pick whose raised plane_z was only the fallback."""
    if not flags.get("rgb_only"):
        return []
    plan = []
    for arm, action in actions.items():
        spec = step.get(arm) or {}
        target = spec.get("to")
        if (spec.get("do") != "ee" or not isinstance(target, dict)
                or action.get("intent") in ("contact", "manipulate")):
            continue
        label = target.get("at")
        if label not in labels:
            continue
        entry = _measurement(runtime, labels[label])
        if not entry or not entry.get("ok") or entry.get("arm") != arm:
            continue
        query = entry.get("query") or {}
        height = (entry.get("result") or {}).get("height") or {}
        if query.get("cam") == "top" and height.get("source") == "fallback":
            plan.append((arm, label, dict(query)))
    return plan
