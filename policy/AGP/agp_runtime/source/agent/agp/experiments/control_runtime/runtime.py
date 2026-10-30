"""Single-action control transactions and reusable visual evidence for Astra."""
from __future__ import annotations

from copy import deepcopy
import fcntl
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
from functools import wraps

from agp.experiments.control_runtime.errors import NotDispatched


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def serial(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return wrapped


def error(code, detail):
    return {"ok": False, "error_code": code, "error": str(detail), "images": []}


class ControlRuntime:
    """Persist before dispatch, collect fresh evidence after every known outcome.

    Semantic assertions such as support or target identity remain Astra's visual
    judgements. This class checks their provenance, not their visual truth.
    """

    def __init__(self, session, adapter):
        self.session = Path(session).resolve()
        self.adapter = adapter
        self.dual_arm = bool(getattr(adapter, "dual_arm", False))
        self.directory = self.session / "control_runtime"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.owner = (self.directory / "owner.lock").open("a")
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.owner.close()
            raise RuntimeError("Another control runtime owns this session") from None
        self.db = sqlite3.connect(self.directory / "journal.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS observations (id INTEGER PRIMARY KEY, revision INTEGER, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS measurements (id INTEGER PRIMARY KEY, observation_id TEXT, query TEXT, value TEXT, valid INTEGER);
            CREATE TABLE IF NOT EXISTS actions (id TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL, response TEXT, result TEXT, prior_state TEXT);
            CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, monotonic_ns INTEGER, kind TEXT, value TEXT);
        """)
        row = self.db.execute("SELECT value FROM state WHERE id=1").fetchone()
        self.state = json.loads(row[0]) if row else {"revision": 0, "observation_id": None,
            "grasp_epoch": 0, "grasp_active": False, "post_grasp_motion_seen": False, "target_confirmation": None}
        if self.dual_arm:
            self.state.setdefault("arms", {arm: {"grasp_epoch": 0, "grasp_active": False,
                "post_grasp_motion_seen": False, "target_confirmation": None} for arm in ("left", "right")})
        task = self.session / "control_task.json"
        self.task = json.loads(task.read_text()) if task.exists() else {"release_requirement": "support"}
        if self.task.get("release_requirement") not in {"drop", "support", "engagement_and_support"}:
            self.close()
            raise ValueError("Task release_requirement must be drop, support or engagement_and_support")
        # Restart discards live-scene assumptions while preserving action outcomes.
        self.state["observation_id"] = None
        self.state["target_confirmation"] = None
        for state in self.state.get("arms", {}).values():
            state["target_confirmation"] = None
        with self.db:
            self.db.execute("UPDATE measurements SET valid=0")
            self._save_state()

    def close(self):
        if getattr(self, "db", None) is not None:
            self.db.close()
            self.db = None
        if not self.owner.closed:
            self.owner.close()

    def _save_state(self):
        self.db.execute("INSERT OR REPLACE INTO state VALUES(1,?)", (encoded(self.state),))

    def _event(self, kind, value):
        self.db.execute("INSERT INTO events(monotonic_ns,kind,value) VALUES(?,?,?)",
                        (time.monotonic_ns(), kind, encoded(value)))

    def _fresh(self, observation_id):
        return bool(observation_id and observation_id == self.state["observation_id"])

    def _observation(self, *, after_ns=0):
        self.state["observation_id"] = None
        result = error("OBSERVATION_FAILED", "No valid capture")
        # Read-only bounded acquisition. No motion is ever retried here.
        for _ in range(3):
            try:
                raw = self.adapter.observe()
            except Exception as exc:
                result = error("OBSERVATION_FAILED", exc)
                break
            if not raw.get("ok"):
                result = dict(raw, images=[])
                break
            images = raw.get("images", [])
            times = raw.get("camera_times", {})
            now_ns = time.monotonic_ns()
            required = set(getattr(self.adapter, "required_cameras", ()))
            labels = [image.get("label") for image in images]
            valid = bool(images) and required.issubset(labels) and len(set(labels)) == len(labels) and all(
                type(times.get(image.get("label"))) is int
                and times[image["label"]] > after_ns
                and -5_000_000 <= now_ns - times[image["label"]] <= 500_000_000
                for image in images)
            if not valid:
                result = error("STALE_CAMERA", "Each returned camera needs a measured timestamp after the action response")
                continue
            try:
                for image in images:
                    path = Path(image["path"]).resolve()
                    path.relative_to(self.session)
                    if not path.is_file():
                        raise ValueError("Capture image is absent")
            except (ValueError, KeyError) as exc:
                result = error("OBSERVATION_FAILED", exc)
                break
            result = dict(raw)
            result.update(revision=self.state["revision"], recorded_monotonic_ns=time.monotonic_ns())
            with self.db:
                cur = self.db.execute("INSERT INTO observations(revision,value) VALUES(?,?)",
                                      (self.state["revision"], encoded(result)))
                result["observation_id"] = f"o{cur.lastrowid}"
                self.db.execute("UPDATE observations SET value=? WHERE id=?", (encoded(result), cur.lastrowid))
                self.state["observation_id"] = result["observation_id"]
                self._save_state()
                self._event("observation", result)
            return result
        with self.db:
            self._save_state()
            self._event("observation_failed", result)
        return result

    @serial
    def observe(self):
        return self._observation()

    def _load_observation(self, observation_id):
        if not self._fresh(observation_id):
            raise ValueError("Use the latest successful observation after the last action")
        row = self.db.execute("SELECT value FROM observations WHERE id=?", (int(observation_id[1:]),)).fetchone()
        return json.loads(row[0])

    @serial
    def measure(self, observation_id, queries):
        if self.dual_arm and self.db.execute("SELECT 1 FROM actions WHERE status NOT IN ('done','rejected') LIMIT 1").fetchone():
            return error("UNRESOLVED_ACTION", "Resolve both arms before creating new geometry evidence")
        try:
            obs = self._load_observation(observation_id)
            if not isinstance(queries, list) or not 1 <= len(queries) <= 32:
                raise ValueError("Provide 1 to 32 queries")
        except (ValueError, TypeError) as exc:
            return error("INVALID_MEASUREMENT", exc)
        results, images = [], []
        for query in queries:
            try:
                if not isinstance(query, dict) or query.keys() - {"arm", "target", "cam", "u", "v", "plane_z", "height_basis", "visibility"}:
                    raise ValueError("Unexpected query fields")
                for field in ("target", "visibility"):
                    self._text(query, field)
                arm = query.get("arm")
                if self.dual_arm and arm not in {"left", "right"}:
                    raise ValueError("Measurement requires arm left or right")
                if not self.dual_arm and arm is not None:
                    raise ValueError("Arm measurements require a dual-arm adapter")
                arm_state = self._arm_state(arm)
                cameras = {"top", "left_wrist", "right_wrist"} if self.dual_arm else {"top", "wrist"}
                if query.get("cam") not in cameras or any(type(query.get(k)) is not int for k in ("u", "v")):
                    raise ValueError("Use a configured camera and integer u,v")
                if query.get("plane_z") is not None:
                    self._numbers([query["plane_z"]], 1)
                    self._text(query, "height_basis")
                key = encoded(query)
                old = self.db.execute("SELECT value FROM measurements WHERE observation_id=? AND query=?",
                                      (observation_id, key)).fetchone()
                if old:
                    entry = dict(json.loads(old[0]), reused=True)
                else:
                    geometry_query = {k: query[k] for k in ("arm", "cam", "u", "v", "plane_z") if k in query}
                    measured = self.adapter.measure(obs["capture"], geometry_query)
                    entry = {"ok": bool(measured.get("ok")), "status": "valid" if measured.get("ok") else "invalid",
                        "target": query["target"], "observation_id": observation_id, "query": query,
                        "grasp_epoch": arm_state["grasp_epoch"], "revision": self.state["revision"],
                        "after_post_grasp_motion": arm_state["post_grasp_motion_seen"],
                        "surface_identity": "model_assertion_unverified",
                        "clearance_semantics": "post_grasp_motion_seen_not_geometric_verification", "result": measured}
                    if arm is not None:
                        entry.update(arm=arm, frame=f"{arm}_base")
                    with self.db:
                        cur = self.db.execute("INSERT INTO measurements(observation_id,query,value,valid) VALUES(?,?,?,?)",
                            (observation_id, key, encoded(entry), int(entry["ok"])))
                        entry["measurement_id"] = f"m{cur.lastrowid}"
                        self.db.execute("UPDATE measurements SET value=? WHERE id=?", (encoded(entry), cur.lastrowid))
                        self._event("measurement", entry)
                results.append(entry)
                detail = entry["result"]
                images.extend(detail.get("images", []))
            except Exception as exc:
                results.append({"ok": False, "status": "invalid", "query": query, "error": str(exc)})
        return {"ok": all(r["ok"] for r in results), "completed": True,
                "observation_id": observation_id, "results": results, "images": images}

    def _arm_state(self, arm=None):
        return self.state if arm is None else self.state["arms"][arm]

    def _check_measurement_arm(self, measurement_id, arm):
        if arm is None or measurement_id is None:
            return
        if not isinstance(measurement_id, str) or not re.fullmatch(r"m[1-9][0-9]*", measurement_id):
            raise ValueError("Invalid measurement_id")
        row = self.db.execute("SELECT value FROM measurements WHERE id=?", (int(measurement_id[1:]),)).fetchone()
        if not row or json.loads(row[0]).get("arm") != arm:
            raise ValueError("Measurement belongs to a different arm or is absent")

    @property
    def supports_estimate(self):
        return (self.dual_arm and getattr(self.adapter, "supports_estimate", False) is True
                and callable(getattr(self.adapter, "estimate", None)))

    @serial
    def estimate(self, observation_id, stages, preview=None):
        """Read-only candidate costs. No grasp transitions or action reservation."""
        if not self.supports_estimate:
            return error("UNSUPPORTED_OPERATION", "Adapter does not support planning estimates")
        if self.db.execute("SELECT 1 FROM actions WHERE status NOT IN ('done','rejected') LIMIT 1").fetchone():
            return error("UNRESOLVED_ACTION", "Resolve the pending action before estimating")
        if not self._fresh(observation_id):
            return error("STALE_OBSERVATION", "Use the latest successful observation")
        try:
            if not isinstance(stages, list) or not 1 <= len(stages) <= 12:
                raise ValueError("Provide 1 to 12 ordered stages")
            if preview is not None:
                if (not isinstance(preview, dict) or set(preview) != {"camera", "samples"}
                        or preview["camera"] not in ("top", "left_wrist", "right_wrist")
                        or not isinstance(preview["samples"], list)
                        or not 1 <= len(preview["samples"]) <= 3
                        or any(type(k) is not int or k < 0 for k in preview["samples"])
                        or len(set(preview["samples"])) != len(preview["samples"])):
                    raise ValueError("Preview requires one camera and 1..3 unique nonnegative integer samples")
                preview = dict(camera=preview["camera"], samples=sorted(preview["samples"]))
            resolved_stages = []
            for stage in deepcopy(stages):
                if not isinstance(stage, dict) or not stage or stage.keys() - {"left", "right"}:
                    raise ValueError("Each stage requires left and/or right")
                resolved = {}
                for arm, spec in stage.items():
                    if (not isinstance(spec, dict) or spec.keys() - {"command", "args", "measurement_id"}
                            or not {"command", "args"} <= spec.keys()):
                        raise ValueError("Estimate arm requires command, args and optional measurement_id")
                    command, args = spec["command"], spec["args"]
                    self._validate_command(command, args)
                    measurement_id = spec.get("measurement_id")
                    if measurement_id is not None:
                        self._check_measurement_arm(measurement_id, arm)
                        row = self.db.execute("SELECT valid FROM measurements WHERE id=?",
                                              (int(measurement_id[1:]),)).fetchone()
                        if not row or not row["valid"]:
                            raise ValueError("Estimate measurement is invalid")
                    resolved[arm] = {"command": command,
                                     "args": self._resolve_args(command, args, measurement_id)}
                resolved_stages.append(resolved)
            if preview is not None and any(spec["command"] == "gripper" for spec in resolved_stages[0].values()):
                raise ValueError("Preview does not support gripper operations in stage 0")
        except (ValueError, TypeError, KeyError) as exc:
            return error("INVALID_INPUT", exc)
        observation = self._load_observation(observation_id)
        try:
            if preview is None:
                result = self.adapter.estimate(resolved_stages)
            else:
                # Saved capture timestamps are never refreshed by planning or rendering.
                source = {key: deepcopy(observation[key]) for key in (
                    "observation_id", "revision", "capture", "captured_monotonic_ns",
                    "camera_times", "state", "images")}
                result = self.adapter.estimate(resolved_stages, preview=preview, preview_source=source)
        except Exception as exc:
            return error("ESTIMATE_FAILED", exc)
        try:
            self._check_estimate_start(observation, result["state"])
            if preview is not None:
                record = result["preview"]
                if record.get("status") == "available":
                    expected = {key: observation[key] for key in (
                        "observation_id", "revision", "capture", "captured_monotonic_ns")}
                    expected.update(camera=preview["camera"],
                        camera_time_monotonic_ns=observation["camera_times"][preview["camera"]],
                        sim_time_s=observation["state"]["sim_time_s"],
                        stage_index=0, sample_indices=preview["samples"])
                    if any(record.get(key) != value for key, value in expected.items()):
                        raise ValueError("Preview source does not match saved observation and request")
                elif record.get("status") != "unavailable" or result.get("images"):
                    raise ValueError("Unavailable preview must not expose images")
        except (ValueError, TypeError, KeyError) as exc:
            return dict(error("INVALID_ESTIMATE", exc), estimate_complete=False,
                        low_level_actions=0, motion_started=False, motion_executed=False)
        return result

    def _check_estimate_start(self, observation, actual):
        """Bind cost to saved observation. Absolute 1e-7 tolerance, no state writes."""
        if (observation["observation_id"] != self.state["observation_id"]
                or observation["revision"] != self.state["revision"]):
            raise ValueError("Estimate observation revision changed")
        expected = observation["state"]
        for key in ("low_level_actions", "remaining_actions"):
            if (type(expected[key]) is not int or type(actual[key]) is not int
                    or expected[key] < 0 or actual[key] != expected[key]):
                raise ValueError(f"Estimate start mismatch: {key}")
        if (type(expected["episode_ended"]) is not bool
                or type(actual["episode_ended"]) is not bool
                or actual["episode_ended"] != expected["episode_ended"]):
            raise ValueError("Estimate episode flag changed")
        self._numbers([expected["sim_time_s"], actual["sim_time_s"]], 2)
        if actual["sim_time_s"] != expected["sim_time_s"]:
            raise ValueError("Estimate simulation time changed")
        if set(expected["arms"]) != {"left", "right"} or set(actual["arms"]) != {"left", "right"}:
            raise ValueError("Estimate requires both observed arms")

        def close(left, right, size):
            self._numbers(left, size)
            self._numbers(right, size)
            if any(abs(a-b) > 1e-7 for a, b in zip(left, right)):
                raise ValueError("Estimate measured arm state changed")

        for arm in ("left", "right"):
            before, after = expected["arms"][arm], actual["arms"][arm]
            close(before["joints_rad"], after["joints"], 6)
            close(before["gripper_joints_m"], after["gripper_measured_m"], 2)
            self._numbers([before["gripper_command"], after["gripper_command"]], 2)
            if before["gripper_command"] != after["gripper_command"]:
                raise ValueError("Estimate gripper command changed")
            left, right = before["tcp_pose_base"], after["tcp_pose_base"]
            self._numbers(left, 7)
            self._numbers(right, 7)
            close(left[:3], right[:3], 3)
            quaternions = []
            for values in (left[3:], right[3:]):
                norm = math.sqrt(sum(v*v for v in values))
                if abs(norm-1) > 1e-5:
                    raise ValueError("Estimate TCP quaternion must be normalized")
                quaternions.append([v/norm for v in values])
            left_q, right_q = quaternions
            if sum(a*b for a, b in zip(left_q, right_q)) < 0:
                right_q = [-v for v in right_q]
            close(left_q, right_q, 4)

    @staticmethod
    def _text(values, key):
        value = values.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} needs concrete visible evidence")
        return value.strip()

    @staticmethod
    def _numbers(values, count):
        if not isinstance(values, (tuple, list)) or len(values) != count or any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
            raise ValueError(f"Expected {count} finite numbers")

    def _validate_action(self, command, args, intent):
        if intent not in {"observe", "grasp", "transfer", "contact", "retreat", "place"}:
            raise ValueError("Unknown action intent")
        self._validate_command(command, args)
        if args.get("speed_profile") == "transit" and intent not in {"observe", "transfer", "retreat"}:
            raise ValueError("Transit is for clear observation, transfer or retreat motion")

    def _validate_command(self, command, args):
        if not isinstance(args, dict):
            raise ValueError("args must be an object")
        allowed = {"home": set(), "move_joints": {"joints"},
                   "move_ee": {"position", "position_offset", "rotation", "mode", "speed_profile"},
                   "move_delta": {"dpos", "drot_deg", "speed_profile"}, "gripper": {"action"}, "hold": {"samples"}}
        motion_options = {"max_samples", "settle_samples", "position_tolerance_m", "rotation_tolerance_rad"}
        for motion in ("home", "move_joints", "move_ee", "move_delta"):
            allowed[motion] |= motion_options
        if command not in allowed:
            raise ValueError(f"Unknown command {command!r}; expected one of {', '.join(sorted(allowed))}")
        unsupported = args.keys() - allowed[command]
        if unsupported:
            raise ValueError(f"Unsupported arguments for {command}: {', '.join(sorted(unsupported))}. "
                             f"Allowed arguments: {', '.join(sorted(allowed[command])) or '(none)'}")
        for field, low, high in (("max_samples", 1, 1000), ("settle_samples", 0, 10), ("samples", 1, 25)):
            if field in args and (type(args[field]) is not int or not low <= args[field] <= high):
                raise ValueError(f"{field} must be an integer in [{low}, {high}]")
        for field, low, high in (("position_tolerance_m", .0005, .01), ("rotation_tolerance_rad", .005, .1)):
            if field in args:
                self._numbers([args[field]], 1)
                if not low <= args[field] <= high:
                    raise ValueError(f"{field} must be in [{low}, {high}]")
        if command == "hold" and "samples" not in args:
            raise ValueError("hold requires samples (1..25)")
        profile = args.get("speed_profile", "precision")
        if profile not in {"precision", "transit"}:
            raise ValueError("Use precision or transit speed")
        if profile == "transit" and command not in {"move_ee", "move_delta"}:
            raise ValueError("Transit is for clear observation, transfer or retreat motion")
        if command == "move_joints":
            self._numbers(args.get("joints"), 6)
        if command == "move_ee":
            if ("position" in args) == ("position_offset" in args):
                raise ValueError("Provide exactly one of position or position_offset")
            position_field = "position_offset" if "position_offset" in args else "position"
            for field, keys in ((position_field, "xyz"), ("rotation", "wxyz")):
                value = args.get(field)
                if not isinstance(value, dict) or set(value) != set(keys):
                    raise ValueError(f"{field} requires {keys}")
                self._numbers([value[k] for k in keys], len(keys))
            if abs(sum(v*v for v in args["rotation"].values()) - 1) > .01:
                raise ValueError("rotation must be a unit quaternion")
            if args.get("mode", "linear") not in {"linear", "plan"}:
                raise ValueError("mode must be linear or plan")
        if command == "move_delta":
            if not (set(args) & {"dpos", "drot_deg"}):
                raise ValueError("Specify a displacement or rotation")
            for key in ("dpos", "drot_deg"):
                if key in args:
                    self._numbers(args[key], 3)
        if command == "gripper" and args.get("action") not in {"open", "close"}:
            raise ValueError("Use open or close with release evidence for opening")
        encoded(args)

    def _check_relative_contact(self, args, evidence, arm=None):
        # A relative adjustment uses the achieved pose, never an invalidated point.
        dpos = args.get("dpos", [])
        if len(dpos) != 3 or not 0 < sum(v * v for v in dpos) <= .02 ** 2:
            raise ValueError("Relative contact translation must be positive and at most 20 mm")
        if any(args.get("drot_deg", [])):
            raise ValueError("Relative contact continuation permits translation only")
        previous_id = self._text(evidence, "previous_action_id")
        row = self.db.execute("SELECT * FROM actions ORDER BY rowid DESC LIMIT 1").fetchone()
        if not row or row["id"] != previous_id or row["status"] != "done":
            raise ValueError("Relative contact must follow the latest completed action")
        previous = json.loads(row["payload"])
        result = json.loads(row["result"])
        action = result.get("action", {})
        if arm is not None:
            previous = previous.get("actions", {}).get(arm, {})
            action = action.get("arms", {}).get(arm, {})
        if (previous.get("intent") != "contact" or not result.get("ok")
                or action.get("motion_outcome") == "held_short"
                or action.get("target_reached") is False
                or result.get("observation", {}).get("observation_id") != self.state["observation_id"]
                or not self._arm_state(arm).get("target_confirmation")):
            raise ValueError("Relative contact needs the successful contact's fresh observation and target provenance")
        self._text(evidence, "target_unchanged")
        self._text(evidence, "adjustment_reason")

    def _check_evidence(self, command, args, intent, measurement_id, evidence, arm=None):
        state = self._arm_state(arm)
        self._check_measurement_arm(measurement_id, arm)
        if args.get("speed_profile") == "transit" or command == "home":
            self._text(evidence, "clearance")
        if command == "move_delta" and intent == "contact" and measurement_id is None:
            self._check_relative_contact(args, evidence, arm)
        elif intent in {"transfer", "contact"} or (command == "move_ee" and "position_offset" in args):
            if not isinstance(measurement_id, str) or not re.fullmatch(r"m[1-9][0-9]*", measurement_id):
                raise ValueError("Transfer/contact requires measurement_id")
            row = self.db.execute("SELECT value,valid FROM measurements WHERE id=?", (int(measurement_id[1:]),)).fetchone()
            if not row or not row["valid"]:
                raise ValueError("Measurement is absent or invalidated by possible contact/grasp change")
            measured = json.loads(row["value"])
            if state["grasp_active"] and (measured["grasp_epoch"] != state["grasp_epoch"] or not measured["after_post_grasp_motion"]):
                raise ValueError("Measure the target after the current grasp and a clearance action")
            self._text(evidence, "target_unchanged")
        if command == "gripper" and args["action"] == "open":
            requirement = evidence.get("release_requirement")
            purpose = evidence.get("release_purpose", "goal")
            if requirement == "empty":
                self._text(evidence, "empty")
                if state["grasp_active"]:
                    self._text(evidence, "object_location")
                    self._text(evidence, "opening_clearance")
                return
            if purpose not in {"goal", "recovery"}:
                raise ValueError("release_purpose must be goal or recovery")
            required = self.task["release_requirement"] if purpose == "goal" else "support"
            if requirement != required:
                raise ValueError(f"This release requires {required}")
            if required == "drop":
                self._text(evidence, "landing")
                return
            self._text(evidence, "support")
            if required == "engagement_and_support":
                self._text(evidence, "engagement")
                if not state["grasp_active"]:
                    raise ValueError("An engagement release requires the current grasp")
                if not state.get("target_confirmation"):
                    # A new visual/geometry check can restore release evidence
                    # without manufacturing a clearance or zero-distance move.
                    if not isinstance(measurement_id, str) or not re.fullmatch(r"m[1-9][0-9]*", measurement_id):
                        raise ValueError("Reconfirm support in the latest observation with measurement_id and target_unchanged")
                    row = self.db.execute("SELECT value,valid FROM measurements WHERE id=?", (int(measurement_id[1:]),)).fetchone()
                    measured = json.loads(row["value"]) if row else {}
                    if (not row or not row["valid"]
                            or measured.get("observation_id") != self.state["observation_id"]
                            or measured.get("grasp_epoch") != state["grasp_epoch"]):
                        raise ValueError("Release reconfirmation requires a valid measurement from the latest observation and current grasp")
                    self._text(evidence, "target_unchanged")

    def _resolve_args(self, command, args, measurement_id):
        if command != "move_ee" or "position_offset" not in args:
            return dict(args)
        if not isinstance(measurement_id, str) or not re.fullmatch(r"m[1-9][0-9]*", measurement_id):
            raise ValueError("position_offset requires a measurement_id")
        row = self.db.execute("SELECT value,valid FROM measurements WHERE id=?", (int(measurement_id[1:]),)).fetchone()
        if not row or not row["valid"]:
            raise ValueError("Referenced position measurement is invalid")
        point = json.loads(row["value"])["result"]["point_base"]
        resolved = dict(args)
        offset = resolved.pop("position_offset")
        resolved["position"] = {axis: point[axis] + offset[axis] for axis in "xyz"}
        self._numbers(list(resolved["position"].values()), 3)
        return resolved

    def _change_revision(self, command, args, intent):
        self.state["revision"] += 1
        self.state["observation_id"] = None
        if command == "gripper" or intent == "contact":
            self.db.execute("UPDATE measurements SET valid=0")
        if command == "gripper" and args["action"] == "close":
            self.state["grasp_epoch"] += 1
            self.state["grasp_active"] = True
            self.state["post_grasp_motion_seen"] = False
            self.state["target_confirmation"] = None

    @serial
    def step(self, action_id, observation_id, command=None, args=None, intent=None, measurement_id=None, evidence=None, *, actions=None):
        evidence = {} if evidence is None else evidence
        try:
            if not isinstance(action_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", action_id):
                raise ValueError("action_id must be a unique short identifier")
            if actions is not None:
                if not self.dual_arm or any(v is not None for v in (command, args, intent, measurement_id)) or evidence:
                    raise ValueError("Use only actions with a dual-arm adapter")
                if not isinstance(actions, dict) or not actions or actions.keys() - {"left", "right"}:
                    raise ValueError("actions must contain left and/or right")
                normalized = {}
                for arm, spec in actions.items():
                    if not isinstance(spec, dict) or spec.keys() - {"command", "args", "intent", "measurement_id", "evidence"}:
                        raise ValueError("Unexpected arm action fields")
                    spec = dict(spec, measurement_id=spec.get("measurement_id"),
                                evidence={} if spec.get("evidence") is None else spec["evidence"])
                    if not isinstance(spec["evidence"], dict):
                        raise ValueError("evidence must be an object")
                    self._validate_action(spec["command"], spec["args"], spec["intent"])
                    normalized[arm] = spec
                payload = {"observation_id": observation_id, "actions": normalized}
            else:
                if self.dual_arm:
                    raise ValueError("Simulation requires explicit actions map")
                if not isinstance(evidence, dict):
                    raise ValueError("evidence must be an object")
                payload = {"observation_id": observation_id, "command": command, "args": args,
                           "intent": intent, "measurement_id": measurement_id, "evidence": evidence}
                self._validate_action(command, args, intent)
            body = encoded(payload)
        except (ValueError, TypeError, KeyError) as exc:
            return error("INVALID_INPUT", exc)
        previous = self.db.execute("SELECT * FROM actions WHERE id=?", (action_id,)).fetchone()
        if previous:
            if previous["payload"] != body:
                return error("ACTION_ID_CONFLICT", "This action_id already names a different request")
            if previous["status"] in {"done", "rejected"}:
                return dict(json.loads(previous["result"]), replayed=True)
            return {"ok": False, "status": "unknown", "action_id": action_id,
                    "error_code": "UNRESOLVED_ACTION", "error": "Use action_status; never resend", "images": []}
        if self.db.execute("SELECT 1 FROM actions WHERE status NOT IN ('done','rejected') LIMIT 1").fetchone():
            return error("UNRESOLVED_ACTION", "Resolve the previous action before issuing another")
        if not self._fresh(observation_id):
            return error("STALE_OBSERVATION", "Use the latest successful observation")
        try:
            if actions is not None:
                resolved = {}
                for arm, spec in payload["actions"].items():
                    self._check_evidence(**spec, arm=arm)
                    resolved[arm] = {"command": spec["command"],
                        "args": self._resolve_args(spec["command"], spec["args"], spec["measurement_id"])}
                command, resolved_args = "actions", {"actions": resolved}
            else:
                self._check_evidence(command, args, intent, measurement_id, evidence)
                resolved_args = self._resolve_args(command, args, measurement_id)
        except (ValueError, TypeError, KeyError) as exc:
            return error("INVALID_EVIDENCE", exc)
        before_state = deepcopy(self.state)
        with self.db:
            self.db.execute("INSERT INTO actions(id,payload,status,prior_state) VALUES(?,?,'pending',?)",
                            (action_id, body, encoded(before_state)))
            if actions is not None:
                self._change_dual_revision(payload["actions"])
            else:
                self._change_revision(command, args, intent)
            self._save_state()
            self._event("action_prepared", {"action_id": action_id, **payload, "resolved_args": resolved_args})
        try:
            response = self.adapter.act(action_id, command, resolved_args)
        except NotDispatched as exc:
            result = dict(error("NOT_DISPATCHED", exc), status="rejected", action_id=action_id)
            with self.db:
                self.state = dict(before_state, observation_id=None, target_confirmation=None,
                                  post_grasp_motion_seen=False)
                for state in self.state.get("arms", {}).values():
                    state.update(target_confirmation=None, post_grasp_motion_seen=False)
                self.db.execute("UPDATE measurements SET valid=0")
                self.db.execute("UPDATE actions SET status='rejected',result=? WHERE id=?",
                                (encoded(result), action_id))
                self._save_state()
                self._event("action_rejected", result)
            return result
        except Exception as exc:
            with self.db:
                self._invalidate_dual_evidence()
                self._save_state()
                self.db.execute("UPDATE actions SET status='unknown' WHERE id=?", (action_id,))
                self._event("action_unknown", {"action_id": action_id, "error": str(exc)})
            return {"ok": False, "status": "unknown", "action_id": action_id,
                    "error_code": "UNRESOLVED_ACTION", "error": str(exc), "images": []}
        return self._complete(action_id, payload, response)

    def _recovery_guidance(self, payload, response, observation, arm=None):
        if (response.get("episode_ended") is True
                or observation.get("state", {}).get("episode_ended") is True):
            return {"kind": "episode_ended", "instructions": (
                "Stop issuing movement and gripper commands. Assess from the final observation. "
                "Episode end alone does not establish success or failure.")}
        if not observation.get("ok"):
            return {"kind": "refresh_observation", "instructions": (
                "Acquire a fresh observation before deciding or issuing recovery movement, closure "
                "or trial lift. Inspect the achieved state and current scene after the action.")}
        if payload["command"] == "gripper":
            if payload["args"]["action"] == "close":
                if not response.get("ok") or response.get("motion_started") is False:
                    return {"kind": "review_close_outcome", "instructions": (
                        "Inspect measured gripper closure and the current scene. Resolve the failed "
                        "or unstarted close and recover valid grasp geometry before considering a "
                        "trial lift. Neither this close result nor an attempted grasp establishes possession.")}
                return {"kind": "verify_grasp_attempt", "instructions": (
                    "Closure is an attempted grasp, not proof of possession. Use a short precision "
                    "trial lift with clearance, then compare whether the object follows the gripper "
                    "and leaves its source. If it did not, reobserve and replan the grasp before transfer.")}
            if arm is not None and (not response.get("ok") or response.get("target_reached") is False
                                    or response.get("motion_started") is False):
                return {"kind": "review_open_outcome", "instructions": (
                    "Opening did not establish release. Inspect measured fingers and object support "
                    "in fresh images before further motion; attempted grasp state is retained.")}
            return None
        if (response.get("motion_outcome") == "held_short"
                or response.get("target_reached") is False):
            return {"kind": "review_achieved_pose", "instructions": (
                "Compare achieved TCP with intended TCP and fresh observed geometry. Remeasure and "
                "replan before dependent close, contact or transfer unless independent evidence "
                "confirms valid intended contact and grasp geometry. Fresh support or engagement "
                "evidence may justify release at an already valid seated pose.")}
        if (response.get("ok") and observation.get("ok") and self._arm_state(arm)["grasp_active"]
                and payload["command"] != "hold"
                and payload["intent"] in {"grasp", "retreat", "observe"}):
            return {"kind": "verify_object_follows", "instructions": (
                "A prior close records an attempted grasp, not possession. Before transfer, compare "
                "the fresh object, source and gripper views to verify that the object followed this "
                "motion. If it stayed at the source or slipped, reobserve and replan the grasp.")}
        return None

    def _invalidate_dual_evidence(self):
        if not self.dual_arm:
            return
        self.db.execute("UPDATE measurements SET valid=0")
        for state in self.state["arms"].values():
            state.update(target_confirmation=None, post_grasp_motion_seen=False)

    def _change_dual_revision(self, actions):
        self.state["revision"] += 1
        self.state["observation_id"] = None
        consequential = {arm for arm, spec in actions.items()
                         if spec["command"] == "gripper" or spec["intent"] == "contact"}
        if consequential:
            self.db.execute("UPDATE measurements SET valid=0")
            for arm, state in self.state["arms"].items():
                # Same-arm contact continuation can retain its provenance, but
                # the other arm changing shared support invalidates it.
                if consequential - {arm}:
                    state["target_confirmation"] = None
        for arm, spec in actions.items():
            if spec["command"] == "gripper" and spec["args"]["action"] == "close":
                state = self.state["arms"][arm]
                state["grasp_epoch"] += 1
                state.update(grasp_active=True, post_grasp_motion_seen=False, target_confirmation=None)

    def _complete_dual(self, action_id, payload, response):
        after_ns = max(time.monotonic_ns(), int(response.get("response_received_monotonic_ns", 0)))
        outcomes = response.get("arms", {})
        prior_row = self.db.execute("SELECT prior_state FROM actions WHERE id=?", (action_id,)).fetchone()
        prior = json.loads(prior_row[0]) if prior_row and prior_row[0] else {}
        shared_counts = [response[key] for key in ("low_level_actions", "trajectory_samples")
                         if key in response]
        globally_unstarted = (
            response.get("motion_started") is False
            and response.get("motion_executed") is False
            and bool(shared_counts)
            and all(type(count) in (int, float) and count == 0 for count in shared_counts)
            and all(
                all(outcome[key] is False for key in ("motion_started", "motion_executed")
                    if key in outcome)
                and all(type(outcome[key]) in (int, float) and outcome[key] == 0
                        for key in ("low_level_actions", "trajectory_samples") if key in outcome)
                for outcome in outcomes.values()))
        uncertain = not response.get("ok") or any(
            not outcomes.get(arm, {}).get("ok")
            or outcomes[arm].get("target_reached") is False
            or outcomes[arm].get("motion_outcome") in {"held_short", "checkpoint"}
            for arm in payload["actions"])
        with self.db:
            self.db.execute("UPDATE actions SET status='responded',response=? WHERE id=?", (encoded(response), action_id))
            if uncertain:
                self._invalidate_dual_evidence()
            for arm, spec in payload["actions"].items():
                state = self.state["arms"][arm]
                outcome = outcomes.get(arm, {})
                # Explicit flags alone cannot override evidence of executed samples.
                # Local zero counts permit one unstarted arm while its partner moved;
                # absent local counts fall back to the shared transaction counts.
                counts = [outcome[key] for key in ("low_level_actions", "trajectory_samples")
                          if key in outcome]
                if not counts:
                    counts = [response[key] for key in ("low_level_actions", "trajectory_samples")
                              if key in response]
                unstarted = (outcome.get("motion_started", response.get("motion_started")) is False
                             and outcome.get("motion_executed", response.get("motion_executed")) is False
                             and bool(counts)
                             and all(type(count) in (int, float) and count == 0 for count in counts))
                if unstarted and arm in prior.get("arms", {}):
                    # Keep fresh-evidence requirements conservative, but restore
                    # the grasp attempt that actually existed before dispatch.
                    for key in ("grasp_epoch", "grasp_active"):
                        state[key] = prior["arms"][arm][key]
                if not outcome.get("ok") or outcome.get("motion_started", response.get("motion_started")) is False:
                    continue
                # A successful open is local even when the other arm failed.
                if (spec["command"] == "gripper" and spec["args"]["action"] == "open"
                        and outcome.get("target_reached") is not False
                        and outcome.get("motion_outcome") not in {"held_short", "checkpoint"}):
                    state.update(grasp_active=False, post_grasp_motion_seen=False, target_confirmation=None)
                else:
                    other_changes_support = any(other != arm and (item["command"] == "gripper" or item["intent"] == "contact")
                                                for other, item in payload["actions"].items())
                    if not uncertain and not other_changes_support and spec["intent"] in {"contact", "transfer"}:
                        state["target_confirmation"] = spec["measurement_id"] or state["target_confirmation"]
                    if (spec["command"] not in {"gripper", "hold"} and state["grasp_active"]
                            and spec["intent"] in {"observe", "retreat", "grasp"}
                            and outcome.get("target_reached") is True
                            and outcome.get("motion_outcome") not in {"held_short", "checkpoint"}
                            and outcome.get("motion_executed", response.get("motion_executed")) is True):
                        state["post_grasp_motion_seen"] = True
            if globally_unstarted:
                # No shared physics advanced. Preserve completed motion history,
                # including the other arm, while requiring fresh target evidence.
                for arm, state in self.state["arms"].items():
                    if arm in prior.get("arms", {}):
                        state["post_grasp_motion_seen"] = prior["arms"][arm]["post_grasp_motion_seen"]
            self._save_state()
            self._event("action_response", {"action_id": action_id, "response": response})
        observation = self._observation(after_ns=after_ns)
        result = {"ok": bool(response.get("ok") and observation.get("ok")
                            and all(outcomes.get(arm, {}).get("ok") for arm in payload["actions"])),
                  "status": "done", "action_id": action_id, "action": response,
                  "observation": observation, "images": observation.get("images", []),
                  "semantic_success": "unassessed"}
        guidance = {}
        for arm, spec in payload["actions"].items():
            outcome = dict(outcomes.get(arm, {"ok": False}), episode_ended=response.get("episode_ended", False))
            outcome.setdefault("motion_started", response.get("motion_started"))
            item = self._recovery_guidance(spec, outcome, observation, arm)
            if item is not None:
                guidance[arm] = item
        if guidance:
            result["recovery_guidance"] = {"arms": guidance}
        with self.db:
            self.db.execute("UPDATE actions SET status='done',result=? WHERE id=?", (encoded(result), action_id))
            self._event("step_completed", result)
        return result

    def _complete(self, action_id, payload, response):
        if "actions" in payload:
            return self._complete_dual(action_id, payload, response)
        after_ns = max(time.monotonic_ns(), int(response.get("response_received_monotonic_ns", 0)))
        with self.db:
            self.db.execute("UPDATE actions SET status='responded',response=? WHERE id=?", (encoded(response), action_id))
            no_motion = (response.get("motion_started") is False
                         and payload["command"] in {"move_ee", "move_delta"})
            held_short = response.get("ok") is True and response.get("motion_outcome") == "held_short"
            target_confirmation = self.state.get("target_confirmation")
            if held_short or (not response.get("ok") and not no_motion):
                self.db.execute("UPDATE measurements SET valid=0")
                self.state["target_confirmation"] = None
                if not held_short:
                    self.state["post_grasp_motion_seen"] = False
            if response.get("ok"):
                if payload["intent"] == "contact" or (payload["intent"] == "transfer" and not held_short):
                    self.state["target_confirmation"] = payload["measurement_id"] or target_confirmation
                if payload["command"] == "gripper" and payload["args"]["action"] == "open":
                    self.state["grasp_active"] = False
                    self.state["post_grasp_motion_seen"] = False
                    self.state["target_confirmation"] = None
                elif not held_short and payload["command"] != "gripper" and self.state["grasp_active"] and payload["intent"] in {"observe", "retreat", "grasp"}:
                    self.state["post_grasp_motion_seen"] = True
            self._save_state()
            self._event("action_response", {"action_id": action_id, "response": response})
        observation = self._observation(after_ns=after_ns)
        result = {"ok": bool(response.get("ok") and observation.get("ok")), "status": "done",
                  "action_id": action_id, "action": response, "observation": observation,
                  "images": observation.get("images", []), "semantic_success": "unassessed"}
        guidance = self._recovery_guidance(payload, response, observation)
        if guidance is not None:
            result["recovery_guidance"] = guidance
        with self.db:
            self.db.execute("UPDATE actions SET status='done',result=? WHERE id=?", (encoded(result), action_id))
            self._event("step_completed", result)
        return result

    @serial
    def action_status(self, action_id):
        row = self.db.execute("SELECT * FROM actions WHERE id=?", (action_id,)).fetchone()
        if not row:
            return error("UNKNOWN_ACTION_ID", "No such action was prepared")
        if row["status"] in {"done", "rejected"}:
            return dict(json.loads(row["result"]), replayed=True)
        response = json.loads(row["response"]) if row["response"] else self.adapter.action_result(action_id)
        if response is None:
            return {"ok": False, "status": "unknown", "action_id": action_id,
                    "error_code": "UNRESOLVED_ACTION", "error": "No durable backend response; motion remains blocked", "images": []}
        return self._complete(action_id, json.loads(row["payload"]), response)

    @serial
    def assess(self, observation_id, outcome, evidence):
        if self.db.execute("SELECT 1 FROM actions WHERE status NOT IN ('done','rejected') LIMIT 1").fetchone():
            return error("UNRESOLVED_ACTION", "Resolve action outcomes before assessing completion")
        if not self._fresh(observation_id):
            return error("STALE_OBSERVATION", "Assess the latest successful observation")
        if outcome not in {"success", "failure", "unknown"} or not isinstance(evidence, dict):
            return error("INVALID_ASSESSMENT", "Use success, failure or unknown and an evidence object")
        try:
            self._text(evidence, "summary")
        except ValueError as exc:
            return error("INVALID_ASSESSMENT", exc)
        result = {"ok": True, "agent_claim": outcome, "evidence": evidence,
                  "observation_id": observation_id, "independent_verdict": "pending", "images": []}
        with self.db:
            self._event("assessment", result)
        path = self.directory / f"assessment_{time.time_ns()}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        return result
