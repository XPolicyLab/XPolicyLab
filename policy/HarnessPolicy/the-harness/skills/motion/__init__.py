"""Bounded arm and gripper motions: planned paths, retiming, arrival checks."""

import math

from harness_policy.runtime.events import Escalation, Event, ExecutionTrace
from harness_policy.runtime import events as ev
from harness_policy.runtime.session import ARMS


GRIPPER_SETTLE_TICKS = 3
# After a path's last waypoint the arm gets this many ticks to catch up; a motion
# that then ends farther than ARRIVAL_TOLERANCE_M (link6, move_eef) or
# ARRIVAL_TOLERANCE_RAD (any joint, move_joints) from its target is BLOCKED.
ARRIVAL_SETTLE_TICKS = 12
# A move_eef stops settling once link6 is this close to its target (the joints may
# still differ by more than ARRIVAL_TOLERANCE_RAD / 5 in a redundant direction).
ARRIVAL_SETTLED_M = 0.003
ARRIVAL_TOLERANCE_M = 0.01
ARRIVAL_TOLERANCE_RAD = 0.05
# A retimed path moves no joint more than this per tick (the URDF velocity limit at 25 Hz).
MAX_RETIMED_JOINT_STEP = 0.12
# home() moves the arms at this speed of the fastest joint (rad/s), below the planner's
# 3.14 rad/s; the acceleration limit (10 rad/s^2) then sets most of the duration: about
# 20 ticks for a 1.5 rad swing, where the planner's own timing took over 100 per arm.
HOME_SPEED = 2.5
# holding(): a closed gripper this much above its empty reading has something inside.
# Without a measured opening (a connection that reports only the last gripper command,
# such as XPolicyLab's), a gripper motion takes this long, the time the gripper needs
# for its full stroke and a firm grip (embodiment gripper_full_stroke_seconds), and
# holding() asks the vision model about the wrist camera's view.
DEFAULT_GRIPPER_STROKE_SECONDS = 0.4
HOLDING_QUESTION = ("This image is from the camera on a robot's gripper; the two fingers are in view. Is an object "
                    "held between the fingers? Answer with exactly YES or NO.")
HOLDING_MARGIN = 0.03
# Motions are timed to this fraction of the shield's velocity and acceleration bounds.
TIMING_MARGIN = 0.9
GRIPPER_TOLERANCE = 0.02
# A gripper whose opening changes less than this per tick has stopped (on an object:
# a compliant grip creeps by about 1e-3 per tick; a free stroke moves about 0.1).
GRIPPER_STALL_PER_TICK = 0.004


class Motion:
    def move_joints(self, arm, joints, speed=None, max_seconds=None, region=None, until=None):
        """speed: rad/s of the fastest joint; None keeps the planner's timing."""
        group = self._arm(arm)
        parameters = {"arm": arm, "joints": list(joints), "speed": speed}
        waypoints, reason = self._joint_path(arm, joints, speed)
        if waypoints is None:
            return self._finish(self._failed("move_joints", parameters, reason))
        return self._follow("move_joints", parameters, group, waypoints,
                            max_seconds, region, until, arm=arm)

    def move_eef(self, arm, pose, speed=None, max_seconds=None, region=None, until=None):
        """speed: m/s of link6 from its current position to the target; None keeps the
        planner's timing."""
        group = self._arm(arm)
        pose = [float(v) for v in pose]
        if len(pose) != 7:
            raise ValueError("pose is [x, y, z, qw, qx, qy, qz] of link6 in the env frame")
        parameters = {"arm": arm, "pose": pose, "speed": speed}
        waypoints, reason = self._eef_path(arm, pose, speed)
        if waypoints is None:
            return self._finish(self._failed("move_eef", parameters, reason))
        return self._follow("move_eef", parameters, group, waypoints,
                            max_seconds, region, until, arm=arm, pose=pose)

    def _joint_path(self, arm, joints, speed):
        """(waypoints, None): `arm`'s planned path to `joints`, timed for `speed` (rad/s
        of the fastest joint; None: the planner's timing) within the joint limits; or
        (None, reason)."""
        group = self._arm(arm)
        waypoints = self.plan_path(arm, joints=joints)
        if waypoints is None:
            return None, "no collision-free path to the joint target"
        start = [self.state["joint_position"][i] for i in group["joints"]]
        if speed is not None:
            travel = max(abs(a - b) for a, b in zip(start, waypoints[-1])) if waypoints else 0.0
            waypoints = self._retime(start, waypoints, travel / self._positive(speed, "speed"))
        return self._within_limits(start, waypoints), None

    def _eef_path(self, arm, pose, speed):
        """(waypoints, None): `arm`'s planned path to the link6 `pose`, timed for
        `speed` (m/s of link6; None: the planner's timing) within the joint limits; or
        (None, reason)."""
        group = self._arm(arm)
        waypoints = self.plan_path(arm, pose=pose)
        if waypoints is None:
            return None, self._unreachable_reason(arm, pose)
        start = [self.state["joint_position"][i] for i in group["joints"]]
        if speed is not None:
            here = self.state[f"{arm.split('_')[0]}_ee_pose"][:3]
            waypoints = self._retime(start, waypoints, math.dist(here, pose[:3]) / self._positive(speed, "speed"))
        return self._within_limits(start, waypoints), None

    def _together(self, skill, targets, speed, max_seconds=None):
        """Move several arms in one bounded execution, all at once: targets is {arm:
        ("joints", joints6) or ("pose", link6 pose)}, timed for `speed` as move_joints
        or move_eef. Returns one trace per arm with that arm's event (COMPLETED,
        BLOCKED, SKILL_ERROR when it had no path, or the event that ended the execution
        before it arrived). The execution's first failure is handled or escalated as
        any motion's; the escalation then carries the per-arm traces (.traces)."""
        paths, events = {}, {}
        for arm, (kind, goal) in targets.items():
            goal = [float(v) for v in goal]
            waypoints, reason = (self._joint_path(arm, goal, speed) if kind == "joints"
                                 else self._eef_path(arm, goal, speed))
            if waypoints is None:
                events[arm] = Event(ev.SKILL_ERROR, skill, {"arm": arm, "reason": reason}, self.state["step"])
            else:
                paths[arm] = (waypoints, goal if kind == "pose" else None)
        parameters = {"arms": list(targets), "speed": speed}
        start, escalation, trace = self.state["step"], None, None
        if paths:
            try:
                trace = self._follow_arms(skill, parameters, paths, max_seconds, events)
            except Escalation as raised:
                escalation, trace = raised, raised.trace
            for arm in paths:
                events.setdefault(arm, trace.event)  # the execution ended before the arm arrived
        samples = trace.samples if trace is not None else []
        traces = [ExecutionTrace(skill, {**parameters, "arm": arm}, start, self.state["step"], events[arm], samples)
                  for arm in targets]
        failed = next((t for t in traces if t.event.kind == ev.SKILL_ERROR), None)
        if escalation is None and failed is not None:
            try:
                self._finish(ExecutionTrace(skill, parameters, start, self.state["step"], failed.event, samples))
            except Escalation as raised:
                escalation = raised
        if escalation is not None:
            escalation.traces = traces
            raise escalation
        return traces

    def _follow_arms(self, skill, parameters, paths, max_seconds, events):
        """Follow one timed path per arm ({arm: (waypoints, pose or None)}) at the same
        ticks; an arm that is through its path holds its last waypoint while it settles
        (as _follow) and then while the others go on. Records each arm's arrival in
        `events` (COMPLETED, or BLOCKED with the arm in its detail); the execution ends
        with the first BLOCKED arrival, else COMPLETED."""
        moving = {arm: {"group": self.groups[arm], "queue": list(waypoints), "pose": pose, "settle": 0,
                        "final": [float(v) for v in waypoints[-1]] if waypoints else None}
                  for arm, (waypoints, pose) in paths.items()}

        def command(state, tick):
            target = self._base_command(state)
            for arm, motion in moving.items():
                group, final = motion["group"], motion["final"]
                if arm in events or final is None:
                    if final is None:
                        events.setdefault(arm, Event(ev.COMPLETED, skill, {"arm": arm}, state["step"]))
                    values = final
                elif motion["queue"]:
                    values = motion["queue"].pop(0)
                else:
                    error = max(abs(state["joint_position"][i] - v) for i, v in zip(group["joints"], final))
                    settled = error <= ARRIVAL_TOLERANCE_RAD / 5 or (
                        motion["pose"] is not None
                        and math.dist(state[f"{arm.split('_')[0]}_ee_pose"][:3], motion["pose"][:3]) <= ARRIVAL_SETTLED_M)
                    if settled or motion["settle"] >= ARRIVAL_SETTLE_TICKS:
                        arrival = self._arrival(skill, arm, group, final, motion["pose"], state)
                        if arrival is not None:
                            arrival.detail["arm"] = arm
                        events[arm] = arrival or Event(ev.COMPLETED, skill, {"arm": arm}, state["step"])
                    motion["settle"] += 1
                    values = final
                for index, value in zip(group["joints"], values or []):
                    target[index] = float(value)
            if all(arm in events for arm in moving):
                return next((events[arm] for arm in moving if events[arm].kind == ev.BLOCKED), None)
            return target
        return self._execute(skill, parameters, command, max_seconds, None, None)

    @staticmethod
    def _positive(value, name):
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive number")
        return value

    def _retime(self, start, waypoints, seconds):
        """Resample a joint path to take `seconds` at the control rate, evenly spaced
        along its joint-space length, but no faster than MAX_RETIMED_JOINT_STEP per
        tick on any joint."""
        points = [list(map(float, start))] + [list(map(float, w)) for w in waypoints]
        lengths = [0.0]
        for a, b in zip(points, points[1:]):
            lengths.append(lengths[-1] + math.dist(a, b))
        total = lengths[-1]
        if total <= 0:
            return waypoints
        largest = sum(max(abs(x - y) for x, y in zip(a, b)) for a, b in zip(points, points[1:]))
        ticks = max(1, math.ceil(seconds * self.hz), math.ceil(largest / MAX_RETIMED_JOINT_STEP))
        result, segment = [], 1
        for k in range(1, ticks + 1):
            target = total * k / ticks
            while segment < len(points) - 1 and lengths[segment] < target:
                segment += 1
            span = lengths[segment] - lengths[segment - 1]
            fraction = 1.0 if span <= 0 else (target - lengths[segment - 1]) / span
            a, b = points[segment - 1], points[segment]
            result.append([x + (y - x) * fraction for x, y in zip(a, b)])
        return result

    def _within_limits(self, start, waypoints):
        """Keep a path's timing when it respects the planner's joint velocity and
        acceleration limits; otherwise retime it along the same path with a
        trapezoidal profile within a margin of them, taking at least as long as before."""
        speed = getattr(self, "joint_speed_limit", None)
        acceleration = getattr(self, "joint_acceleration_limit", None)
        if not waypoints or speed is None or acceleration is None:
            return waypoints
        points = [list(map(float, start))] + [list(map(float, w)) for w in waypoints]
        deltas = [[y - x for x, y in zip(p, q)] for p, q in zip(points, points[1:])] + [[0.0] * len(start)]
        previous = [0.0] * len(start)
        within = True
        v, a = speed / self.hz, acceleration / self.hz ** 2  # rad per tick, per tick^2
        for delta in deltas:
            if any(abs(d) > v or abs(d - b) > a for d, b in zip(delta, previous)):
                within = False
                break
            previous = delta
        if within:
            return waypoints
        v, a = TIMING_MARGIN * v, TIMING_MARGIN * a
        # Path parameter: the largest joint change, so its rate bounds every joint's.
        lengths = [0.0]
        for delta in deltas[:-1]:
            lengths.append(lengths[-1] + max(abs(d) for d in delta))
        total = lengths[-1]
        if total <= 0:
            return waypoints
        shortest = 2 * math.sqrt(total / a) if total <= v * v / a else total / v + v / a
        ticks = max(len(waypoints), math.ceil(shortest))
        cruise = (a * ticks - math.sqrt(max(a * a * ticks * ticks - 4 * a * total, 0.0))) / 2
        ramp = cruise / a
        result, segment = [], 1
        for k in range(1, ticks + 1):
            if k <= ramp:
                s = 0.5 * a * k * k
            elif k <= ticks - ramp:
                s = 0.5 * a * ramp * ramp + cruise * (k - ramp)
            else:
                s = total - 0.5 * a * (ticks - k) ** 2
            s = min(s, total) if k < ticks else total
            while segment < len(points) - 1 and lengths[segment] < s:
                segment += 1
            span = lengths[segment] - lengths[segment - 1]
            fraction = 1.0 if span <= 0 else (s - lengths[segment - 1]) / span
            p, q = points[segment - 1], points[segment]
            result.append([x + (y - x) * fraction for x, y in zip(p, q)])
        return result

    def open_gripper(self, arm, width=1.0, max_seconds=1.5):
        """Command the gripper of `arm` to `width` (1 open, 0 closed) and wait, at most
        `max_seconds`, until the opening settles (without a measured opening: for the
        gripper's full stroke time). Returns the trace; running out of time is COMPLETED."""
        return self._gripper("open_gripper", arm, float(width), max_seconds)

    def close_gripper(self, arm, max_seconds=1.5):
        """Close the gripper; the trace's event detail gives the measured opening and
        whether something is between the fingers (holding)."""
        trace = self._gripper("close_gripper", arm, 0.0, max_seconds)
        if not self._measured_gripper():
            trace.event.detail = {**(trace.event.detail or {}), "opening": None, "holding": self.holding(arm)}
            return trace
        opening = self._opening(arm)
        trace.event.detail = {**(trace.event.detail or {}), "opening": round(opening, 3),
                              "holding": opening > self._empty_opening() + HOLDING_MARGIN}
        return trace

    def holding(self, arm):
        """True when the gripper of `arm` is closed on something: it was last commanded
        closed and its measured opening is clearly above the empty closed reading of
        this embodiment (without a measured opening: the vision model sees an object
        between the fingers in the wrist camera's view, asked once per robot state)."""
        if getattr(self, "_gripper_commands", {}).get(arm, 1.0) > 0.5:
            return False
        if not self._measured_gripper():
            return self._looked_holding(arm, look=True)
        return self._opening(arm) > self._empty_opening() + HOLDING_MARGIN

    def _holding_known(self, arm):
        """holding() without a new look: what the last look said (for the many internal
        checks that only rank estimates near the fingers)."""
        if getattr(self, "_gripper_commands", {}).get(arm, 1.0) > 0.5:
            return False
        if not self._measured_gripper():
            return self._looked_holding(arm, look=False)
        return self._opening(arm) > self._empty_opening() + HOLDING_MARGIN

    def _measured_gripper(self):
        return (getattr(self, "profile", None) or {}).get("gripper_feedback", "measured") == "measured"

    def _looked_holding(self, arm, look):
        """The vision model's answer about the wrist camera's view, cached per arm and
        robot step (a new look only when the robot moved since)."""
        cache = self.__dict__.setdefault("_holding_looks", {})
        step = (getattr(self, "state", None) or {}).get("step")
        known = cache.get(arm)
        if known is not None and (known[0] == step or not look):
            return known[1]
        if not look:
            return False
        model = getattr(self, "vision", None) or getattr(self, "pointer", None)
        if model is None:
            raise RuntimeError("holding() needs a vision model when the gripper opening is not measured")
        camera = f"cam_{arm.split('_')[0]}_wrist"
        observation = self.get_observation(cameras=[camera])
        answer = model.ask([observation.images[camera]], HOLDING_QUESTION, system="")
        held = answer.strip().upper().startswith("YES")
        cache[arm] = (self.state.get("step"), held)
        return held

    def _opening(self, arm):
        return float(self.state["joint_position"][self._arm(arm)["gripper"]])

    def _empty_opening(self):
        return float((getattr(self, "profile", None) or {}).get("gripper_empty_opening", 0.185))

    def hold(self, seconds):
        """Keep the current targets for a number of seconds (sim time)."""
        target = self._base_command(self.state)
        return self._execute("hold", {"seconds": seconds}, lambda state, tick: target,
                             max_seconds=seconds, region=None, until=None, timeout_is_nominal=True)

    def home(self, arms=ARMS, speed=HOME_SPEED):
        """Move the arms back to the joints they started the episode at (grippers as they
        are): the pose a learned policy's demonstrations start from. All arms move at
        once in one execution, at `speed` rad/s of the fastest joint (None: the
        planner's slower timing). Returns one trace per arm."""
        arms = [arms] if isinstance(arms, str) else list(arms)
        return self._together("home", {arm: ("joints", self._start_of(arm)) for arm in arms}, speed)

    def _start_of(self, arm):
        group = self._arm(arm)
        return [self._start_joints[i] for i in group["joints"]]

    def _follow(self, skill, parameters, group, waypoints, max_seconds, region, until, arm, pose=None):
        queue = list(waypoints)
        final = [float(v) for v in waypoints[-1]] if waypoints else None
        settle = [0]

        def command(state, tick):
            if queue:
                values = queue.pop(0)
            else:
                if final is None:
                    return None
                error = max(abs(state["joint_position"][i] - v) for i, v in zip(group["joints"], final))
                settled = error <= ARRIVAL_TOLERANCE_RAD / 5 or (
                    pose is not None
                    and math.dist(state[f"{arm.split('_')[0]}_ee_pose"][:3], pose[:3]) <= ARRIVAL_SETTLED_M)
                if settled or settle[0] >= ARRIVAL_SETTLE_TICKS:
                    return self._arrival(skill, arm, group, final, pose, state)
                settle[0] += 1
                values = final
            target = self._base_command(state)
            for index, value in zip(group["joints"], values):
                target[index] = float(value)
            return target
        return self._execute(skill, parameters, command, max_seconds, region, until)

    def _arrival(self, skill, arm, group, final, pose, state):
        """None when the arm reached the motion's target, else a BLOCKED event."""
        joint_error = max(abs(state["joint_position"][i] - v) for i, v in zip(group["joints"], final))
        reached = [float(v) for v in state[f"{arm.split('_')[0]}_ee_pose"][:3]]
        if pose is not None:
            gap = math.dist(reached, pose[:3])
            if gap <= ARRIVAL_TOLERANCE_M:
                return None
        elif joint_error <= ARRIVAL_TOLERANCE_RAD:
            return None
        detail = {"reached_link6": [round(v, 4) for v in reached],
                  "reached_fingertips": [round(v, 4) for v in self.fingertip_of(state[f"{arm.split('_')[0]}_ee_pose"])],
                  "max_joint_error_rad": round(joint_error, 4),
                  "reason": "the arm stopped short of the target; contact or a load is holding it back"}
        if pose is not None:
            detail["target_link6"] = [round(v, 4) for v in pose[:3]]
            detail["gap_m"] = round(gap, 4)
        return Event(ev.BLOCKED, skill, detail, state["step"])

    def _gripper(self, skill, arm, value, max_seconds):
        group = self._arm(arm)
        index = group["gripper"]
        settled = []
        self.__dict__.setdefault("_holding_looks", {}).pop(arm, None)
        if not self._measured_gripper():
            # Only the command is reported: give the gripper its full stroke, then stop.
            seconds = float((getattr(self, "profile", None) or {}).get("gripper_full_stroke_seconds",
                                                                         DEFAULT_GRIPPER_STROKE_SECONDS))
            ticks = max(1, math.ceil(seconds * self.hz))

            def timed(state, tick):
                if tick >= ticks:
                    return None
                target = self._base_command(state)
                target[index] = value
                return target
            return self._execute(skill, {"arm": arm, "value": value}, timed, max_seconds, None, None,
                                 timeout_is_nominal=True)

        def command(state, tick):
            opening = state["joint_position"][index]
            settled.append(abs(opening - value) <= GRIPPER_TOLERANCE
                           or (len(settled) > 1 and abs(opening - self._last_opening) < GRIPPER_STALL_PER_TICK))
            self._last_opening = opening
            if len(settled) >= GRIPPER_SETTLE_TICKS and all(settled[-GRIPPER_SETTLE_TICKS:]):
                return None
            target = self._base_command(state)
            target[index] = value
            return target
        self._last_opening = self.state["joint_position"][index]
        return self._execute(skill, {"arm": arm, "value": value}, command, max_seconds, None, None,
                             timeout_is_nominal=True)

    def _unreachable_reason(self, arm, pose):
        """Say whether IK or the path failed, with the distance from the arm base."""
        base = self.profile["arm_bases"][arm][:3]
        distance = math.dist(base, pose[:3])
        where = (f"target link6 at {[round(v, 3) for v in pose[:3]]} is {distance:.2f} m from the "
                 f"{arm} base at {[round(v, 3) for v in base]}")
        if self.ik(arm, pose) is None:
            return (f"no IK solution: {where}; the position may be out of reach or the orientation "
                    "infeasible there (try a reachable orientation, e.g. tilt the approach axis down)")
        return f"IK solution exists but no collision-free path (self or table collision): {where}"
