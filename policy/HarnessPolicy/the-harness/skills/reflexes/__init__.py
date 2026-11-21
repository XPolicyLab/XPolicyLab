"""Ready-made reactive behaviors over the reactive layer (optional: configs/ablations/reflexes.toml).

intercept(arm, "o2") catches an object that moves past, such as one on a conveyor, with a
controller that runs every control tick: it follows the object's look (its image patch) in
the head camera's images, aims the jaws at where the object will be, comes down when they
are above it and closes while still following, all without asking the model. A program sees
the scene only between motions; a moving object is gone by the time a looked-up position
is reached, which is why this runs in the reactive layer.
"""

import math

import numpy as np

from harness_policy.geometry import down_with_yaw, pixel_to_plane
from skills.eef import MIN_GRASP_Z


HOVER_M = 0.08        # the fingertips follow this far above the grasp height
ALIGN_M = 0.015       # within this (xy) of where the object will be, the jaws come down
ARRIVE_M = 0.012      # this close to the grasp height, the jaws close
CLOSE_TICKS = 12      # ticks the jaws close while still following
LIFT_M = 0.10         # the lift after a catch
# Lifts tried in turn (rise, share of the way back toward the arm's base), the first
# within reach is made: straight up, else up and back.
LIFTS = ((LIFT_M, 0.0), (LIFT_M, 0.3), (LIFT_M / 2, 0.5))
LEAD_S = 0.3          # while following, aim where the object will be this much later
# Coming down and closing, the jaws aim where the object is (plus two ticks of command
# latency): with LEAD_S they closed ~3 cm ahead of it and missed or held it by an edge.
CLOSE_LEAD_S = 0.08
WINDOW_PX = 32        # half-size of the image window searched around the last sighting (a belt moves ~10 px a frame)
LOST_FRAMES = 5       # camera frames without a sighting before the controller gives up
# With MIN_TRACK sightings the motion is fitted, and the arm coming over the object (it
# hides it from the head camera well before the jaws arrive) no longer stops the catch:
# the controller goes on with the fitted motion for up to COAST_FRAMES unseen frames.
MIN_TRACK = 3
COAST_FRAMES = 30
MIN_PIXELS = 12       # pixels of the object's colour that count as a sighting
COLOR_MIN_TOL = 30.0  # smallest colour distance (sum over RGB) that still matches
CAMERA_HZ = 10.0      # camera frames per second while following
# The most the commanded link6 pose moves in one control tick, so that no tick asks for a
# joint jump the shield refuses (0.3 rad on the ARX X5): from wherever the arm is, the
# first ticks bring the open jaws over the object instead of jumping there.
MAX_STEP_M = 0.02     # 0.5 m/s at 25 Hz, faster than a conveyor
MAX_TURN_RAD = 0.12   # the wrist's turn to the grasp yaw
# The patch tracker: the object's grey-level patch (its mask's box and a margin) is
# compared by normalised correlation, first every COARSE_STRIDE-th pixel at every
# COARSE_STRIDE-th place of the search window, then every PATCH_STRIDE-th pixel at every
# place near the best one (a few ms a frame for a 60-pixel patch, within the reactive
# budget). A colour alone matched the belt and the arm around a white object in the
# paper runs (tens of thousands of pixels), and the controller followed those.
COARSE_STRIDE = 4
PATCH_STRIDE = 2
PATCH_MARGIN_PX = 6   # the patch extends the mask's box by this much (the object's outline)
MIN_MATCH = 0.6       # correlation that counts as a sighting
# A sighting this far from where the fitted motion puts the object is something else
# (the arm over it, another object); the velocity is a straight-line fit of this many
# recent sightings, not of two frames.
GATE_M = 0.04
HISTORY = 8


class ColorTracker:
    """Follows one object in a camera's images by its colour: the median colour of its
    mask, searched in a window around the last pixel where it was seen. numpy only, well
    within the reactive layer's per-tick budget."""

    def __init__(self, image, mask):
        # locate keeps geometry.Mask objects (pixels in .array); a plain array works too.
        mask = np.asarray(getattr(mask, "array", mask), dtype=bool)
        pixels = np.asarray(image, dtype=np.float32)[mask]
        if len(pixels) < MIN_PIXELS:
            raise ValueError("the object's mask is too small to track")
        self.color = np.median(pixels, axis=0)
        spread = float(np.median(np.abs(pixels - self.color).sum(axis=1)))
        self.tolerance = max(COLOR_MIN_TOL, 2.5 * spread)
        ys, xs = np.nonzero(mask)
        self.uv = (float(xs.mean()), float(ys.mean()))

    def find(self, image, window=WINDOW_PX):
        """The object's pixel (u, v) in `image` near its last one, or None; a sighting
        moves the search window with it."""
        height, width = image.shape[:2]
        u0, v0 = int(round(self.uv[0])), int(round(self.uv[1]))
        x0, x1 = max(0, u0 - window), min(width, u0 + window + 1)
        y0, y1 = max(0, v0 - window), min(height, v0 + window + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float32)
        ys, xs = np.nonzero(np.abs(patch - self.color).sum(axis=2) < self.tolerance)
        if len(xs) < MIN_PIXELS:
            return None
        self.uv = (float(xs.mean()) + x0, float(ys.mean()) + y0)
        return self.uv


class PatchTracker:
    """Follows one object in a camera's images by its look: the grey-level patch of its
    mask's box (plus PATCH_MARGIN_PX), found by normalised correlation in a window around
    the last sighting. Unlike a colour, the patch does not match a belt or table of a
    similar colour, and an arm over the object lowers the match instead of moving it.
    numpy only, a few milliseconds a frame."""

    def __init__(self, image, mask):
        mask = np.asarray(getattr(mask, "array", mask), dtype=bool)
        if int(mask.sum()) < MIN_PIXELS:
            raise ValueError("the object's mask is too small to track")
        height, width = mask.shape
        ys, xs = np.nonzero(mask)
        y0, x0 = max(0, ys.min() - PATCH_MARGIN_PX), max(0, xs.min() - PATCH_MARGIN_PX)
        y1, x1 = min(height, ys.max() + 1 + PATCH_MARGIN_PX), min(width, xs.max() + 1 + PATCH_MARGIN_PX)
        self.patch = _grey(np.asarray(image)[y0:y1, x0:x1])
        if self.patch.size < MIN_PIXELS or float(self.patch.std()) < 1e-3:
            raise ValueError("the object's patch has no texture to track")
        # A coarse template finds the place in the window, a finer one refines it nearby.
        self.templates = {stride: _normalised(self.patch[::stride, ::stride]) for stride in (COARSE_STRIDE, PATCH_STRIDE)}
        # The sighting is the mask's centre: its place relative to the patch's corner.
        self.center = (float(xs.mean()) - x0, float(ys.mean()) - y0)
        self.uv = (float(xs.mean()), float(ys.mean()))
        self.match = 1.0

    def find(self, image, window=WINDOW_PX):
        """The object's pixel (u, v) in `image` near its last one, or None when nothing in
        the window matches it well enough; a sighting moves the window with it."""
        image = np.asarray(image)
        corner = (int(round(self.uv[1] - self.center[1])), int(round(self.uv[0] - self.center[0])))
        coarse = self._best(image, corner, window, COARSE_STRIDE)
        if coarse is None:
            return None
        fine = self._best(image, coarse[0], COARSE_STRIDE, 1, stride=PATCH_STRIDE)
        if fine is None:
            return None
        (top, left), self.match = fine
        if self.match < MIN_MATCH:
            return None
        self.uv = (left + self.center[0], top + self.center[1])
        return self.uv

    def _best(self, image, corner, reach, step, stride=None):
        """((top, left), correlation) of the best patch corner within `reach` pixels of
        `corner`, every `step` pixels, comparing every `stride`-th pixel (default `step`)."""
        stride = stride or step
        template = self.templates[stride]
        height, width = image.shape[:2]
        span_y, span_x = self.patch.shape
        top0, left0 = max(0, corner[0] - reach), max(0, corner[1] - reach)
        bottom, right = min(height, corner[0] + reach + span_y), min(width, corner[1] + reach + span_x)
        if bottom - top0 < span_y or right - left0 < span_x:
            return None
        region = _grey(image[top0:bottom, left0:right])
        windows = np.lib.stride_tricks.sliding_window_view(region, (span_y, span_x))[::step, ::step, ::stride, ::stride]
        flat = windows.reshape(windows.shape[0], windows.shape[1], -1)
        mean = flat.mean(axis=2, keepdims=True)
        std = flat.std(axis=2)
        score = ((flat - mean) @ template) / (template.size * np.maximum(std, 1e-3))
        score[std < 1e-3] = -1.0
        row, col = np.unravel_index(int(np.argmax(score)), score.shape)
        return (top0 + row * step, left0 + col * step), float(score[row, col])


def _grey(image):
    return np.asarray(image, dtype=np.float32)[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _normalised(patch):
    return ((patch - patch.mean()) / max(float(patch.std()), 1e-3)).ravel()


def fit_velocity(history):
    """The (vx, vy) of a straight-line fit of [(t, x, y), ...] (0, 0 for fewer than two)."""
    if len(history) < 2:
        return (0.0, 0.0)
    t = np.array([h[0] for h in history])
    t = t - t.mean()
    denominator = float((t * t).sum())
    if denominator <= 0.0:
        return (0.0, 0.0)
    return tuple(float((t * (np.array([h[k] for h in history]) - np.mean([h[k] for h in history]))).sum()
                       / denominator) for k in (1, 2))


def step_toward(current, target, max_step=MAX_STEP_M, max_turn=MAX_TURN_RAD):
    """The link6 pose [x, y, z, qw, qx, qy, qz] at most `max_step` metres and `max_turn`
    radians from `current` on the way to `target`."""
    current, target = np.asarray(current, dtype=float), np.asarray(target, dtype=float)
    delta = target[:3] - current[:3]
    distance = float(np.linalg.norm(delta))
    position = target[:3] if distance <= max_step else current[:3] + delta * (max_step / distance)
    q0, q1 = current[3:] / np.linalg.norm(current[3:]), target[3:] / np.linalg.norm(target[3:])
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    angle = 2.0 * math.acos(min(1.0, dot))
    if angle <= max_turn:
        quaternion = q1
    else:
        half, fraction = angle / 2.0, max_turn / angle
        quaternion = (math.sin((1.0 - fraction) * half) * q0 + math.sin(fraction * half) * q1) / math.sin(half)
    return [float(v) for v in position] + [float(v) for v in quaternion]


def intercept_controller(tracker, camera, grasp_z, orientation, arm, link6_for_tip, lead_seconds=LEAD_S):
    """The controller of intercept: fn(obs) -> command or None, with its own state. It
    follows the object at HOVER_M above `grasp_z` toward where it will be in
    `lead_seconds`, comes down once the jaws are within ALIGN_M of that point (aiming then
    at where it is, CLOSE_LEAD_S), closes at
    the grasp height while still following, and finishes CLOSE_TICKS later. Each tick's
    pose is at most MAX_STEP_M and MAX_TURN_RAD from the last one (from the arm's own pose
    on the first tick). It gives up (None) when the object is lost for LOST_FRAMES frames
    before its motion is fitted (MIN_TRACK sightings), or for COAST_FRAMES after, before
    the jaws came down. `state["phase"]` says how far it got."""
    side = arm.split("_")[0]
    state = {"xy": None, "t": None, "v": (0.0, 0.0), "frame": None, "lost": 0, "phase": "follow",
             "closing": 0, "sightings": 0, "rejected": 0, "command": None, "history": []}

    def controller(obs):
        frame = obs.get("observation")
        if frame is not None and frame is not state["frame"]:
            state["frame"] = frame
            previous = tracker.uv
            uv = tracker.find(frame.images[camera])
            point = None if uv is None else pixel_to_plane(frame, camera, uv[0], uv[1], grasp_z)
            t = float(obs["t"])
            if point is not None and len(state["history"]) >= 3:
                # Far from where the fitted motion puts it: something else (the arm over it).
                expected = (state["xy"][0] + state["v"][0] * (t - state["t"]),
                            state["xy"][1] + state["v"][1] * (t - state["t"]))
                if math.hypot(point[0] - expected[0], point[1] - expected[1]) > GATE_M:
                    state["rejected"] += 1
                    tracker.uv = previous
                    point = None
            if point is None:
                state["lost"] += 1
            else:
                state["lost"] = 0
                state["sightings"] += 1
                state["history"] = (state["history"] + [(t, float(point[0]), float(point[1]))])[-HISTORY:]
                state["v"] = fit_velocity(state["history"])
                state["xy"], state["t"] = (float(point[0]), float(point[1])), t
        # Once its motion is fitted, the object is followed through the frames the arm hides
        # it (a belt moves it steadily): the controller coasts on the fit for COAST_FRAMES.
        patience = COAST_FRAMES if len(state["history"]) >= MIN_TRACK else LOST_FRAMES
        if state["phase"] == "follow" and (state["xy"] is None or state["lost"] >= patience):
            if state["lost"] >= patience or obs["t"] > 1.0:
                state["phase"] = "lost"
                return None
            return {}
        ahead = float(obs["t"]) - state["t"] + (lead_seconds if state["phase"] == "follow" else CLOSE_LEAD_S)
        target = (state["xy"][0] + state["v"][0] * ahead, state["xy"][1] + state["v"][1] * ahead)
        here = obs[f"{side}_ee"]  # link6; pointing down, its xy is the fingertips'
        if state["phase"] == "follow" and math.hypot(here[0] - target[0], here[1] - target[1]) < ALIGN_M:
            state["phase"] = "descend"
        tip_z = grasp_z + (HOVER_M if state["phase"] == "follow" else 0.0)
        pose = link6_for_tip([target[0], target[1], tip_z], orientation)
        state["command"] = step_toward(state["command"] or obs[f"{side}_ee_pose"], pose)
        command = {f"{side}_eef": state["command"]}
        if state["phase"] == "descend" and abs(here[2] - pose[2]) < ARRIVE_M:
            state["phase"] = "close"
        if state["phase"] == "close":
            command[f"{side}_gripper"] = 0.0
            state["closing"] += 1
            if state["closing"] > CLOSE_TICKS:
                state["phase"] = "closed"
                return None
        return command

    return controller, state


class Reflexes:
    def intercept(self, arm, object, max_seconds=20.0, camera="cam_head", lead_seconds=LEAD_S):
        """Catch `object` (an id from the last locate) with `arm` while it moves past, as
        on a conveyor, then lift it LIFT_M. A controller follows the object's image patch
        in the camera's images every control tick, aims the open jaws at where it will be,
        comes down when they are above it and closes while still following; the model is
        not asked while it runs. Locate the object just before (its mask in `camera`
        gives its look); the jaws must be open. The lift goes straight up, or, where that
        is out of reach (the far side of a belt), partly back toward the arm's base. Returns {"caught": bool, "phase",
        "sightings", "controller": trace, "lift": trace or None}; "caught" means the
        jaws closed at the object, so check the grip (the scene after the program, or
        holding) before relying on it."""
        scene = {item["id"]: item for item in self.__dict__.get("_eef_scene") or []}
        if object not in scene:
            raise ValueError(f"{object!r} is not an object of the last locate; locate it first")
        observation, masks = self.__dict__["_eef_masks"]
        mask = masks.get(object, {}).get(camera)
        if mask is None:
            raise ValueError(f"{object} is not seen by {camera}; locate it again when it is in view")
        item = scene[object]
        tracker = PatchTracker(observation.images[camera], mask)
        angle = float(item.get("angle_deg") or 0.0)
        # Jaws closing across the object's long side, as pick grasps it.
        orientation = down_with_yaw(math.radians(((angle + 180.0) % 180.0) - 90.0))
        # The lowest straight-down grasp, as pick's: a flat object's own height would put the
        # jaws into the table, where they never arrive to close.
        grasp_z = max(MIN_GRASP_Z, float(item["position"][2]))
        controller, state = intercept_controller(tracker, camera, grasp_z, orientation, arm,
                                                 self.link6_for_tip, lead_seconds)
        trace = self.run_controller(controller, max_seconds=max_seconds, cameras=[camera], camera_hz=CAMERA_HZ)
        caught = state["phase"] == "closed"
        lift = None
        if caught:
            pose = list(self.get_state()[f"{arm.split('_')[0]}_ee_pose"])
            base = ((getattr(self, "profile", None) or {}).get("arm_bases") or {}).get(arm)
            ik = getattr(self, "ik", None)
            for rise, back in LIFTS:
                target = list(pose)
                target[2] += rise
                if base is not None:
                    target[0] += back * (float(base[0]) - pose[0])
                    target[1] += back * (float(base[1]) - pose[1])
                if ik is None or ik(arm, target) is not None:
                    lift = self.move_eef(arm, target, max_seconds=3.0)
                    break
        return {"caught": caught, "phase": state["phase"], "sightings": state["sightings"],
                "rejected": state["rejected"], "controller": trace, "lift": lift}
