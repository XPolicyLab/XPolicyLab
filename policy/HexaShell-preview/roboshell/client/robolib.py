"""robolib: the robot from Python, for `robo py` cells. Standard library + numpy only.

Every function sends one robo command (or none) and returns its feedback as a dict (keys readable as attributes too).
A command that fails raises RoboError (its feedback is in `.feedback`) unless called with check=False; once the episode
is over every command raises EpisodeOver. With VERBOSE = True (the default) each command prints one summary line.
The manual (AGENTS.md, section "Python") describes the functions.
"""
import base64
import json
import math
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robo as _robo  # noqa: E402  (the command line client: HTTP and observation download)

OBS = _robo.OBS_DIR
VERBOSE = True
CAMERAS = ("head", "wrist_l", "wrist_r")
_cache = {}
_views = [0]


class RoboError(Exception):
    def __init__(self, message, feedback=None):
        super().__init__(message)
        self.feedback = feedback or {}


class EpisodeOver(RoboError):
    pass


class Result(dict):
    """The feedback of a command: a dict whose keys can also be read as attributes."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def _fmt(values):
    return "[" + ", ".join(f"{v:.3f}" for v in values) + "]"


def _summary(name, payload, reply, code):
    parts = [name]
    if payload.get("arm"):
        parts.append(payload["arm"])
    if reply.get("error"):
        parts.append(f"ERROR {reply['error']}")
    if reply.get("plan_ok") is False:
        parts.append(f"FAILED {reply.get('plan_fail_reason')}: {reply.get('plan_detail') or ''}".rstrip(": "))
    if reply.get("reached_tcp"):
        parts.append(f"at {_fmt(reply['reached_tcp']['pos'])} err {reply.get('error_m', 0) * 1000:.0f} mm/{reply.get('error_deg', 0):.1f} deg")
    if "fingers_stopped" in reply:
        parts.append(f"gripper {reply.get('gripper')} opening {reply.get('opening_m')} m stopped={reply['fingers_stopped']}")
    if "reachable" in reply:
        parts.append(f"reachable={reply['reachable']}")
    if reply.get("reachable_nearby") is not None and not reply.get("reachable", False):
        parts.append(f"nearby options {len(reply['reachable_nearby'])}")
    if reply.get("sim_time_left_s") is not None:
        parts.append(f"sim left {reply['sim_time_left_s']} s")
    if reply.get("episode_over"):
        parts.append("EPISODE OVER")
    if code not in (0, None):
        parts.append(f"exit {code}")
    print("  > " + " | ".join(str(p) for p in parts))


def cmd(name, check=True, **args):
    """Send any robo command; arguments by name as on the command line (`--dx 0.1` -> dx=0.1)."""
    payload = {"cmd": name}
    payload.update({k: v for k, v in args.items() if v is not None})
    try:
        reply = _robo.post(payload)
    except Exception as error:   # timeouts and connection errors: the command may still be running
        raise RoboError(f"{name}: no reply from robo-server ({error}); call status() before trying again")
    code = int(reply.pop("exit_code", 0))
    updated = reply.pop("obs_updated", False)
    files = reply.pop("files", None)
    if files:
        os.makedirs(OBS, exist_ok=True)
        written = []
        for file_name, data in files.items():
            path = os.path.join(OBS, os.path.basename(file_name))
            with open(path, "wb") as handle:
                handle.write(base64.b64decode(data))
            written.append(path)
        reply["files"] = written
    if updated:
        _cache.clear()
        try:
            _robo.download_obs()
        except Exception as error:
            reply["obs_error"] = f"observation download failed: {error}"
    if VERBOSE and name not in ("pixels",):
        _summary(name, payload, reply, code)
    result = Result(reply)
    if code == _robo.EXIT_EPISODE_OVER or (reply.get("episode_over") and name != "done"):
        raise EpisodeOver(f"{name}: the episode is over", result)
    if check and code in (_robo.EXIT_BAD_ARGS, _robo.EXIT_EXEC_FAILED):
        raise RoboError(f"{name} failed: {reply.get('error') or reply.get('plan_fail_reason') or reply}", result)
    return result


# ---- robot ----------------------------------------------------------------
def obs():
    """Refresh obs/ (free) and return the state."""
    cmd("obs", check=False)
    return state()


def status():
    return cmd("status", check=False)


def move(arm, dx=0.0, dy=0.0, dz=0.0, check=True):
    """Translate the TCP in the world frame (each axis at most 0.20 m), orientation unchanged."""
    return cmd("move", arm=arm, dx=float(dx), dy=float(dy), dz=float(dz), check=check)


def _vec(value):
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return [float(v) for v in np.asarray(value, dtype=float).reshape(-1)]


def goto(arm, x=None, y=None, z=None, approach=None, open=None, path="line", check=True):
    """Move the TCP to (x, y, z) in a straight line (path="joint": joint interpolation, not a straight line).
    approach: "down", "forward", "down45" or a direction vector; open: "x", "y", "z", an angle in degrees about
    the approach direction, or a vector. None keeps the current position coordinate / direction."""
    return cmd("goto", arm=arm, x=None if x is None else float(x), y=None if y is None else float(y),
               z=None if z is None else float(z), approach=_vec(approach), open=_vec(open), path=path, check=check)


def reach(arm, x=None, y=None, z=None, approach=None, open=None):
    """Is this TCP pose reachable (inverse kinematics only; nothing moves, no simulation time)? Lists nearby
    reachable poses when it is not."""
    return cmd("reach", arm=arm, x=None if x is None else float(x), y=None if y is None else float(y),
               z=None if z is None else float(z), approach=_vec(approach), open=_vec(open), check=False)


def rotate(arm, roll=0.0, pitch=0.0, yaw=0.0, frame="world", check=True):
    return cmd("rotate", arm=arm, roll=float(roll), pitch=float(pitch), yaw=float(yaw), frame=frame, check=check)


def point(arm, direction, open, check=True):
    return cmd("point", arm=arm, preset=direction, open=open, check=check)


def gripper(arm, value, check=True):
    """Set the opening (1 open, 0 closed). The feedback says whether the fingers stopped on something
    (fingers_stopped) and the opening between them (opening_m)."""
    value = {"open": 1.0, "close": 0.0}.get(value, value)
    return cmd("gripper", arm=arm, value=float(value), check=check)


def open_gripper(arm, value=1.0):
    return gripper(arm, value)


def close_gripper(arm):
    return gripper(arm, 0.0)


def home(arm="both", check=True):
    return cmd("home", arm=arm, check=check)


def wait(sec):
    return cmd("wait", sec=float(sec))


def done():
    """End the episode now (see the rules in the manual)."""
    return cmd("done", check=False)


def tilt(degrees, heading):
    """Approach direction tilted `degrees` from straight down, leaning towards world-xy heading `heading`
    (degrees, 0 = +x, 90 = +y)."""
    t, h = math.radians(degrees), math.radians(heading)
    return [math.cos(h) * math.sin(t), math.sin(h) * math.sin(t), -math.cos(t)]


# ---- observation ------------------------------------------------------------
def state():
    """obs/state.json: both arms, the cameras, the time left."""
    if not os.path.exists(os.path.join(OBS, "state.json")):
        cmd("obs", check=False)
    with open(os.path.join(OBS, "state.json")) as handle:
        return json.load(handle)


def tcp(arm):
    """Position, approach direction, opening direction and gripper of one arm, from the latest observation."""
    s = state()[arm]
    return Result(pos=np.array(s["tcp_pos"]), approach=np.array(s["approach_dir"]), open=np.array(s["open_dir"]),
                  gripper=s["gripper"], gripper_observed=s.get("gripper_observed"))


def img(camera):
    """The latest image of a camera as a numpy array (480 x 640 x 3, uint8, RGB; index [v, u])."""
    if camera not in CAMERAS:
        raise ValueError(f"camera must be one of {CAMERAS}")
    if camera not in _cache:
        reply = cmd("pixels", camera="all")
        for name in CAMERAS:
            _cache[name] = np.load(os.path.join(OBS, f"{name}_rgb.npy"))
        if VERBOSE:
            print(f"  > images loaded: {', '.join(CAMERAS)}")
    return _cache[camera]


def K(camera):
    return np.array(state()["cameras"][camera]["intrinsics"], dtype=float)


def T(camera):
    """Camera-to-world 4x4 (camera axes: x right, y down, z forward)."""
    return np.array(state()["cameras"][camera]["extrinsics_world"], dtype=float)


def ray(camera, u, v):
    """(origin, unit direction) in the world of the ray through pixel (u, v)."""
    k, t = K(camera), T(camera)
    d = t[:3, :3] @ np.array([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], 1.0])
    return t[:3, 3].copy(), d / np.linalg.norm(d)


def pixel_to_plane(camera, u, v, z):
    """World point where the ray through pixel (u, v) meets the horizontal plane at height z (None if it does not)."""
    origin, d = ray(camera, u, v)
    if abs(d[2]) < 1e-9 or (z - origin[2]) / d[2] <= 0:
        return None
    return origin + (z - origin[2]) / d[2] * d


def project(point, camera):
    """Pixel (u, v) of a world point in a camera, or None if it is behind the camera."""
    k, t = K(camera), T(camera)
    p = np.linalg.inv(t) @ np.append(np.asarray(point, dtype=float), 1.0)
    if p[2] <= 1e-6:
        return None
    return np.array([k[0, 0] * p[0] / p[2] + k[0, 2], k[1, 1] * p[1] / p[2] + k[1, 2]])


def triangulate(camera_a, uv_a, camera_b, uv_b):
    """World point closest to the rays through pixel uv_a of camera_a and uv_b of camera_b (the same physical point
    seen in both). Returns (point, gap): gap is the distance between the rays in metres; a large gap means the two
    pixels are not the same point."""
    oa, da = ray(camera_a, *uv_a)
    ob, db = ray(camera_b, *uv_b)
    w = oa - ob
    a, b, c, d, e = da @ da, da @ db, db @ db, da @ w, db @ w
    denom = a * c - b * b
    if abs(denom) < 1e-12:
        return None, float("inf")
    s, t = (b * e - c * d) / denom, (a * e - b * d) / denom
    pa, pb = oa + s * da, ob + t * db
    return (pa + pb) / 2, float(np.linalg.norm(pa - pb))


# ---- tools ------------------------------------------------------------------
def tools():
    """The extra commands of this manual (name -> argument names)."""
    return {c["name"]: [a["name"] for a in c.get("args", [])] for c in _robo.extra_commands()}


def tool(name, *positional, check=True, **args):
    """Run an extra command. Positional values fill the command's positional arguments (e.g. the arm) in order."""
    spec = {c["name"]: c for c in _robo.extra_commands()}.get(name)
    if spec is None:
        raise RoboError(f"unknown command {name!r}; available: {sorted(tools())}")
    names = [a["name"] for a in spec.get("args", []) if a.get("positional")]
    if len(positional) > len(names):
        raise RoboError(f"{name} takes {len(names)} positional argument(s): {names}")
    args.update(dict(zip(names, positional)))
    return cmd(name, check=check, **args)


def table_z(refresh=False):
    """World z of the table top, measured from the images (table_plane)."""
    reply = tool("table_plane", refresh=1 if refresh else 0, check=False)
    return reply.get("table_z")


def segment(camera="head", min_area=None, check=True):
    """Object regions in one camera image (`robo segment`); the overlay is written to obs/seg_<camera>.png."""
    return tool("segment", camera=camera, min_area=min_area, check=check)


def locate(camera="head", id=None, u=None, v=None, camera_b=None, check=True):
    """3D top centre, size and yaw of one object from two or more cameras (`robo locate`)."""
    return tool("locate", camera=camera, id=id, u=u, v=v, camera_b=camera_b, check=check)


# ---- pictures for looking -----------------------------------------------------
_DIGITS = {  # 3 x 5 pixel digits for labels
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111",
    "4": "101101111001001", "5": "111100111001111", "6": "111100111101111", "7": "111001010010010",
    "8": "111101111101111", "9": "111101111001111", "-": "000000111000000", ".": "000000000000010",
}


def png_bytes(image):
    """Encode an H x W x 3 (or H x W) uint8 array as PNG."""
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    image = np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8)[:, :, :3])
    h, w = image.shape[:2]
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(h))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def save_png(path, image):
    with open(path, "wb") as handle:
        handle.write(png_bytes(image))
    return path


def _label(canvas, x, y, text, color, scale):
    for i, ch in enumerate(str(text)):
        pattern = _DIGITS.get(ch)
        if pattern is None:
            continue
        for k, bit in enumerate(pattern):
            if bit == "1":
                r, c = k // 3, k % 3
                y0, x0 = y + r * scale, x + i * 4 * scale + c * scale
                canvas[max(0, y0):max(0, y0 + scale), max(0, x0):max(0, x0 + scale)] = color


def show(camera_or_image, points=(), boxes=(), mask=None, crop=None, scale=1, path=None):
    """Write a picture to look at (with your image viewer) and return its path.
    points: [(u, v), ...] drawn as numbered crosses; boxes: [(u0, v0, u1, v1), ...] numbered rectangles;
    mask: boolean H x W array tinted; crop: (u0, v0, u1, v1) region to keep; scale: integer zoom (after cropping).
    Coordinates are pixels of the full image."""
    image = img(camera_or_image) if isinstance(camera_or_image, str) else np.asarray(camera_or_image)
    canvas = image.copy() if image.ndim == 3 else np.repeat(np.asarray(image)[:, :, None], 3, axis=2).copy()
    canvas = canvas.astype(np.uint8)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        canvas[m] = (0.5 * canvas[m] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    h, w = canvas.shape[:2]
    for n, (u0, v0, u1, v1) in enumerate(boxes):
        c = colors[n % len(colors)]
        u0, v0, u1, v1 = [int(round(t)) for t in (u0, v0, u1, v1)]
        u0, u1, v0, v1 = max(0, u0), min(w - 1, u1), max(0, v0), min(h - 1, v1)
        canvas[v0:v1 + 1, [u0, u1]] = c
        canvas[[v0, v1], u0:u1 + 1] = c
        _label(canvas, u0 + 2, max(0, v0 - 12), n, c, 2)
    for n, (u, v) in enumerate(points):
        c = colors[n % len(colors)]
        u, v = int(round(u)), int(round(v))
        canvas[max(0, v - 6):v + 7, max(0, u - 1):u + 2] = c
        canvas[max(0, v - 1):v + 2, max(0, u - 6):u + 7] = c
        _label(canvas, u + 5, v + 5, n, c, 2)
    if crop is not None:
        u0, v0, u1, v1 = [int(round(t)) for t in crop]
        canvas = canvas[max(0, v0):v1, max(0, u0):u1]
    if scale and int(scale) > 1:
        canvas = np.repeat(np.repeat(canvas, int(scale), axis=0), int(scale), axis=1)
    if path is None:
        _views[0] += 1
        path = os.path.join(OBS, f"view_{_views[0]:03d}.png")
    return save_png(path, canvas)
