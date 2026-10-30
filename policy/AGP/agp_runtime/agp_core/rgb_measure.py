"""`measure` without depth (switch OURS_RGB_ONLY; agp_core/adapter.py routes measurements here when it is on).

Runs in the MCP server process (numpy + PIL only) on the files agp_core/rgb_observe.py wrote into the session's
capture directories: the three RGB images, calibration.json (public camera constants; no depth) and
robot_state.json (official joints / ee_pose / gripper command and the URDF link poses). It never reads a depth file
and has no access to the simulator.

A query {arm, cam, u, v, plane_z?} gives a point in that arm's base frame (table top z = 0):
  * The pixel ray uses the pixel-centre convention: pixel (u, v) = image[v, u] has its centre at continuous
    coordinates (u + 0.5, v + 0.5) (agp_core/public_camera.py; not K^-1 [u, v, 1] at the pixel corner).
  * plane_z given: the point where the ray meets the horizontal plane z = plane_z (height source 'plane').
  * plane_z absent: the height of the picked surface from the epipolar matcher (agp_core/zmatch.py). A top pick is
    matched into the measuring arm's wrist image (the other wrist if the point is not in view there); a wrist pick
    into the top image. The view that supplies the template or the partner image may come from an earlier capture
    of this session when the robot covers the pixel in the latest one (the top camera is fixed; this assumes the
    target has not moved since). Height source 'match' when the two-way check and the margin gate pass, else
    'uncertain' with the best guess. Each matched point returns a review image: both views with the pick and its
    match marked.
  * Every query is checked against the robot's own silhouette (public URDF link hulls, agp_core/x5_link_hulls.json,
    at the capture's joint states, projected with the public camera model): `on_robot` names the arm whose
    outline contains the pixel, i.e. the pixel shows the robot, not the scene behind it.

Raised targets (matching is the default for them):
  * plane_z at or below TABLE_PLANE_M (the table, z = 0, or a support plane within 5 mm of it): the exact plane
    intersection, unchanged (height source 'plane').
  * plane_z above TABLE_PLANE_M (a raised surface): the matcher runs first, exactly as without plane_z. Its height
    is used when it is a confident 'match' within AGREE_M of plane_z (source 'match', height.plane_z records the
    given value); otherwise the given plane is the fallback (source 'fallback', height.fallback_reason 'uncertain',
    'none' or 'disagrees', height.match the matcher's own result). A fallback returns no review image.
  * A 'match' at an end of the swept height range is reported as 'uncertain' (edge_rule).
  * Earlier captures are not used where a closed gripper may carry the target: a top pick whose matcher window
    touches the silhouette of a gripper that was last commanded closed, and a pick in the wrist image of an arm
    whose gripper was last commanded closed. A held object moves with the gripper, so an earlier top image shows
    the scene without it there (in recorded sessions such views gave confident matches on the surface below).
"""
import json
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw

from migration.pad_projection import selected_point_pad_projection
from agp_core import zmatch

CAMERAS = ("top", "left_wrist", "right_wrist")
ARMS = ("left", "right")
HULLS = Path(__file__).with_name("x5_link_hulls.json")
NEAR_M = 0.02                    # silhouettes are clipped at this optical depth (m)
OWN_WRIST_LINKS = ("link7", "link8")   # the only own-arm links a wrist camera sees (its fingers)
EARLIER_CAPTURES = 40            # how far back a view may be taken from
WINDOW_PX = 12                   # largest matcher window radius (zmatch.VARIANTS) at full resolution
WINDOW_MARGIN_PX = 3
MIN_FREE_FRACTION = 0.9          # share of a partner corridor that must be free of the robot
TABLE_PLANE_M = 0.005            # plane_z at or below this is exact (table or a support plane near it)
AGREE_M = 0.02                   # a match farther than this from a raised plane_z is not used
GRIPPER_LINKS = ("link6", "link7", "link8", "camera", "camera_base")   # the gripper's links (compact.GRIPPER_LINKS)


# ----------------------------------------------------------------------------- small geometry
def pixel_ray(K, T, u, v):
    """Ray (origin, unit direction) through the centre of pixel index (u, v) of camera (K, T = frame <- optical)."""
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    d = T[:3, :3] @ np.array([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], 1.0])
    return T[:3, 3].copy(), d / np.linalg.norm(d)


def intersect_z(origin, direction, z):
    """Point where origin + s * direction (s > 0) meets the plane at height z, or None."""
    if abs(direction[2]) < 1e-9:
        return None
    s = (z - origin[2]) / direction[2]
    return None if not np.isfinite(s) or s <= 0 else origin + s * direction


def project(K, T, X):
    """Continuous image coordinates (pixel centres at +0.5) and optical depth of points X (frame of T)."""
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    xc = (np.asarray(X, float) - T[:3, 3]) @ T[:3, :3]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = xc[..., :2] / xc[..., 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, xc[..., 2]


def clip_near(points, near):
    """Vertices of the convex hull of `points` (optical frame, (N, 3)) clipped to z >= near.

    The clipped hull's vertices are the points in front of the plane plus the points where hull edges cross it;
    the crossings of every front/behind pair lie on or inside the clipped hull, so adding all of them gives the
    same convex hull without knowing the edges."""
    points = np.asarray(points, float)
    front = points[points[:, 2] >= near]
    behind = points[points[:, 2] < near]
    if len(front) == 0 or len(behind) == 0:
        return front
    a = front[:, None, :]
    b = behind[None, :, :]
    s = (near - a[..., 2]) / (b[..., 2] - a[..., 2])
    crossing = (a + s[..., None] * (b - a)).reshape(-1, 3)
    return np.vstack([front, crossing])


def convex_hull_2d(points):
    """Counter-clockwise convex hull (Andrew's monotone chain) of (N, 2) points."""
    pts = np.unique(np.round(np.asarray(points, float), 6), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(tuple(p))
        return out
    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def inside_convex(polygon, x, y, margin=0.0):
    """True if (x, y) lies inside the counter-clockwise convex polygon, or within `margin` px of it."""
    if len(polygon) < 3:
        return False
    a = polygon
    b = np.roll(polygon, -1, axis=0)
    edge = b - a
    length = np.linalg.norm(edge, axis=1)
    cross = (edge[:, 0] * (y - a[:, 1]) - edge[:, 1] * (x - a[:, 0])) / np.where(length > 0, length, 1)
    return bool(np.all(cross >= -margin))


# ----------------------------------------------------------------------------- capture files
@lru_cache(maxsize=1)
def link_hulls():
    """{link: [convex part vertices (N, 3)]} and {link: (mesh vertices, faces)} for the links with triangles."""
    data = json.loads(HULLS.read_text())
    hulls = {name: [np.asarray(part, float) for part in link["parts"]] for name, link in data["links"].items()}
    meshes = {name: (np.asarray(link["mesh_vertices"], float), np.asarray(link["faces"], int))
              for name, link in data["links"].items() if "faces" in link}
    return hulls, meshes


def clip_polygon_near(polygon, near):
    """Sutherland-Hodgman clip of one planar polygon (optical frame, (n, 3)) to z >= near."""
    out = []
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i + 1) % len(polygon)]
        if a[2] >= near:
            out.append(a)
        if (a[2] >= near) != (b[2] >= near):
            out.append(a + (near - a[2]) / (b[2] - a[2]) * (b - a))
    return np.asarray(out)


def _to_pixels(K, optical):
    return optical[..., :2] / optical[..., 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def _in_triangles(triangles, x, y):
    """True where (x, y) lies inside any of the (n, 3, 2) triangles (either orientation)."""
    if len(triangles) == 0:
        return False
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    def side(p, q):
        return (q[:, 0] - p[:, 0]) * (y - p[:, 1]) - (q[:, 1] - p[:, 1]) * (x - p[:, 0])
    s1, s2, s3 = side(a, b), side(b, c), side(c, a)
    return bool(np.any(((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))))


class Capture:
    """One capture directory written by agp_core/rgb_observe.py (cached per process)."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.id = self.directory.name
        self.calibration = json.loads((self.directory / "calibration.json").read_text())
        state = self.directory / "robot_state.json"
        self.robot = json.loads(state.read_text()) if state.is_file() else None
        self._images, self._masks, self._shapes = {}, {}, {}

    def image(self, cam):
        if cam not in self._images:
            with Image.open(self.directory / f"{cam}.png") as img:
                self._images[cam] = np.asarray(img.convert("RGB"))
        return self._images[cam]

    def K(self, cam):
        return np.asarray(self.calibration[cam]["K"], float)

    def T(self, cam, arm):
        return np.asarray(self.calibration[cam]["T_base_optical_by_arm"][arm], float)

    def tcp(self, arm):
        return np.asarray(self.calibration["top"]["T_base_tcp_by_arm"][arm], float)

    def size(self, cam):
        K = self.K(cam)
        return int(round(2 * K[1, 2])), int(round(2 * K[0, 2]))     # (height, width): principal point at centre

    def shapes(self, cam):
        """The robot seen by `cam`, in continuous image coordinates: [(arm, link, kind, data)] with kind 'hull'
        (one convex polygon: the projected hull of one convex part of the link) or 'mesh' (own fingers in a wrist
        camera: (n, 3, 2) triangles plus a list of near-plane-clipped polygons). A wrist camera's own fingers are
        drawn only while the gripper is commanded open (then they sit at the open limit); a closed gripper stops on
        whatever it holds and its opening is not observed, so its fingers are left out of that wrist view. In the
        other views the fingers are drawn at the commanded end position (robot_state.json finger_q_assumed_m). Grip
        width (OURS_GRIP_WIDTH): a gripper commanded to a partial width (gripper_width_command) has its own fingers
        drawn in its wrist view too, at that width's finger target (finger_q_assumed_m)."""
        if cam in self._shapes:
            return self._shapes[cam]
        out = []
        if self.robot is not None:
            K = self.K(cam)
            T = np.asarray(self.calibration[cam]["T_world_optical"], float)
            hulls, meshes = link_hulls()
            own = cam.split("_")[0] if cam.endswith("_wrist") else None
            for arm, state in self.robot["arms"].items():
                for link, pose in state["T_world_links"].items():
                    if link not in hulls or (arm == own and link not in OWN_WRIST_LINKS):
                        continue
                    pose = np.asarray(pose, float)
                    if arm == own and link in meshes:
                        if float(state.get("gripper_command", 1.0)) < 0.5 and state.get("gripper_width_command") is None:
                            continue          # closed: the fingers stop on what they hold; the opening is unobserved
                        vertices, faces = meshes[link]
                        optical = ((vertices @ pose[:3, :3].T + pose[:3, 3]) - T[:3, 3]) @ T[:3, :3]
                        tris = optical[faces]
                        front = (tris[..., 2] >= NEAR_M).all(1)
                        partial = (tris[..., 2] >= NEAR_M).any(1) & ~front
                        clipped = [_to_pixels(K, c) for c in (clip_polygon_near(t, NEAR_M) for t in tris[partial])
                                   if len(c) >= 3]
                        out.append((arm, link, "mesh", (_to_pixels(K, tris[front]), clipped)))
                        continue
                    for part in hulls[link]:
                        world = part @ pose[:3, :3].T + pose[:3, 3]
                        optical = clip_near((world - T[:3, 3]) @ T[:3, :3], NEAR_M)
                        if len(optical) >= 3:
                            out.append((arm, link, "hull", convex_hull_2d(_to_pixels(K, optical))))
        self._shapes[cam] = out
        return out

    def robot_mask(self, cam):
        """HxW bool: pixels inside the robot's projected silhouette in `cam`."""
        if cam not in self._masks:
            self._masks[cam] = self._draw(cam, lambda arm, link: True)
        return self._masks[cam]

    def _draw(self, cam, select):
        """HxW bool mask of the shapes of `cam` whose (arm, link) passes `select`."""
        h, w = self.size(cam)
        canvas = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(canvas)

        def fill(polygon):
            if len(polygon) >= 3:
                draw.polygon([tuple(p) for p in np.clip(polygon, -1e4, 1e4)], fill=255)
        for arm, link, kind, data in self.shapes(cam):
            if not select(arm, link):
                continue
            if kind == "hull":
                fill(data)
            else:
                for triangle in data[0]:
                    fill(triangle)
                for polygon in data[1]:
                    fill(polygon)
        return np.asarray(canvas) > 0

    def gripper_closed(self, arm):
        """True when `arm`'s gripper was last commanded closed (robot_state.json gripper_command < 0.5). With
        OURS_GRIP_WIDTH a partial width (gripper_width_command) counts as closed here, since it may hold what it
        sees."""
        if self.robot is None:
            return False
        if (self.robot["arms"].get(arm) or {}).get("gripper_width_command") is not None:
            return True
        return float((self.robot["arms"].get(arm) or {}).get("gripper_command", 1.0)) < 0.5

    def closed_gripper_mask(self, cam):
        """HxW bool: pixels inside the silhouette of a gripper (GRIPPER_LINKS) whose gripper is closed."""
        key = ("closed_gripper", cam)
        if key not in self._masks:
            closed = {arm for arm in ARMS if self.gripper_closed(arm)}
            self._masks[key] = (self._draw(cam, lambda arm, link: arm in closed and link in GRIPPER_LINKS)
                                if closed else np.zeros(self.size(cam), bool))
        return self._masks[key]

    def on_robot(self, cam, u, v):
        """(arm, link) whose silhouette contains the centre of pixel (u, v), or None."""
        x, y = u + 0.5, v + 0.5
        for arm, link, kind, data in self.shapes(cam):
            if kind == "hull":
                hit = inside_convex(data, x, y)
            else:
                hit = _in_triangles(data[0], x, y) or any(inside_convex(convex_hull_2d(c), x, y) for c in data[1])
            if hit:
                return arm, link
        return None


@lru_cache(maxsize=64)
def _capture(directory):
    return Capture(directory)


def load_capture(session, capture):
    session = Path(session).resolve()
    directory = (session / "captures" / str(capture)).resolve()
    if not directory.is_relative_to(session) or not (directory / "calibration.json").is_file():
        raise ValueError(f"No saved capture {capture} in this session")
    return _capture(str(directory))


def earlier_captures(session, capture, limit=EARLIER_CAPTURES):
    """Capture ids of this session older than `capture`, newest first (ids are monotonic ns)."""
    root = Path(session) / "captures"
    ids = sorted((int(p.name) for p in root.iterdir() if p.name.isdigit() and int(p.name) < int(capture)
                  and (p / "robot_state.json").is_file()), reverse=True)
    return [str(i) for i in ids[:limit]]


def _window_free(mask, x, y, radius):
    """No robot pixel within the square window of `radius` px around continuous (x, y); False if off-image."""
    h, w = mask.shape
    i, j = int(np.floor(y)), int(np.floor(x))
    if not (0 <= i < h and 0 <= j < w):
        return False
    return not mask[max(0, i - radius):i + radius + 1, max(0, j - radius):j + radius + 1].any()


# ----------------------------------------------------------------------------- view selection
def template_capture(session, current, cam, u, v):
    """Capture whose `cam` image shows pixel (u, v)'s window free of the robot: the current one if possible,
    else the newest earlier one (only for the fixed top camera), else the current one."""
    radius = WINDOW_PX + WINDOW_MARGIN_PX
    if _window_free(current.robot_mask(cam), u + 0.5, v + 0.5, radius) or cam != "top":
        return current
    if not _window_free(current.closed_gripper_mask(cam), u + 0.5, v + 0.5, radius):
        return current      # a closed gripper next to the pick may hold the target, which then moved
    for capture in earlier_captures(session, current.id):
        other = load_capture(session, capture)
        if _window_free(other.robot_mask(cam), u + 0.5, v + 0.5, radius):
            return other
    return current


def corridor(K, T, p, Kb, Tb, hs):
    """Continuous pixels in camera b of the points of pixel p's ray at the swept heights (inside b only)."""
    points = np.array([zmatch.ray_point(K, T, p, h) for h in hs])
    uv, depth = project(Kb, Tb, points)
    return uv[depth > 0.02]


def free_fraction(mask, uv, radius):
    h, w = mask.shape
    inside = [(x, y) for x, y in uv if 0 <= x < w and 0 <= y < h]
    if not inside:
        return 0.0
    return float(np.mean([_window_free(mask, x, y, radius) for x, y in inside]))


def partner_capture(session, current, cam_b, ray_view, p, arm, allow_earlier=True):
    """Capture for the partner view `cam_b`: the current one if most of the pick's corridor in it is free of the
    robot; for the fixed top camera an earlier capture with a freer corridor may be used (not with
    allow_earlier=False: a pick in the wrist image of an arm whose gripper is closed)."""
    radius = WINDOW_PX + WINDOW_MARGIN_PX
    hs = zmatch.HS[zmatch.HS < min(ray_view.T[2, 3], current.T(cam_b, arm)[2, 3]) - zmatch.CAMERA_CLEARANCE_M]
    uv = corridor(ray_view.K, ray_view.T, p, current.K(cam_b), current.T(cam_b, arm), hs)
    best = (free_fraction(current.robot_mask(cam_b), uv, radius), current)
    if best[0] >= MIN_FREE_FRACTION or cam_b != "top" or not allow_earlier:
        return best[1]
    for capture in earlier_captures(session, current.id):
        other = load_capture(session, capture)
        fraction = free_fraction(other.robot_mask(cam_b), uv, radius)
        if fraction >= MIN_FREE_FRACTION:
            return other
        if fraction > best[0]:
            best = (fraction, other)
    return best[1]


# ----------------------------------------------------------------------------- review image
def _panel(img, centre, top_view):
    """320x240 panel: a 160x120 crop around `centre` at x2 for the top camera, the whole image at x0.5 otherwise.
    Returns the panel and a function mapping continuous image coordinates into it."""
    h, w = img.shape[:2]
    if top_view:
        x0 = int(np.clip(round(centre[0]) - 80, 0, w - 160))
        y0 = int(np.clip(round(centre[1]) - 60, 0, h - 120))
        panel = Image.fromarray(img[y0:y0 + 120, x0:x0 + 160]).resize((320, 240), Image.NEAREST)
        return panel, lambda x, y: ((x - x0) * 2, (y - y0) * 2)
    panel = Image.fromarray(img).resize((320, 240), Image.BILINEAR)
    return panel, lambda x, y: (x * 0.5, y * 0.5)


def review_image(path, view_a, cam_a, p, view_b, cam_b, pb, path_b, title, status):
    """Pick (red cross) in view A and its match (magenta X, orange if uncertain) in view B, side by side, with
    the searched line of sight in B in yellow."""
    left, map_a = _panel(view_a, p, cam_a == "top")
    right, map_b = _panel(view_b, pb if pb is not None else (320, 240), cam_b == "top")
    draw = ImageDraw.Draw(left)
    x, y = map_a(*p)
    draw.line([(x - 9, y), (x + 9, y)], fill=(255, 0, 0), width=2)
    draw.line([(x, y - 9), (x, y + 9)], fill=(255, 0, 0), width=2)
    draw = ImageDraw.Draw(right)
    if len(path_b) >= 2:
        draw.line([map_b(px, py) for px, py in path_b], fill=(255, 255, 0), width=1)
    if pb is not None:
        colour = (255, 0, 255) if status == "match" else (255, 140, 0)
        x, y = map_b(*pb)
        draw.line([(x - 7, y - 7), (x + 7, y + 7)], fill=colour, width=2)
        draw.line([(x - 7, y + 7), (x + 7, y - 7)], fill=colour, width=2)
    canvas = Image.new("RGB", (640, 256), (0, 0, 0))
    canvas.paste(left, (0, 16))
    canvas.paste(right, (320, 16))
    ImageDraw.Draw(canvas).text((4, 2), title, fill=(255, 255, 255))
    canvas.save(path, compress_level=1)


# ----------------------------------------------------------------------------- measurement
def edge_rule(result, view_a, view_b):
    """A 'match' whose height is an end of the searched range (zmatch.HS below both cameras) is not a located peak,
    since the score may keep rising beyond it; it becomes 'uncertain' (edge: true). Seen once in recorded episodes: a
    top pick 'matched' at the lowest swept height, 5 mm below the table."""
    if result.get("status") != "match" or result.get("h") is None:
        return result
    searched = zmatch.HS[zmatch.HS < min(view_a.T[2, 3], view_b.T[2, 3]) - zmatch.CAMERA_CLEARANCE_M]
    if searched.size and (result["h"] <= searched[0] + 1e-9 or result["h"] >= searched[-1] - 1e-9):
        return dict(result, status="uncertain", edge=True)
    return result


def _match(session, current, query, review=True):
    """Run the matcher for a query without plane_z; returns (height dict, review image or None). review=False (the
    landing-surface check of agp_core/grasp_depth.py): the same height, no review image written."""
    arm, cam, u, v = query["arm"], query["cam"], query["u"], query["v"]
    p = (u + 0.5, v + 0.5)
    if cam == "top":
        source = template_capture(session, current, "top", u, v)
        partners = [f"{arm}_wrist", f"{'right' if arm == 'left' else 'left'}_wrist"]
        allow_earlier = True
    else:
        source = current
        partners = ["top"]
        # the wrist camera of a closed gripper may be looking at what that gripper holds
        allow_earlier = not current.gripper_closed(cam.split("_")[0])
    view_a = zmatch.View(source.image(cam), source.K(cam), source.T(cam, arm))
    attempts = []
    for cam_b in partners:
        partner = partner_capture(session, current, cam_b, view_a, p, arm, allow_earlier=allow_earlier)
        view_b = zmatch.View(partner.image(cam_b), partner.K(cam_b), partner.T(cam_b, arm),
                             partner.robot_mask(cam_b))
        result = edge_rule(zmatch.match_height(view_a, view_b, p), view_a, view_b)
        attempts.append((cam_b, partner, view_b, result))
        if result["status"] != "none":
            break
    cam_b, partner, view_b, result = attempts[-1]
    height = {"source": {"match": "match", "uncertain": "uncertain"}.get(result["status"], "none"),
              "z": result["h"], "partner": cam_b,
              "template_capture": source.id, "partner_capture": partner.id,
              "older_view": source.id != current.id or partner.id != current.id,
              "variant": result.get("variant"), **({"edge": True} if result.get("edge") else {}),
              "peak": result.get("peak"), "second": result.get("second"), "margin": result.get("margin"),
              "two_way_z": result.get("lr_h"),
              "gate": {"two_way_tolerance_m": zmatch.LR_TOLERANCE_M, "margin": zmatch.MARGIN_GATE,
                       "order": list(zmatch.RULE_ORDER)},
              "tried": [{"partner": c, "status": r["status"]} for c, _, _, r in attempts]}
    if result["status"] == "none" or not review:
        return height, None
    X = zmatch.ray_point(view_a.K, view_a.T, p, result["h"])
    uv, _ = project(view_b.K, view_b.T, X[None])
    hs = zmatch.HS[zmatch.HS < min(view_a.T[2, 3], view_b.T[2, 3]) - zmatch.CAMERA_CLEARANCE_M]
    path_b = corridor(view_a.K, view_a.T, p, view_b.K, view_b.T, hs[::8])
    name = f"match_{cam}_{u}_{v}_{arm}.png"
    title = (f"{cam} {u},{v} -> {cam_b}: z {result['h'] * 1000:.1f} mm "
             f"{'match' if result['status'] == 'match' else 'UNCERTAIN'} ({result['variant']})"
             + ("; older top image" if height["older_view"] else ""))
    review_image(current.directory / name, source.image(cam), cam, p, partner.image(cam_b), cam_b,
                 tuple(uv[0]), path_b, title, result["status"])
    return height, {"label": f"match {cam} {u},{v}", "path": str(current.directory / name)}


def measure(session, capture, query):
    """RGB-only measurement of one query (see the module docstring). Raises ValueError on invalid input."""
    session = Path(session).resolve()
    if not re.fullmatch(r"[0-9]+", str(capture)):
        raise ValueError("Invalid capture identity")
    cam, arm = query.get("cam"), query.get("arm")
    if cam not in CAMERAS or arm not in ARMS:
        raise ValueError("RGB-only measurement needs cam top|left_wrist|right_wrist and arm left|right")
    u, v = query["u"], query["v"]
    current = load_capture(session, capture)
    h, w = current.size(cam)
    if not (0 <= v < h and 0 <= u < w):
        raise ValueError("Pixel outside image")
    if "plane_z" in query and not np.isfinite(query["plane_z"]):
        raise ValueError("plane_z must be finite")
    robot = current.on_robot(cam, u, v)
    on_robot = None if robot is None else {"arm": robot[0], "link": robot[1]}
    origin, direction = pixel_ray(current.K(cam), current.T(cam, arm), u, v)
    images = []
    if "plane_z" in query and float(query["plane_z"]) <= TABLE_PLANE_M:
        height = {"source": "plane", "z": float(query["plane_z"])}
    elif "plane_z" in query:
        # a raised plane_z is only the fallback; the matcher decides first.
        given = float(query["plane_z"])
        matched, image = _match(session, current, query)
        if matched["source"] == "match" and abs(matched["z"] - given) <= AGREE_M:
            height = dict(matched, plane_z=given)
            images.append(image)
        else:
            reason = "disagrees" if matched["source"] == "match" else matched["source"]
            height = {"source": "fallback", "z": given, "plane_z": given, "fallback_reason": reason,
                      "agree_m": AGREE_M, "match": matched}
    else:
        height, image = _match(session, current, query)
        if image is not None:
            images.append(image)
    if height["z"] is None:
        other = "top image" if cam != "top" else "left_wrist or right_wrist image"
        return {"ok": False, "rgb_only": True, "height": height, "on_robot": on_robot, "images": [],
                "error": (f"No height match: the point is not visible in the {other}. Give plane_z, or move "
                          f"the {arm} wrist camera above the target and measure again.")}
    point = intersect_z(origin, direction, height["z"])
    if point is None:
        return {"ok": False, "rgb_only": True, "height": height, "on_robot": on_robot, "images": images,
                "error": "The pixel ray does not reach that height in front of the camera"}
    point_tcp = np.linalg.solve(current.tcp(arm), np.r_[point, 1.0])[:3]
    return {
        "ok": True,
        "rgb_only": True,
        "point_base": dict(zip("xyz", point.tolist())),
        "point_tcp": dict(zip("xyz", point_tcp.tolist())),
        "tcp_frame": arm + "_tcp",
        "pad_projection": selected_point_pad_projection(point_tcp, "assumed_plane_intersection"),
        "height": height,
        "on_robot": on_robot,
        "camera_distance_m": float(np.linalg.norm(point - origin)),
        "pixel_center": [u + 0.5, v + 0.5],
        "frame": f"{arm}_base",
        "arm": arm,
        "images": images,
    }
