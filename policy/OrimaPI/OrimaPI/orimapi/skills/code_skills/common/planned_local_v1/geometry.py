"""Current RGB coarse identities -> locally refined metric evidence.

The object silhouette model is an estimate, not an object pose oracle. A
point on a horizontal surface can use local plane stereo; nonplanar contact
features require explicit multiview correspondences. Nothing reads simulator
objects, depth, segmentation truth, reward predicates or layout files.
"""
from copy import copy, deepcopy

import cv2
import numpy as np

from orimapi.skills.code_skills.common.geometry import rgb
from orimapi.skills.code_skills.fill_pen_holder.pen_geometry import (
    refine_mask, select_region,
)

CAMERAS = {"head": "cam_high", "wrist_l": "cam_left_wrist", "wrist_r": "cam_right_wrist"}


def vector(value, length=3):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"Expected {length} finite coordinates")
    return result


def unit(value):
    result = vector(value)
    if np.linalg.norm(result) < 1e-9:
        raise ValueError("Degenerate geometric axis")
    return result / np.linalg.norm(result)


def validate_box(coarse, image):
    h, w = image.shape[:2]
    box = coarse.get("box")
    if not isinstance(box, list) or len(box) != 4 or any(type(x) is not int for x in box):
        raise ValueError("Coarse identity requires four original-image integer bounds")
    x0, y0, x1, y1 = box
    if not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h:
        raise ValueError("Coarse box outside current head image")
    point = vector(coarse.get("point", [(x0+x1)/2, (y0+y1)/2]), 2)
    if not x0 <= point[0] < x1 or not y0 <= point[1] < y1:
        raise ValueError("Identity marker must lie in its current box")
    return {**coarse, "point": point.tolist()}


def refine_pixel(image, uv):
    """Subpixel corner refinement, at most the original 3px search window.

    A textureless patch stays unchanged and is explicitly not a correspondence
    certificate. Metric triangulation still validates the rays.
    """
    uv = vector(uv, 2)
    h, w = image.shape[:2]
    if not 0 <= uv[0] < w or not 0 <= uv[1] < h:
        raise ValueError("Feature outside the current calibrated image")
    if not 4 <= uv[0] < w-4 or not 4 <= uv[1] < h-4:
        return uv
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    u, v = np.rint(uv).astype(int)
    if gray[v-3:v+4, u-3:u+4].std() < 1:
        return uv
    result = cv2.cornerSubPix(
        gray, uv.astype(np.float32).reshape(1, 1, 2), (3, 3), (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, .01),
    ).reshape(2)
    return result if np.linalg.norm(result-uv) <= 3 else uv


def frame(a, b, c):
    """A directed feature and a third non-collinear feature define a frame."""
    x = unit(vector(b)-a)
    z = unit(np.cross(x, vector(c)-a))
    return np.column_stack((x, np.cross(z, x), z))


def docking_pose(tcp, source_points, destination_points):
    """Rigidly register three measured material features, including polarity."""
    sa, sb, sc = map(vector, source_points)
    da, db, dc = map(vector, destination_points)
    rotation = frame(da, db, dc) @ frame(sa, sb, sc).T
    result = np.asarray(tcp, float).copy()
    result[:3, :3] = rotation @ result[:3, :3]
    result[:3, 3] = da + rotation @ (result[:3, 3]-sa)
    return result


def row_slots(measurements, start, end):
    """Rank measured heights, allocate nonoverlapping footprints on a live row."""
    start, end = vector(start), vector(end)
    direction = end-start
    span = np.linalg.norm(direction)
    if span < 1e-9:
        raise ValueError("Row endpoints coincide")
    direction /= span
    ranked = sorted(measurements, key=lambda k: measurements[k]["height"])
    # Conservative horizontal footprint: long side, not the doll's pixel size.
    widths = np.array([measurements[k]["length"] for k in ranked])
    if span < widths.sum():
        raise ValueError("Measured row is too short for the five measured footprints")
    gap = (span-widths.sum()) / max(1, len(widths)-1)
    slots, offset = {}, 0.
    for identity, width in zip(ranked, widths):
        slots[identity] = (start+direction*(offset+width/2)).tolist()
        offset += width+gap
    return ranked, slots


class Measurements:
    def __init__(self, owner):
        self.owner = owner

    def scene(self):
        module = self.owner.api.operations.modules["scene"]
        result = module.scene_from_api(self.owner.api)
        if result.table_z is None:
            # Public scene calibration is already explicitly permitted and
            # used by RobotGeometry/the native planner. Do NOT turn an
            # unavailable RGB-plane fit into a made-up measured height.
            result = module.Scene(result.cams, float(self.owner.api.robot.table),
                                  result.arms, result.model, result.signature)
            self.owner.emit("public_table_calibration_fallback",
                            step=self.owner.latest.step, table_z=result.table_z,
                            source="public RobotGeometry table_config; NOT RGB measurement",
                            not_container_or_socket_height=True)
        return module, result

    def segment(self):
        with self.owner.api.operations.modules["scene"].table_plane._OneThread():
            module, scene = self.scene()
            result, code = module.cmd_segment(scene, "head", 15)
        result.pop("files", None)  # Do not put base64 image blobs in GPT history.
        if code:
            raise ValueError(f"Local segmentation: {result}")
        return result

    def feature(self, coarse, name):
        spec = coarse.get("features", {}).get(name)
        if not isinstance(spec, dict):
            raise ValueError(f"Need visible calibrated feature {name}, not an invented coordinate")
        views = {k: v for k, v in spec.items() if k in CAMERAS}
        if not views or set(spec)-set(CAMERAS)-{"plane"}:
            raise ValueError("Features use head/wrist_l/wrist_r pixels and optional plane=table")
        first, *others = views
        uv = refine_pixel(rgb(self.owner.latest.images[CAMERAS[first]]), views[first])
        args = dict(camera=first, u=float(uv[0]), v=float(uv[1]))
        if spec.get("plane") == "table":
            args["z"] = self.owner.api.robot.table
        elif "plane" in spec:
            raise ValueError("Only the public tabletop can be supplied without measuring height")
        elif others:
            second = others[0]
            other = refine_pixel(rgb(self.owner.latest.images[CAMERAS[second]]), views[second])
            args.update(camera2=second, u2=float(other[0]), v2=float(other[1]))
        # Automatic mode is horizontal-plane stereo only, not arbitrary 3D.
        result = self.owner.tool("rgb_point", args)
        point = vector(result["point_world"])
        self.owner.emit("feature_measurement", step=self.owner.latest.step, name=name,
                        result=result, assumptions="horizontal surface unless paired RGB rays",
                        sensor_ground_truth=False)
        return point

    def object(self, coarse):
        try:
            return self.silhouette_object(coarse)
        except ValueError as error:
            fields = {"top", "base", "long_a", "long_b", "short_a", "short_b"}
            if not fields <= coarse.get("features", {}).keys():
                raise ValueError(f"{error}; alternative: supply current multiview "
                                 f"features {sorted(fields)}") from error
            points = {key: self.feature(coarse, key) for key in fields}
            top, base = float(points["top"][2]), float(points["base"][2])
            long_axis = points["long_b"][:2]-points["long_a"][:2]
            short_axis = points["short_b"][:2]-points["short_a"][:2]
            length, width = float(np.linalg.norm(long_axis)), float(np.linalg.norm(short_axis))
            if min(top-base, length, width) <= 0:
                raise ValueError("Explicit RGB features do not establish positive object extents")
            center = (points["long_a"]+points["long_b"])/2
            center[2] = (top+base)/2
            value = dict(center=center.tolist(), top_z=top, base_z=base,
                         height=top-base, length=length, width=width,
                         yaw_deg=float(np.degrees(np.arctan2(long_axis[1], long_axis[0]))),
                         step=self.owner.latest.step, raw=None,
                         scope="current explicit multiview feature extents; not contact truth")
            self.owner.emit("alternative_object_measurement", result=value,
                            previous_failure=str(error), sensor_ground_truth=False)
            return value

    def silhouette_object(self, coarse):
        owner = self.owner
        image = rgb(owner.latest.images["cam_high"])
        coarse = validate_box(coarse, image)
        module = owner.api.operations.modules["scene"]
        with module.table_plane._OneThread():
            _, scene = self.scene()
            segmentation = scene.segment("head", min_area=15)
            region = select_region(segmentation["objects"], coarse["box"])
            index = next(i for i, row in enumerate(segmentation["objects"])
                         if row["id"] == region["id"])
            mask = refine_mask(image, segmentation["masks"][index], coarse)
            ys, xs = np.nonzero(mask)
            refined = deepcopy(region)
            refined.update(bbox=[int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
                           centroid_px=[float(xs.mean()), float(ys.mean())], area_px=int(mask.sum()))
            local = copy(scene)
            original = scene.segment
            local.segment = lambda camera, min_area=15: (
                {"objects": [refined], "masks": [mask]} if camera == "head"
                else original(camera, min_area))
            result = module.locate(local, "head", mask, refined, "auto")
        if not result.get("plan_ok"):
            # Alternative tool, same fresh observation; no hallucinated height.
            # Automatic patch stereo cannot establish full object dimensions.
            raise ValueError(f"Object multiview measurement failed: {result}; "
                             "use measure_feature or a measured view change")
        extents = result["extents"]
        base = float(result.get("base_height", owner.api.robot.table))
        top = float(result["top_z"])
        value = {
            "center": [*map(float, result["footprint_xy"]), (base+top)/2],
            "top_z": top, "base_z": base, "height": top-base,
            "length": float(extents["length"]), "width": float(extents["width"]),
            "yaw_deg": float(result["yaw_deg"]), "step": owner.latest.step,
            "scope": "silhouette estimate, not physical attachment/containment",
            "raw": result,
        }
        if min(value["height"], value["width"], value["length"]) <= 0:
            raise ValueError("Measurement does not establish positive object extents")
        owner.emit("object_measurement", step=owner.latest.step, result=value,
                   sensor_ground_truth=False)
        return value
