"""Generation skill_choices extracted from local DataGen v19; no simulation, Eval layouts, hashes or manifest IO."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml
from shapely.geometry import MultiPoint, Point, box
from shapely.ops import unary_union

from .placement import _input_record, _unit

SCHEMA_VERSION = "robodojo.train_layout.v1"
MANIFEST_SCHEMA_VERSION = "robodojo.train_layout_manifest.v1"
GENERATOR_VERSION = "independent_config_catalog_v19"
WHOLE_LAYOUT_MAX_ATTEMPTS = 512
SUPPORTED_TASKS = (
    "align_blocks",
    "build_tower",
    "cover_blocks",
    "deposit_coin",
    "fasten_screws",
    "fill_egg_holder",
    "fill_pen_holder",
    "general_pickup",
    "hang_mugs",
    "hang_mugs_random",
    "insert_tubes",
    "make_kong",
    "make_toast",
    "make_toast_random",
    "match_and_pick_from_conveyor",
    "organize_table",
    "pack_objects_into_box",
    "pack_objects_into_box_random",
    "stack_blocks",
    "stack_blocks_random",
    "stack_blocks_by_language",
    "plug_in_charger",
    "press_by_number",
    "play_stacking_toy",
    "play_Xylophone",
    "arrange_largest_number",
    "arrange_largest_number_random",
    "insert_key",
    "play_tic_tac_toe",
    "push_T",
    "push_T_random",
    "stack_bowls",
    "stack_bowls_random",
    "store_tools_in_toolbox",
    "store_laptop_and_headphones",
    "store_laptop_and_headphones_random",
    "sort_nesting_dolls_by_size",
    "sort_nesting_dolls_by_size_random",
    "swap_T",
    "swap_blocks",
    "imitate_sorting_sequence",
)
OBJECT_TYPES = ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid")


class TrainLayoutError(RuntimeError):
    """A Train_Layout contract failed; callers must not fall back to Eval_Layout."""


class _PlacementExhausted(TrainLayoutError):
    """A sampled whole-layout candidate cannot fit under the declared config."""


@dataclass(frozen=True)
class _Placed:
    plane: str
    x: float
    y: float
    half_x: float
    half_y: float
    footprint: tuple[tuple[float, float], ...] | None = None


def _choice(values: list[Any], seed: int, task: str, layout_id: int, field: str) -> Any:
    if not values:
        raise TrainLayoutError(f"empty candidate set for {task}:{field}")
    return values[
        min(int(_unit(seed, task, layout_id, field) * len(values)), len(values) - 1)
    ]


def _sample_limit(value: Any, u: float, *, field: str) -> float:
    if not isinstance(value, list) or not value:
        raise TrainLayoutError(
            f"{field} must be a two-element numeric range or non-empty union of two-element ranges"
        )
    if len(value) == 2 and all((isinstance(item, (int, float)) for item in value)):
        lo, hi = sorted((float(item) for item in value))
        return lo + (hi - lo) * u
    intervals = value
    if not all((isinstance(item, list) and len(item) == 2 for item in intervals)):
        raise TrainLayoutError(
            f"{field} union must contain two-element numeric intervals"
        )
    if not all(
        (
            all((isinstance(endpoint, (int, float)) for endpoint in interval))
            for interval in intervals
        )
    ):
        raise TrainLayoutError(
            f"{field} union must contain two-element numeric intervals"
        )
    selected = intervals[min(int(u * len(intervals)), len(intervals) - 1)]
    lo, hi = sorted((float(item) for item in selected))
    local_u = u * len(intervals) % 1.0
    return lo + (hi - lo) * local_u


def _sample_rotate_deg(value: Any, u: float) -> tuple[Any, float]:
    """Mirror ``ClutteredGenerator._sample_rotate_angle`` deterministically."""
    if value is None:
        return (0.0, 0.0)
    if isinstance(value, (int, float)) and (not isinstance(value, bool)):
        magnitude = float(value)
        if not math.isfinite(magnitude) or magnitude < 0.0:
            raise TrainLayoutError("rotate_deg scalar must be finite and non-negative")
        return (deepcopy(value), -magnitude + 2.0 * magnitude * float(u))
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(
            (
                isinstance(item, (int, float)) and (not isinstance(item, bool))
                for item in value
            )
        )
    ):
        low, high = sorted((float(item) for item in value))
        if not math.isfinite(low) or not math.isfinite(high):
            raise TrainLayoutError("rotate_deg range must be finite")
        return (deepcopy(value), low + (high - low) * float(u))
    raise TrainLayoutError("rotate_deg must be a scalar or a two-element numeric range")


def _quat_multiply(left: list[float], right: list[float]) -> list[float]:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    result = [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
    norm = math.sqrt(sum((component * component for component in result)))
    if norm <= 0:
        raise TrainLayoutError("zero-norm placement quaternion")
    return [component / norm for component in result]


def _quat_inverse(quaternion: list[float]) -> list[float]:
    if len(quaternion) != 4:
        raise TrainLayoutError("invalid placement quaternion")
    w, x, y, z = (float(value) for value in quaternion)
    squared_norm = w * w + x * x + y * y + z * z
    if squared_norm <= 0:
        raise TrainLayoutError("zero-norm placement quaternion")
    return [w / squared_norm, -x / squared_norm, -y / squared_norm, -z / squared_norm]


def _quat_rotate_vector(quaternion: list[float], vector: list[float]) -> list[float]:
    if len(quaternion) != 4 or len(vector) != 3:
        raise TrainLayoutError("invalid quaternion/vector placement transform")
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0:
        raise TrainLayoutError("zero-norm placement quaternion")
    w, x, y, z = (w / norm, x / norm, y / norm, z / norm)
    vx, vy, vz = (float(value) for value in vector)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def _asset_paths(
    assets_root: Path, object_type: str, category: str, index: int
) -> tuple[Path, Path]:
    root = assets_root / "Object" / "RoboDojo" / object_type / category / f"{index:05d}"
    metadata = root / "metadata.json"
    candidates = (root / "object.usdz", root / "object.usd")
    asset = next((path for path in candidates if path.is_file()), None)
    if not metadata.is_file() or asset is None:
        raise TrainLayoutError(
            f"incomplete asset catalog entry: {object_type}/{category}/{index:05d}"
        )
    return (metadata, asset)


def _catalog_indices(assets_root: Path, object_type: str, category: str) -> list[int]:
    root = assets_root / "Object" / "RoboDojo" / object_type / category
    result = []
    if root.is_dir():
        for path in root.iterdir():
            if path.is_dir() and path.name.isdigit():
                try:
                    _asset_paths(assets_root, object_type, category, int(path.name))
                except TrainLayoutError:
                    continue
                result.append(int(path.name))
    return sorted(result)


def _allowed_indices(
    assets_root: Path, object_type: str, category_cfg: dict[str, Any]
) -> list[int]:
    category = str(category_cfg.get("name", ""))
    available = _catalog_indices(assets_root, object_type, category)
    requested = category_cfg.get("index")
    if requested is None:
        return available
    requested_values = [int(value) for value in requested]
    missing = sorted(set(requested_values) - set(available))
    if missing:
        raise TrainLayoutError(
            f"task config references missing {object_type}/{category} indices: {missing}"
        )
    return requested_values


_ASSET_METADATA: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}


def _load_asset_metadata(metadata_path: Path) -> dict[str, Any]:
    """Return parsed, read-only asset metadata, reusing an unchanged file."""
    stat = metadata_path.stat()
    key = (str(metadata_path), stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)
    metadata = _ASSET_METADATA.get(key)
    if metadata is None:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _ASSET_METADATA[key] = metadata
    return metadata


def _metadata_geometry(metadata: dict[str, Any]) -> tuple[float, float, float]:
    geometry = metadata.get("geometry") or {}
    bbox = geometry.get("bbox")
    if not isinstance(bbox, list) or len(bbox) < 3:
        bbox = (geometry.get("aligned_bbox") or {}).get("extents")
    if not isinstance(bbox, list) or len(bbox) < 3:
        vertices = (geometry.get("oriented_bbox") or {}).get("vertices")
        if (
            isinstance(vertices, list)
            and vertices
            and all((isinstance(row, list) and len(row) == 3 for row in vertices))
        ):
            bbox = [
                max((float(row[axis]) for row in vertices))
                - min((float(row[axis]) for row in vertices))
                for axis in range(3)
            ]
        else:
            raise TrainLayoutError("asset metadata lacks a 3D bounding box")
    return tuple((float(v) for v in bbox[:3]))


def _metadata_bbox_vertices(metadata: dict[str, Any]) -> list[list[float]]:
    vertices = ((metadata.get("geometry") or {}).get("oriented_bbox") or {}).get(
        "vertices"
    )
    if not (
        isinstance(vertices, list)
        and len(vertices) >= 4
        and all(
            (
                isinstance(row, list)
                and len(row) == 3
                and all((isinstance(value, (int, float)) for value in row))
                for row in vertices
            )
        )
    ):
        raise TrainLayoutError("asset metadata lacks oriented bbox vertices")
    return [[float(value) for value in row] for row in vertices]


def _placement_base_orientation(
    metadata: dict[str, Any], common: dict[str, Any]
) -> list[float]:
    if common.get("zlim") is not None:
        base = common.get("qpos", [1.0, 0.0, 0.0, 0.0])
    else:
        placement = _placement_entry(metadata, common)
        base = placement.get("orientation")
    if not isinstance(base, list) or len(base) != 4:
        raise TrainLayoutError("asset has no valid placement orientation")
    return _quat_multiply([1.0, 0.0, 0.0, 0.0], [float(value) for value in base])


def _placement_entry(
    metadata: dict[str, Any], common: dict[str, Any]
) -> dict[str, Any]:
    placements = (metadata.get("active") or {}).get("place") or {}
    if not isinstance(placements, dict) or not placements:
        raise TrainLayoutError("asset has no active placement frames")
    requested = common.get("place_tag")
    if requested is None:
        requested_tags = ["up"]
    elif isinstance(requested, str):
        requested_tags = [requested]
    elif (
        isinstance(requested, list)
        and requested
        and all((isinstance(tag, str) and tag for tag in requested))
    ):
        requested_tags = list(requested)
    else:
        raise TrainLayoutError("place_tag must be a string or nonempty string list")
    matches = [
        placements[tag]
        for tag in requested_tags
        if isinstance(placements.get(tag), dict)
        and isinstance(
            (placements[tag].get("projection_circle") or {}).get("center"), list
        )
    ]
    if len(matches) != 1:
        raise TrainLayoutError(
            f"independent generation requires exactly one declared placement frame; requested={requested_tags}, available={sorted(placements)}"
        )
    return matches[0]


def _placement_orientation(
    metadata: dict[str, Any], common: dict[str, Any], yaw: float
) -> list[float]:
    base = _placement_base_orientation(metadata, common)
    yaw_quat = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
    return _quat_multiply(yaw_quat, base)


def _plane_height(common: dict[str, Any], scene: dict[str, Any]) -> float:
    plane = str(common.get("relative_plane", "Table")).split("/", 1)[0]
    if plane == "Table":
        table = scene.get("Table") or {}
        pos, scale = (table.get("default_pos"), table.get("scale"))
        if not isinstance(pos, list) or not isinstance(scale, list):
            raise TrainLayoutError("scene config lacks Table position/scale")
        return float(pos[2]) + float(scale[2]) / 2.0
    if plane == "Ground":
        ground = scene.get("Ground") or {}
        position = ground.get("default_pos")
        thickness = ground.get("thickness")
        if (
            not isinstance(position, list)
            or len(position) != 3
            or (not isinstance(thickness, (int, float)))
        ):
            raise TrainLayoutError("scene config lacks Ground position/thickness")
        return float(position[2]) + float(thickness) / 2.0
    raise TrainLayoutError(
        f"unsupported relative plane without an independent fixture: {plane}"
    )


def _placement_origin_offset(
    metadata: dict[str, Any], common: dict[str, Any]
) -> list[float]:
    if common.get("zlim") is not None:
        return [0.0, 0.0, 0.0]
    center = (_placement_entry(metadata, common).get("projection_circle") or {}).get(
        "center"
    )
    if not isinstance(center, list) or len(center) != 7:
        raise TrainLayoutError(
            "asset metadata lacks a 7D active.place.up.projection_circle.center"
        )
    rotated = _quat_rotate_vector(
        _placement_base_orientation(metadata, common),
        [float(value) for value in center[:3]],
    )
    return [-value for value in rotated]


def _overlaps(
    candidate: _Placed,
    occupied: Iterable[_Placed],
    margin: float,
    prohibited: list[Any],
) -> bool:
    for existing in occupied:
        if existing.plane != candidate.plane:
            continue
        if candidate.footprint is not None and existing.footprint is not None:
            candidate_polygon = MultiPoint(candidate.footprint).convex_hull
            existing_polygon = MultiPoint(existing.footprint).convex_hull
            if candidate_polygon.distance(existing_polygon) < margin:
                return True
        elif (
            abs(existing.x - candidate.x) < existing.half_x + candidate.half_x + margin
            and abs(existing.y - candidate.y)
            < existing.half_y + candidate.half_y + margin
        ):
            return True
    for area in prohibited:
        if not isinstance(area, list) or len(area) != 4:
            raise TrainLayoutError(
                "ProhibitedArea entries must contain four coordinates"
            )
        x0, x1 = sorted((float(area[0]), float(area[2])))
        y0, y1 = sorted((float(area[1]), float(area[3])))
        if (
            candidate.x + candidate.half_x > x0
            and candidate.x - candidate.half_x < x1
            and (candidate.y + candidate.half_y > y0)
            and (candidate.y - candidate.half_y < y1)
        ):
            return True
    return False


def _make_instance(
    *,
    source_root: Path,
    assets_root: Path,
    scene: dict[str, Any],
    task: str,
    generation_seed: int,
    layout_id: int,
    object_type: str,
    category: str,
    category_idx: int,
    group: str | None,
    label: str,
    common: dict[str, Any],
    occupied: list[_Placed],
    prohibited: list[Any],
    max_attempts: int = 512,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata_path, asset_path = _asset_paths(
        assets_root, object_type, category, category_idx
    )
    metadata = _load_asset_metadata(metadata_path)
    size_x, size_y, _ = _metadata_geometry(metadata)
    rotate_deg, sampled_rotate_deg = _sample_rotate_deg(
        common.get("rotate_deg", 0.0),
        _unit(generation_seed, task, layout_id, f"{label}:yaw"),
    )
    rotate_rand = bool(common.get("rotate_rand", False))
    yaw = math.radians(sampled_rotate_deg) if rotate_rand else 0.0
    if task == "general_pickup" and common.get("zlim") is None:
        placement_center = (
            _placement_entry(metadata, common).get("projection_circle") or {}
        ).get("center")
        if not isinstance(placement_center, list) or len(placement_center) != 7:
            raise TrainLayoutError("asset placement center is invalid")
        yaw_quaternion = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
        orientation = _quat_multiply(
            yaw_quaternion,
            _quat_inverse([float(value) for value in placement_center[3:]]),
        )
        origin_offset = [
            -value
            for value in _quat_rotate_vector(
                orientation, [float(value) for value in placement_center[:3]]
            )
        ]
        rotated_bbox = [
            _quat_rotate_vector(orientation, vertex)
            for vertex in _metadata_bbox_vertices(metadata)
        ]
        min_x = min((vertex[0] for vertex in rotated_bbox))
        max_x = max((vertex[0] for vertex in rotated_bbox))
        min_y = min((vertex[1] for vertex in rotated_bbox))
        max_y = max((vertex[1] for vertex in rotated_bbox))
        half_x = (max_x - min_x) / 2.0
        half_y = (max_y - min_y) / 2.0
    else:
        orientation = _placement_orientation(metadata, common, yaw)
        origin_offset = _placement_origin_offset(metadata, common)
        rotated_bbox = None
        half_x = (abs(math.cos(yaw)) * size_x + abs(math.sin(yaw)) * size_y) / 2.0
        half_y = (abs(math.sin(yaw)) * size_x + abs(math.cos(yaw)) * size_y) / 2.0
    margin = float(common.get("margin", 0.01))
    plane = str(common.get("relative_plane", "Table")).split("/", 1)[0]
    selected = None
    for attempt in range(max_attempts):
        x = _sample_limit(
            common.get("xlim"),
            _unit(generation_seed, task, layout_id, f"{label}:x", attempt),
            field="xlim",
        )
        y = _sample_limit(
            common.get("ylim"),
            _unit(generation_seed, task, layout_id, f"{label}:y", attempt),
            field="ylim",
        )
        candidate = _Placed(
            plane=plane,
            x=x + origin_offset[0],
            y=y + origin_offset[1],
            half_x=half_x,
            half_y=half_y,
            footprint=tuple(
                (
                    (x + origin_offset[0] + corner_x, y + origin_offset[1] + corner_y)
                    for corner_x, corner_y in (
                        ((vertex[0], vertex[1]) for vertex in rotated_bbox)
                        if rotated_bbox is not None
                        else (
                            (-size_x / 2.0, -size_y / 2.0),
                            (-size_x / 2.0, size_y / 2.0),
                            (size_x / 2.0, size_y / 2.0),
                            (size_x / 2.0, -size_y / 2.0),
                        )
                    )
                )
            )
            if task == "general_pickup"
            else tuple(
                (
                    (
                        x
                        + origin_offset[0]
                        + corner_x * math.cos(yaw)
                        - corner_y * math.sin(yaw),
                        y
                        + origin_offset[1]
                        + corner_x * math.sin(yaw)
                        + corner_y * math.cos(yaw),
                    )
                    for corner_x, corner_y in (
                        (-size_x / 2.0, -size_y / 2.0),
                        (-size_x / 2.0, size_y / 2.0),
                        (size_x / 2.0, size_y / 2.0),
                        (size_x / 2.0, -size_y / 2.0),
                    )
                )
            )
            if task == "align_blocks"
            else None,
        )
        if not _overlaps(candidate, occupied, margin, prohibited):
            selected = candidate
            break
    if selected is None:
        raise _PlacementExhausted(
            f"could not place {task}:{label} within independent config constraints"
        )
    occupied.append(selected)
    z_offset = 0.0
    if common.get("zlim") is not None:
        z_offset = _sample_limit(
            common["zlim"],
            _unit(generation_seed, task, layout_id, f"{label}:z"),
            field="zlim",
        )
    position = [
        selected.x,
        selected.y,
        _plane_height(common, scene) + z_offset + origin_offset[2],
    ]
    physics = deepcopy(metadata.get("physics") or {})
    physics["type"] = object_type.lower()
    record = {
        "category": category,
        "category_idx": category_idx,
        "xlim": deepcopy(common.get("xlim")),
        "ylim": deepcopy(common.get("ylim")),
        "zlim": deepcopy(common.get("zlim")),
        "qpos": deepcopy(common.get("qpos", [1.0, 0.0, 0.0, 0.0])),
        "rotate_deg": rotate_deg,
        "rotate_rand": rotate_rand,
        "relative_plane": common.get("relative_plane", "Table"),
        "place_tag": common.get("place_tag"),
        "margin": margin,
        "check_mode": common.get("check_mode", "bbox"),
        "need_check_stable": bool(common.get("need_check_stable", True)),
        "label": label,
        "default_pos": position,
        "default_ori": orientation,
        "scale": deepcopy(common.get("scale", [1.0, 1.0, 1.0])),
        "physics": physics,
        "visual": deepcopy(common.get("visual", {})),
    }
    if group is not None:
        record["group"] = group
    inputs = [
        _input_record(metadata_path, source_root, "object_metadata"),
        _input_record(asset_path, source_root, "object_asset"),
    ]
    return (record, inputs)


def _selected_groups(
    task: str, config: dict[str, Any], assets_root: Path, seed: int, layout_id: int
) -> list[tuple[str, dict[str, Any], dict[str, Any], list[int]]]:
    result = []
    for object_type in (key for key in config if key in OBJECT_TYPES):
        for group_number, group_cfg in enumerate(config.get(object_type) or []):
            categories = group_cfg.get("category") or []
            select = group_cfg.get("select_mode") or {}
            if task in {"stack_blocks", "stack_blocks_random"}:
                if (
                    object_type != "Rigid"
                    or group_number != 0
                    or select.get("mode") != "hierarchical"
                ):
                    raise TrainLayoutError(f"{task} config has unexpected semantics")
                category_cfg = _choice(
                    categories, seed, task, layout_id, "hierarchical_category"
                )
                candidates = _allowed_indices(assets_root, object_type, category_cfg)
                ordered = sorted(
                    candidates,
                    key=lambda value: _unit(
                        seed, task, layout_id, f"block_index:{value}"
                    ),
                )
                if len(ordered) < 3:
                    raise TrainLayoutError(
                        "stack_blocks requires three unique catalog assets"
                    )
                result.append((object_type, group_cfg, category_cfg, ordered[:3]))
                continue
            if task == "general_pickup":
                if object_type != "Rigid" or group_number != 0:
                    raise TrainLayoutError(
                        "general_pickup standard config has unexpected object groups"
                    )
                if int(select.get("nums", 0)) != 1 or list(
                    select.get("label") or []
                ) != ["target"]:
                    raise TrainLayoutError(
                        "general_pickup requires exactly one target-labelled object"
                    )
                category_cfg = _choice(
                    categories, seed, task, layout_id, "target_category"
                )
                candidates = _allowed_indices(assets_root, object_type, category_cfg)
                index = _choice(
                    candidates,
                    seed,
                    task,
                    layout_id,
                    f"target_asset:{category_cfg.get('name')}",
                )
                result.append((object_type, group_cfg, category_cfg, [index]))
                continue
            if len(categories) != 1:
                raise TrainLayoutError(
                    f"{task}:{object_type}:{group_number} requires exactly one category"
                )
            category_cfg = categories[0]
            candidates = _allowed_indices(assets_root, object_type, category_cfg)
            count = int(select.get("nums", 1))
            mode = str(select.get("mode", "allow_duplicate"))
            if task == "play_stacking_toy" and object_type == "Rigid":
                ordered = sorted(
                    candidates,
                    key=lambda value: _unit(
                        seed, task, layout_id, f"stacking_group:{group_number}:{value}"
                    ),
                )
                used = {
                    indices[0]
                    for typ, _, cat, indices in result
                    if typ == "Rigid" and cat.get("name") == category_cfg.get("name")
                }
                available = [value for value in ordered if value not in used]
                if not available:
                    raise TrainLayoutError(
                        "play_stacking_toy requires distinct catalog asset per configured group"
                    )
                indices = [available[0]] * count
            elif mode in {"same", "allow_duplicate"}:
                indices = [
                    _choice(
                        candidates,
                        seed,
                        task,
                        layout_id,
                        f"{object_type}:{group_number}:asset",
                    )
                ] * count
            elif mode == "unique":
                ordered = sorted(
                    candidates,
                    key=lambda value: _unit(
                        seed, task, layout_id, f"{object_type}:{group_number}:{value}"
                    ),
                )
                if len(ordered) < count:
                    raise TrainLayoutError("not enough unique catalog assets")
                indices = ordered[:count]
            else:
                raise TrainLayoutError(
                    f"unsupported select mode without Eval template: {mode}"
                )
            result.append((object_type, group_cfg, category_cfg, indices))
    return result


def _populate_imitate_sorting_sequence(
    *,
    source_root: Path,
    assets_root: Path,
    task: str,
    config: dict[str, Any],
    scene: dict[str, Any],
    generation_seed: int,
    layout_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Sample five category-distinct target/support pairs from task config."""
    geometry_groups = list(config.get("Geometry") or [])
    rigid_groups = list(config.get("Rigid") or [])
    if len(geometry_groups) != 2 or len(rigid_groups) != 6:
        raise TrainLayoutError(f"{task} config group contract drifted")
    target_group = rigid_groups[0]
    target_select = target_group.get("select_mode") or {}
    target_labels = list(target_select.get("label") or [])
    categories = list(target_group.get("category") or [])
    if (
        target_select.get("mode") != "hierarchical"
        or target_labels != ["t0", "t1", "t2", "t3", "t4"]
        or len(categories) != 5
    ):
        raise TrainLayoutError(f"{task} target hierarchy contract drifted")
    for index, group in enumerate(rigid_groups[1:]):
        select = group.get("select_mode") or {}
        if (
            select.get("mode") != "same_as_label"
            or select.get("same_label") != target_labels[index]
            or list(select.get("label") or []) != [f"aim{index}"]
        ):
            raise TrainLayoutError(f"{task} paired aim contract drifted at {index}")
    prohibited = config.get("Prohibited_Area") or []
    last_error: Exception | None = None
    for whole_attempt in range(WHOLE_LAYOUT_MAX_ATTEMPTS):
        layout: dict[str, Any] = {}
        inputs: list[dict[str, Any]] = []
        occupied: list[_Placed] = []
        try:
            for group_number, group in enumerate(geometry_groups):
                category_cfg = list(group.get("category") or [])[0]
                candidates = _allowed_indices(assets_root, "Geometry", category_cfg)
                selected_index = _choice(
                    candidates,
                    generation_seed,
                    task,
                    layout_id,
                    f"whole:{whole_attempt}:basket:{group_number}",
                )
                label = str((group.get("select_mode") or {}).get("label")[0])
                instance, records = _make_instance(
                    source_root=source_root,
                    assets_root=assets_root,
                    scene=scene,
                    task=task,
                    generation_seed=generation_seed,
                    layout_id=layout_id,
                    object_type="Geometry",
                    category=str(category_cfg["name"]),
                    category_idx=selected_index,
                    group=category_cfg.get("group"),
                    label=label,
                    common=deepcopy(group.get("common") or {}),
                    occupied=occupied,
                    prohibited=prohibited,
                )
                layout.setdefault("Geometry", {}).setdefault(
                    str(category_cfg["name"]), []
                ).append(instance)
                inputs.extend(records)
            ordered_categories = sorted(
                categories,
                key=lambda item: _unit(
                    generation_seed,
                    task,
                    layout_id,
                    f"whole:{whole_attempt}:category:{item.get('name')}",
                ),
            )
            for pair_index, (target_label, category_cfg) in enumerate(
                zip(target_labels, ordered_categories, strict=True)
            ):
                candidates = _allowed_indices(assets_root, "Rigid", category_cfg)
                category_index = _choice(
                    candidates,
                    generation_seed,
                    task,
                    layout_id,
                    f"whole:{whole_attempt}:asset:{target_label}",
                )
                metadata_path, _ = _asset_paths(
                    assets_root, "Rigid", str(category_cfg["name"]), category_index
                )
                metadata = _load_asset_metadata(metadata_path)
                placement_frames = (metadata.get("active") or {}).get("place") or {}
                placement_tags = sorted(
                    (
                        tag
                        for tag, record in placement_frames.items()
                        if isinstance(record, dict)
                        and isinstance(
                            (record.get("projection_circle") or {}).get("center"), list
                        )
                    )
                )
                if not placement_tags:
                    raise TrainLayoutError(
                        f"{task}:{target_label} asset has no declared placement frame"
                    )
                placement_tag = _choice(
                    placement_tags,
                    generation_seed,
                    task,
                    layout_id,
                    f"whole:{whole_attempt}:place_tag:{target_label}",
                )
                for label, group in (
                    (target_label, target_group),
                    (f"aim{target_label[1:]}", rigid_groups[pair_index + 1]),
                ):
                    common = deepcopy(group.get("common") or {})
                    common["place_tag"] = placement_tag
                    instance, records = _make_instance(
                        source_root=source_root,
                        assets_root=assets_root,
                        scene=scene,
                        task=task,
                        generation_seed=generation_seed,
                        layout_id=layout_id,
                        object_type="Rigid",
                        category=str(category_cfg["name"]),
                        category_idx=category_index,
                        group=category_cfg.get("group"),
                        label=label,
                        common=common,
                        occupied=occupied,
                        prohibited=prohibited,
                    )
                    layout.setdefault("Rigid", {}).setdefault(
                        str(category_cfg["name"]), []
                    ).append(instance)
                    inputs.extend(records)
            return (layout, inputs, whole_attempt + 1)
        except _PlacementExhausted as exc:
            last_error = exc
    raise TrainLayoutError(
        f"could not place {task} after {WHOLE_LAYOUT_MAX_ATTEMPTS} attempts; last_error={last_error}"
    )


def _populate_general_pickup(
    *,
    source_root: Path,
    assets_root: Path,
    config: dict[str, Any],
    scene: dict[str, Any],
    generation_seed: int,
    layout_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Sample the official one-target plus category-unique clutter contract."""
    selected = _selected_groups(
        "general_pickup", config, assets_root, generation_seed, layout_id
    )
    if len(selected) != 1:
        raise TrainLayoutError("general_pickup must select exactly one target")
    object_type, group_cfg, category_cfg, indices = selected[0]
    if object_type != "Rigid" or len(indices) != 1:
        raise TrainLayoutError("general_pickup target selection contract drifted")
    layout: dict[str, Any] = {"Rigid": {}}
    occupied: list[_Placed] = []
    inputs: list[dict[str, Any]] = []
    target, target_inputs = _make_instance(
        source_root=source_root,
        assets_root=assets_root,
        scene=scene,
        task="general_pickup",
        generation_seed=generation_seed,
        layout_id=layout_id,
        object_type="Rigid",
        category=str(category_cfg["name"]),
        category_idx=indices[0],
        group=category_cfg.get("group"),
        label="target",
        common=deepcopy(group_cfg.get("common") or {}),
        occupied=occupied,
        prohibited=[],
    )
    layout["Rigid"].setdefault(str(category_cfg["name"]), []).append(target)
    inputs.extend(target_inputs)
    placed_clutter, clutter_inputs = _populate_declared_clutter(
        source_root=source_root,
        assets_root=assets_root,
        task="general_pickup",
        config=config,
        scene=scene,
        generation_seed=generation_seed,
        layout_id=layout_id,
        layout=layout,
        occupied=occupied,
        prohibited=[],
    )
    inputs.extend(clutter_inputs)
    if placed_clutter < 8:
        raise _PlacementExhausted(
            "general_pickup independent clutter placement produced fewer than eight objects"
        )
    return (layout, inputs)


def _populate_declared_clutter(
    *,
    source_root: Path,
    assets_root: Path,
    task: str,
    config: dict[str, Any],
    scene: dict[str, Any],
    generation_seed: int,
    layout_id: int,
    layout: dict[str, Any],
    occupied: list[_Placed] | None = None,
    occupied_polygons: list[Any] | None = None,
    prohibited: list[Any],
) -> tuple[int, list[dict[str, Any]]]:
    """Materialize every declared category-unique clutter group."""
    if (occupied is None) == (occupied_polygons is None):
        raise TrainLayoutError(
            f"{task} clutter requires exactly one placement representation"
        )
    inputs: list[dict[str, Any]] = []
    total_placed = 0
    for group_index, clutter_cfg in enumerate(config.get("Clutter") or []):
        if not isinstance(clutter_cfg, dict):
            raise TrainLayoutError(f"{task} clutter group must be a mapping")
        count = int(clutter_cfg.get("nums", 0))
        yaml_path = clutter_cfg.get("yaml_path")
        if count < 1 or clutter_cfg.get("mode") != "category_unique":
            raise TrainLayoutError(f"{task} requires category_unique clutter groups")
        if not isinstance(yaml_path, str) or not yaml_path:
            raise TrainLayoutError(f"{task} clutter yaml_path is invalid")
        clutter_path = assets_root / "Object" / "RoboDojo" / yaml_path
        try:
            clutter_document = yaml.safe_load(clutter_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise TrainLayoutError(
                f"cannot read {task} clutter catalog: {exc}"
            ) from exc
        catalog = (clutter_document or {}).get("Clutter")
        if not isinstance(catalog, dict) or len(catalog) < count:
            raise TrainLayoutError(f"{task} clutter catalog is incomplete")
        inputs.append(_input_record(clutter_path, source_root, "clutter_catalog"))
        placeable_by_category: dict[str, list[int]] = {}
        for category, indices in catalog.items():
            if not isinstance(indices, list):
                continue
            placeable = []
            for index in indices:
                category_idx = int(index)
                metadata_path, _ = _asset_paths(
                    assets_root, "Clutter", category, category_idx
                )
                metadata = _load_asset_metadata(metadata_path)
                try:
                    _metadata_bbox_vertices(metadata)
                    _placement_entry(metadata, common={})
                except TrainLayoutError:
                    continue
                placeable.append(category_idx)
            if placeable:
                placeable_by_category[category] = placeable
        category_order = sorted(
            placeable_by_category,
            key=lambda category: _unit(
                generation_seed,
                task,
                layout_id,
                f"clutter_category_order:{category}"
                if task == "general_pickup"
                else f"clutter_group:{group_index}:category_order:{category}",
            ),
        )
        common = {
            key: deepcopy(clutter_cfg.get(key))
            for key in (
                "xlim",
                "ylim",
                "zlim",
                "qpos",
                "rotate_deg",
                "rotate_rand",
                "relative_plane",
                "place_tag",
                "margin",
                "check_mode",
                "need_check_stable",
                "scale",
                "physics",
                "visual",
            )
            if key in clutter_cfg
        }
        placed_in_group = 0
        for slot, category in enumerate(category_order):
            if placed_in_group == count:
                break
            category_idx = int(
                _choice(
                    placeable_by_category[category],
                    generation_seed,
                    task,
                    layout_id,
                    f"clutter_asset:{category}"
                    if task == "general_pickup"
                    else f"clutter_group:{group_index}:asset:{category}",
                )
            )
            try:
                if occupied_polygons is not None:
                    clutter, clutter_inputs = _make_table_surface_instance(
                        source_root=source_root,
                        assets_root=assets_root,
                        scene=scene,
                        task=task,
                        generation_seed=generation_seed,
                        layout_id=layout_id,
                        object_type="Clutter",
                        category=category,
                        category_idx=category_idx,
                        group=None,
                        label=f"clutter_{group_index}_{slot}",
                        common=common,
                        occupied_polygons=occupied_polygons,
                        prohibited=prohibited,
                        max_attempts=100,
                        cluttered=True,
                    )
                else:
                    clutter, clutter_inputs = _make_instance(
                        source_root=source_root,
                        assets_root=assets_root,
                        scene=scene,
                        task=task,
                        generation_seed=generation_seed,
                        layout_id=layout_id,
                        object_type="Clutter",
                        category=category,
                        category_idx=category_idx,
                        group=None,
                        label=f"clutter_{group_index}_{slot}",
                        common=common,
                        occupied=occupied,
                        prohibited=prohibited,
                        max_attempts=100,
                    )
            except _PlacementExhausted:
                continue
            physics = deepcopy(clutter.get("physics") or {})
            physics["type"] = "rigid"
            layout.setdefault("Rigid", {}).setdefault(category, []).append(
                {
                    "category_idx": category_idx,
                    "physics": physics,
                    "scale": deepcopy(clutter.get("scale", [1.0, 1.0, 1.0])),
                    "type": "cluttered",
                    "clutter_idx": group_index,
                    "yaml_path": yaml_path,
                    "relative_plane": clutter_cfg.get("relative_plane", "Table"),
                    "default_pos": clutter["default_pos"],
                    "default_ori": clutter["default_ori"],
                }
            )
            inputs.extend(clutter_inputs)
            placed_in_group += 1
        total_placed += placed_in_group
    return (total_placed, inputs)


def _populate_stack_bowls_random(
    *,
    source_root: Path,
    assets_root: Path,
    config: dict[str, Any],
    scene: dict[str, Any],
    generation_seed: int,
    layout_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task = "stack_bowls_random"
    selected = _selected_groups(task, config, assets_root, generation_seed, layout_id)
    if len(selected) != 1 or len(selected[0][3]) != 3:
        raise TrainLayoutError("stack_bowls_random requires exactly three bowls")
    layout: dict[str, Any] = {}
    inputs: list[dict[str, Any]] = []
    occupied: list[_Placed] = []
    prohibited = list(config.get("ProhibitedArea") or [])
    object_type, group_cfg, category_cfg, indices = selected[0]
    labels = list((group_cfg.get("select_mode") or {}).get("label") or [])
    for label, index in zip(labels, indices, strict=True):
        bowl, bowl_inputs = _make_instance(
            source_root=source_root,
            assets_root=assets_root,
            scene=scene,
            task=task,
            generation_seed=generation_seed,
            layout_id=layout_id,
            object_type=object_type,
            category=str(category_cfg["name"]),
            category_idx=index,
            group=category_cfg.get("group"),
            label=str(label),
            common=deepcopy(group_cfg.get("common") or {}),
            occupied=occupied,
            prohibited=prohibited,
        )
        layout.setdefault(object_type, {}).setdefault(
            str(category_cfg["name"]), []
        ).append(bowl)
        inputs.extend(bowl_inputs)
    _, clutter_inputs = _populate_declared_clutter(
        source_root=source_root,
        assets_root=assets_root,
        task=task,
        config=config,
        scene=scene,
        generation_seed=generation_seed,
        layout_id=layout_id,
        layout=layout,
        occupied=occupied,
        prohibited=prohibited,
    )
    inputs.extend(clutter_inputs)
    return (layout, inputs)


def _populate_arrange_largest_number(
    *,
    source_root: Path,
    assets_root: Path,
    config: dict[str, Any],
    scene: dict[str, Any],
    task: str,
    generation_seed: int,
    layout_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Implement the task's hierarchical digits and linked-count mat contract."""
    if task not in {"arrange_largest_number", "arrange_largest_number_random"}:
        raise TrainLayoutError(f"unsupported largest-number task: {task}")
    rigid_groups = config.get("Rigid") or []
    mat_groups = config.get("Geometry") or []
    if len(rigid_groups) != 1 or len(mat_groups) != 1:
        raise TrainLayoutError(
            "arrange_largest_number requires one rigid digit group and one geometry mat group"
        )
    rigid_group, mat_group = (rigid_groups[0], mat_groups[0])
    rigid_select = rigid_group.get("select_mode") or {}
    mat_select = mat_group.get("select_mode") or {}
    if (
        rigid_select.get("mode") != "hierarchical"
        or rigid_select.get("instance_sample_mode") != "unique"
        or rigid_select.get("label_prefix") != ["digit"]
        or (mat_select.get("mode") != "linked_count")
        or (mat_select.get("linked_label_prefix") != "digit")
        or (mat_select.get("label_prefix") != ["mat"])
    ):
        raise TrainLayoutError(
            "arrange_largest_number task-config selection semantics drifted"
        )
    counts = [int(value) for value in rigid_select.get("select_instance_nums") or []]
    if counts != [4, 5]:
        raise TrainLayoutError("arrange_largest_number must select four or five digits")
    count = counts[
        1 if _unit(generation_seed, task, layout_id, "digit_count") >= 0.5 else 0
    ]
    rigid_categories = list(rigid_group.get("category") or [])
    category_cfg = _choice(
        rigid_categories, generation_seed, task, layout_id, "hierarchical_category"
    )
    digit_indices = sorted(
        _allowed_indices(assets_root, "Rigid", category_cfg),
        key=lambda value: _unit(
            generation_seed, task, layout_id, f"digit_index:{value}"
        ),
    )[:count]
    if len(digit_indices) != count or len(set(digit_indices)) != count:
        raise TrainLayoutError(
            "arrange_largest_number lacks enough unique digits in one texture group"
        )
    mat_categories = list(mat_group.get("category") or [])
    if len(mat_categories) != 1:
        raise TrainLayoutError("arrange_largest_number requires one mat category")
    mat_category = mat_categories[0]
    if _allowed_indices(assets_root, "Geometry", mat_category) != [1]:
        raise TrainLayoutError("arrange_largest_number mat catalog binding drifted")
    dynamic = deepcopy(mat_group.get("dynamic_layout") or {})
    if dynamic != {"type": "x_symmetric_line", "center": 0.0, "spacing": 0.085}:
        raise TrainLayoutError(
            "arrange_largest_number symmetric mat-row contract drifted"
        )
    layout: dict[str, Any] = {}
    occupied: list[_Placed] = []
    occupied_polygons: list[Any] | None = (
        [] if task == "arrange_largest_number_random" else None
    )
    inputs: list[dict[str, Any]] = []
    shared_y = -0.1 + 0.05 * _unit(generation_seed, task, layout_id, "mat_shared_y")
    mat_common_base = deepcopy(mat_group.get("common") or {})
    for index in range(count):
        x = float(dynamic["center"]) + (index - (count - 1) / 2.0) * float(
            dynamic["spacing"]
        )
        common = {**mat_common_base, "xlim": [x, x], "ylim": [shared_y, shared_y]}
        if occupied_polygons is not None:
            record, records = _make_table_surface_instance(
                source_root=source_root,
                assets_root=assets_root,
                scene=scene,
                task=task,
                generation_seed=generation_seed,
                layout_id=layout_id,
                object_type="Geometry",
                category=str(mat_category["name"]),
                category_idx=1,
                group=mat_category.get("group"),
                label=f"mat_{index}",
                common=common,
                occupied_polygons=occupied_polygons,
                prohibited=[],
            )
        else:
            record, records = _make_instance(
                source_root=source_root,
                assets_root=assets_root,
                scene=scene,
                task=task,
                generation_seed=generation_seed,
                layout_id=layout_id,
                object_type="Geometry",
                category=str(mat_category["name"]),
                category_idx=1,
                group=mat_category.get("group"),
                label=f"mat_{index}",
                common=common,
                occupied=occupied,
                prohibited=[],
            )
        record.update(
            {
                "symmetric_index": index,
                "symmetric_total": count,
                "dynamic_layout": dynamic,
                "shared_y": shared_y,
            }
        )
        layout.setdefault("Geometry", {}).setdefault(
            str(mat_category["name"]), []
        ).append(record)
        inputs.extend(records)
    rigid_common = deepcopy(rigid_group.get("common") or {})
    prohibited = list(config.get("ProhibitedArea") or [])
    for index, asset_index in enumerate(digit_indices):
        if occupied_polygons is not None:
            record, records = _make_table_surface_instance(
                source_root=source_root,
                assets_root=assets_root,
                scene=scene,
                task=task,
                generation_seed=generation_seed,
                layout_id=layout_id,
                object_type="Rigid",
                category=str(category_cfg["name"]),
                category_idx=asset_index,
                group=category_cfg.get("group"),
                label=f"digit_{index}",
                common=rigid_common,
                occupied_polygons=occupied_polygons,
                prohibited=prohibited,
            )
        else:
            record, records = _make_instance(
                source_root=source_root,
                assets_root=assets_root,
                scene=scene,
                task=task,
                generation_seed=generation_seed,
                layout_id=layout_id,
                object_type="Rigid",
                category=str(category_cfg["name"]),
                category_idx=asset_index,
                group=category_cfg.get("group"),
                label=f"digit_{index}",
                common=rigid_common,
                occupied=occupied,
                prohibited=prohibited,
            )
        layout.setdefault("Rigid", {}).setdefault(str(category_cfg["name"]), []).append(
            record
        )
        inputs.extend(records)
    if task == "arrange_largest_number_random":
        placed_clutter, clutter_inputs = _populate_declared_clutter(
            source_root=source_root,
            assets_root=assets_root,
            task=task,
            config=config,
            scene=scene,
            generation_seed=generation_seed,
            layout_id=layout_id,
            layout=layout,
            occupied_polygons=occupied_polygons,
            prohibited=prohibited,
        )
        inputs.extend(clutter_inputs)
        if placed_clutter != 20:
            raise _PlacementExhausted(
                f"{task} requires exactly twenty placed clutter objects"
            )
    return (layout, inputs)


def _populate_arrange_largest_number_random(
    *,
    source_root: Path,
    assets_root: Path,
    config: dict[str, Any],
    scene: dict[str, Any],
    generation_seed: int,
    layout_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Retry the complete random layout until all twenty clutter objects fit."""
    last_error: _PlacementExhausted | None = None
    for whole_attempt in range(WHOLE_LAYOUT_MAX_ATTEMPTS):
        sampling_id = _whole_layout_sampling_id(layout_id, whole_attempt)
        try:
            layout, inputs = _populate_arrange_largest_number(
                source_root=source_root,
                assets_root=assets_root,
                config=config,
                scene=scene,
                task="arrange_largest_number_random",
                generation_seed=generation_seed,
                layout_id=sampling_id,
            )
        except _PlacementExhausted as exc:
            last_error = exc
            continue
        return (layout, inputs, whole_attempt + 1)
    raise TrainLayoutError(
        f"could not place arrange_largest_number_random after {WHOLE_LAYOUT_MAX_ATTEMPTS} deterministic whole-layout attempts; last_error={last_error}"
    )


def _populate_align_blocks(
    *,
    source_root: Path,
    assets_root: Path,
    config: dict[str, Any],
    scene: dict[str, Any],
    generation_seed: int,
    layout_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Place the held straightedge independently from mutually disjoint cubes.

    The triangular prism is a movable task tool/target, not a support fixture.
    Its projected physical footprint must nevertheless exclude initial cube
    placements: contact at reset makes the thin tool tip or roll before the
    controller can grasp it. Cubes must also remain mutually disjoint. This
    rule is derived from task geometry and does not consult Eval layouts.
    """
    task = "align_blocks"
    groups = config.get("Rigid") or []
    if len(groups) != 2:
        raise TrainLayoutError("align_blocks requires target and cube groups")
    target_group, cube_group = groups
    target_select = target_group.get("select_mode") or {}
    cube_select = cube_group.get("select_mode") or {}
    if (
        [item.get("name") for item in target_group.get("category") or []]
        != ["triangular_prism"]
        or target_select.get("mode") != "allow_duplicate"
        or target_select.get("label") != ["target"]
        or ([item.get("name") for item in cube_group.get("category") or []] != ["cube"])
        or (cube_select.get("mode") != "same")
        or (cube_select.get("label") != ["cube0", "cube1", "cube2"])
    ):
        raise TrainLayoutError("align_blocks task-config semantics drifted")
    selected = _selected_groups(task, config, assets_root, generation_seed, layout_id)
    if len(selected) != 2:
        raise TrainLayoutError("align_blocks selection did not produce two groups")
    layout: dict[str, Any] = {}
    inputs: list[dict[str, Any]] = []
    target_occupied: list[_Placed] = []
    cube_occupied: list[_Placed] = []
    prohibited = list(config.get("ProhibitedArea") or [])
    for group_index, (object_type, group_cfg, category_cfg, indices) in enumerate(
        selected
    ):
        labels = list((group_cfg.get("select_mode") or {}).get("label") or [])
        if len(labels) != len(indices):
            raise TrainLayoutError("align_blocks label/count mismatch")
        occupied = target_occupied if group_index == 0 else cube_occupied
        if group_index == 1 and (not cube_occupied):
            cube_occupied.extend(target_occupied)
        for label, index in zip(labels, indices, strict=True):
            instance_common = deepcopy(group_cfg.get("common") or {})
            record, records = _make_instance(
                source_root=source_root,
                assets_root=assets_root,
                scene=scene,
                task=task,
                generation_seed=generation_seed,
                layout_id=layout_id,
                object_type=object_type,
                category=str(category_cfg["name"]),
                category_idx=index,
                group=category_cfg.get("group"),
                label=str(label),
                common=instance_common,
                occupied=occupied,
                prohibited=prohibited,
            )
            layout.setdefault(object_type, {}).setdefault(
                str(category_cfg["name"]), []
            ).append(record)
            inputs.extend(records)
    return (layout, inputs)


def _materialize_default_camera_stand(
    *, source_root: Path, assets_root: Path, scene: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Materialize the fixed default-scene camera stand without Eval ancestry."""
    groups = scene.get("Geometry") or []
    if len(groups) != 1:
        raise TrainLayoutError("default scene must declare exactly one Geometry group")
    group = groups[0]
    common = deepcopy(group.get("common") or {})
    categories = group.get("category") or []
    selection = group.get("select_mode") or {}
    if (
        len(categories) != 1
        or categories[0].get("name") != "camera_stand"
        or categories[0].get("index") != [0]
        or (selection.get("nums") != 1)
        or (selection.get("mode") != "unique")
        or (selection.get("label") != ["camera_stand"])
        or (common.get("xlim") != [0.0, 0.0])
        or (common.get("ylim") != [-0.47, -0.47])
        or (common.get("zlim") != [0.715, 0.715])
        or (common.get("qpos") != [0.707, -0.707, 0.0, 0.0])
        or (common.get("rotate_rand") is not False)
        or (common.get("relative_plane") != "Ground")
    ):
        raise TrainLayoutError("default camera_stand scene contract drifted")
    metadata_path, asset_path = _asset_paths(assets_root, "Geometry", "camera_stand", 0)
    metadata = _load_asset_metadata(metadata_path)
    ground = scene.get("Ground") or {}
    ground_position = ground.get("default_pos") or []
    if (
        ground.get("geometry") != "cube"
        or len(ground_position) != 3
        or (not isinstance(ground.get("thickness"), (int, float)))
    ):
        raise TrainLayoutError("default Ground contract is invalid")
    ground_top = float(ground_position[2]) + float(ground["thickness"]) / 2.0
    physics = deepcopy(metadata.get("physics") or {})
    physics["type"] = "geometry"
    record = {
        "category": "camera_stand",
        "category_idx": 0,
        "group": None,
        "xlim": deepcopy(common["xlim"]),
        "ylim": deepcopy(common["ylim"]),
        "zlim": deepcopy(common["zlim"]),
        "qpos": deepcopy(common["qpos"]),
        "rotate_deg": common.get("rotate_deg", 0),
        "rotate_rand": False,
        "relative_plane": "Ground",
        "place_tag": common.get("place_tag"),
        "margin": float(common.get("margin", 0.01)),
        "check_mode": common.get("check_mode", "bbox"),
        "need_check_stable": bool(common.get("need_check_stable", True)),
        "label": "camera_stand",
        "default_pos": [
            float(common["xlim"][0]),
            float(common["ylim"][0]),
            ground_top + float(common["zlim"][0]),
        ],
        "default_ori": deepcopy(common["qpos"]),
        "scale": deepcopy(common.get("scale", [1.0, 1.0, 1.0])),
        "physics": physics,
        "visual": deepcopy(common.get("visual", {})),
    }
    return (
        record,
        [
            _input_record(metadata_path, source_root, "object_metadata"),
            _input_record(asset_path, source_root, "object_asset"),
        ],
    )


def _whole_layout_sampling_id(layout_id: int, attempt: int) -> int:
    """Give every rejection-resampling pass a disjoint deterministic RNG key."""
    if layout_id < 0 or attempt < 0 or attempt >= WHOLE_LAYOUT_MAX_ATTEMPTS:
        raise TrainLayoutError("invalid whole-layout rejection-sampling identity")
    if attempt == 0:
        return layout_id
    return -((layout_id + 1 << 16) + attempt)


def _support_frame_from_parent(
    *,
    source_root: Path,
    assets_root: Path,
    parent: dict[str, Any],
    relative_plane: str,
    compact_support_key: bool = False,
) -> tuple[list[float], list[float], float]:
    """Resolve a canonical ``label/support-tag/index`` passive support frame.

    RoboDojo's relative-plane path names the already sampled parent by label
    and then the passive-support metadata key.  The returned frame is computed
    only from that parent's frozen pose and catalog metadata; no Eval layout or
    simulator state is consulted.
    """
    parts = str(relative_plane).split("/")
    if len(parts) == 3 and all(parts):
        parent_label, support_name, support_index = parts
        support_key = f"{support_name}/{support_index}"
    elif compact_support_key and len(parts) == 2 and all(parts):
        parent_label, support_key = parts
    else:
        raise TrainLayoutError("relative support plane must be label/support-tag/index")
    if parent.get("label") != parent_label:
        raise TrainLayoutError(
            f"relative support parent label mismatch: {relative_plane}"
        )
    metadata_path, _ = _asset_paths(
        assets_root,
        str(parent["physics"]["type"]).capitalize(),
        str(parent["category"]),
        int(parent["category_idx"]),
    )
    metadata = _load_asset_metadata(metadata_path)
    support = ((metadata.get("passive") or {}).get("support") or {}).get(support_key)
    if not isinstance(support, dict):
        raise TrainLayoutError(f"parent metadata lacks passive support {support_key}")
    centers = support.get("center")
    radii = support.get("radius")
    if (
        not isinstance(centers, list)
        or len(centers) != 1
        or (not isinstance(centers[0], list))
        or (len(centers[0]) != 7)
        or (not isinstance(radii, list))
        or (len(radii) != 1)
    ):
        raise TrainLayoutError(
            "independent relative-plane generation requires exactly one 7D passive support frame and radius"
        )
    local = [float(value) for value in centers[0]]
    radius = float(radii[0])
    if not math.isfinite(radius) or radius <= 0.0:
        raise TrainLayoutError("passive support radius must be positive")
    parent_pos = [float(value) for value in parent["default_pos"]]
    parent_ori = [float(value) for value in parent["default_ori"]]
    offset = _quat_rotate_vector(parent_ori, local[:3])
    frame_pos = [parent_pos[index] + offset[index] for index in range(3)]
    frame_ori = _quat_multiply(parent_ori, local[3:])
    return (frame_pos, frame_ori, radius)


def _oriented_bbox_vertices(metadata: dict[str, Any]) -> list[list[float]]:
    vertices = ((metadata.get("geometry") or {}).get("oriented_bbox") or {}).get(
        "vertices"
    )
    if (
        not isinstance(vertices, list)
        or len(vertices) < 4
        or any((not isinstance(point, list) or len(point) != 3 for point in vertices))
    ):
        raise TrainLayoutError("asset metadata lacks oriented bbox vertices")
    return [[float(value) for value in point] for point in vertices]


def _placement_candidates(
    metadata: dict[str, Any], common: dict[str, Any]
) -> list[dict[str, Any]]:
    placements = (metadata.get("active") or {}).get("place") or {}
    if not isinstance(placements, dict):
        raise TrainLayoutError("asset has no active placement frames")
    requested = common.get("place_tag")
    if requested is None:
        tags = list(placements)
    elif isinstance(requested, str):
        tags = [requested]
    elif (
        isinstance(requested, list)
        and requested
        and all((isinstance(tag, str) and tag for tag in requested))
    ):
        tags = list(requested)
    else:
        raise TrainLayoutError("place_tag must be a string or nonempty string list")
    candidates = []
    for tag in tags:
        value = placements.get(tag)
        center = (
            (value.get("projection_circle") or {}).get("center")
            if isinstance(value, dict)
            else None
        )
        if isinstance(center, list) and len(center) == 7:
            candidates.append({"tag": tag, "center": [float(v) for v in center]})
    if not candidates:
        raise TrainLayoutError("asset has no requested projection-circle frame")
    return candidates


def _quat_inverse(quaternion: list[float]) -> list[float]:
    norm_squared = sum((float(value) ** 2 for value in quaternion))
    if norm_squared <= 0.0:
        raise TrainLayoutError("zero-norm placement quaternion")
    w, x, y, z = (float(value) for value in quaternion)
    return [w / norm_squared, -x / norm_squared, -y / norm_squared, -z / norm_squared]


def _make_table_surface_instance(
    *,
    source_root: Path,
    assets_root: Path,
    scene: dict[str, Any],
    task: str,
    generation_seed: int,
    layout_id: int,
    object_type: str,
    category: str,
    category_idx: int,
    group: str | None,
    label: str,
    common: dict[str, Any],
    occupied_polygons: list[Any],
    prohibited: list[Any],
    max_attempts: int = 512,
    cluttered: bool = False,
    allow_geometry_origin: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Exact polygon/retry semantics for a Table-surface config group."""
    metadata_path, asset_path = _asset_paths(
        assets_root, object_type, category, category_idx
    )
    metadata = _load_asset_metadata(metadata_path)
    fixed_z = common.get("zlim") is not None
    vertices = _oriented_bbox_vertices(metadata)
    has_active_place = bool((metadata.get("active") or {}).get("place") or {})
    geometry_origin = allow_geometry_origin and (not has_active_place)
    placements = (
        [{"tag": None, "center": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}]
        if fixed_z or geometry_origin
        else _placement_candidates(metadata, common)
    )
    rotate_deg = deepcopy(common.get("rotate_deg", 0.0))
    rotate_rand = bool(common.get("rotate_rand", False))
    margin = float(common.get("margin", 0.01))
    check_mode = str(common.get("check_mode", "bbox"))
    if check_mode not in {"bbox", "point", "none", "enforce"}:
        raise TrainLayoutError(f"unsupported Table placement check_mode: {check_mode}")
    table = scene.get("Table") or {}
    table_pos, table_scale = (table.get("default_pos"), table.get("scale"))
    if not isinstance(table_pos, list) or not isinstance(table_scale, list):
        raise TrainLayoutError("scene config lacks Table position/scale")
    table_region = box(
        float(table_pos[0]) - float(table_scale[0]) / 2.0 + 0.05,
        float(table_pos[1]) - float(table_scale[1]) / 2.0 + 0.05,
        float(table_pos[0]) + float(table_scale[0]) / 2.0 - 0.05,
        float(table_pos[1]) + float(table_scale[1]) / 2.0 - 0.05,
    )
    prohibited_polygons = []
    for area in prohibited:
        if not isinstance(area, list) or len(area) != 4:
            raise TrainLayoutError(
                "ProhibitedArea entries must contain four coordinates"
            )
        prohibited_polygons.append(
            box(
                min(float(area[0]), float(area[2])),
                min(float(area[1]), float(area[3])),
                max(float(area[0]), float(area[2])),
                max(float(area[1]), float(area[3])),
            )
        )
    selected_pos = None
    selected_ori = None
    selected_polygon = None
    selected_tag = None
    for attempt in range(max_attempts):
        placement = _choice(
            placements, generation_seed, task, layout_id, f"{label}:placement:{attempt}"
        )
        center = placement["center"]
        base_ori = (
            [float(value) for value in common.get("qpos", [1.0, 0.0, 0.0, 0.0])]
            if fixed_z
            else [1.0, 0.0, 0.0, 0.0]
            if geometry_origin
            else _quat_inverse(center[3:])
        )
        if (
            len(base_ori) != 4
            or not all((math.isfinite(value) for value in base_ori))
            or sum((value * value for value in base_ori)) <= 0.0
        ):
            raise TrainLayoutError("fixed-z qpos must be a finite nonzero quaternion")
        _, sampled_deg = _sample_rotate_deg(
            common.get("rotate_deg", 0.0),
            _unit(generation_seed, task, layout_id, f"{label}:yaw", attempt),
        )
        yaw = math.radians(sampled_deg) if rotate_rand else 0.0
        yaw_quat = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
        orientation = _quat_multiply(yaw_quat, base_ori)
        origin_offset = (
            [0.0, 0.0, 0.0]
            if fixed_z
            else [0.0, 0.0, -min((vertex[2] for vertex in vertices))]
            if geometry_origin
            else [-value for value in _quat_rotate_vector(base_ori, center[:3])]
        )
        sampled_x = _sample_limit(
            common.get("xlim"),
            _unit(generation_seed, task, layout_id, f"{label}:x", attempt),
            field="xlim",
        )
        sampled_y = _sample_limit(
            common.get("ylim"),
            _unit(generation_seed, task, layout_id, f"{label}:y", attempt),
            field="ylim",
        )
        z_offset = (
            _sample_limit(
                common["zlim"],
                _unit(generation_seed, task, layout_id, f"{label}:z", attempt),
                field="zlim",
            )
            if fixed_z
            else origin_offset[2]
        )
        position = [
            sampled_x + origin_offset[0],
            sampled_y + origin_offset[1],
            _plane_height(common, scene) + z_offset,
        ]
        rotated_vertices = [
            _quat_rotate_vector(orientation, vertex) for vertex in vertices
        ]
        polygon = MultiPoint(
            [
                (position[0] + rotated[0], position[1] + rotated[1])
                for rotated in rotated_vertices
            ]
        ).convex_hull.buffer(margin)
        local_z_max = z_offset + max((vertex[2] for vertex in rotated_vertices))
        if cluttered and (
            position[1] + polygon.bounds[1] < 0.0
            and local_z_max > 0.1
            or (position[1] + polygon.bounds[1] < -0.2 and local_z_max > 0.05)
        ):
            continue
        if check_mode == "bbox" and (not table_region.contains(polygon)):
            continue
        if check_mode != "enforce" and (
            any((polygon.intersects(other) for other in occupied_polygons))
            or any((polygon.intersects(other) for other in prohibited_polygons))
        ):
            continue
        selected_pos = position
        selected_ori = orientation
        selected_polygon = polygon
        selected_tag = placement["tag"]
        break
    if selected_pos is None:
        raise _PlacementExhausted(
            f"could not place {task}:{label} with canonical polygon checks"
        )
    occupied_polygons.append(selected_polygon)
    physics = deepcopy(metadata.get("physics") or {})
    physics["type"] = object_type.lower()
    record = {
        "category": category,
        "category_idx": category_idx,
        "xlim": deepcopy(common.get("xlim")),
        "ylim": deepcopy(common.get("ylim")),
        "zlim": deepcopy(common.get("zlim")),
        "qpos": deepcopy(common.get("qpos", [1.0, 0.0, 0.0, 0.0])),
        "rotate_deg": rotate_deg,
        "rotate_rand": rotate_rand,
        "relative_plane": "Table",
        "place_tag": common.get("place_tag"),
        "sampled_place_tag": selected_tag,
        "placement_mode": "fixed_z_origin_pose"
        if fixed_z
        else "surface_geometry_origin"
        if geometry_origin
        else "surface_projection_circle",
        "margin": margin,
        "check_mode": check_mode,
        "need_check_stable": bool(common.get("need_check_stable", True)),
        "label": label,
        "default_pos": selected_pos,
        "default_ori": selected_ori,
        "scale": deepcopy(common.get("scale", [1.0, 1.0, 1.0])),
        "physics": physics,
        "visual": deepcopy(common.get("visual", {})),
    }
    if group is not None:
        record["group"] = group
    return (
        record,
        [
            _input_record(metadata_path, source_root, "object_metadata"),
            _input_record(asset_path, source_root, "object_asset"),
        ],
    )


def _make_relative_support_instance(
    *,
    source_root: Path,
    assets_root: Path,
    task: str,
    generation_seed: int,
    layout_id: int,
    object_type: str,
    category: str,
    category_idx: int,
    group: str | None,
    label: str,
    common: dict[str, Any],
    parent: dict[str, Any],
    occupied_polygons: list[Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Place an object in the parent's passive-support frame.

    This mirrors the relevant ``ClutteredGenerator`` semantics: sample the
    placement center in the intersection of the configured XY region and the
    passive support circle; solve object origin from the active placement
    frame; rotate around support-local Z; require the buffered exact oriented
    bbox footprint to remain in the support circle for ``bbox`` mode; and
    reject collisions between siblings on the same support.
    """
    relative_plane = str(common.get("relative_plane", ""))
    frame_pos, frame_ori, support_radius = _support_frame_from_parent(
        source_root=source_root,
        assets_root=assets_root,
        parent=parent,
        relative_plane=relative_plane,
        compact_support_key=task == "make_kong",
    )
    relative_parts = relative_plane.split("/")
    if len(relative_parts) == 3 and all(relative_parts):
        parent_label, support_name, support_index = relative_parts
        support_key = f"{support_name}/{support_index}"
    elif task == "make_kong" and len(relative_parts) == 2 and all(relative_parts):
        parent_label, support_key = relative_parts
    else:
        raise TrainLayoutError(
            f"{task} relative support plane must use its frozen public path contract"
        )
    parent_metadata_path, _ = _asset_paths(
        assets_root,
        str(parent["physics"]["type"]).capitalize(),
        str(parent["category"]),
        int(parent["category_idx"]),
    )
    parent_metadata = _load_asset_metadata(parent_metadata_path)
    support_metadata = (
        (parent_metadata.get("passive") or {}).get("support") or {}
    ).get(support_key)
    support_local_frame = (
        support_metadata.get("center", [None])[0]
        if isinstance(support_metadata, dict)
        else None
    )
    if (
        parent_label != parent.get("label")
        or not isinstance(support_local_frame, list)
        or len(support_local_frame) != 7
    ):
        raise TrainLayoutError(
            f"invalid support provenance binding for {relative_plane}"
        )
    metadata_path, asset_path = _asset_paths(
        assets_root, object_type, category, category_idx
    )
    metadata = _load_asset_metadata(metadata_path)
    fixed_z = common.get("zlim") is not None
    if fixed_z:
        placements = []
        base_ori = [float(value) for value in common.get("qpos", [1.0, 0.0, 0.0, 0.0])]
        if (
            len(base_ori) != 4
            or not all((math.isfinite(value) for value in base_ori))
            or sum((value * value for value in base_ori)) <= 0.0
        ):
            raise TrainLayoutError(
                "fixed-z relative-support qpos must be a finite nonzero quaternion"
            )
        origin_offset = [0.0, 0.0, 0.0]
    else:
        placements = _placement_candidates(metadata, common)
        if len(placements) != 1:
            raise TrainLayoutError(
                "relative support child requires exactly one active placement frame for these canonical tasks"
            )
        center = placements[0]["center"]
        base_ori = _quat_inverse(center[3:])
        origin_offset = [-value for value in _quat_rotate_vector(base_ori, center[:3])]
    rotate_deg, _ = _sample_rotate_deg(
        common.get("rotate_deg", 0.0),
        _unit(generation_seed, task, layout_id, f"{label}:yaw"),
    )
    rotate_rand = bool(common.get("rotate_rand", False))
    margin = float(common.get("margin", 0.01))
    if not math.isfinite(margin) or margin < 0.0:
        raise TrainLayoutError("relative support margin must be non-negative")
    check_mode = str(common.get("check_mode", "bbox"))
    if check_mode not in {"bbox", "point", "none", "enforce"}:
        raise TrainLayoutError(f"unsupported relative support check_mode: {check_mode}")
    support_region = Point(0.0, 0.0).buffer(support_radius)
    vertices = _oriented_bbox_vertices(metadata)
    selected_local_pos: list[float] | None = None
    selected_local_ori: list[float] | None = None
    selected_polygon = None
    for attempt in range(512):
        _, sampled_rotate_deg = _sample_rotate_deg(
            common.get("rotate_deg", 0.0),
            _unit(generation_seed, task, layout_id, f"{label}:yaw", attempt),
        )
        yaw = math.radians(sampled_rotate_deg) if rotate_rand else 0.0
        yaw_quat = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
        local_ori = _quat_multiply(yaw_quat, base_ori)
        sampled_x = _sample_limit(
            common.get("xlim"),
            _unit(generation_seed, task, layout_id, f"{label}:x", attempt),
            field="xlim",
        )
        sampled_y = _sample_limit(
            common.get("ylim"),
            _unit(generation_seed, task, layout_id, f"{label}:y", attempt),
            field="ylim",
        )
        if not support_region.intersects(Point(sampled_x, sampled_y)):
            continue
        sampled_z = (
            _sample_limit(
                common["zlim"],
                _unit(generation_seed, task, layout_id, f"{label}:z", attempt),
                field="zlim",
            )
            if fixed_z
            else origin_offset[2]
        )
        local_pos = [
            sampled_x + origin_offset[0],
            sampled_y + origin_offset[1],
            sampled_z,
        ]
        transformed_xy = []
        for vertex in vertices:
            rotated = _quat_rotate_vector(local_ori, vertex)
            transformed_xy.append(
                (local_pos[0] + rotated[0], local_pos[1] + rotated[1])
            )
        polygon = MultiPoint(transformed_xy).convex_hull.buffer(margin)
        if check_mode == "bbox" and (not support_region.contains(polygon)):
            continue
        if check_mode != "enforce" and any(
            (polygon.intersects(existing) for existing in occupied_polygons)
        ):
            continue
        selected_local_pos = local_pos
        selected_local_ori = local_ori
        selected_polygon = polygon
        break
    if (
        selected_local_pos is None
        or selected_local_ori is None
        or selected_polygon is None
    ):
        raise _PlacementExhausted(f"could not place {task}:{label} on {relative_plane}")
    occupied_polygons.append(selected_polygon)
    world_offset = _quat_rotate_vector(frame_ori, selected_local_pos)
    position = [frame_pos[index] + world_offset[index] for index in range(3)]
    orientation = _quat_multiply(selected_local_ori, frame_ori)
    physics = deepcopy(metadata.get("physics") or {})
    physics["type"] = object_type.lower()
    record = {
        "category": category,
        "category_idx": category_idx,
        "xlim": deepcopy(common.get("xlim")),
        "ylim": deepcopy(common.get("ylim")),
        "zlim": deepcopy(common.get("zlim")),
        "qpos": deepcopy(common.get("qpos", [1.0, 0.0, 0.0, 0.0])),
        "rotate_deg": rotate_deg,
        "rotate_rand": rotate_rand,
        "relative_plane": relative_plane,
        "place_tag": common.get("place_tag"),
        "sampled_place_tag": None if fixed_z else placements[0]["tag"],
        "placement_mode": "fixed_z_origin_pose"
        if fixed_z
        else "surface_projection_circle",
        "margin": margin,
        "check_mode": check_mode,
        "need_check_stable": bool(common.get("need_check_stable", True)),
        "label": label,
        "support_binding": {
            "parent_label": parent_label,
            "parent_category": parent["category"],
            "parent_category_idx": int(parent["category_idx"]),
            "support_key": support_key,
            "parent_metadata_path": parent_metadata_path.resolve().as_posix(),
            "support_local_frame": [float(value) for value in support_local_frame],
            "support_radius_m": float(support_radius),
            "support_world_frame": [
                *[float(value) for value in frame_pos],
                *[float(value) for value in frame_ori],
            ],
            "frame_composition": "parent_world_times_support_local",
            "child_world_quaternion_order": "child_local_times_support_world",
        },
        "default_pos": position,
        "default_ori": orientation,
        "scale": deepcopy(common.get("scale", [1.0, 1.0, 1.0])),
        "physics": physics,
        "visual": deepcopy(common.get("visual", {})),
    }
    if group is not None:
        record["group"] = group
    return (
        record,
        [
            _input_record(metadata_path, source_root, "object_metadata"),
            _input_record(asset_path, source_root, "object_asset"),
        ],
    )


def _conveyor_support_contract(
    *, source_root: Path, assets_root: Path, parent: dict[str, Any], relative_plane: str
) -> dict[str, Any]:
    """Resolve the native conveyor's multi-circle passive support surface."""
    parts = relative_plane.split("/")
    if len(parts) != 3 or parent.get("label") != parts[0]:
        raise TrainLayoutError(
            "match conveyor relative plane must bind conveyor/conveyor/0"
        )
    support_key = f"{parts[1]}/{parts[2]}"
    metadata_path, _ = _asset_paths(
        assets_root,
        str(parent["physics"]["type"]).capitalize(),
        str(parent["category"]),
        int(parent["category_idx"]),
    )
    metadata = _load_asset_metadata(metadata_path)
    support = ((metadata.get("passive") or {}).get("support") or {}).get(support_key)
    centers = support.get("center") if isinstance(support, dict) else None
    radii = support.get("radius") if isinstance(support, dict) else None
    if (
        not isinstance(centers, list)
        or not centers
        or any((not isinstance(center, list) or len(center) != 7 for center in centers))
        or (not isinstance(radii, list))
        or (len(radii) != len(centers))
    ):
        raise TrainLayoutError(
            "native conveyor support must declare aligned 7D centers and radii"
        )
    clean_centers = [[float(value) for value in center] for center in centers]
    clean_radii = [float(value) for value in radii]
    if any((not math.isfinite(radius) or radius <= 0.0 for radius in clean_radii)):
        raise TrainLayoutError("native conveyor support radii must be positive")
    anchor = clean_centers[0]
    anchor_inverse = _quat_inverse(anchor[3:])
    local_regions: list[dict[str, Any]] = []
    polygons = []
    for center, radius in zip(clean_centers, clean_radii, strict=True):
        relative_position = _quat_rotate_vector(
            anchor_inverse, [center[index] - anchor[index] for index in range(3)]
        )
        local_regions.append({"center": relative_position, "radius_m": radius})
        polygons.append(
            Point(relative_position[0], relative_position[1]).buffer(radius)
        )
    parent_position = [float(value) for value in parent["default_pos"]]
    parent_orientation = [float(value) for value in parent["default_ori"]]
    anchor_offset = _quat_rotate_vector(parent_orientation, anchor[:3])
    anchor_world_position = [
        parent_position[index] + anchor_offset[index] for index in range(3)
    ]
    anchor_world_orientation = _quat_multiply(parent_orientation, anchor[3:])
    return {
        "support_key": support_key,
        "parent_metadata_path": metadata_path.resolve().as_posix(),
        "anchor_local_frame": anchor,
        "anchor_world_position": anchor_world_position,
        "anchor_world_orientation": anchor_world_orientation,
        "local_regions": local_regions,
        "region": unary_union(polygons),
    }


def _make_conveyor_child_instance(
    *,
    source_root: Path,
    assets_root: Path,
    task: str,
    generation_seed: int,
    layout_id: int,
    category: str,
    category_idx: int,
    group: str | None,
    label: str,
    common: dict[str, Any],
    parent: dict[str, Any],
    support: dict[str, Any],
    occupied_polygons: list[Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Place one rigid object in the conveyor's anchored support frame."""
    metadata_path, asset_path = _asset_paths(
        assets_root, "Rigid", category, category_idx
    )
    metadata = _load_asset_metadata(metadata_path)
    placements = _placement_candidates(metadata, common)
    vertices = _oriented_bbox_vertices(metadata)
    rotate_deg = deepcopy(common.get("rotate_deg", 0.0))
    rotate_rand = bool(common.get("rotate_rand", False))
    margin = float(common.get("margin", 0.01))
    check_mode = str(common.get("check_mode", "bbox"))
    if check_mode not in {"bbox", "point", "none", "enforce"}:
        raise TrainLayoutError("unsupported conveyor placement check_mode")
    selected: tuple[list[float], list[float], Any, str] | None = None
    support_region = support["region"]
    for attempt in range(512):
        placement = _choice(
            placements, generation_seed, task, layout_id, f"{label}:placement:{attempt}"
        )
        center = placement["center"]
        base_orientation = _quat_inverse(center[3:])
        origin_offset = [
            -value for value in _quat_rotate_vector(base_orientation, center[:3])
        ]
        _, sampled_rotation = _sample_rotate_deg(
            common.get("rotate_deg", 0.0),
            _unit(generation_seed, task, layout_id, f"{label}:yaw", attempt),
        )
        yaw = math.radians(sampled_rotation) if rotate_rand else 0.0
        local_orientation = _quat_multiply(
            [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], base_orientation
        )
        sampled_x = _sample_limit(
            common.get("xlim"),
            _unit(generation_seed, task, layout_id, f"{label}:x", attempt),
            field="xlim",
        )
        sampled_y = _sample_limit(
            common.get("ylim"),
            _unit(generation_seed, task, layout_id, f"{label}:y", attempt),
            field="ylim",
        )
        if not support_region.intersects(Point(sampled_x, sampled_y)):
            continue
        local_position = [
            sampled_x + origin_offset[0],
            sampled_y + origin_offset[1],
            origin_offset[2],
        ]
        polygon = MultiPoint(
            [
                (
                    local_position[0]
                    + _quat_rotate_vector(local_orientation, vertex)[0],
                    local_position[1]
                    + _quat_rotate_vector(local_orientation, vertex)[1],
                )
                for vertex in vertices
            ]
        ).convex_hull.buffer(margin)
        if check_mode == "bbox" and (not support_region.contains(polygon)):
            continue
        if check_mode != "enforce" and any(
            (polygon.intersects(existing) for existing in occupied_polygons)
        ):
            continue
        selected = (local_position, local_orientation, polygon, str(placement["tag"]))
        break
    if selected is None:
        raise _PlacementExhausted(
            f"could not place {task}:{label} on native conveyor support"
        )
    local_position, local_orientation, polygon, placement_tag = selected
    occupied_polygons.append(polygon)
    world_offset = _quat_rotate_vector(
        support["anchor_world_orientation"], local_position
    )
    position = [
        support["anchor_world_position"][index] + world_offset[index]
        for index in range(3)
    ]
    orientation = _quat_multiply(local_orientation, support["anchor_world_orientation"])
    physics = deepcopy(metadata.get("physics") or {})
    physics["type"] = "rigid"
    record = {
        "category": category,
        "category_idx": category_idx,
        "xlim": deepcopy(common.get("xlim")),
        "ylim": deepcopy(common.get("ylim")),
        "zlim": deepcopy(common.get("zlim")),
        "qpos": deepcopy(common.get("qpos", [1.0, 0.0, 0.0, 0.0])),
        "rotate_deg": rotate_deg,
        "rotate_rand": rotate_rand,
        "relative_plane": str(common.get("relative_plane")),
        "place_tag": common.get("place_tag"),
        "sampled_place_tag": placement_tag,
        "placement_mode": "native_conveyor_multi_region_support",
        "margin": margin,
        "check_mode": check_mode,
        "need_check_stable": bool(common.get("need_check_stable", True)),
        "label": label,
        "support_binding": {
            "parent_label": parent["label"],
            "parent_category": parent["category"],
            "parent_category_idx": int(parent["category_idx"]),
            "support_key": support["support_key"],
            "parent_metadata_path": support["parent_metadata_path"],
            "anchor_center_index": 0,
            "anchor_local_frame": deepcopy(support["anchor_local_frame"]),
            "support_regions": deepcopy(support["local_regions"]),
            "anchor_world_frame": [
                *support["anchor_world_position"],
                *support["anchor_world_orientation"],
            ],
            "frame_composition": "parent_world_times_first_support_center",
            "child_world_quaternion_order": "child_local_times_support_world",
        },
        "default_pos": position,
        "default_ori": orientation,
        "scale": deepcopy(common.get("scale", [1.0, 1.0, 1.0])),
        "physics": physics,
        "visual": deepcopy(common.get("visual", {})),
    }
    if group is not None:
        record["group"] = group
    return (
        record,
        [
            _input_record(metadata_path, source_root, "object_metadata"),
            _input_record(asset_path, source_root, "object_asset"),
        ],
    )


def _populate_match_and_pick_from_conveyor(
    *,
    source_root: Path,
    assets_root: Path,
    task: str,
    config: dict[str, Any],
    scene: dict[str, Any],
    generation_seed: int,
    layout_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Generate the native conveyor plus one exact remembered/matching pair."""
    dynamic_groups = list(config.get("Dynamic") or [])
    rigid_groups = list(config.get("Rigid") or [])
    if len(dynamic_groups) != 1 or len(rigid_groups) != 3:
        raise TrainLayoutError("match conveyor object-group contract drifted")
    conveyor_group = dynamic_groups[0]
    conveyor_select = conveyor_group.get("select_mode") or {}
    conveyor_categories = list(conveyor_group.get("category") or [])
    conveyor_common = deepcopy(conveyor_group.get("common") or {})
    if (
        len(conveyor_categories) != 1
        or conveyor_categories[0].get("name") != "conveyor"
        or _allowed_indices(assets_root, "Dynamic", conveyor_categories[0]) != [0]
        or (conveyor_select.get("nums") != 1)
        or (conveyor_select.get("mode") != "unique")
        or (list(conveyor_select.get("label") or []) != ["conveyor"])
        or (conveyor_common.get("relative_plane") != "Ground")
        or (conveyor_common.get("xlim") != [-2.08, -2.08])
        or (conveyor_common.get("ylim") != [0.175, 0.175])
        or (conveyor_common.get("zlim") != [0.0, 0.0])
        or (conveyor_common.get("qpos") != [0, 0, 0, 1])
    ):
        raise TrainLayoutError("native Dynamic conveyor contract drifted")
    expected_select = (
        ("same_as_label", ["target0"], "target1", 1),
        ("unique", ["target1"], None, 1),
        ("unique", ["other0", "other1", "other2"], None, 3),
    )
    category_signatures: list[list[tuple[str, tuple[int, ...]]]] = []
    for group, (mode, labels, same_label, count) in zip(
        rigid_groups, expected_select, strict=True
    ):
        select = group.get("select_mode") or {}
        common = group.get("common") or {}
        if (
            select.get("nums") != count
            or select.get("mode") != mode
            or list(select.get("label") or []) != labels
            or (select.get("same_label") != same_label)
            or (common.get("relative_plane") != "conveyor/conveyor/0")
            or (common.get("check_mode") != "point")
        ):
            raise TrainLayoutError("match conveyor linked selection contract drifted")
        signature = [
            (
                str(category["name"]),
                tuple(_allowed_indices(assets_root, "Rigid", category)),
            )
            for category in list(group.get("category") or [])
        ]
        if not signature or len(set(signature)) != len(signature):
            raise TrainLayoutError("match conveyor catalog identities must be unique")
        category_signatures.append(signature)
    if not all(
        (signature == category_signatures[0] for signature in category_signatures[1:])
    ):
        raise TrainLayoutError("match conveyor role catalogs must be identical")
    identities = [
        (category, category_idx)
        for category, indices in category_signatures[0]
        for category_idx in indices
    ]
    if len(identities) < 4:
        raise TrainLayoutError("match conveyor needs one target and three distractors")
    category_configs = {
        str(category["name"]): category
        for category in list(rigid_groups[1].get("category") or [])
    }
    last_error: _PlacementExhausted | None = None
    for whole_attempt in range(WHOLE_LAYOUT_MAX_ATTEMPTS):
        sampling_id = _whole_layout_sampling_id(layout_id, whole_attempt)
        target_identity = _choice(
            identities, generation_seed, task, sampling_id, "target_identity"
        )
        distractor_identities = sorted(
            [identity for identity in identities if identity != target_identity],
            key=lambda identity: _unit(
                generation_seed,
                task,
                sampling_id,
                f"distractor_identity:{identity[0]}:{identity[1]}",
            ),
        )[:3]
        metadata_path, asset_path = _asset_paths(assets_root, "Dynamic", "conveyor", 0)
        metadata = _load_asset_metadata(metadata_path)
        _, sampled_rotation = _sample_rotate_deg(
            conveyor_common.get("rotate_deg", 0.0),
            _unit(generation_seed, task, sampling_id, "conveyor:yaw"),
        )
        yaw = (
            math.radians(sampled_rotation)
            if bool(conveyor_common.get("rotate_rand", False))
            else 0.0
        )
        conveyor_orientation = _quat_multiply(
            [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
            [float(value) for value in conveyor_common["qpos"]],
        )
        conveyor_position = [
            _sample_limit(
                conveyor_common["xlim"],
                _unit(generation_seed, task, sampling_id, "conveyor:x"),
                field="xlim",
            ),
            _sample_limit(
                conveyor_common["ylim"],
                _unit(generation_seed, task, sampling_id, "conveyor:y"),
                field="ylim",
            ),
            _plane_height(conveyor_common, scene)
            + _sample_limit(
                conveyor_common["zlim"],
                _unit(generation_seed, task, sampling_id, "conveyor:z"),
                field="zlim",
            ),
        ]
        conveyor_physics = deepcopy(metadata.get("physics") or {})
        conveyor_physics["type"] = "dynamic"
        conveyor = {
            "category": "conveyor",
            "category_idx": 0,
            "xlim": deepcopy(conveyor_common["xlim"]),
            "ylim": deepcopy(conveyor_common["ylim"]),
            "zlim": deepcopy(conveyor_common["zlim"]),
            "qpos": deepcopy(conveyor_common["qpos"]),
            "rotate_deg": deepcopy(conveyor_common.get("rotate_deg", 0.0)),
            "rotate_rand": bool(conveyor_common.get("rotate_rand", False)),
            "relative_plane": "Ground",
            "place_tag": conveyor_common.get("place_tag"),
            "margin": float(conveyor_common.get("margin", 0.01)),
            "check_mode": str(conveyor_common.get("check_mode", "bbox")),
            "need_check_stable": bool(conveyor_common.get("need_check_stable", True)),
            "label": "conveyor",
            "placement_mode": "fixed_z_origin_pose",
            "default_pos": conveyor_position,
            "default_ori": conveyor_orientation,
            "scale": deepcopy(conveyor_common.get("scale", [1.0, 1.0, 1.0])),
            "physics": conveyor_physics,
            "visual": deepcopy(conveyor_common.get("visual", {})),
        }
        support = _conveyor_support_contract(
            source_root=source_root,
            assets_root=assets_root,
            parent=conveyor,
            relative_plane="conveyor/conveyor/0",
        )
        layout: dict[str, Any] = {"Dynamic": {"conveyor": [conveyor]}}
        inputs = [
            _input_record(metadata_path, source_root, "object_metadata"),
            _input_record(asset_path, source_root, "object_asset"),
        ]
        occupied_polygons: list[Any] = []
        placements = [
            (rigid_groups[0], "target0", target_identity),
            (rigid_groups[1], "target1", target_identity),
            *[
                (rigid_groups[2], f"other{index}", identity)
                for index, identity in enumerate(distractor_identities)
            ],
        ]
        try:
            for group, label, (category, category_idx) in placements:
                category_cfg = category_configs[category]
                category_placement = {
                    key: deepcopy(category_cfg[key])
                    for key in (
                        "xlim",
                        "ylim",
                        "zlim",
                        "qpos",
                        "rotate_deg",
                        "rotate_rand",
                        "place_tag",
                        "margin",
                        "check_mode",
                        "need_check_stable",
                    )
                    if key in category_cfg
                }
                common = {**deepcopy(group.get("common") or {}), **category_placement}
                instance, records = _make_conveyor_child_instance(
                    source_root=source_root,
                    assets_root=assets_root,
                    task=task,
                    generation_seed=generation_seed,
                    layout_id=sampling_id,
                    category=category,
                    category_idx=category_idx,
                    group=category_cfg.get("group"),
                    label=label,
                    common=common,
                    parent=conveyor,
                    support=support,
                    occupied_polygons=occupied_polygons,
                )
                layout.setdefault("Rigid", {}).setdefault(category, []).append(instance)
                inputs.extend(records)
        except _PlacementExhausted as exc:
            last_error = exc
            continue
        return (layout, inputs, whole_attempt + 1)
    raise TrainLayoutError(
        f"could not place {task} after {WHOLE_LAYOUT_MAX_ATTEMPTS} deterministic whole-layout attempts; last_error={last_error}"
    )


def _populate_relative_support_task(
    *,
    source_root: Path,
    assets_root: Path,
    task: str,
    config: dict[str, Any],
    scene: dict[str, Any],
    generation_seed: int,
    layout_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Generate the two canonical object-relative support-frame tasks."""
    expected: dict[str, list[dict[str, Any]]] = {
        "fill_egg_holder": [
            {
                "type": "Geometry",
                "category": "egg_basket",
                "common": {
                    "xlim": [-0.4, 0.4],
                    "ylim": [-0.2, -0.1],
                    "rotate_rand": True,
                    "rotate_deg": 30,
                    "relative_plane": "Table",
                },
                "select": {"nums": [1], "mode": "same", "label": ["egg_basket"]},
            },
            {
                "type": "Rigid",
                "category": "egg",
                "common": {
                    "xlim": [-0.1, 0.1],
                    "ylim": [-0.1, 0.1],
                    "rotate_rand": True,
                    "rotate_deg": 45,
                    "relative_plane": "egg_basket/egg_basket/0",
                    "margin": 0,
                },
                "select": {
                    "nums": [4],
                    "mode": "allow_duplicate",
                    "label": ["target0", "target1", "target2", "target3"],
                },
            },
            {
                "type": "Articulation",
                "category": "egg_holder",
                "common": {
                    "xlim": [-0.05, 0.05],
                    "ylim": [-0.2, -0.1],
                    "relative_plane": "Table",
                },
                "select": {
                    "nums": [1],
                    "mode": "allow_duplicate",
                    "label": ["egg_holder"],
                },
            },
        ],
        "deposit_coin": [
            {
                "type": "Geometry",
                "category": "vertical_coin_stand",
                "common": {
                    "xlim": [-0.43, 0.43],
                    "ylim": [-0.2, -0.05],
                    "rotate_rand": True,
                    "rotate_deg": 30,
                    "relative_plane": "Table",
                },
                "select": {
                    "nums": [1],
                    "mode": "allow_duplicate",
                    "label": ["vertical_coin_stand"],
                },
            },
            {
                "type": "Geometry",
                "category": "piggy_bank",
                "common": {
                    "xlim": [[-0.4, -0.3], [0.3, 0.4]],
                    "ylim": [-0.2, -0.15],
                    "rotate_rand": True,
                    "rotate_deg": 15,
                    "relative_plane": "Table",
                    "margin": 0.04,
                },
                "select": {
                    "nums": [1],
                    "mode": "allow_duplicate",
                    "label": ["piggy_bank"],
                },
            },
            {
                "type": "Rigid",
                "category": "coin",
                "common": {
                    "xlim": [0.0, 0.0],
                    "ylim": [0.0, 0.0],
                    "rotate_rand": False,
                    "relative_plane": "vertical_coin_stand/vertical_coin_stand/0",
                    "check_mode": "point",
                },
                "select": {"nums": [1], "mode": "allow_duplicate", "label": ["coin0"]},
            },
        ],
    }
    specs = expected.get(task)
    if specs is None:
        raise TrainLayoutError(f"{task} has no relative-support generation contract")
    actual_groups = [
        (object_type, group)
        for object_type in config
        if object_type in OBJECT_TYPES
        for group in config.get(object_type) or []
    ]
    if len(actual_groups) != len(specs):
        raise TrainLayoutError(f"{task} object-group count/order drifted")
    validated = []
    for group_number, ((object_type, group_cfg), spec) in enumerate(
        zip(actual_groups, specs, strict=True)
    ):
        categories = group_cfg.get("category") or []
        if (
            object_type != spec["type"]
            or len(categories) != 1
            or categories[0] != {"name": spec["category"]}
            or (deepcopy(group_cfg.get("common") or {}) != spec["common"])
            or (deepcopy(group_cfg.get("select_mode") or {}) != spec["select"])
        ):
            raise TrainLayoutError(
                f"{task}:group{group_number} relative-support semantics drifted"
            )
        indices = _allowed_indices(assets_root, object_type, categories[0])
        if indices != [0]:
            raise TrainLayoutError(
                f"{task}:group{group_number} catalog binding drifted"
            )
        validated.append((object_type, group_cfg, categories[0], spec))
    last_error: _PlacementExhausted | None = None
    for whole_attempt in range(WHOLE_LAYOUT_MAX_ATTEMPTS):
        sampling_id = _whole_layout_sampling_id(layout_id, whole_attempt)
        layout: dict[str, Any] = {}
        table_occupied: list[Any] = []
        support_occupied: dict[str, list[Any]] = {}
        records_by_label: dict[str, dict[str, Any]] = {}
        inputs: list[dict[str, Any]] = []
        try:
            for group_number, (object_type, group_cfg, category_cfg, spec) in enumerate(
                validated
            ):
                common = deepcopy(group_cfg.get("common") or {})
                labels = list(spec["select"]["label"])
                candidates = _allowed_indices(assets_root, object_type, category_cfg)
                mode = spec["select"]["mode"]
                if mode == "same":
                    selected_indices = [
                        _choice(
                            candidates,
                            generation_seed,
                            task,
                            sampling_id,
                            f"{object_type}:{group_number}:asset",
                        )
                    ] * len(labels)
                else:
                    selected_indices = [
                        _choice(
                            candidates,
                            generation_seed,
                            task,
                            sampling_id,
                            f"{object_type}:{group_number}:asset:{index}",
                        )
                        for index in range(len(labels))
                    ]
                for label, category_idx in zip(labels, selected_indices, strict=True):
                    relative_plane = str(common["relative_plane"])
                    if relative_plane in {"Table", "Ground"}:
                        if relative_plane != "Table":
                            raise TrainLayoutError(
                                f"{task}:{label} unexpectedly uses Ground"
                            )
                        instance, instance_inputs = _make_table_surface_instance(
                            source_root=source_root,
                            assets_root=assets_root,
                            scene=scene,
                            task=task,
                            generation_seed=generation_seed,
                            layout_id=sampling_id,
                            object_type=object_type,
                            category=spec["category"],
                            category_idx=category_idx,
                            group=category_cfg.get("group"),
                            label=label,
                            common=common,
                            occupied_polygons=table_occupied,
                            prohibited=list(config.get("ProhibitedArea") or []),
                        )
                    else:
                        parent_label = relative_plane.split("/", 1)[0]
                        parent = records_by_label.get(parent_label)
                        if parent is None:
                            raise TrainLayoutError(
                                f"{task}:{label} support parent was not declared earlier in config order"
                            )
                        instance, instance_inputs = _make_relative_support_instance(
                            source_root=source_root,
                            assets_root=assets_root,
                            task=task,
                            generation_seed=generation_seed,
                            layout_id=sampling_id,
                            object_type=object_type,
                            category=spec["category"],
                            category_idx=category_idx,
                            group=category_cfg.get("group"),
                            label=label,
                            common=common,
                            parent=parent,
                            occupied_polygons=support_occupied.setdefault(
                                relative_plane, []
                            ),
                        )
                    layout.setdefault(object_type, {}).setdefault(
                        spec["category"], []
                    ).append(instance)
                    inputs.extend(instance_inputs)
                    records_by_label[label] = instance
        except _PlacementExhausted as exc:
            last_error = exc
            continue
        return (layout, inputs, whole_attempt + 1)
    raise TrainLayoutError(
        f"could not place {task} after {WHOLE_LAYOUT_MAX_ATTEMPTS} deterministic whole-layout attempts; last_error={last_error}"
    )


def _populate_make_kong(
    *,
    source_root: Path,
    assets_root: Path,
    config: dict[str, Any],
    scene: dict[str, Any],
    generation_seed: int,
    layout_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Generate the public 13-group Mahjong contract without Eval ancestry."""
    task = "make_kong"
    expected = [
        {
            "type": "Rigid",
            "category": "mahjong",
            "common": {
                "xlim": [[-0.325, -0.325], [-0.279, -0.279], [-0.233, -0.233]],
                "ylim": [-0.15, -0.15],
                "rotate_rand": False,
                "relative_plane": "Table",
                "place_tag": "straight_front",
                "margin": 0.0,
            },
            "select": {
                "nums": 3,
                "mode": "same",
                "allow_global_same": False,
                "label": ["mahjong0_0", "mahjong0_1", "mahjong0_2"],
            },
        },
        {
            "type": "Rigid",
            "category": "mahjong",
            "common": {
                "xlim": [[-0.187, -0.187], [-0.141, -0.141], [-0.095, -0.095]],
                "ylim": [-0.15, -0.15],
                "rotate_rand": False,
                "relative_plane": "Table",
                "place_tag": "straight_front",
                "margin": 0.0,
            },
            "select": {
                "nums": 3,
                "mode": "same",
                "allow_global_same": False,
                "label": ["mahjong1_0", "mahjong1_1", "mahjong1_2"],
            },
        },
        {
            "type": "Rigid",
            "category": "mahjong",
            "common": {
                "xlim": [[-0.049, -0.049], [-0.003, -0.003], [0.043, 0.043]],
                "ylim": [-0.15, -0.15],
                "rotate_rand": False,
                "relative_plane": "Table",
                "place_tag": "straight_front",
                "margin": 0.0,
            },
            "select": {
                "nums": 3,
                "mode": "same",
                "allow_global_same": False,
                "label": ["mahjong2_0", "mahjong2_1", "mahjong2_2"],
            },
        },
        {
            "type": "Rigid",
            "category": "mahjong",
            "common": {
                "xlim": [[0.089, 0.089], [0.135, 0.135], [0.181, 0.181]],
                "ylim": [-0.15, -0.15],
                "rotate_rand": False,
                "relative_plane": "Table",
                "place_tag": "straight_front",
                "margin": 0.0,
            },
            "select": {
                "nums": 3,
                "mode": "same",
                "allow_global_same": False,
                "label": ["mahjong3_0", "mahjong3_1", "mahjong3_2"],
            },
        },
        {
            "type": "Rigid",
            "category": "mahjong",
            "common": {
                "xlim": [[0.227, 0.227], [0.273, 0.273]],
                "ylim": [-0.15, -0.15],
                "rotate_rand": False,
                "relative_plane": "Table",
                "place_tag": "straight_front",
                "margin": 0.0,
            },
            "select": {
                "nums": 2,
                "mode": "same",
                "allow_global_same": False,
                "label": ["mahjong4_0", "mahjong4_1"],
            },
        },
        *[
            {
                "type": "Rigid",
                "category": "mahjong",
                "common": {
                    "xlim": [0.0, 0.0]
                    if index == 5
                    else [0.046, 0.046]
                    if index == 6
                    else [-0.046, -0.046]
                    if index == 7
                    else [-0.092, -0.092],
                    "ylim": [0.05, 0.05],
                    "rotate_rand": False,
                    "relative_plane": "Table",
                    "place_tag": "straight_back",
                    "margin": 0.0,
                },
                "select": {
                    "nums": 1,
                    "mode": "same_index_as_label",
                    "same_label": f"mahjong{index - 5}_0",
                    "label": [f"mahjong{index}_0"],
                },
            }
            for index in range(5, 9)
        ],
        {
            "type": "Rigid",
            "category": "mahjong",
            "common": {
                "xlim": [-0.4, -0.4],
                "ylim": [-0.05, -0.05],
                "rotate_rand": False,
                "relative_plane": "Table",
                "place_tag": "down",
                "margin": 0.0,
            },
            "select": {"nums": 1, "mode": "unique", "label": ["other0"]},
        },
        {
            "type": "Rigid",
            "category": "mahjong",
            "common": {
                "xlim": [-0.4, -0.4],
                "ylim": [-0.002, -0.002],
                "rotate_rand": False,
                "relative_plane": "Table",
                "place_tag": "down",
                "margin": 0.0,
            },
            "select": {"nums": 1, "mode": "unique", "label": ["other1"]},
        },
        {
            "type": "Rigid",
            "category": "mahjong",
            "common": {
                "xlim": [0.0, 0.0],
                "ylim": [0.0, 0.0],
                "rotate_rand": False,
                "place_tag": "down",
                "margin": 0.0,
                "relative_plane": "other1/mahjong_0",
                "check_mode": "point",
            },
            "select": {"nums": 1, "mode": "unique", "label": ["other2"]},
        },
        {
            "type": "Rigid",
            "category": "mahjong",
            "common": {
                "xlim": [0.0, 0.0],
                "ylim": [0.0, 0.0],
                "rotate_rand": False,
                "place_tag": "down",
                "margin": 0.0,
                "relative_plane": "other0/mahjong_0",
                "check_mode": "point",
            },
            "select": {
                "nums": 1,
                "mode": "same_index_as_label",
                "same_label": "mahjong4_0",
                "label": ["mahjong9_0"],
            },
        },
    ]
    if list(config) != ["Rigid"]:
        raise TrainLayoutError("make_kong config group order drifted")
    groups = list(config.get("Rigid") or [])
    if len(groups) != len(expected):
        raise TrainLayoutError("make_kong requires exactly thirteen Rigid groups")
    candidates = list(range(42))
    validated: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for group_number, (group_cfg, spec) in enumerate(
        zip(groups, expected, strict=True)
    ):
        categories = group_cfg.get("category") or []
        if (
            spec["type"] != "Rigid"
            or len(categories) != 1
            or categories[0] != {"name": "mahjong"}
            or (deepcopy(group_cfg.get("common") or {}) != spec["common"])
            or (deepcopy(group_cfg.get("select_mode") or {}) != spec["select"])
        ):
            raise TrainLayoutError(
                f"make_kong:group{group_number} public config semantics drifted"
            )
        if _allowed_indices(assets_root, "Rigid", categories[0]) != candidates:
            raise TrainLayoutError("make_kong mahjong candidate catalog drifted")
        validated.append((group_cfg, spec))
    catalog_inputs: list[dict[str, Any]] = []
    for category_idx in candidates:
        metadata_path, asset_path = _asset_paths(
            assets_root, "Rigid", "mahjong", category_idx
        )
        catalog_inputs.extend(
            [
                _input_record(metadata_path, source_root, "candidate_object_metadata"),
                _input_record(asset_path, source_root, "candidate_object_asset"),
            ]
        )
    last_error: _PlacementExhausted | None = None
    for whole_attempt in range(WHOLE_LAYOUT_MAX_ATTEMPTS):
        sampling_id = _whole_layout_sampling_id(layout_id, whole_attempt)
        layout: dict[str, Any] = {}
        table_occupied: list[Any] = []
        support_occupied: dict[str, list[Any]] = {}
        records_by_label: dict[str, dict[str, Any]] = {}
        selected_by_label: dict[str, int] = {}
        globally_used: set[int] = set()
        inputs: list[dict[str, Any]] = list(catalog_inputs)
        try:
            for group_number, (group_cfg, spec) in enumerate(validated):
                select = spec["select"]
                mode = select["mode"]
                labels = list(select["label"])
                if mode == "same":
                    available = [
                        index for index in candidates if index not in globally_used
                    ]
                    index = _choice(
                        available,
                        generation_seed,
                        task,
                        sampling_id,
                        f"group:{group_number}:asset",
                    )
                    selected_indices = [index] * len(labels)
                    globally_used.add(index)
                elif mode == "unique":
                    available = [
                        index for index in candidates if index not in globally_used
                    ]
                    index = _choice(
                        available,
                        generation_seed,
                        task,
                        sampling_id,
                        f"group:{group_number}:asset",
                    )
                    selected_indices = [index]
                    globally_used.add(index)
                elif mode == "same_index_as_label":
                    same_label = select["same_label"]
                    if same_label not in selected_by_label:
                        raise TrainLayoutError(
                            f"make_kong group{group_number} references unknown {same_label}"
                        )
                    selected_indices = [selected_by_label[same_label]]
                    if selected_indices[0] not in candidates:
                        raise TrainLayoutError(
                            "make_kong linked index is outside catalog"
                        )
                else:
                    raise TrainLayoutError(
                        f"make_kong group{group_number} selection mode drifted"
                    )
                common = deepcopy(group_cfg.get("common") or {})
                for label, category_idx in zip(labels, selected_indices, strict=True):
                    relative_plane = str(common.get("relative_plane", "Table"))
                    if relative_plane == "Table":
                        instance, instance_inputs = _make_table_surface_instance(
                            source_root=source_root,
                            assets_root=assets_root,
                            scene=scene,
                            task=task,
                            generation_seed=generation_seed,
                            layout_id=sampling_id,
                            object_type="Rigid",
                            category="mahjong",
                            category_idx=category_idx,
                            group=None,
                            label=label,
                            common=common,
                            occupied_polygons=table_occupied,
                            prohibited=list(config.get("ProhibitedArea") or []),
                        )
                    else:
                        parent_label = relative_plane.split("/", 1)[0]
                        parent = records_by_label.get(parent_label)
                        if parent is None:
                            raise TrainLayoutError(
                                f"make_kong:{label} support parent {parent_label} was not materialized earlier"
                            )
                        instance, instance_inputs = _make_relative_support_instance(
                            source_root=source_root,
                            assets_root=assets_root,
                            task=task,
                            generation_seed=generation_seed,
                            layout_id=sampling_id,
                            object_type="Rigid",
                            category="mahjong",
                            category_idx=category_idx,
                            group=None,
                            label=label,
                            common=common,
                            parent=parent,
                            occupied_polygons=support_occupied.setdefault(
                                relative_plane, []
                            ),
                        )
                    layout.setdefault("Rigid", {}).setdefault("mahjong", []).append(
                        instance
                    )
                    inputs.extend(instance_inputs)
                    records_by_label[label] = instance
                    selected_by_label[label] = category_idx
        except _PlacementExhausted as exc:
            last_error = exc
            continue
        return (layout, inputs, whole_attempt + 1)
    raise TrainLayoutError(
        f"could not place {task} after {WHOLE_LAYOUT_MAX_ATTEMPTS} deterministic whole-layout attempts; last_error={last_error}"
    )


def _populate_swap_blocks(
    *,
    source_root: Path,
    assets_root: Path,
    task: str,
    config: dict[str, Any],
    scene: dict[str, Any],
    generation_seed: int,
    layout_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Generate the native ordered two-of-three initial occupancy.

    The three mats and button are fixed fixtures.  The two block labels draw an
    ordered pair without replacement from ``mat0..mat2``; all six initial
    occupancy states therefore share one outcome-independent sampling law.
    Each block is materialized on the selected mat's audited passive support
    frame while retaining the bare mat label required by
    ``Func_Parser.find_relative_plane``.
    """
    expected = [
        ("Geometry", "cube_cushion", [1], ["mat0"], "Table"),
        ("Geometry", "cube_cushion", [1], ["mat1"], "Table"),
        ("Geometry", "cube_cushion", [1], ["mat2"], "Table"),
        ("Rigid", "cube", [3], ["target0"], ["mat0", "mat1", "mat2"]),
        ("Rigid", "cube", [3], ["target1"], ["mat0", "mat1", "mat2"]),
        ("Articulation", "SpringButton", [2], ["button0"], "Table"),
    ]
    actual = [
        (object_type, group)
        for object_type in config
        if object_type in OBJECT_TYPES
        for group in config.get(object_type) or []
    ]
    if len(actual) != len(expected):
        raise TrainLayoutError("swap_blocks object-group count/order drifted")
    validated: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
    for group_number, ((object_type, group), spec) in enumerate(
        zip(actual, expected, strict=True)
    ):
        expected_type, category, indices, labels, relative_plane = spec
        categories = group.get("category") or []
        select = group.get("select_mode") or {}
        common = group.get("common") or {}
        if (
            object_type != expected_type
            or len(categories) != 1
            or categories[0].get("name") != category
            or (list(categories[0].get("index") or []) != indices)
            or (select.get("mode") != "allow_duplicate")
            or (int(select.get("nums", 0)) != 1)
            or (list(select.get("label") or []) != labels)
            or (common.get("relative_plane") != relative_plane)
        ):
            raise TrainLayoutError(
                f"swap_blocks group {group_number} selection semantics drifted"
            )
        if _allowed_indices(assets_root, object_type, categories[0]) != indices:
            raise TrainLayoutError(
                f"swap_blocks group {group_number} catalog binding drifted"
            )
        validated.append((object_type, group, categories[0], labels[0]))
    last_error: _PlacementExhausted | None = None
    for whole_attempt in range(WHOLE_LAYOUT_MAX_ATTEMPTS):
        sampling_id = _whole_layout_sampling_id(layout_id, whole_attempt)
        mat_order = sorted(
            ("mat0", "mat1", "mat2"),
            key=lambda label: _unit(
                generation_seed,
                task,
                sampling_id,
                f"ordered_distinct_initial_occupancy:{label}",
            ),
        )
        target_mat = {"target0": mat_order[0], "target1": mat_order[1]}
        layout: dict[str, Any] = {}
        records_by_label: dict[str, dict[str, Any]] = {}
        table_occupied: list[Any] = []
        support_occupied: dict[str, list[Any]] = {}
        inputs: list[dict[str, Any]] = []
        try:
            for object_type, group, category_cfg, label in validated:
                common = deepcopy(group.get("common") or {})
                if object_type == "Rigid":
                    mat_label = target_mat[label]
                    support_path = f"{mat_label}/default/0"
                    common["relative_plane"] = support_path
                    instance, instance_inputs = _make_relative_support_instance(
                        source_root=source_root,
                        assets_root=assets_root,
                        task=task,
                        generation_seed=generation_seed,
                        layout_id=sampling_id,
                        object_type=object_type,
                        category=str(category_cfg["name"]),
                        category_idx=3,
                        group=category_cfg.get("group"),
                        label=label,
                        common=common,
                        parent=records_by_label[mat_label],
                        occupied_polygons=support_occupied.setdefault(support_path, []),
                    )
                    instance["relative_plane"] = mat_label
                else:
                    instance, instance_inputs = _make_table_surface_instance(
                        source_root=source_root,
                        assets_root=assets_root,
                        scene=scene,
                        task=task,
                        generation_seed=generation_seed,
                        layout_id=sampling_id,
                        object_type=object_type,
                        category=str(category_cfg["name"]),
                        category_idx=int((category_cfg.get("index") or [0])[0]),
                        group=category_cfg.get("group"),
                        label=label,
                        common=common,
                        occupied_polygons=table_occupied,
                        prohibited=list(config.get("ProhibitedArea") or []),
                    )
                layout.setdefault(object_type, {}).setdefault(
                    str(category_cfg["name"]), []
                ).append(instance)
                records_by_label[label] = instance
                inputs.extend(instance_inputs)
        except _PlacementExhausted as exc:
            last_error = exc
            continue
        return (layout, inputs, whole_attempt + 1)
    raise TrainLayoutError(
        f"could not place swap_blocks after {WHOLE_LAYOUT_MAX_ATTEMPTS} deterministic whole-layout attempts; last_error={last_error}"
    )


def _populate_later_fixture_task(
    *,
    source_root: Path,
    assets_root: Path,
    task: str,
    config: dict[str, Any],
    scene: dict[str, Any],
    generation_seed: int,
    layout_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Mirror canonical whole-layout rejection for movable-first fixtures.

    ``hang_mugs`` and ``insert_tubes`` intentionally declare their movable
    objects before the destination fixture.  Reordering those groups would
    change the task distribution.  Canonical generation instead rejects the
    *entire* candidate when that later fixture cannot fit, then samples all
    objects again.  This deterministic implementation preserves that order,
    all declared ranges, and selection modes.
    """
    expected = {
        "hang_mugs": {
            "Rigid": (
                "mug",
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                3,
                "allow_duplicate",
                ["mug0", "mug1", "mug2"],
            ),
            "Geometry": ("cup_holder", [0], 1, "allow_duplicate", ["cup_holder"]),
        },
        "insert_tubes": {
            "Rigid": ("test_tube", [2], 3, "same", ["tube0", "tube1", "tube2"]),
            "Geometry": ("test_tube_slot", [0], 1, "allow_duplicate", ["slot"]),
        },
    }
    contract = expected.get(task)
    if contract is None:
        raise TrainLayoutError(f"{task} has no later-fixture generation contract")
    actual_types = [key for key in config if key in OBJECT_TYPES]
    if actual_types != ["Rigid", "Geometry"]:
        raise TrainLayoutError(f"{task} object-group order drifted")
    groups: list[tuple[str, dict[str, Any], dict[str, Any], list[str]]] = []
    for object_type in actual_types:
        configured_groups = config.get(object_type) or []
        if len(configured_groups) != 1:
            raise TrainLayoutError(f"{task}:{object_type} group count drifted")
        group_cfg = configured_groups[0]
        categories = group_cfg.get("category") or []
        if len(categories) != 1:
            raise TrainLayoutError(f"{task}:{object_type} category count drifted")
        category_cfg = categories[0]
        select = group_cfg.get("select_mode") or {}
        category, indices, count, mode, labels = contract[object_type]
        if (
            category_cfg.get("name") != category
            or list(category_cfg.get("index") or []) != indices
            or int(select.get("nums", 0)) != count
            or (select.get("mode") != mode)
            or (list(select.get("label") or []) != labels)
        ):
            raise TrainLayoutError(f"{task}:{object_type} selection semantics drifted")
        if _allowed_indices(assets_root, object_type, category_cfg) != indices:
            raise TrainLayoutError(f"{task}:{object_type} catalog binding drifted")
        groups.append((object_type, group_cfg, category_cfg, labels))
    last_error: _PlacementExhausted | None = None
    for whole_attempt in range(WHOLE_LAYOUT_MAX_ATTEMPTS):
        sampling_id = _whole_layout_sampling_id(layout_id, whole_attempt)
        layout: dict[str, Any] = {}
        occupied: list[_Placed] = []
        inputs: list[dict[str, Any]] = []
        try:
            for group_number, (
                object_type,
                group_cfg,
                category_cfg,
                labels,
            ) in enumerate(groups):
                select = group_cfg["select_mode"]
                candidates = _allowed_indices(assets_root, object_type, category_cfg)
                if select["mode"] == "same":
                    selected_indices = [
                        _choice(
                            candidates,
                            generation_seed,
                            task,
                            sampling_id,
                            f"{object_type}:{group_number}:asset",
                        )
                    ] * len(labels)
                else:
                    selected_indices = [
                        _choice(
                            candidates,
                            generation_seed,
                            task,
                            sampling_id,
                            f"{object_type}:{group_number}:asset:{instance_index}",
                        )
                        for instance_index in range(len(labels))
                    ]
                for label, category_idx in zip(labels, selected_indices, strict=True):
                    instance, records = _make_instance(
                        source_root=source_root,
                        assets_root=assets_root,
                        scene=scene,
                        task=task,
                        generation_seed=generation_seed,
                        layout_id=sampling_id,
                        object_type=object_type,
                        category=str(category_cfg["name"]),
                        category_idx=category_idx,
                        group=category_cfg.get("group"),
                        label=label,
                        common=deepcopy(group_cfg.get("common") or {}),
                        occupied=occupied,
                        prohibited=list(config.get("ProhibitedArea") or []),
                    )
                    layout.setdefault(object_type, {}).setdefault(
                        str(category_cfg["name"]), []
                    ).append(instance)
                    inputs.extend(records)
        except _PlacementExhausted as exc:
            last_error = exc
            continue
        return (layout, inputs, whole_attempt + 1)
    raise TrainLayoutError(
        f"could not place {task} after {WHOLE_LAYOUT_MAX_ATTEMPTS} deterministic whole-layout attempts; last_error={last_error}"
    )


def _populate_hang_mugs_random(
    *,
    source_root: Path,
    assets_root: Path,
    config: dict[str, Any],
    scene: dict[str, Any],
    generation_seed: int,
    layout_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Generate the official random-half hang-mugs Train contract.

    The random variant is deliberately its own skill_choice: it binds the five
    random-half mug assets, rack index 1, and the declared twenty-object
    category-unique clutter group before sampling a complete layout.  A
    candidate is rejected as a whole when the rack or any clutter object
    cannot be placed, so the result cannot alias a successful base layout.
    """
    task = "hang_mugs_random"
    expected_rigid_common = {
        "xlim": [-0.45, 0.45],
        "ylim": [-0.25, 0.02],
        "rotate_rand": True,
        "rotate_deg": [15, 165],
        "relative_plane": "Table",
        "margin": 0.05,
    }
    expected_rigid_select = {
        "nums": 3,
        "mode": "allow_duplicate",
        "label": ["mug0", "mug1", "mug2"],
    }
    expected_geometry_common = {
        "xlim": [-0.2, 0.2],
        "ylim": [0.0, 0.05],
        "rotate_rand": True,
        "rotate_deg": 25,
        "relative_plane": "Table",
        "margin": 0.03,
    }
    expected_geometry_select = {
        "nums": 1,
        "mode": "allow_duplicate",
        "label": ["cup_holder"],
    }
    expected_clutter = {
        "xlim": [[-0.6, 0.6]],
        "ylim": [[-0.5, 0.5]],
        "relative_plane": "Table",
        "yaml_path": "Clutter/clutter.yml",
        "nums": 20,
        "mode": "category_unique",
        "rotate_rand": True,
        "rotate_deg": 30,
        "margin": 0.015,
    }
    expected_prohibited = [[-0.35, -0.5, 0.35, -0.4]]
    expected_rigid_category = {"name": "mug", "index": [10, 11, 12, 13, 14]}
    expected_geometry_category = {"name": "cup_holder", "index": [1]}
    if list(config) != ["Rigid", "Geometry", "Clutter", "ProhibitedArea"]:
        raise TrainLayoutError(f"{task} object-group order drifted")
    rigid_groups = config.get("Rigid") or []
    geometry_groups = config.get("Geometry") or []
    clutter_groups = config.get("Clutter") or []
    if len(rigid_groups) != 1 or len(geometry_groups) != 1 or len(clutter_groups) != 1:
        raise TrainLayoutError(f"{task} object-group cardinality drifted")
    rigid_group = rigid_groups[0]
    geometry_group = geometry_groups[0]
    rigid_categories = rigid_group.get("category") or []
    geometry_categories = geometry_group.get("category") or []
    if (
        deepcopy(rigid_group.get("common") or {}) != expected_rigid_common
        or deepcopy(rigid_group.get("select_mode") or {}) != expected_rigid_select
        or rigid_categories != [expected_rigid_category]
        or (deepcopy(geometry_group.get("common") or {}) != expected_geometry_common)
        or (
            deepcopy(geometry_group.get("select_mode") or {})
            != expected_geometry_select
        )
        or (geometry_categories != [expected_geometry_category])
        or (deepcopy(clutter_groups[0]) != expected_clutter)
        or (deepcopy(config.get("ProhibitedArea") or []) != expected_prohibited)
    ):
        raise TrainLayoutError(f"{task} random-half config semantics drifted")
    rigid_category = rigid_categories[0]
    geometry_category = geometry_categories[0]
    random_mug_indices = [10, 11, 12, 13, 14]
    if _allowed_indices(assets_root, "Rigid", rigid_category) != random_mug_indices:
        raise TrainLayoutError(f"{task} random mug catalog binding drifted")
    if _allowed_indices(assets_root, "Geometry", geometry_category) != [1]:
        raise TrainLayoutError(f"{task} random rack catalog binding drifted")
    catalog_inputs: list[dict[str, Any]] = []
    for object_type, category, indices in (
        ("Rigid", "mug", random_mug_indices),
        ("Geometry", "cup_holder", [1]),
    ):
        for category_idx in indices:
            metadata_path, asset_path = _asset_paths(
                assets_root, object_type, category, category_idx
            )
            catalog_inputs.extend(
                [
                    _input_record(
                        metadata_path, source_root, "candidate_object_metadata"
                    ),
                    _input_record(asset_path, source_root, "candidate_object_asset"),
                ]
            )
    last_error: _PlacementExhausted | None = None
    for whole_attempt in range(WHOLE_LAYOUT_MAX_ATTEMPTS):
        sampling_id = _whole_layout_sampling_id(layout_id, whole_attempt)
        layout: dict[str, Any] = {}
        occupied: list[_Placed] = []
        inputs: list[dict[str, Any]] = list(catalog_inputs)
        try:
            selected_indices = [
                _choice(
                    random_mug_indices,
                    generation_seed,
                    task,
                    sampling_id,
                    f"Rigid:0:asset:{instance_index}",
                )
                for instance_index in range(3)
            ]
            for label, category_idx in zip(
                expected_rigid_select["label"], selected_indices, strict=True
            ):
                mug, mug_inputs = _make_instance(
                    source_root=source_root,
                    assets_root=assets_root,
                    scene=scene,
                    task=task,
                    generation_seed=generation_seed,
                    layout_id=sampling_id,
                    object_type="Rigid",
                    category="mug",
                    category_idx=category_idx,
                    group=None,
                    label=label,
                    common=deepcopy(expected_rigid_common),
                    occupied=occupied,
                    prohibited=expected_prohibited,
                )
                layout.setdefault("Rigid", {}).setdefault("mug", []).append(mug)
                inputs.extend(mug_inputs)
            rack, rack_inputs = _make_instance(
                source_root=source_root,
                assets_root=assets_root,
                scene=scene,
                task=task,
                generation_seed=generation_seed,
                layout_id=sampling_id,
                object_type="Geometry",
                category="cup_holder",
                category_idx=1,
                group=None,
                label="cup_holder",
                common=deepcopy(expected_geometry_common),
                occupied=occupied,
                prohibited=expected_prohibited,
            )
            layout.setdefault("Geometry", {}).setdefault("cup_holder", []).append(rack)
            inputs.extend(rack_inputs)
            placed_clutter, clutter_inputs = _populate_declared_clutter(
                source_root=source_root,
                assets_root=assets_root,
                task=task,
                config=config,
                scene=scene,
                generation_seed=generation_seed,
                layout_id=sampling_id,
                layout=layout,
                occupied=occupied,
                prohibited=expected_prohibited,
            )
            if placed_clutter != 20:
                raise _PlacementExhausted(
                    f"{task} requires exactly twenty placed clutter objects"
                )
            inputs.extend(clutter_inputs)
        except _PlacementExhausted as exc:
            last_error = exc
            continue
        return (layout, inputs, whole_attempt + 1)
    raise TrainLayoutError(
        f"could not place {task} after {WHOLE_LAYOUT_MAX_ATTEMPTS} deterministic whole-layout attempts; last_error={last_error}"
    )


def _populate_push_t_random(
    *,
    source_root: Path,
    assets_root: Path,
    config: dict[str, Any],
    scene: dict[str, Any],
    generation_seed: int,
    layout_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Generate the official random-half Push T Train contract.

    Push T random is intentionally independent from the base skill_choice.  It binds
    the random-half rigid T (asset 1), the target cushion, and both declared
    clutter bands before sampling a complete layout.  A band placement
    shortfall rejects the complete candidate and resamples the whole layout;
    the generic object-group path is therefore never allowed to silently drop
    the Clutter section.
    """
    task = "push_T_random"
    expected_common = {
        "xlim": [-0.45, 0.45],
        "ylim": [-0.25, -0.1],
        "rotate_deg": 360,
        "rotate_rand": True,
        "relative_plane": "Table",
    }
    expected_select = {"nums": 1, "mode": "allow_duplicate"}
    expected_rigid_category = {"name": "t", "index": [1]}
    expected_geometry_category = {"name": "t_cushion"}
    expected_clutter = [
        {
            "xlim": [-0.6, 0.6],
            "ylim": [[-0.5, -0.25], [0.0, 0.5]],
            "relative_plane": "Table",
            "yaml_path": "Clutter/clutter.yml",
            "nums": 10,
            "mode": "category_unique",
            "rotate_rand": True,
            "rotate_deg": 30,
            "margin": 0.015,
        },
        {
            "xlim": [[-0.6, -0.45], [0.45, 0.6]],
            "ylim": [-0.25, 0.0],
            "relative_plane": "Table",
            "yaml_path": "Clutter/clutter.yml",
            "nums": 10,
            "mode": "category_unique",
            "rotate_rand": True,
            "rotate_deg": 30,
            "margin": 0.015,
        },
    ]
    expected_prohibited = [[-0.35, -0.5, 0.35, -0.4]]
    if list(config) != ["Rigid", "Geometry", "Clutter", "ProhibitedArea"]:
        raise TrainLayoutError(f"{task} object-group order drifted")
    rigid_groups = config.get("Rigid") or []
    geometry_groups = config.get("Geometry") or []
    clutter_groups = config.get("Clutter") or []
    if len(rigid_groups) != 1 or len(geometry_groups) != 1 or len(clutter_groups) != 2:
        raise TrainLayoutError(f"{task} object-group cardinality drifted")
    rigid_group = rigid_groups[0]
    geometry_group = geometry_groups[0]
    rigid_categories = rigid_group.get("category") or []
    geometry_categories = geometry_group.get("category") or []
    if (
        deepcopy(rigid_group.get("common") or {}) != expected_common
        or deepcopy(rigid_group.get("select_mode") or {})
        != {**expected_select, "label": ["t"]}
        or rigid_categories != [expected_rigid_category]
        or (deepcopy(geometry_group.get("common") or {}) != expected_common)
        or (
            deepcopy(geometry_group.get("select_mode") or {})
            != {**expected_select, "label": ["target_t"]}
        )
        or (geometry_categories != [expected_geometry_category])
        or ([deepcopy(group) for group in clutter_groups] != expected_clutter)
        or (deepcopy(config.get("ProhibitedArea") or []) != expected_prohibited)
    ):
        raise TrainLayoutError(f"{task} random-half config semantics drifted")
    if _allowed_indices(assets_root, "Rigid", rigid_categories[0]) != [1]:
        raise TrainLayoutError(f"{task} random T catalog binding drifted")
    if _allowed_indices(assets_root, "Geometry", geometry_categories[0]) != [0]:
        raise TrainLayoutError(f"{task} T cushion catalog binding drifted")
    catalog_inputs: list[dict[str, Any]] = []
    for object_type, category, category_idx in (
        ("Rigid", "t", 1),
        ("Geometry", "t_cushion", 0),
    ):
        metadata_path, asset_path = _asset_paths(
            assets_root, object_type, category, category_idx
        )
        catalog_inputs.extend(
            [
                _input_record(metadata_path, source_root, "candidate_object_metadata"),
                _input_record(asset_path, source_root, "candidate_object_asset"),
            ]
        )
    last_error: _PlacementExhausted | None = None
    for whole_attempt in range(WHOLE_LAYOUT_MAX_ATTEMPTS):
        sampling_id = _whole_layout_sampling_id(layout_id, whole_attempt)
        layout: dict[str, Any] = {}
        occupied: list[_Placed] = []
        inputs: list[dict[str, Any]] = list(catalog_inputs)
        try:
            target, target_inputs = _make_instance(
                source_root=source_root,
                assets_root=assets_root,
                scene=scene,
                task=task,
                generation_seed=generation_seed,
                layout_id=sampling_id,
                object_type="Rigid",
                category="t",
                category_idx=1,
                group=None,
                label="t",
                common=deepcopy(expected_common),
                occupied=occupied,
                prohibited=expected_prohibited,
            )
            layout.setdefault("Rigid", {}).setdefault("t", []).append(target)
            inputs.extend(target_inputs)
            cushion, cushion_inputs = _make_instance(
                source_root=source_root,
                assets_root=assets_root,
                scene=scene,
                task=task,
                generation_seed=generation_seed,
                layout_id=sampling_id,
                object_type="Geometry",
                category="t_cushion",
                category_idx=0,
                group=None,
                label="target_t",
                common=deepcopy(expected_common),
                occupied=occupied,
                prohibited=expected_prohibited,
            )
            layout.setdefault("Geometry", {}).setdefault("t_cushion", []).append(
                cushion
            )
            inputs.extend(cushion_inputs)
            placed_clutter, clutter_inputs = _populate_declared_clutter(
                source_root=source_root,
                assets_root=assets_root,
                task=task,
                config=config,
                scene=scene,
                generation_seed=generation_seed,
                layout_id=sampling_id,
                layout=layout,
                occupied=occupied,
                prohibited=expected_prohibited,
            )
            if placed_clutter != 20:
                raise _PlacementExhausted(
                    f"{task} requires exactly twenty placed clutter objects"
                )
            inputs.extend(clutter_inputs)
        except _PlacementExhausted as exc:
            last_error = exc
            continue
        return (layout, inputs, whole_attempt + 1)
    raise TrainLayoutError(
        f"could not place {task} after {WHOLE_LAYOUT_MAX_ATTEMPTS} deterministic whole-layout attempts; last_error={last_error}"
    )


def _populate_config_ordered_table_task(
    *,
    source_root: Path,
    assets_root: Path,
    task: str,
    config: dict[str, Any],
    scene: dict[str, Any],
    generation_seed: int,
    layout_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Generate table-only groups with canonical cross-group selection rules."""
    expected: dict[str, list[dict[str, Any]]] = {
        "fill_pen_holder": [
            {
                "type": "Rigid",
                "category": "pen",
                "indices": [0, 1, 4],
                "count": 2,
                "mode": "allow_duplicate",
                "labels": ["target0", "target1"],
            },
            {
                "type": "Rigid",
                "category": "oil_pen",
                "indices": [0, 1],
                "count": 2,
                "mode": "allow_duplicate",
                "labels": ["target2", "target3"],
            },
            {
                "type": "Rigid",
                "category": "pen_holder",
                "indices": [0],
                "count": 1,
                "mode": "unique",
                "labels": ["pen_holder"],
            },
        ],
        "fasten_screws": [
            {
                "type": "Rigid",
                "category": "factory_nut",
                "indices": [0, 1, 2, 3, 4],
                "count": 3,
                "mode": "unique",
                "labels": ["nut0", "nut1", "nut2"],
            },
            *[
                {
                    "type": "Geometry",
                    "category": "factory_bolt",
                    "indices": [0, 1, 2, 3, 4],
                    "count": 1,
                    "mode": "same_index_as_label",
                    "same_label": f"nut{index}",
                    "labels": [f"bolt{index}"],
                }
                for index in range(3)
            ],
        ],
        "cover_blocks": [
            {
                "type": "Rigid",
                "category": "cube",
                "indices": [4],
                "count": 1,
                "mode": "unique",
                "labels": ["red"],
            },
            {
                "type": "Rigid",
                "category": "cube",
                "indices": [6],
                "count": 1,
                "mode": "unique",
                "labels": ["green"],
            },
            {
                "type": "Rigid",
                "category": "cube",
                "indices": [5],
                "count": 1,
                "mode": "unique",
                "labels": ["blue"],
            },
            *[
                {
                    "type": "Rigid",
                    "category": "cup",
                    "indices": [1],
                    "count": 1,
                    "mode": "allow_duplicate",
                    "labels": [f"cup{index}"],
                }
                for index in range(3)
            ],
        ],
    }
    specs = expected.get(task)
    if specs is None:
        raise TrainLayoutError(f"{task} has no ordered-table generation contract")
    actual_groups: list[tuple[str, dict[str, Any]]] = [
        (object_type, group)
        for object_type in config
        if object_type in OBJECT_TYPES
        for group in config.get(object_type) or []
    ]
    if len(actual_groups) != len(specs):
        raise TrainLayoutError(f"{task} object-group count/order drifted")
    validated: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for group_number, ((object_type, group_cfg), spec) in enumerate(
        zip(actual_groups, specs, strict=True)
    ):
        categories = group_cfg.get("category") or []
        select = group_cfg.get("select_mode") or {}
        if len(categories) != 1:
            raise TrainLayoutError(f"{task}:group{group_number} category count drifted")
        category_cfg = categories[0]
        raw_count = select.get("nums")
        normalized_count = (
            int(raw_count[0])
            if isinstance(raw_count, list) and len(raw_count) == 1
            else int(raw_count)
            if isinstance(raw_count, int) and (not isinstance(raw_count, bool))
            else None
        )
        if (
            object_type != spec["type"]
            or category_cfg.get("name") != spec["category"]
            or normalized_count != spec["count"]
            or (select.get("mode") != spec["mode"])
            or (list(select.get("label") or []) != spec["labels"])
            or (select.get("same_label") != spec.get("same_label"))
            or (str((group_cfg.get("common") or {}).get("relative_plane")) != "Table")
        ):
            raise TrainLayoutError(
                f"{task}:group{group_number} selection/order semantics drifted"
            )
        indices = _allowed_indices(assets_root, object_type, category_cfg)
        if indices != spec["indices"]:
            raise TrainLayoutError(
                f"{task}:group{group_number} catalog binding drifted"
            )
        validated.append((object_type, group_cfg, category_cfg, spec))
    last_error: _PlacementExhausted | None = None
    for whole_attempt in range(WHOLE_LAYOUT_MAX_ATTEMPTS):
        sampling_id = _whole_layout_sampling_id(layout_id, whole_attempt)
        layout: dict[str, Any] = {}
        occupied: list[_Placed] = []
        inputs: list[dict[str, Any]] = []
        selected_by_label: dict[str, int] = {}
        try:
            for group_number, (object_type, group_cfg, category_cfg, spec) in enumerate(
                validated
            ):
                candidates = spec["indices"]
                if spec["mode"] == "unique":
                    selected_indices = sorted(
                        candidates,
                        key=lambda value: _unit(
                            generation_seed,
                            task,
                            sampling_id,
                            f"{object_type}:{group_number}:asset:{value}",
                        ),
                    )[: spec["count"]]
                elif spec["mode"] == "allow_duplicate":
                    selected_indices = [
                        _choice(
                            candidates,
                            generation_seed,
                            task,
                            sampling_id,
                            f"{object_type}:{group_number}:asset:{index}",
                        )
                        for index in range(spec["count"])
                    ]
                elif spec["mode"] == "same_index_as_label":
                    same_label = spec["same_label"]
                    if same_label not in selected_by_label:
                        raise TrainLayoutError(
                            f"{task}:group{group_number} references unavailable label {same_label}"
                        )
                    selected_indices = [selected_by_label[same_label]]
                    if selected_indices[0] not in candidates:
                        raise TrainLayoutError(
                            f"{task}:group{group_number} linked asset is missing"
                        )
                else:
                    raise TrainLayoutError(
                        f"{task}:group{group_number} selection mode drifted"
                    )
                if len(selected_indices) != spec["count"]:
                    raise TrainLayoutError(
                        f"{task}:group{group_number} selection cardinality failed"
                    )
                category_placement = {
                    key: deepcopy(category_cfg[key])
                    for key in (
                        "xlim",
                        "ylim",
                        "zlim",
                        "qpos",
                        "rotate_deg",
                        "rotate_rand",
                        "place_tag",
                        "margin",
                        "check_mode",
                        "need_check_stable",
                    )
                    if key in category_cfg
                }
                common = {
                    **deepcopy(group_cfg.get("common") or {}),
                    **category_placement,
                }
                for label, category_idx in zip(
                    spec["labels"], selected_indices, strict=True
                ):
                    instance, records = _make_instance(
                        source_root=source_root,
                        assets_root=assets_root,
                        scene=scene,
                        task=task,
                        generation_seed=generation_seed,
                        layout_id=sampling_id,
                        object_type=object_type,
                        category=spec["category"],
                        category_idx=category_idx,
                        group=category_cfg.get("group"),
                        label=label,
                        common=common,
                        occupied=occupied,
                        prohibited=list(config.get("ProhibitedArea") or []),
                    )
                    layout.setdefault(object_type, {}).setdefault(
                        spec["category"], []
                    ).append(instance)
                    inputs.extend(records)
                    selected_by_label[label] = category_idx
        except _PlacementExhausted as exc:
            last_error = exc
            continue
        return (layout, inputs, whole_attempt + 1)
    raise TrainLayoutError(
        f"could not place {task} after {WHOLE_LAYOUT_MAX_ATTEMPTS} deterministic whole-layout attempts; last_error={last_error}"
    )


def _exact_ordered_table_contract(task: str) -> list[dict[str, Any]]:
    """Frozen canonical config contracts for exact-polygon table tasks."""
    organize = [
        (
            "Geometry",
            "drawer",
            [0],
            {
                "xlim": [-0.45, -0.25],
                "ylim": [0.15, 0.15],
                "rotate_rand": False,
                "relative_plane": "Table",
            },
            "allow_duplicate",
            ["drawer"],
        ),
        (
            "Geometry",
            "cube_cushion",
            [2],
            {
                "xlim": [-0.15, -0.15],
                "ylim": [0.1, 0.1],
                "rotate_rand": False,
                "relative_plane": "Table",
                "margin": 0.0,
            },
            "allow_duplicate",
            ["cube_cushion"],
        ),
        (
            "Geometry",
            "monitor",
            [0, 1, 2],
            {
                "xlim": [-0.05, 0.15],
                "ylim": [0.2, 0.25],
                "rotate_rand": False,
                "relative_plane": "Table",
            },
            "allow_duplicate",
            ["monitor"],
        ),
        (
            "Geometry",
            "mousemat",
            [0],
            {
                "xlim": [0.35, 0.5],
                "ylim": [-0.1, 0.05],
                "rotate_rand": False,
                "relative_plane": "Table",
            },
            "allow_duplicate",
            ["mousemat"],
        ),
        (
            "Geometry",
            "frame",
            [0],
            {
                "xlim": [0.1, 0.1],
                "ylim": [0.0, 0.0],
                "relative_plane": "Table",
                "rotate_rand": False,
                "check_mode": "enforce",
                "margin": 0,
            },
            "allow_duplicate",
            ["frame"],
        ),
        (
            "Rigid",
            "mouse",
            [4, 6, 7],
            {
                "xlim": [0.0, 0.45],
                "ylim": [-0.3, 0.05],
                "rotate_rand": True,
                "rotate_deg": 45,
                "relative_plane": "Table",
            },
            "allow_duplicate",
            ["mouse"],
        ),
        (
            "Rigid",
            "alarm",
            [0],
            {
                "xlim": [-0.45, 0.0],
                "ylim": [-0.3, 0.05],
                "rotate_rand": False,
                "relative_plane": "Table",
            },
            "allow_duplicate",
            ["alarm"],
        ),
        (
            "Rigid",
            "keyboard",
            [0, 1, 2],
            {
                "xlim": [-0.05, 0.05],
                "ylim": [-0.2, -0.15],
                "rotate_rand": False,
                "relative_plane": "Table",
            },
            "allow_duplicate",
            ["keyboard"],
        ),
        (
            "Rigid",
            "garage",
            [5, 6, 7, 8, 11],
            {
                "xlim": [-0.45, 0.0],
                "ylim": [-0.3, 0.05],
                "rotate_rand": True,
                "rotate_deg": 45,
                "relative_plane": "Table",
                "place_tag": "up",
            },
            "allow_duplicate",
            ["garage"],
        ),
    ]
    pack = [
        (
            "Rigid",
            "car",
            [0, 6, 7],
            {
                "xlim": [-0.45, 0.45],
                "ylim": [-0.25, 0.0],
                "rotate_deg": 360,
                "rotate_rand": True,
                "relative_plane": "Table",
            },
            "unique",
            ["car"],
        ),
        (
            "Rigid",
            "electric_toothbrush",
            [0, 1, 2, 3],
            {
                "xlim": [-0.45, 0.45],
                "ylim": [-0.25, 0.0],
                "rotate_deg": 360,
                "rotate_rand": True,
                "relative_plane": "Table",
            },
            "unique",
            ["electric_toothbrush"],
        ),
        (
            "Rigid",
            "hammer",
            [0, 1, 2, 3, 4],
            {
                "xlim": [-0.45, 0.45],
                "ylim": [-0.25, 0.0],
                "rotate_deg": 360,
                "rotate_rand": True,
                "relative_plane": "Table",
            },
            "unique",
            ["hammer"],
        ),
        (
            "Rigid",
            "shoe",
            [0, 1, 2, 3, 4, 5, 6, 7],
            {
                "xlim": [-0.45, 0.45],
                "ylim": [-0.25, 0.0],
                "rotate_deg": 360,
                "rotate_rand": True,
                "relative_plane": "Table",
            },
            "unique",
            ["shoe"],
        ),
        (
            "Rigid",
            "box",
            [0],
            {
                "xlim": [-0.3, 0.3],
                "ylim": [-0.1, 0.0],
                "rotate_deg": 25,
                "rotate_rand": True,
                "relative_plane": "Table",
            },
            "unique",
            ["box"],
        ),
    ]
    pack_random = [
        (
            "Rigid",
            "car",
            [1, 2, 3, 4, 5],
            {
                "xlim": [-0.45, 0.45],
                "ylim": [-0.25, 0.0],
                "rotate_deg": 360,
                "rotate_rand": True,
                "relative_plane": "Table",
            },
            "unique",
            ["car"],
        ),
        (
            "Rigid",
            "electric_toothbrush",
            [4, 5, 6],
            {
                "xlim": [-0.45, 0.45],
                "ylim": [-0.25, 0.0],
                "rotate_deg": 360,
                "rotate_rand": True,
                "relative_plane": "Table",
            },
            "unique",
            ["electric_toothbrush"],
        ),
        (
            "Rigid",
            "hammer",
            [6, 7, 8],
            {
                "xlim": [-0.45, 0.45],
                "ylim": [-0.25, 0.0],
                "rotate_deg": 360,
                "rotate_rand": True,
                "relative_plane": "Table",
            },
            "unique",
            ["hammer"],
        ),
        (
            "Rigid",
            "shoe",
            list(range(8, 18)),
            {
                "xlim": [-0.45, 0.45],
                "ylim": [-0.25, 0.0],
                "rotate_deg": 360,
                "rotate_rand": True,
                "relative_plane": "Table",
            },
            "unique",
            ["shoe"],
        ),
        (
            "Rigid",
            "box",
            [0],
            {
                "xlim": [-0.3, 0.3],
                "ylim": [-0.1, 0.0],
                "rotate_deg": 25,
                "rotate_rand": True,
                "relative_plane": "Table",
            },
            "unique",
            ["box"],
        ),
    ]
    rows = {
        "organize_table": organize,
        "pack_objects_into_box": pack,
        "pack_objects_into_box_random": pack_random,
    }.get(task)
    if rows is None:
        raise TrainLayoutError(f"{task} has no exact ordered-table contract")
    result = [
        {
            "type": object_type,
            "category": category,
            "indices": indices,
            "common": common,
            "select": {"nums": 1, "mode": mode, "label": labels},
        }
        for object_type, category, indices, common, mode, labels in rows
    ]
    if task == "organize_table":
        result[4]["select"]["nums"] = [1]
    return result


def _populate_exact_ordered_table_task(
    *,
    source_root: Path,
    assets_root: Path,
    task: str,
    config: dict[str, Any],
    scene: dict[str, Any],
    generation_seed: int,
    layout_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Generate schema-frozen, config-ordered tasks with exact polygons."""
    specs = _exact_ordered_table_contract(task)
    if task == "pack_objects_into_box_random":
        if list(config) != ["Rigid"]:
            raise TrainLayoutError(
                "pack_objects_into_box_random config group order drifted"
            )
        if (
            config.get("Clutter") is not None
            or config.get("ProhibitedArea") is not None
        ):
            raise TrainLayoutError(
                "pack_objects_into_box_random must not inherit clutter or prohibited areas"
            )
    actual_groups = [
        (object_type, group)
        for object_type in config
        if object_type in OBJECT_TYPES
        for group in config.get(object_type) or []
    ]
    if len(actual_groups) != len(specs):
        raise TrainLayoutError(f"{task} object-group count/order drifted")
    validated = []
    catalog_inputs: list[dict[str, Any]] = []
    for group_number, ((object_type, group_cfg), spec) in enumerate(
        zip(actual_groups, specs, strict=True)
    ):
        categories = group_cfg.get("category") or []
        if (
            object_type != spec["type"]
            or len(categories) != 1
            or categories[0].get("name") != spec["category"]
            or (deepcopy(group_cfg.get("common") or {}) != spec["common"])
            or (deepcopy(group_cfg.get("select_mode") or {}) != spec["select"])
        ):
            raise TrainLayoutError(
                f"{task}:group{group_number} exact config semantics drifted"
            )
        category_cfg = categories[0]
        indices = _allowed_indices(assets_root, object_type, category_cfg)
        if indices != spec["indices"]:
            raise TrainLayoutError(
                f"{task}:group{group_number} catalog binding drifted"
            )
        for category_idx in indices:
            metadata_path, asset_path = _asset_paths(
                assets_root, object_type, spec["category"], category_idx
            )
            catalog_inputs.extend(
                [
                    _input_record(
                        metadata_path, source_root, "candidate_object_metadata"
                    ),
                    _input_record(asset_path, source_root, "candidate_object_asset"),
                ]
            )
        validated.append((object_type, group_cfg, category_cfg, spec))
    last_error: _PlacementExhausted | None = None
    for whole_attempt in range(WHOLE_LAYOUT_MAX_ATTEMPTS):
        sampling_id = _whole_layout_sampling_id(layout_id, whole_attempt)
        layout: dict[str, Any] = {}
        occupied_polygons: list[Any] = []
        inputs: list[dict[str, Any]] = list(catalog_inputs)
        try:
            for group_number, (object_type, group_cfg, category_cfg, spec) in enumerate(
                validated
            ):
                candidates = spec["indices"]
                mode = spec["select"]["mode"]
                if mode == "unique":
                    selected_indices = sorted(
                        candidates,
                        key=lambda value: _unit(
                            generation_seed,
                            task,
                            sampling_id,
                            f"{object_type}:{group_number}:asset:{value}",
                        ),
                    )[:1]
                elif mode == "allow_duplicate":
                    selected_indices = [
                        _choice(
                            candidates,
                            generation_seed,
                            task,
                            sampling_id,
                            f"{object_type}:{group_number}:asset:0",
                        )
                    ]
                else:
                    raise TrainLayoutError(
                        f"{task}:group{group_number} selection mode drifted"
                    )
                label = spec["select"]["label"][0]
                instance, instance_inputs = _make_table_surface_instance(
                    source_root=source_root,
                    assets_root=assets_root,
                    scene=scene,
                    task=task,
                    generation_seed=generation_seed,
                    layout_id=sampling_id,
                    object_type=object_type,
                    category=spec["category"],
                    category_idx=selected_indices[0],
                    group=category_cfg.get("group"),
                    label=label,
                    common=deepcopy(group_cfg.get("common") or {}),
                    occupied_polygons=occupied_polygons,
                    prohibited=list(config.get("ProhibitedArea") or []),
                    allow_geometry_origin=task == "pack_objects_into_box_random"
                    and spec["category"] == "hammer"
                    and (selected_indices[0] == 7),
                )
                layout.setdefault(object_type, {}).setdefault(
                    spec["category"], []
                ).append(instance)
                inputs.extend(instance_inputs)
        except _PlacementExhausted as exc:
            last_error = exc
            continue
        return (layout, inputs, whole_attempt + 1)
    raise TrainLayoutError(
        f"could not place {task} after {WHOLE_LAYOUT_MAX_ATTEMPTS} deterministic whole-layout attempts; last_error={last_error}"
    )


def _populate_sort_nesting_dolls(
    *,
    source_root: Path,
    assets_root: Path,
    task: str,
    config: dict[str, Any],
    scene: dict[str, Any],
    generation_seed: int,
    layout_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Generate one complete canonical texture family of five nesting dolls."""
    expected_common = {
        "xlim": [-0.45, 0.45],
        "ylim": [-0.25, -0.05],
        "rotate_rand": True,
        "rotate_deg": 45,
        "relative_plane": "Table",
        "margin": 0.02,
    }
    category_starts = (
        (15, 20) if task == "sort_nesting_dolls_by_size_random" else (0, 5, 10)
    )
    expected_categories = [
        {
            "name": "matryoshka_dolls",
            "index": list(range(start, start + 5)),
            "group": f"texture{group}",
        }
        for group, start in enumerate(category_starts)
    ]
    expected_select = {
        "mode": "hierarchical",
        "select_category_nums": [1, 1],
        "select_instance_nums": [5, 5],
        "instance_sample_mode": "unique",
        "label": ["doll0", "doll1", "doll2", "doll3", "doll4"],
    }
    object_groups = [
        (object_type, group)
        for object_type in config
        if object_type in OBJECT_TYPES
        for group in config.get(object_type) or []
    ]
    if len(object_groups) != 1:
        raise TrainLayoutError(f"{task} object-group count/order drifted")
    object_type, group_cfg = object_groups[0]
    categories = group_cfg.get("category") or []
    if (
        object_type != "Rigid"
        or deepcopy(group_cfg.get("common") or {}) != expected_common
        or categories != expected_categories
        or (deepcopy(group_cfg.get("select_mode") or {}) != expected_select)
    ):
        raise TrainLayoutError(f"{task} hierarchical semantics drifted")
    for category_cfg in categories:
        if (
            _allowed_indices(assets_root, object_type, category_cfg)
            != category_cfg["index"]
        ):
            raise TrainLayoutError(f"{task} texture catalog binding drifted")
    catalog_inputs: list[dict[str, Any]] = []
    for category_idx in (
        category_idx
        for category_cfg in expected_categories
        for category_idx in category_cfg["index"]
    ):
        metadata_path, asset_path = _asset_paths(
            assets_root, "Rigid", "matryoshka_dolls", category_idx
        )
        catalog_inputs.extend(
            [
                _input_record(metadata_path, source_root, "candidate_object_metadata"),
                _input_record(asset_path, source_root, "candidate_object_asset"),
            ]
        )
    last_error: _PlacementExhausted | None = None
    for whole_attempt in range(WHOLE_LAYOUT_MAX_ATTEMPTS):
        sampling_id = _whole_layout_sampling_id(layout_id, whole_attempt)
        category_cfg = _choice(
            categories, generation_seed, task, sampling_id, "hierarchical_category"
        )
        indices = sorted(
            category_cfg["index"],
            key=lambda value: _unit(
                generation_seed, task, sampling_id, f"hierarchical_instance:{value}"
            ),
        )
        layout: dict[str, Any] = {}
        occupied_polygons: list[Any] = []
        inputs: list[dict[str, Any]] = list(catalog_inputs)
        try:
            for label, category_idx in zip(
                expected_select["label"], indices, strict=True
            ):
                instance, instance_inputs = _make_table_surface_instance(
                    source_root=source_root,
                    assets_root=assets_root,
                    scene=scene,
                    task=task,
                    generation_seed=generation_seed,
                    layout_id=sampling_id,
                    object_type="Rigid",
                    category="matryoshka_dolls",
                    category_idx=category_idx,
                    group=category_cfg["group"],
                    label=label,
                    common=expected_common,
                    occupied_polygons=occupied_polygons,
                    prohibited=list(config.get("ProhibitedArea") or []),
                )
                layout.setdefault("Rigid", {}).setdefault(
                    "matryoshka_dolls", []
                ).append(instance)
                inputs.extend(instance_inputs)
            if task == "sort_nesting_dolls_by_size_random":
                placed_clutter, clutter_inputs = _populate_declared_clutter(
                    source_root=source_root,
                    assets_root=assets_root,
                    task=task,
                    config=config,
                    scene=scene,
                    generation_seed=generation_seed,
                    layout_id=sampling_id,
                    layout=layout,
                    occupied_polygons=occupied_polygons,
                    prohibited=list(config.get("ProhibitedArea") or []),
                )
                inputs.extend(clutter_inputs)
                if placed_clutter != 20:
                    raise _PlacementExhausted(
                        f"{task} requires exactly twenty placed clutter objects"
                    )
        except _PlacementExhausted as exc:
            last_error = exc
            continue
        return (layout, inputs, whole_attempt + 1)
    raise TrainLayoutError(
        f"could not place {task} after {WHOLE_LAYOUT_MAX_ATTEMPTS} deterministic whole-layout attempts; last_error={last_error}"
    )


def _support_dag_contract(task: str) -> list[dict[str, Any]]:
    """Schema-frozen contracts for parent/child passive-support tasks."""
    make_toast = [
        {
            "type": "Articulation",
            "common": {
                "xlim": [0.04, 0.35],
                "ylim": [-0.12, -0.03],
                "zlim": [0.08, 0.08],
                "rotate_rand": True,
                "rotate_deg": [-105, -75],
                "relative_plane": "Table",
            },
            "categories": [{"name": "toaster", "index": [0, 1]}],
            "select": {"nums": 1, "mode": "unique", "label": ["toaster"]},
        },
        {
            "type": "Geometry",
            "common": {
                "xlim": [-0.35, 0.0],
                "ylim": [-0.2, 0.0],
                "rotate_rand": True,
                "rotate_deg": 15,
                "relative_plane": "Table",
            },
            "categories": [{"name": "bread_shelf"}],
            "select": {"nums": 1, "mode": "allow_duplicate", "label": ["bread_shelf"]},
        },
    ]
    for index in range(4):
        make_toast.append(
            {
                "type": "Rigid",
                "common": {
                    "xlim": [-0.02, -0.02],
                    "ylim": [-0.007, -0.007],
                    "rotate_rand": False,
                    "relative_plane": f"bread_shelf/bread_shelf/{index}",
                    "check_mode": "none",
                    "place_tag": "up",
                },
                "categories": [{"name": "bread", "index": [0]}],
                "select": {
                    "nums": 1,
                    "mode": "allow_duplicate",
                    "label": [f"bread_{index}"],
                },
            }
        )
    make_toast_random = deepcopy(make_toast)
    make_toast_random[0]["categories"] = [{"name": "toaster", "index": [2, 4]}]
    xylophone = [
        {
            "type": "Rigid",
            "common": {
                "xlim": [-0.45, 0.0],
                "ylim": [-0.2, 0.0],
                "rotate_rand": True,
                "rotate_deg": 30,
                "relative_plane": "Table",
                "margin": 0.02,
            },
            "categories": [
                {"name": "mallet_stand", "place_tag": "up_side", "xlim": [-0.2, 0.05]},
                {"name": "mallet_stand", "place_tag": "up", "xlim": [-0.45, -0.2]},
            ],
            "select": {"nums": 1, "mode": "same", "label": ["mallet_stand"]},
        },
        {
            "type": "Rigid",
            "common": {
                "xlim": [0.0, 0.0],
                "ylim": [0.0, 0.0],
                "rotate_rand": False,
                "relative_plane": "mallet_stand/mallet_stand/0",
                "check_mode": "point",
            },
            "categories": [{"name": "mallet"}],
            "select": {"nums": 1, "mode": "same", "label": ["mallet"]},
        },
        {
            "type": "Geometry",
            "common": {
                "xlim": [-0.1, 0.0],
                "ylim": [-0.05, 0.05],
                "rotate_rand": True,
                "rotate_deg": 15,
                "relative_plane": "Table",
                "margin": 0.02,
            },
            "categories": [{"name": "xylophone"}],
            "select": {"nums": 1, "mode": "same", "label": ["xylophone"]},
        },
    ]
    store_laptop_and_headphones = [
        {
            "type": "Articulation",
            "common": {
                "xlim": [-0.05, 0.05],
                "ylim": [-0.08, -0.08],
                "zlim": [0.02, 0.02],
                "rotate_rand": False,
                "relative_plane": "laptop_stand/laptop_stand/0",
                "check_mode": "none",
            },
            "categories": [{"name": "laptop", "index": [0]}],
            "select": {"nums": 1, "mode": "allow_duplicate", "label": ["laptop"]},
        },
        {
            "type": "Rigid",
            "common": {
                "xlim": [-0.4, 0.4],
                "ylim": [-0.25, 0.0],
                "rotate_rand": True,
                "rotate_deg": 30,
                "relative_plane": "Table",
            },
            "categories": [{"name": "headset", "index": [2, 3, 4, 5, 6]}],
            "select": {"nums": 1, "mode": "unique", "label": ["headset"]},
        },
        {
            "type": "Geometry",
            "common": {
                "xlim": [0.04, 0.09],
                "ylim": [0.02, 0.02],
                "rotate_deg": [-10, 0],
                "rotate_rand": True,
                "relative_plane": "Table",
            },
            "categories": [{"name": "laptop_stand", "index": [1]}],
            "select": {"nums": 1, "mode": "allow_duplicate", "label": ["laptop_stand"]},
        },
        {
            "type": "Geometry",
            "common": {
                "xlim": [0.3, 0.35],
                "ylim": [-0.15, -0.05],
                "zlim": [0.025, 0.025],
                "rotate_rand": True,
                "rotate_deg": [75, 105],
                "relative_plane": "Table",
            },
            "categories": [{"name": "vertical_laptop_storage_rack"}],
            "select": {
                "nums": 1,
                "mode": "allow_duplicate",
                "label": ["vertical_laptop_storage_rack"],
            },
        },
        {
            "type": "Geometry",
            "common": {
                "xlim": [-0.35, -0.25],
                "ylim": [-0.02, 0.05],
                "rotate_deg": [0, 30],
                "rotate_rand": True,
                "relative_plane": "Table",
            },
            "categories": [{"name": "headset_stand"}],
            "select": {"nums": 1, "mode": "allow_duplicate", "label": ["stand"]},
        },
    ]
    store_laptop_and_headphones_random = deepcopy(store_laptop_and_headphones)
    store_laptop_and_headphones_random[0]["categories"] = [
        {"name": "laptop", "index": [1, 2, 3]}
    ]
    store_laptop_and_headphones_random[1]["categories"] = [
        {"name": "headset", "index": [8, 9, 10]}
    ]
    contracts = {
        "make_toast": make_toast,
        "make_toast_random": make_toast_random,
        "play_Xylophone": xylophone,
        "store_laptop_and_headphones": store_laptop_and_headphones,
        "store_laptop_and_headphones_random": store_laptop_and_headphones_random,
    }
    if task not in contracts:
        raise TrainLayoutError(f"{task} has no support-DAG contract")
    return contracts[task]


def _populate_support_dag_task(
    *,
    source_root: Path,
    assets_root: Path,
    task: str,
    config: dict[str, Any],
    scene: dict[str, Any],
    generation_seed: int,
    layout_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, list[dict[str, str]], list[int]]:
    """Generate schema-frozen Table/support DAGs without Eval ancestry.

    Config identities and serialized output grouping remain in declaration
    order.  Physical materialization uses a stable topological order so a
    child may reference a parent declared later in the YAML.
    """
    specs = _support_dag_contract(task)
    if task in {"make_toast_random", "store_laptop_and_headphones_random"}:
        clutter_count = 15 if task == "make_toast_random" else 20
        expected_clutter = [
            {
                "xlim": [[-0.6, 0.6]],
                "ylim": [[-0.5, 0.5]],
                "relative_plane": "Table",
                "yaml_path": "Clutter/clutter.yml",
                "nums": clutter_count,
                "mode": "category_unique",
                "rotate_rand": True,
                "rotate_deg": 30,
                "margin": 0.015,
            }
        ]
        if config.get("Clutter") != expected_clutter:
            raise TrainLayoutError(f"{task} clutter sampling contract drifted")
        if config.get("ProhibitedArea") != [[-0.35, -0.5, 0.35, -0.4]]:
            raise TrainLayoutError(f"{task} ProhibitedArea contract drifted")
    actual_groups = [
        (object_type, group)
        for object_type in config
        if object_type in OBJECT_TYPES
        for group in config.get(object_type) or []
    ]
    if len(actual_groups) != len(specs):
        raise TrainLayoutError(f"{task} support-DAG group count/order drifted")
    validated = []
    catalog_inputs: list[dict[str, Any]] = []
    for group_number, ((object_type, group_cfg), spec) in enumerate(
        zip(actual_groups, specs, strict=True)
    ):
        categories = group_cfg.get("category") or []
        if (
            object_type != spec["type"]
            or deepcopy(group_cfg.get("common") or {}) != spec["common"]
            or categories != spec["categories"]
            or (deepcopy(group_cfg.get("select_mode") or {}) != spec["select"])
        ):
            raise TrainLayoutError(
                f"{task}:group{group_number} support-DAG semantics drifted"
            )
        category_candidates = []
        for alternative_index, category_cfg in enumerate(categories):
            indices = _allowed_indices(assets_root, object_type, category_cfg)
            if not indices:
                raise TrainLayoutError(
                    f"{task}:group{group_number} has empty candidate catalog"
                )
            category_candidates.append((alternative_index, category_cfg, indices))
            for category_idx in indices:
                metadata_path, asset_path = _asset_paths(
                    assets_root, object_type, str(category_cfg["name"]), category_idx
                )
                catalog_inputs.extend(
                    [
                        _input_record(
                            metadata_path, source_root, "candidate_object_metadata"
                        ),
                        _input_record(
                            asset_path, source_root, "candidate_object_asset"
                        ),
                    ]
                )
        validated.append((object_type, group_cfg, category_candidates, spec))
    labels_to_group: dict[str, int] = {}
    for group_number, (_, _, _, spec) in enumerate(validated):
        labels = spec["select"]["label"]
        if len(labels) != 1:
            raise TrainLayoutError(
                f"{task}:group{group_number} must declare exactly one label"
            )
        label = str(labels[0])
        if label in labels_to_group:
            raise TrainLayoutError(f"{task} duplicate support-DAG label {label}")
        labels_to_group[label] = group_number
    support_edges = [
        {
            "child": spec["select"]["label"][0],
            "relative_plane": spec["common"]["relative_plane"],
            "parent": spec["common"]["relative_plane"].split("/", 1)[0],
        }
        for spec in specs
        if spec["common"]["relative_plane"] not in {"Table", "Ground"}
    ]
    dependencies: dict[int, int | None] = {}
    for group_number, (_, _, _, spec) in enumerate(validated):
        relative_plane = str(spec["common"]["relative_plane"])
        if relative_plane in {"Table", "Ground"}:
            dependencies[group_number] = None
            continue
        parent_label = relative_plane.split("/", 1)[0]
        parent_group = labels_to_group.get(parent_label)
        if parent_group is None:
            raise TrainLayoutError(
                f"{task}:group{group_number} references unknown support parent {parent_label}"
            )
        if parent_group == group_number:
            raise TrainLayoutError(f"{task}:group{group_number} cannot support itself")
        dependencies[group_number] = parent_group
    materialization_order: list[int] = []
    remaining = set(range(len(validated)))
    while remaining:
        ready = [
            group_number
            for group_number in sorted(remaining)
            if dependencies[group_number] is None
            or dependencies[group_number] in materialization_order
        ]
        if not ready:
            raise TrainLayoutError(f"{task} support-DAG contains a cycle")
        group_number = ready[0]
        materialization_order.append(group_number)
        remaining.remove(group_number)
    last_error: _PlacementExhausted | None = None
    for whole_attempt in range(WHOLE_LAYOUT_MAX_ATTEMPTS):
        sampling_id = _whole_layout_sampling_id(layout_id, whole_attempt)
        layout: dict[str, Any] = {}
        table_occupied: list[Any] = []
        support_occupied: dict[str, list[Any]] = {}
        records_by_label: dict[str, dict[str, Any]] = {}
        records_by_group: dict[int, tuple[str, str, dict[str, Any]]] = {}
        inputs = list(catalog_inputs)
        try:
            for group_number in materialization_order:
                object_type, group_cfg, category_candidates, spec = validated[
                    group_number
                ]
                alternative_index, category_cfg, candidates = _choice(
                    category_candidates,
                    generation_seed,
                    task,
                    sampling_id,
                    f"{object_type}:{group_number}:category",
                )
                mode = spec["select"]["mode"]
                if mode in {"same", "allow_duplicate"}:
                    category_idx = _choice(
                        candidates,
                        generation_seed,
                        task,
                        sampling_id,
                        f"{object_type}:{group_number}:asset",
                    )
                elif mode == "unique":
                    category_idx = sorted(
                        candidates,
                        key=lambda value: _unit(
                            generation_seed,
                            task,
                            sampling_id,
                            f"{object_type}:{group_number}:asset:{value}",
                        ),
                    )[0]
                else:
                    raise TrainLayoutError(
                        f"{task}:group{group_number} selection mode drifted"
                    )
                label = spec["select"]["label"][0]
                category_overrides = {
                    key: deepcopy(value)
                    for key, value in category_cfg.items()
                    if key not in {"name", "index", "group"}
                }
                common = {
                    **deepcopy(group_cfg.get("common") or {}),
                    **category_overrides,
                }
                relative_plane = str(common["relative_plane"])
                if relative_plane == "Table":
                    instance, instance_inputs = _make_table_surface_instance(
                        source_root=source_root,
                        assets_root=assets_root,
                        scene=scene,
                        task=task,
                        generation_seed=generation_seed,
                        layout_id=sampling_id,
                        object_type=object_type,
                        category=str(category_cfg["name"]),
                        category_idx=category_idx,
                        group=category_cfg.get("group"),
                        label=label,
                        common=common,
                        occupied_polygons=table_occupied,
                        prohibited=list(config.get("ProhibitedArea") or []),
                    )
                else:
                    parent_label = relative_plane.split("/", 1)[0]
                    parent = records_by_label.get(parent_label)
                    if parent is None:
                        raise TrainLayoutError(
                            f"{task}:{label} support parent {parent_label} was not materialized before its child"
                        )
                    instance, instance_inputs = _make_relative_support_instance(
                        source_root=source_root,
                        assets_root=assets_root,
                        task=task,
                        generation_seed=generation_seed,
                        layout_id=sampling_id,
                        object_type=object_type,
                        category=str(category_cfg["name"]),
                        category_idx=category_idx,
                        group=category_cfg.get("group"),
                        label=label,
                        common=common,
                        parent=parent,
                        occupied_polygons=support_occupied.setdefault(
                            relative_plane, []
                        ),
                    )
                instance["category_alternative_index"] = alternative_index
                inputs.extend(instance_inputs)
                records_by_label[label] = instance
                records_by_group[group_number] = (
                    object_type,
                    str(category_cfg["name"]),
                    instance,
                )
            if task in {"make_toast_random", "store_laptop_and_headphones_random"}:
                placed_clutter, clutter_inputs = _populate_declared_clutter(
                    source_root=source_root,
                    assets_root=assets_root,
                    task=task,
                    config=config,
                    scene=scene,
                    generation_seed=generation_seed,
                    layout_id=sampling_id,
                    layout=layout,
                    occupied_polygons=table_occupied,
                    prohibited=list(config.get("ProhibitedArea") or []),
                )
                inputs.extend(clutter_inputs)
                expected_clutter_count = 15 if task == "make_toast_random" else 20
                if placed_clutter != expected_clutter_count:
                    raise _PlacementExhausted(
                        f"{task} requires exactly {expected_clutter_count} category-unique clutter objects"
                    )
        except _PlacementExhausted as exc:
            last_error = exc
            continue
        for group_number in range(len(validated)):
            object_type, category, instance = records_by_group[group_number]
            layout.setdefault(object_type, {}).setdefault(category, []).append(instance)
        return (layout, inputs, whole_attempt + 1, support_edges, materialization_order)
    raise TrainLayoutError(
        f"could not place {task} after {WHOLE_LAYOUT_MAX_ATTEMPTS} deterministic whole-layout attempts; last_error={last_error}"
    )


def generate_train_layout(
    *,
    source_root: str | Path,
    assets_root: str | Path,
    task: str,
    generation_seed: int,
    layout_id: int,
    env_cfg_type: str = "arx_x5",
) -> dict[str, Any]:
    """Generate one layout without consulting Eval_Layout."""
    if task not in SUPPORTED_TASKS:
        raise TrainLayoutError(f"unsupported train-layout task: {task}")
    source_root = Path(source_root).resolve()
    assets_root = Path(assets_root).resolve()
    config_path = source_root / "task" / "RoboDojo" / "config" / f"{task}.yml"
    registry_path = source_root / "task/RoboDojo/config/_task.yml"
    registry = (
        yaml.safe_load(registry_path.read_text()) if registry_path.is_file() else {}
    )
    scene_name = registry.get("tasks", {}).get(task, {}).get("scene_config", "default")
    if not isinstance(scene_name, str) or not scene_name.isidentifier():
        raise TrainLayoutError("Invalid public scene configuration name")
    scene_path = source_root / "env_cfg" / "scene" / (scene_name + ".yml")
    if not config_path.is_file() or not scene_path.is_file():
        raise TrainLayoutError("required task/scene config is missing")
    for forbidden in (
        source_root / "Assets" / "Eval_Layout",
        assets_root / "Eval_Layout",
    ):
        if config_path.is_relative_to(forbidden) or scene_path.is_relative_to(
            forbidden
        ):
            raise TrainLayoutError("Eval_Layout cannot be a generation ancestor")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    scene = yaml.safe_load(scene_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(scene, dict):
        raise TrainLayoutError("task/scene config must decode to mappings")
    layout: dict[str, Any] = {}
    task_semantics: str | None = None
    whole_layout_attempts: int | None = None
    whole_layout_group_order: list[str] | None = None
    support_graph: list[dict[str, str]] | None = None
    support_materialization_order: list[int] | None = None
    prohibited = config.get("ProhibitedArea") or []
    source_inputs = [
        _input_record(Path(__file__), source_root, "generator_source"),
        _input_record(config_path, source_root, "task_config"),
        _input_record(scene_path, source_root, "scene_config"),
    ]
    if registry_path.is_file():
        source_inputs.append(_input_record(registry_path, source_root, "task_registry"))
    if task == "arrange_largest_number":
        layout, inputs = _populate_arrange_largest_number(
            source_root=source_root,
            assets_root=assets_root,
            config=config,
            scene=scene,
            task=task,
            generation_seed=generation_seed,
            layout_id=layout_id,
        )
        source_inputs.extend(inputs)
        task_semantics = (
            "unique_same_texture_digits_plus_linked_count_symmetric_mat_row"
        )
    elif task == "arrange_largest_number_random":
        layout, inputs, whole_layout_attempts = _populate_arrange_largest_number_random(
            source_root=source_root,
            assets_root=assets_root,
            config=config,
            scene=scene,
            generation_seed=generation_seed,
            layout_id=layout_id,
        )
        source_inputs.extend(inputs)
        task_semantics = "four_or_five_unique_random_half_same_texture_digits_plus_linked_count_symmetric_mat_row_and_twenty_category_unique_clutter_with_exact_polygon_rejection"
        whole_layout_group_order = ["Geometry", "Rigid", "Clutter"]
    elif task == "align_blocks":
        layout, inputs = _populate_align_blocks(
            source_root=source_root,
            assets_root=assets_root,
            config=config,
            scene=scene,
            generation_seed=generation_seed,
            layout_id=layout_id,
        )
        source_inputs.extend(inputs)
        task_semantics = (
            "held_straightedge_target_separate_from_mutually_disjoint_cubes"
        )
    elif task == "general_pickup":
        layout, inputs = _populate_general_pickup(
            source_root=source_root,
            assets_root=assets_root,
            config=config,
            scene=scene,
            generation_seed=generation_seed,
            layout_id=layout_id,
        )
        source_inputs.extend(inputs)
        task_semantics = "one_target_plus_ten_category_unique_declared_catalog_clutter_with_bounded_per_object_rejection"
    elif task == "stack_bowls_random":
        layout, inputs = _populate_stack_bowls_random(
            source_root=source_root,
            assets_root=assets_root,
            config=config,
            scene=scene,
            generation_seed=generation_seed,
            layout_id=layout_id,
        )
        source_inputs.extend(inputs)
        task_semantics = "three_same_model_random_half_bowls_plus_two_declared_category_unique_clutter_bands"
    elif task == "hang_mugs_random":
        layout, inputs, whole_layout_attempts = _populate_hang_mugs_random(
            source_root=source_root,
            assets_root=assets_root,
            config=config,
            scene=scene,
            generation_seed=generation_seed,
            layout_id=layout_id,
        )
        source_inputs.extend(inputs)
        task_semantics = "three_random_half_mugs_plus_random_half_rack_and_twenty_category_unique_clutter_with_whole_layout_rejection"
        whole_layout_group_order = ["Rigid", "Geometry", "Clutter"]
    elif task == "push_T_random":
        layout, inputs, whole_layout_attempts = _populate_push_t_random(
            source_root=source_root,
            assets_root=assets_root,
            config=config,
            scene=scene,
            generation_seed=generation_seed,
            layout_id=layout_id,
        )
        source_inputs.extend(inputs)
        task_semantics = "random_half_t_plus_t_cushion_and_two_category_unique_clutter_bands_with_whole_layout_rejection"
        whole_layout_group_order = ["Rigid", "Geometry", "Clutter"]
    elif task in {"hang_mugs", "insert_tubes"}:
        layout, inputs, whole_layout_attempts = _populate_later_fixture_task(
            source_root=source_root,
            assets_root=assets_root,
            task=task,
            config=config,
            scene=scene,
            generation_seed=generation_seed,
            layout_id=layout_id,
        )
        source_inputs.extend(inputs)
        task_semantics = (
            "config_ordered_whole_layout_rejection_sampling_for_later_fixture"
        )
        whole_layout_group_order = ["Rigid", "Geometry"]
    elif task in {"fasten_screws", "fill_pen_holder", "cover_blocks"}:
        layout, inputs, whole_layout_attempts = _populate_config_ordered_table_task(
            source_root=source_root,
            assets_root=assets_root,
            task=task,
            config=config,
            scene=scene,
            generation_seed=generation_seed,
            layout_id=layout_id,
        )
        source_inputs.extend(inputs)
        task_semantics = (
            "three_color_cubes_permuted_over_three_columns_with_fixed_downward_cups"
            if task == "cover_blocks"
            else "config_ordered_table_groups_with_cross_group_selection_and_whole_layout_rejection_sampling"
        )
        whole_layout_group_order = [
            object_type
            for object_type in config
            if object_type in OBJECT_TYPES
            for _ in config.get(object_type) or []
        ]
    elif task in {
        "organize_table",
        "pack_objects_into_box",
        "pack_objects_into_box_random",
    }:
        layout, inputs, whole_layout_attempts = _populate_exact_ordered_table_task(
            source_root=source_root,
            assets_root=assets_root,
            task=task,
            config=config,
            scene=scene,
            generation_seed=generation_seed,
            layout_id=layout_id,
        )
        source_inputs.extend(inputs)
        if task == "pack_objects_into_box_random":
            task_semantics = "random_half_pack_asset_family_with_config_ordered_exact_polygon_table_sampling_and_complete_candidate_catalog_binding"
        else:
            task_semantics = "config_ordered_exact_polygon_table_sampling_with_complete_candidate_catalog_binding"
        whole_layout_group_order = [
            object_type
            for object_type in config
            if object_type in OBJECT_TYPES
            for _ in config.get(object_type) or []
        ]
    elif task == "make_kong":
        layout, inputs, whole_layout_attempts = _populate_make_kong(
            source_root=source_root,
            assets_root=assets_root,
            config=config,
            scene=scene,
            generation_seed=generation_seed,
            layout_id=layout_id,
        )
        source_inputs.extend(inputs)
        task_semantics = "config_ordered_same_index_mahjong_rows_with_public_passive_support_relative_discard_and_declaration_tiles"
        whole_layout_group_order = ["Rigid"] * 13
    elif task in {"sort_nesting_dolls_by_size", "sort_nesting_dolls_by_size_random"}:
        layout, inputs, whole_layout_attempts = _populate_sort_nesting_dolls(
            source_root=source_root,
            assets_root=assets_root,
            task=task,
            config=config,
            scene=scene,
            generation_seed=generation_seed,
            layout_id=layout_id,
        )
        source_inputs.extend(inputs)
        if task == "sort_nesting_dolls_by_size_random":
            task_semantics = "one_complete_random_half_texture_family_of_five_unique_ranked_dolls_plus_twenty_category_unique_clutter_with_exact_polygon_rejection"
            whole_layout_group_order = ["Rigid", "Clutter"]
        else:
            task_semantics = "one_complete_texture_family_of_five_unique_ranked_dolls_with_exact_polygon_rejection"
            whole_layout_group_order = ["Rigid"]
    elif task in {
        "make_toast",
        "make_toast_random",
        "play_Xylophone",
        "store_laptop_and_headphones",
        "store_laptop_and_headphones_random",
    }:
        (
            layout,
            inputs,
            whole_layout_attempts,
            support_graph,
            support_materialization_order,
        ) = _populate_support_dag_task(
            source_root=source_root,
            assets_root=assets_root,
            task=task,
            config=config,
            scene=scene,
            generation_seed=generation_seed,
            layout_id=layout_id,
        )
        source_inputs.extend(inputs)
        if task == "store_laptop_and_headphones":
            task_semantics = "config_identity_ordered_topologically_materialized_passive_support_dag_with_fixed_z_support_and_complete_candidate_catalog_binding"
        elif task == "store_laptop_and_headphones_random":
            task_semantics = "random_half_laptop_and_headset_asset_families_plus_config_identity_ordered_topologically_materialized_passive_support_dag_and_twenty_category_unique_clutter_with_exact_prohibited_area_rejection"
        elif task == "make_toast_random":
            task_semantics = "random_half_toaster_asset_family_plus_config_ordered_passive_support_dag_and_fifteen_category_unique_clutter_with_exact_prohibited_area_rejection"
        else:
            task_semantics = "config_ordered_passive_support_dag_with_category_overrides_and_complete_candidate_catalog_binding"
        whole_layout_group_order = [
            object_type
            for object_type in config
            if object_type in OBJECT_TYPES
            for _ in config.get(object_type) or []
        ]
        if task in {"make_toast_random", "store_laptop_and_headphones_random"}:
            whole_layout_group_order.append("Clutter")
    elif task in {"fill_egg_holder", "deposit_coin"}:
        layout, inputs, whole_layout_attempts = _populate_relative_support_task(
            source_root=source_root,
            assets_root=assets_root,
            task=task,
            config=config,
            scene=scene,
            generation_seed=generation_seed,
            layout_id=layout_id,
        )
        source_inputs.extend(inputs)
        task_semantics = "config_ordered_object_relative_passive_support_frame_sampling_with_whole_layout_rejection"
        whole_layout_group_order = [
            object_type
            for object_type in config
            if object_type in OBJECT_TYPES
            for _ in config.get(object_type) or []
        ]
    elif task == "match_and_pick_from_conveyor":
        layout, inputs, whole_layout_attempts = _populate_match_and_pick_from_conveyor(
            source_root=source_root,
            assets_root=assets_root,
            task=task,
            config=config,
            scene=scene,
            generation_seed=generation_seed,
            layout_id=layout_id,
        )
        source_inputs.extend(inputs)
        task_semantics = "native_dynamic_conveyor_with_exact_remembered_matching_identity_and_three_unique_distractors"
        whole_layout_group_order = ["Dynamic", "Rigid", "Rigid", "Rigid"]
    elif task == "swap_blocks":
        layout, inputs, whole_layout_attempts = _populate_swap_blocks(
            source_root=source_root,
            assets_root=assets_root,
            task=task,
            config=config,
            scene=scene,
            generation_seed=generation_seed,
            layout_id=layout_id,
        )
        source_inputs.extend(inputs)
        task_semantics = "uniform_ordered_distinct_two_of_three_mat_occupancy_with_passive_support_frame_placement"
        whole_layout_group_order = [
            object_type
            for object_type in config
            if object_type in OBJECT_TYPES
            for _ in config.get(object_type) or []
        ]
    elif task == "imitate_sorting_sequence":
        layout, inputs, whole_layout_attempts = _populate_imitate_sorting_sequence(
            source_root=source_root,
            assets_root=assets_root,
            task=task,
            config=config,
            scene=scene,
            generation_seed=generation_seed,
            layout_id=layout_id,
        )
        source_inputs.extend(inputs)
        task_semantics = (
            "five_category_distinct_target_support_pairs_with_train_identity_order"
        )
        whole_layout_group_order = ["Geometry", "Geometry"] + ["Rigid"] * 6
    else:
        occupied: list[_Placed] = []
        for object_type, group_cfg, category_cfg, indices in _selected_groups(
            task, config, assets_root, generation_seed, layout_id
        ):
            common = deepcopy(group_cfg.get("common") or {})
            labels = list((group_cfg.get("select_mode") or {}).get("label") or [])
            if len(labels) != len(indices):
                raise TrainLayoutError(f"{task}:{object_type} label/count mismatch")
            category = str(category_cfg["name"])
            group = category_cfg.get("group")
            for label, index in zip(labels, indices, strict=True):
                instance, inputs = _make_instance(
                    source_root=source_root,
                    assets_root=assets_root,
                    scene=scene,
                    task=task,
                    generation_seed=generation_seed,
                    layout_id=layout_id,
                    object_type=object_type,
                    category=category,
                    category_idx=index,
                    group=group,
                    label=str(label),
                    common=common,
                    occupied=occupied,
                    prohibited=prohibited,
                )
                layout.setdefault(object_type, {}).setdefault(category, []).append(
                    instance
                )
                source_inputs.extend(inputs)
    camera_stand, camera_stand_inputs = _materialize_default_camera_stand(
        source_root=source_root, assets_root=assets_root, scene=scene
    )
    layout.setdefault("Geometry", {}).setdefault("camera_stand", []).append(
        camera_stand
    )
    source_inputs.extend(camera_stand_inputs)
    layout["Room"] = {
        "default": scene["Room"]["default"],
        "default_pos": scene["Room"]["default_pos"],
        "default_rot": scene["Room"]["default_ori"],
        "scale": scene["Room"]["scale"],
    }
    if "Table" in scene:
        layout["Table"] = deepcopy(scene["Table"])
    layout["Ground"] = deepcopy(scene["Ground"])
    background = deepcopy(scene["Background"])
    background["category_name"] = background.pop("default")
    layout["Background"] = background
    unique_inputs = {record["path"]: record for record in source_inputs}
    normalized_prohibited_areas = [
        [
            min(float(area[0]), float(area[2])),
            min(float(area[1]), float(area[3])),
            max(float(area[0]), float(area[2])),
            max(float(area[1]), float(area[3])),
        ]
        for area in prohibited
        if isinstance(area, list) and len(area) == 4
    ]
    if len(normalized_prohibited_areas) != len(prohibited):
        raise TrainLayoutError("ProhibitedArea entries must contain four coordinates")
    layout["_train_layout"] = {
        "schema_version": SCHEMA_VERSION,
        "layout_domain": "train",
        "generator_version": GENERATOR_VERSION,
        "task": task,
        "env_cfg_type": env_cfg_type,
        "generation_seed": int(generation_seed),
        "layout_id": int(layout_id),
        "ancestry_policy": "task_config_plus_scene_config_plus_object_catalog_only_no_eval_ancestor",
        "omitted_scene_fixtures": [],
        "omitted_task_fixtures": [],
        **(
            {
                "prohibited_area_semantics": "canonical_minx_miny_maxx_maxy",
                "prohibited_area_rectangles": normalized_prohibited_areas,
                "prohibited_area_source": {
                    "path": config_path.relative_to(source_root).as_posix()
                },
            }
            if prohibited
            else {}
        ),
        **({"task_semantics": task_semantics} if task_semantics else {}),
        **({"support_graph": support_graph} if support_graph is not None else {}),
        **(
            {
                "support_materialization_order": support_materialization_order,
                "support_materialization_order_labels": [
                    next(
                        (
                            spec["select"]["label"][0]
                            for group_number, spec in enumerate(
                                _support_dag_contract(task)
                            )
                            if group_number == materialized_group
                        )
                    )
                    for materialized_group in support_materialization_order
                ],
            }
            if support_materialization_order is not None
            else {}
        ),
        **(
            {
                "whole_layout_rejection_sampling": {
                    "attempts": whole_layout_attempts,
                    "rejected_attempts": whole_layout_attempts - 1,
                    "max_attempts": WHOLE_LAYOUT_MAX_ATTEMPTS,
                    "group_order": whole_layout_group_order,
                    "outcome_conditioning": False,
                }
            }
            if whole_layout_attempts is not None
            else {}
        ),
        "source_inputs": [unique_inputs[key] for key in sorted(unique_inputs)],
    }
    return layout
