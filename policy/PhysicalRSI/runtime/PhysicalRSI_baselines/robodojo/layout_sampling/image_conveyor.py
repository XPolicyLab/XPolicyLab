"""Photo identity binding and fixed conveyor fixture proposals."""

import json
from copy import deepcopy

from . import catalog as c
from .placement import _input_record


def bind_photo(requests, config, assets_root):
    groups = config.get("Geometry", [])
    if len(groups) != 1:
        raise c.TrainLayoutError("Expected one conveyor image group")
    group = groups[0]
    select = group["select_mode"]
    if (
        select["mode"] != "label_map"
        or select["label"] != ["photo"]
        or group["category"] != [{"name": "photo"}]
    ):
        raise c.TrainLayoutError("Conveyor photo mapping contract changed")
    target = next(row for row in requests if row["label"] == select["key_label"])
    path = assets_root / "Object/RoboDojo/Geometry/photo/map.json"
    mapping = json.loads(path.read_text())
    identity = f"{target['category']}/{target['index']:05d}"
    if identity not in mapping or not str(mapping[identity]).isdigit():
        raise c.TrainLayoutError("No public photo mapping for " + identity)
    index = int(mapping[identity])
    c._asset_paths(assets_root, "Geometry", "photo", index)
    result = [
        *requests,
        dict(
            kind="Geometry",
            category="photo",
            index=index,
            group=None,
            label="photo",
            common=deepcopy(group["common"]),
        ),
    ]
    result.sort(key=lambda row: row["label"] != "conveyor")
    return (
        result,
        path,
        dict(target_label=target["label"], target_identity=identity, photo_index=index),
    )


def place_fixture(*, parent, common, **kwargs):
    """Fixed-z fixture in world XY relative to conveyor root, bbox-top Z.

    This geometric convention is recorded for native scene validation; no
    historical evaluation layout supplies board/basket transforms.
    """
    path, _ = c._asset_paths(
        kwargs["assets_root"], "Dynamic", parent["category"], parent["category_idx"]
    )
    vertices = c._oriented_bbox_vertices(c._load_asset_metadata(path))
    z_top = parent["default_pos"][2] + max(
        c._quat_rotate_vector(parent["default_ori"], v)[2] for v in vertices
    )
    local = dict(common, relative_plane="Table", check_mode="enforce")
    if common.get("zlim") is None:
        raise c.TrainLayoutError("Conveyor board/basket requires explicit z offset")
    record, inputs = c._make_table_surface_instance(
        **kwargs,
        common=local,
        scene={"Table": dict(default_pos=[0, 0, -0.5], scale=[20, 20, 1])},
        occupied_polygons=[],
        prohibited=[],
    )
    record["default_pos"] = [
        record["default_pos"][0] + parent["default_pos"][0],
        record["default_pos"][1] + parent["default_pos"][1],
        record["default_pos"][2] + z_top,
    ]
    record["relative_plane"] = common["relative_plane"]
    record["check_mode"] = common.get("check_mode", "bbox")
    record["fixture_binding"] = dict(
        parent_label=parent["label"],
        frame_convention="world_axes_parent_origin_xy_world_bbox_top_z",
        physics_status="native_scene_not_validated",
    )
    inputs.append(_input_record(path, kwargs["source_root"], "conveyor_metadata"))
    return record, inputs
