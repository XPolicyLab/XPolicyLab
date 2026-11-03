"""Public-config container fills, pending native settling/containment checks.

Bare container labels use a local bounding-box top frame. This is an explicit
geometric proposal convention, not a claim of physical containment validity.
"""

from copy import deepcopy

from . import catalog as c
from .placement import _input_record


def place_container_child(*, parent, common, **kwargs):
    if parent.get("scale", [1, 1, 1]) != [1, 1, 1]:
        raise c.TrainLayoutError("Scaled container frame requires native validation")
    assets = kwargs["assets_root"]
    metadata_path, _ = c._asset_paths(
        assets,
        parent["physics"]["type"].capitalize(),
        parent["category"],
        parent["category_idx"],
    )
    metadata = c._load_asset_metadata(metadata_path)
    vertices = c._oriented_bbox_vertices(metadata)
    top = [
        (min(v[i] for v in vertices) + max(v[i] for v in vertices)) / 2
        for i in range(2)
    ]
    top.append(max(v[2] for v in vertices))
    local_common = dict(common, relative_plane="Table")
    # Use the existing metadata-based child placement in local coordinates.
    # The public enforce mode intentionally allows fill particles to overlap
    # before settling; it cannot be treated as a stability certificate.
    record, inputs = c._make_table_surface_instance(
        **kwargs,
        common=local_common,
        scene={"Table": {"default_pos": [0, 0, -0.5], "scale": [2, 2, 1]}},
        occupied_polygons=[],
        prohibited=[],
    )
    local_pos = record["default_pos"]
    offset = c._quat_rotate_vector(
        parent["default_ori"], [top[i] + local_pos[i] for i in range(3)]
    )
    record["default_pos"] = [parent["default_pos"][i] + offset[i] for i in range(3)]
    record["default_ori"] = c._quat_multiply(
        parent["default_ori"], record["default_ori"]
    )
    record["relative_plane"] = common["relative_plane"]
    record["container_binding"] = dict(
        parent_label=parent["label"],
        parent_category=parent["category"],
        parent_category_idx=parent["category_idx"],
        frame_convention="parent_local_bbox_top_center",
        local_frame=[*top, 1, 0, 0, 0],
        parent_position=deepcopy(parent["default_pos"]),
        parent_orientation=deepcopy(parent["default_ori"]),
        physics_status="settling_and_containment_not_validated",
    )
    inputs.append(
        _input_record(metadata_path, kwargs["source_root"], "container_metadata")
    )
    return record, inputs
