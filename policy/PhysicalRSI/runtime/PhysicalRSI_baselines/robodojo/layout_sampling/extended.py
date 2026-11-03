"""Additional public-config samplers; no benchmark-layout templates.

Selection follows the public category/instance modes. Physics and distribution
qualification are separate from producing a geometrically valid candidate.
"""

from copy import deepcopy
from pathlib import Path

import yaml

from . import catalog as c
from .containers import place_container_child
from .equation import complete_requests
from .image_conveyor import bind_photo, place_fixture
from .placement import _input_record, _unit

SUPPORTED_TASKS = {
    "classify_objects",
    "classify_objects_by_language",
    "fold_clothes",
    "fold_clothes_random",
    "sweep_blocks",
    "sweep_blocks_random",
    "put_bottles_into_dustbin",
    "solve_equation",
    "pour_balls_into_vase",
    "pour_by_language",
    "pour_liquid_into_cup",
    "pour_liquid_into_cup_random",
    "pick_from_conveyor_by_image",
}


def _count(value, random):
    if isinstance(value, int):
        return value
    if len(value) == 1:
        return value[0]
    if len(value) == 2:
        low, high = value
        return low + min(int(random * (high - low + 1)), high - low)
    raise c.TrainLayoutError("Unsupported count distribution")


def select_instances(config, assets_root, *, task, seed, counter):
    """Expand category/instance groups into uniquely labeled object requests."""
    result = []
    for kind, groups in config.items():
        if kind not in c.OBJECT_TYPES:
            continue
        for group_index, group in enumerate(groups):
            common = group["common"]
            categories = group["category"]
            selection = group["select_mode"]
            prefix = f"{kind}:{group_index}"

            def random(field):
                return _unit(seed, task, counter, prefix + ":" + field)

            def ordered(values, field):
                return sorted(values, key=lambda index: random(f"{field}:{index}"))

            mode = selection["mode"]
            if mode == "label_map" and task == "pick_from_conveyor_by_image":
                continue  # Bound after selecting the referenced target identity.
            picks = []
            if mode == "hierarchical":
                ncat = _count(
                    selection.get("select_category_nums", [len(categories)]),
                    random("ncat"),
                )
                if not 1 <= ncat <= len(categories):
                    raise c.TrainLayoutError("Invalid hierarchical category count")
                prefixes = selection["label_prefix"]
                for position, cat_index in enumerate(
                    ordered(range(len(categories)), "category")[:ncat]
                ):
                    category = categories[cat_index]
                    indices = c._allowed_indices(assets_root, kind, category)
                    number = _count(
                        selection["select_instance_nums"], random(f"ninst:{position}")
                    )
                    instance_mode = selection["instance_sample_mode"]
                    if instance_mode == "unique":
                        if number > len(indices):
                            raise c.TrainLayoutError("Insufficient unique instances")
                        selected = ordered(indices, f"instances:{position}")[:number]
                    elif instance_mode == "allow_duplicate":
                        selected = [
                            indices[
                                min(
                                    int(
                                        random(f"instance:{position}:{i}")
                                        * len(indices)
                                    ),
                                    len(indices) - 1,
                                )
                            ]
                            for i in range(number)
                        ]
                    else:
                        raise c.TrainLayoutError(
                            "Unsupported hierarchical instance mode"
                        )
                    label_offset = len(picks)
                    picks.extend(
                        (
                            category,
                            index,
                            f"{prefixes[position]}_{i}"
                            if isinstance(prefixes, list)
                            else f"{prefixes}_{label_offset + i}",
                        )
                        for i, index in enumerate(selected)
                    )
            else:
                pool = [
                    (cat_index, index)
                    for cat_index, category in enumerate(categories)
                    for index in c._allowed_indices(assets_root, kind, category)
                ]
                if not pool:
                    raise c.TrainLayoutError("Empty asset catalog")
                number = _count(selection["nums"], random("number"))
                labels = selection["label"]
                if len(labels) != number:
                    raise c.TrainLayoutError(
                        "Selection count and explicit labels differ"
                    )
                if mode == "unique":
                    if number > len(pool):
                        raise c.TrainLayoutError("Insufficient unique assets")
                    selected = ordered(range(len(pool)), "instance")[:number]
                elif mode in {"same", "allow_duplicate"}:
                    selected = [
                        min(
                            int(
                                random("same" if mode == "same" else f"instance:{i}")
                                * len(pool)
                            ),
                            len(pool) - 1,
                        )
                        for i in range(number)
                    ]
                else:
                    raise c.TrainLayoutError("Unsupported selection mode: " + mode)
                picks.extend(
                    (categories[pool[i][0]], pool[i][1], label)
                    for i, label in zip(selected, labels, strict=True)
                )
            for category, index, label in picks:
                settings = dict(
                    common,
                    **{
                        k: v
                        for k, v in category.items()
                        if k not in {"name", "index", "group"}
                    },
                )
                result.append(
                    dict(
                        kind=kind,
                        category=category["name"],
                        index=index,
                        group=category.get("group"),
                        label=label,
                        common=settings,
                    )
                )
    if len({row["label"] for row in result}) != len(result):
        raise c.TrainLayoutError("Duplicate labels after hierarchical selection")
    return result


def generate_train_layout(
    *, source_root, assets_root, task, generation_seed, layout_id
):
    if task not in SUPPORTED_TASKS:
        raise c.TrainLayoutError("No extended sampler for " + task)
    source_root, assets_root = Path(source_root), Path(assets_root)
    config_path = source_root / "task/RoboDojo/config" / (task + ".yml")
    registry_path = source_root / "task/RoboDojo/config/_task.yml"
    registry = (
        yaml.safe_load(registry_path.read_text()) if registry_path.is_file() else {}
    )
    scene_name = registry.get("tasks", {}).get(task, {}).get("scene_config", "default")
    if not isinstance(scene_name, str) or not scene_name.isidentifier():
        raise c.TrainLayoutError("Invalid public scene configuration name")
    scene_path = source_root / "env_cfg/scene" / (scene_name + ".yml")
    config, scene = (
        yaml.safe_load(path.read_text()) for path in (config_path, scene_path)
    )
    last_error = None
    for attempt in range(c.WHOLE_LAYOUT_MAX_ATTEMPTS):
        counter = c._whole_layout_sampling_id(layout_id, attempt)
        requests = select_instances(
            config, assets_root, task=task, seed=generation_seed, counter=counter
        )
        equation = None
        if task == "solve_equation":
            requests, equation = complete_requests(
                requests, assets_root=assets_root, seed=generation_seed, counter=counter
            )
        layout, polygons, ground_occupied, inputs = {}, [], [], []
        photo = None
        if task == "pick_from_conveyor_by_image":
            requests, map_path, photo = bind_photo(requests, config, assets_root)
            inputs.append(_input_record(map_path, source_root, "photo_identity_map"))
        parents, support_occupied = {}, {}
        try:
            for row in requests:
                kwargs = dict(
                    source_root=source_root,
                    assets_root=assets_root,
                    scene=scene,
                    task=task,
                    generation_seed=generation_seed,
                    layout_id=counter,
                    object_type=row["kind"],
                    category=row["category"],
                    category_idx=row["index"],
                    group=row["group"],
                    label=row["label"],
                    common=row["common"],
                    prohibited=config.get("ProhibitedArea", []),
                )
                plane = row["common"]["relative_plane"]
                if plane == "Table":
                    record, used = c._make_table_surface_instance(
                        **kwargs,
                        occupied_polygons=polygons,
                        allow_geometry_origin=row["kind"] == "Geometry",
                    )
                elif plane == "Ground":
                    record, used = c._make_instance(**kwargs, occupied=ground_occupied)
                else:
                    parent = parents.get(plane.split("/", 1)[0])
                    if parent is None:
                        raise c.TrainLayoutError(
                            "Missing or out-of-order support parent"
                        )
                    relative_kwargs = {
                        key: value
                        for key, value in kwargs.items()
                        if key not in {"scene", "prohibited"}
                    }
                    if task == "pick_from_conveyor_by_image" and "/" not in plane:
                        record, used = place_fixture(**relative_kwargs, parent=parent)
                    elif task == "pick_from_conveyor_by_image":
                        support = c._conveyor_support_contract(
                            source_root=source_root,
                            assets_root=assets_root,
                            parent=parent,
                            relative_plane=plane,
                        )
                        child_kwargs = {
                            k: v
                            for k, v in relative_kwargs.items()
                            if k != "object_type"
                        }
                        record, used = c._make_conveyor_child_instance(
                            **child_kwargs,
                            parent=parent,
                            support=support,
                            occupied_polygons=support_occupied.setdefault(plane, []),
                        )
                    elif "/" not in plane and task.startswith("pour_"):
                        record, used = place_container_child(
                            **relative_kwargs, parent=parent
                        )
                    else:
                        record, used = c._make_relative_support_instance(
                            **relative_kwargs,
                            parent=parent,
                            occupied_polygons=support_occupied.setdefault(plane, []),
                        )
                parents[row["label"]] = record
                layout.setdefault(row["kind"], {}).setdefault(
                    row["category"], []
                ).append(record)
                inputs.extend(used)
            if "Clutter" in config:
                _, used = c._populate_declared_clutter(
                    source_root=source_root,
                    assets_root=assets_root,
                    task=task,
                    config=config,
                    scene=scene,
                    generation_seed=generation_seed,
                    layout_id=counter,
                    layout=layout,
                    occupied_polygons=polygons,
                    prohibited=config.get("ProhibitedArea", []),
                )
                inputs.extend(used)
        except c._PlacementExhausted as error:
            last_error = error
            continue
        stand, used = c._materialize_default_camera_stand(
            source_root=source_root, assets_root=assets_root, scene=scene
        )
        layout.setdefault("Geometry", {}).setdefault("camera_stand", []).append(stand)
        inputs.extend(used)
        layout["Room"] = dict(
            default=scene["Room"]["default"],
            default_pos=scene["Room"]["default_pos"],
            default_rot=scene["Room"]["default_ori"],
            scale=scene["Room"]["scale"],
        )
        for key in ("Table", "Ground"):
            if key in scene:
                layout[key] = deepcopy(scene[key])
        background = deepcopy(scene["Background"])
        background["category_name"] = background.pop("default")
        layout["Background"] = background
        inputs.extend(
            _input_record(path, source_root, role)
            for path, role in (
                (config_path, "task_config"),
                (scene_path, "scene_config"),
                (Path(__file__), "generator_source"),
            )
        )
        if registry_path.is_file():
            inputs.append(_input_record(registry_path, source_root, "task_registry"))
        layout["_train_layout"] = dict(
            generator="extended_public_config_v1",
            task=task,
            generation_seed=generation_seed,
            layout_id=layout_id,
            attempts=attempt + 1,
            source_inputs=inputs,
            sampling="hierarchical_selection_whole_layout_geometry_rejection",
        )
        if equation is not None:
            layout["_train_layout"]["equation"] = equation
        if photo is not None:
            layout["_train_layout"]["photo_binding"] = photo
        return layout
    raise c.TrainLayoutError(
        f"No geometric candidate after {c.WHOLE_LAYOUT_MAX_ATTEMPTS} attempts: {last_error}"
    )
