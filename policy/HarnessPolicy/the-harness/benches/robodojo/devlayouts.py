"""Development layouts for RoboDojo, disjoint from the evaluation packs.

The public RoboDojo checkout ships only replay: ``SceneManager`` spawns what
``LayoutManager.load_saved_layout`` reads from an ``Eval_Layout`` JSON, and a
``None`` layout spawns no objects. The step that turned a task config into a
layout JSON (object selection plus placement) is not in the release; only its
planar sampler, ``utils.cluttered_generator.ClutteredGenerator``, is.

This module rebuilds that step with the released sampler so development
episodes can use the normal replay path:

* A template is drawn from the task's evaluation layouts (all packs). It
  supplies the fields the release does not let us derive: resolved object
  identities for complex select modes, physics, visuals, room, table, ground,
  background, and parent-relative placements.
* Object identities are redrawn from the task config where the select mode is
  simple (one category, ``same``/``unique``/``allow_duplicate``, no other entry
  refers to the label, no child placed on it, physics equal to the asset
  metadata). Clutter is regenerated from the task's ``Clutter`` entries.
* Every object on the table is re-placed with ``ClutteredGenerator`` using the
  instance's own ``xlim``/``ylim``/``zlim``/``qpos``/``rotate_*``/``margin``/
  ``check_mode``/``place_tag`` and the asset's oriented bbox and place frames,
  inside the table top inset by 5 cm, avoiding the task's prohibited areas.
  Objects placed relative to another object keep their relative pose. Objects
  on the ground (the camera stand) keep their pose.
* The conveyor tasks have no table: objects on the conveyor are re-placed
  within their config ranges in the conveyor's frame (discrete slots are
  shuffled among the objects that share them, continuous ranges resampled with
  ``CONVEYOR_MIN_GAP_M`` between objects) and turned by up to
  ``CONVEYOR_YAW_DEG``; objects off the conveyor keep their pose.
* A layout is rejected, and the next attempt drawn, if it repeats an
  evaluation layout: same labelled objects and assets, each within
  ``MIN_EVAL_DISTANCE_M`` and ``MIN_EVAL_ANGLE_DEG`` of its eval pose.

Generation is deterministic: the RNG is seeded from ``(task, seed, attempt)``
and the global numpy / ``random`` states are restored afterwards. Layouts are
written once and replayed from disk after that.

Importing this module does not import Isaac, torch, numpy, or RoboDojo.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
import zlib

from .tasks import BENCHMARK, LAYOUT_PACK_SEEDS, layout_dir


GENERATOR = "harness-robodojo-dev-v1"
OBJECT_TYPES = ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid")
SIMPLE_MODES = ("same", "unique", "allow_duplicate")
TABLE_INSET_M = 0.05  # LayoutManager.cluttered_generator_init
MIN_EVAL_DISTANCE_M = 0.01
MIN_EVAL_ANGLE_DEG = 5.0
MAX_ATTEMPTS = 50
CLUTTER_TRIES_PER_OBJECT = 3
# Origin of the conveyor's "conveyor/conveyor/0" placement plane relative to the
# conveyor asset (x), from the evaluation layouts: an object with xlim slot s sits
# at conveyor x + 0.445 + s.
CONVEYOR_BELT_X_OFFSET = 0.445
CONVEYOR_MIN_GAP_M = 0.12
CONVEYOR_YAW_DEG = 10.0


def dev_layout_path(directory, task, seed):
    return Path(directory) / f"{task}_{seed}.json"


def validate_dev_seed(seed):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def rng_seed(task, seed, attempt):
    return zlib.crc32(f"{GENERATOR}:{task}:{seed}:{attempt}".encode())


MAX_UNSTABLE_SKIPS = 8


def _meta_path(path):
    return Path(path).with_name(Path(path).name + ".meta")


def ensure_dev_layout(checkout, env_config, directory, task, seed, skip=0):
    """Path of the development layout ``seed`` for ``task``, generating it on first use.

    ``skip`` selects the next acceptable layout in the seed's deterministic
    sequence after ``skip`` earlier ones were found unstable in simulation. The
    layout file keeps RoboDojo's format; the skip count is in a ``.meta`` sidecar.
    """
    validate_dev_seed(seed)
    path = dev_layout_path(directory, task, seed)
    if path.exists():
        meta = _meta_path(path)
        recorded = json.loads(meta.read_text()).get("skip", 0) if meta.exists() else 0
        if recorded >= skip:
            return path
    layout = generate_dev_layout(checkout, env_config, task, seed, skip=skip)
    _meta_path(path).parent.mkdir(parents=True, exist_ok=True)
    _meta_path(path).write_text(json.dumps({"task": task, "seed": seed, "skip": skip, "generator": GENERATOR}))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(layout, indent=2))
    os.replace(temporary, path)
    return path


def eval_layout_files(checkout, env_config, task):
    pattern = re.compile(rf"{re.escape(task)}_(\d+)\.json")
    files = []
    for pack in LAYOUT_PACK_SEEDS:
        directory = layout_dir(checkout, env_config, pack)
        if not directory.is_dir():
            continue
        matches = [(int(m.group(1)), p) for p in directory.iterdir() if (m := pattern.fullmatch(p.name))]
        files.extend(p for _i, p in sorted(matches))
    return files


def discard_dev_layout(path):
    """Set an unstable layout aside (kept for inspection) so the next one is generated."""
    path = Path(path)
    if path.exists():
        os.replace(path, path.with_name(path.name + ".unstable"))


def generate_dev_layout(checkout, env_config, task, seed, skip=0):
    import random

    import numpy as np
    import yaml

    checkout = Path(checkout).absolute()
    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))
    templates = eval_layout_files(checkout, env_config, task)
    if not templates:
        raise RuntimeError(f"No RoboDojo evaluation layouts for task {task!r} to take a template from")
    evals = [json.loads(p.read_text()) for p in templates]
    config = yaml.safe_load((checkout / "task" / BENCHMARK / "config" / f"{task}.yml").read_text()) or {}
    objects = checkout / "Assets" / "Object" / BENCHMARK

    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        failures = []
        for attempt in range(MAX_ATTEMPTS):
            value = rng_seed(task, seed, attempt)
            np.random.seed(value)
            random.seed(value)
            index = int(np.random.randint(len(templates)))
            try:
                layout = _sample(np, deepcopy(evals[index]), config, objects)
            except _PlacementError as error:
                failures.append(str(error))
                continue
            same = _coincident_eval(layout, evals)
            if same is not None:
                failures.append(f"repeats evaluation layout {templates[same]}")
                continue
            if skip > 0:
                skip -= 1
                failures.append("skipped (unstable in simulation)")
                continue
            return layout
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    raise RuntimeError(f"No dev layout for {task} seed {seed} after {MAX_ATTEMPTS} attempts: {failures[-3:]}")


class _PlacementError(Exception):
    pass


def _instances(layout):
    for kind in OBJECT_TYPES:
        for category, entries in (layout.get(kind) or {}).items():
            for inst in entries:
                yield kind, category, inst


def _metadata(objects, kind, category, index):
    path = objects / kind / category / f"{int(index):05d}" / "metadata.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):  # missing or empty, as utils.load_file.load_object_metadata
        return None


def _parent_key(inst):
    plane = inst.get("relative_plane")
    if plane in (None, "Table", "Ground"):
        return None
    return str(plane).split("/")[0]


def _find_parent(layout, key):
    by_category = None
    for _kind, category, inst in _instances(layout):
        if inst.get("label") == key:
            return inst
        if by_category is None and category == key and inst.get("type") != "cluttered":
            by_category = inst
    return by_category


def _referenced_labels(config):
    exact, prefixes = set(), set()
    for kind in OBJECT_TYPES:
        for entry in config.get(kind) or []:
            mode = (entry or {}).get("select_mode") or {}
            for key in ("same_label", "key_label"):
                if mode.get(key):
                    exact.add(str(mode[key]))
            if mode.get("linked_label_prefix"):
                prefixes.add(str(mode["linked_label_prefix"]))
    return exact, prefixes


def _redraw_identities(np, layout, config, objects):
    exact, prefixes = _referenced_labels(config)
    parents = {_parent_key(inst) for _k, _c, inst in _instances(layout)} - {None}
    for kind in OBJECT_TYPES:
        for entry in config.get(kind) or []:
            mode = (entry or {}).get("select_mode") or {}
            categories = entry.get("category") or []
            labels = mode.get("label") or []
            if mode.get("mode") not in SIMPLE_MODES or len(categories) != 1 or not labels:
                continue
            name = categories[0].get("name")
            choices = [int(i) for i in categories[0].get("index") or []]
            if len(choices) < 2 or name in parents:
                continue
            if any(l in exact or l in parents or any(str(l).startswith(p) for p in prefixes) for l in labels):
                continue
            insts = [i for i in (layout.get(kind) or {}).get(name, []) if i.get("label") in labels]
            if not insts:
                continue
            if not all(_physics_matches(objects, kind, name, inst) for inst in insts):
                continue
            choices = [i for i in choices if _placeable(_metadata(objects, kind, name, i), insts[0])]
            if not choices or (mode["mode"] == "unique" and len(choices) < len(insts)):
                continue
            if mode["mode"] == "same":
                drawn = [choices[int(np.random.randint(len(choices)))]] * len(insts)
            elif mode["mode"] == "unique":
                drawn = [choices[i] for i in np.random.permutation(len(choices))[: len(insts)]]
            else:
                drawn = [choices[int(np.random.randint(len(choices)))] for _ in insts]
            for inst, index in zip(insts, drawn):
                physics = dict(_metadata(objects, kind, name, index).get("physics") or {})
                if "type" in inst.get("physics", {}):
                    physics["type"] = inst["physics"]["type"]
                inst["category_idx"] = int(index)
                inst["physics"] = physics


def _physics_matches(objects, kind, category, inst):
    metadata = _metadata(objects, kind, category, inst.get("category_idx", 0))
    if metadata is None:
        return False
    physics = {k: v for k, v in (inst.get("physics") or {}).items() if k != "type"}
    return physics == (metadata.get("physics") or {})


def _placeable(metadata, inst):
    if not metadata:
        return False
    if not ((metadata.get("geometry") or {}).get("oriented_bbox") or {}).get("vertices"):
        return False
    return inst.get("zlim") is not None or bool((metadata.get("active") or {}).get("place"))


def _region(xlim, ylim):
    """Allowed placement region. A list of ranges is a union; ``[a, a]`` is a single value,
    so ``xlim: [[-0.2, -0.2], [0, 0], [0.2, 0.2]]`` is a choice of three slots."""
    from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, box
    from shapely.ops import unary_union

    def ranges(values):
        return [values] if values and not isinstance(values[0], (list, tuple)) else list(values)

    shapes = []
    for x0, x1 in ranges(xlim):
        for y0, y1 in ranges(ylim):
            if x0 == x1 and y0 == y1:
                shapes.append(Point(x0, y0))
            elif x0 == x1 or y0 == y1:
                shapes.append(LineString([(x0, y0), (x1, y1)]))
            else:
                shapes.append(box(x0, y0, x1, y1))
    if len(shapes) == 1:
        return shapes[0]
    if all(isinstance(s, Point) for s in shapes):
        return MultiPoint(shapes)
    if all(isinstance(s, LineString) for s in shapes):
        return MultiLineString(shapes)
    return unary_union(shapes)


def _degenerate(values):
    if values is None:
        return True
    if isinstance(values[0], (list, tuple)):
        return False
    return float(values[0]) == float(values[1])


def _scaled_config(metadata, scale):
    import numpy as np

    scale = np.asarray(scale if scale is not None else [1.0, 1.0, 1.0], dtype=float)
    config = deepcopy(metadata)
    bbox = config["geometry"]["oriented_bbox"]
    bbox["vertices"] = (np.asarray(bbox["vertices"], dtype=float) * scale).tolist()
    for data in ((config.get("active") or {}).get("place") or {}).values():
        center = (data.get("projection_circle") or {}).get("center")
        if center is not None:
            data["projection_circle"]["center"] = [*(np.asarray(center[:3]) * scale).tolist(), *center[3:]]
    return config


def _place(generator, metadata, inst, *, rotate_rand, rotate_deg, margin, place_tag, check_mode, region,
           zlim=None, qpos=None, cluttered=False, name="model"):
    ok, pose, _polygon = generator.add_model_from_config(
        config=_scaled_config(metadata, inst.get("scale")),
        place_tag=place_tag,
        rotate_rand=bool(rotate_rand),
        rotate_deg=rotate_deg,
        margin=float(margin or 0.0),
        max_attempts=100,
        name=name,
        allowed_region=region,
        zlim=zlim,
        qpos=qpos,
        cluttered=cluttered,
        check_mode=check_mode or "bbox",
    )
    if ok:
        inst["default_pos"] = [float(v) for v in pose[:3]]
        inst["default_ori"] = [float(v) for v in pose[3:]]
    return ok


def _matrix(pos, ori):
    import numpy as np
    import transforms3d as t3d

    out = np.eye(4)
    out[:3, :3] = t3d.quaternions.quat2mat(ori)
    out[:3, 3] = pos
    return out


def _sample(np, layout, config, objects):
    """Re-place ``layout`` in place. Returns it and the labels of randomized objects."""
    from shapely.geometry import box
    import transforms3d as t3d
    from utils.cluttered_generator import ClutteredGenerator

    _redraw_identities(np, layout, config, objects)
    if layout.get("Table") is None and (layout.get("Dynamic") or {}).get("conveyor"):
        return _sample_conveyor(np, layout)
    old = {id(inst): (list(inst["default_pos"]), list(inst["default_ori"])) for _k, _c, inst in _instances(layout)}

    # Clutter is regenerated below from the task config.
    for kind in OBJECT_TYPES:
        group = layout.get(kind) or {}
        for category in list(group):
            group[category] = [i for i in group[category] if i.get("type") != "cluttered"]
            if not group[category]:
                del group[category]

    table = layout.get("Table")
    if table is None:
        raise RuntimeError("Dev layouts need a table scene; the conveyor tasks are not supported")
    position, scale = table["default_pos"], table["scale"]
    height = position[2] + scale[2] / 2
    generator = ClutteredGenerator()
    generator.reset(
        box(-scale[0] / 2 + TABLE_INSET_M + position[0], -scale[1] / 2 + TABLE_INSET_M + position[1],
            scale[0] / 2 - TABLE_INSET_M + position[0], scale[1] / 2 - TABLE_INSET_M + position[1]),
        frame=np.array([0.0, 0.0, height, 1.0, 0.0, 0.0, 0.0]),
    )
    for area in (config.get("ProhibitedArea") or []) + (config.get("Prohibited_Area") or []):
        generator.add_box_prohibited_area(*[float(v) for v in area])

    # Fixed objects first, then object types in task-config order (a stand before
    # the pieces scattered around it), then the template's order.
    table_items = [(k, c, i) for k, c, i in _instances(layout) if i.get("relative_plane") == "Table"]
    kinds = [k for k in config if k in OBJECT_TYPES]
    fixed = [t for t in table_items if _degenerate(t[2].get("xlim")) and _degenerate(t[2].get("ylim"))
             and not t[2].get("rotate_rand")]
    movable = sorted((t for t in table_items if t not in fixed),
                     key=lambda t: kinds.index(t[0]) if t[0] in kinds else len(kinds))
    for kind, category, inst in fixed + movable:
        metadata = _metadata(objects, kind, category, inst.get("category_idx", 0))
        if not _placeable(metadata, inst):
            raise RuntimeError(f"Asset {kind}/{category}/{inst.get('category_idx', 0):05d} has no readable "
                               "metadata.json with a bbox and place frames (a git-LFS pointer?)")
        if not _place(generator, metadata, inst, rotate_rand=inst.get("rotate_rand"),
                      rotate_deg=inst.get("rotate_deg"), margin=inst.get("margin"),
                      place_tag=inst.get("place_tag"), check_mode=inst.get("check_mode"),
                      region=_region(inst["xlim"], inst["ylim"]), zlim=inst.get("zlim"), qpos=inst.get("qpos"),
                      name=str(inst.get("label", category))):
            raise _PlacementError(f"could not place {inst.get('label', category)}")

    # Objects placed on another object keep their pose relative to it.
    pending = [i for _k, _c, i in _instances(layout) if _parent_key(i) is not None]
    for _round in range(len(pending) + 1):
        remaining = []
        for inst in pending:
            parent = _find_parent(layout, _parent_key(inst))
            if parent is None or parent is inst:
                continue
            if _parent_key(parent) is not None and any(parent is p for p in pending):
                remaining.append(inst)
                continue
            before = _matrix(*old[id(parent)])
            after = _matrix(parent["default_pos"], parent["default_ori"])
            child = after @ np.linalg.inv(before) @ _matrix(*old[id(inst)])
            inst["default_pos"] = child[:3, 3].tolist()
            quaternion = t3d.quaternions.mat2quat(child[:3, :3])
            inst["default_ori"] = (quaternion / np.linalg.norm(quaternion)).tolist()
            old[id(inst)] = (inst["default_pos"], inst["default_ori"])
        if len(remaining) == len(pending):
            break
        pending = remaining

    _regenerate_clutter(np, generator, layout, config, objects)
    return layout


def _regenerate_clutter(np, generator, layout, config, objects):
    import yaml

    task_categories = {c for _k, c, _i in _instances(layout)}
    rigid = layout.setdefault("Rigid", {})
    for clutter_idx, entry in enumerate(config.get("Clutter") or []):
        pool = yaml.safe_load((objects / entry["yaml_path"]).read_text()).get("Clutter") or {}
        categories = sorted(c for c in pool if c not in task_categories)
        mode = entry.get("mode", "allow_duplicate")
        wanted = int(entry.get("nums", 0))
        candidates = []
        order = [categories[i] for i in np.random.permutation(len(categories))]
        if mode == "category_unique":
            candidates = [(c, pool[c][int(np.random.randint(len(pool[c])))]) for c in order]
        else:
            pairs = [(c, int(i)) for c in categories for i in pool[c]]
            if mode == "unique":
                candidates = [pairs[i] for i in np.random.permutation(len(pairs))]
            else:
                candidates = [pairs[int(np.random.randint(len(pairs)))] for _ in range(wanted * CLUTTER_TRIES_PER_OBJECT)]
        placed = 0
        for category, index in candidates[: wanted * CLUTTER_TRIES_PER_OBJECT]:
            if placed >= wanted:
                break
            metadata = _metadata(objects, "Clutter", category, index)
            inst = {"category_idx": int(index), "scale": [1.0, 1.0, 1.0], "type": "cluttered",
                    "clutter_idx": clutter_idx, "yaml_path": entry["yaml_path"],
                    "relative_plane": entry.get("relative_plane", "Table")}
            if not _placeable(metadata, inst):
                continue
            if _place(generator, metadata, inst, rotate_rand=entry.get("rotate_rand"),
                      rotate_deg=entry.get("rotate_deg"), margin=entry.get("margin"), place_tag=None,
                      check_mode="bbox", region=_region(entry["xlim"], entry["ylim"]), cluttered=True,
                      name=f"clutter_{category}"):
                rigid.setdefault(category, []).append(inst)
                placed += 1


def _sample_conveyor(np, layout):
    """Re-place the objects on the conveyor (see the module docstring)."""
    import math

    conveyor = layout["Dynamic"]["conveyor"][0]["default_pos"]
    origins = {"conveyor": (conveyor[0], conveyor[1]),
               "conveyor/conveyor/0": (conveyor[0] + CONVEYOR_BELT_X_OFFSET, conveyor[1])}

    def ranges(lim):
        """[(lo, hi)] choices for one axis: a list of fixed slots or one interval."""
        if lim and isinstance(lim[0], (list, tuple)):
            return [tuple(map(float, item)) for item in lim]
        return [tuple(map(float, lim))]

    def turn(quaternion, degrees):
        half = math.radians(degrees) / 2
        yaw = (math.cos(half), 0.0, 0.0, math.sin(half))
        w1, x1, y1, z1 = yaw
        w2, x2, y2, z2 = quaternion
        return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]

    movers = [inst for kind, _c, inst in _instances(layout) if kind == "Rigid"
              and inst.get("relative_plane") in origins and inst.get("xlim") is not None]
    groups = {}
    for inst in movers:
        groups.setdefault(json.dumps([inst["relative_plane"], inst["xlim"], inst.get("ylim")]), []).append(inst)
    for members in groups.values():
        sample = members[0]
        ox, oy = origins[sample["relative_plane"]]
        xs, ys = ranges(sample["xlim"]), ranges(sample.get("ylim") or [0.0, 0.0])
        slots = len(xs) > 1
        if slots and len(xs) < len(members):
            raise _PlacementError(f"{len(members)} objects for {len(xs)} conveyor slots")
        order = [int(i) for i in np.random.permutation(len(xs))] if slots else []
        placed = []
        for number, inst in enumerate(members):
            old_x, old_y = inst["default_pos"][0], inst["default_pos"][1]
            # Where the template's object was sampled; moving by the difference keeps
            # the asset's own offset from its placement point.
            if slots:
                was_x = min((lo for lo, _hi in xs), key=lambda lo: abs(ox + lo - old_x))
            else:
                was_x = min(max(old_x - ox, xs[0][0]), xs[0][1])
            was_y = min(max(old_y - oy, ys[0][0]), ys[0][1])
            for _try in range(50):
                rel_x = xs[order[number]][0] if slots else float(np.random.uniform(*xs[0]))
                rel_y = float(np.random.uniform(*ys[0]))
                x, y = old_x + (rel_x - was_x), old_y + (rel_y - was_y)
                if slots or all(math.dist((x, y), other) >= CONVEYOR_MIN_GAP_M for other in placed):
                    break
            else:
                raise _PlacementError("no room on the conveyor")
            placed.append((x, y))
            inst["default_pos"] = [x, y, inst["default_pos"][2]]
            inst["default_ori"] = turn(inst["default_ori"], float(np.random.uniform(-CONVEYOR_YAW_DEG,
                                                                                     CONVEYOR_YAW_DEG)))
    return layout


def _coincident_eval(layout, evals):
    """Index of an eval layout this one repeats, or None.

    Two layouts coincide when they have the same labelled objects and every one
    has the same asset, lies within ``MIN_EVAL_DISTANCE_M`` in xy and within
    ``MIN_EVAL_ANGLE_DEG`` in orientation.
    """
    import math

    def objects(data):
        return {(kind, i["label"]): (category, i.get("category_idx"), i["default_pos"], i["default_ori"])
                for kind, category, i in _instances(data) if i.get("label")}

    def angle(q1, q2):
        dot = min(1.0, abs(sum(a * b for a, b in zip(q1, q2))) / (math.hypot(*q1) * math.hypot(*q2)))
        return math.degrees(2 * math.acos(dot))

    mine = objects(layout)
    for index, other in enumerate(evals):
        theirs = objects(other)
        if set(theirs) != set(mine):
            continue
        if all(mine[k][:2] == theirs[k][:2]
               and math.dist(mine[k][2][:2], theirs[k][2][:2]) < MIN_EVAL_DISTANCE_M
               and angle(mine[k][3], theirs[k][3]) < MIN_EVAL_ANGLE_DEG for k in mine):
            return index
    return None
