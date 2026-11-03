"""End-effector tools by object id: Z1's locate, pick, release, transfer and push.

Z1 (archived: tag archive/pre-default-agent, baselines/roboprobe/z1.py) gave a model
calling one tool per turn these tools on RoboProbe's end-effector loop. Here they are
bounded executions like every other skill, for programs:

- locate("red mug; white basket") measures every instance SAM 3 finds in the three
  cameras: a position triangulated from two views that agree within 3 cm, neither cut
  off by its image's border (else: assumed 2 cm above the table), its footprint
  (length, width, long-side angle) from the largest view, which arms reach it straight
  down or tilted, and how the jaws would close on it. The objects become the scene, labelled o1, o2, ... (left to right);
  a close-up of each object is attached to the next turn. The located names stay
  tracked: after every program they are measured again (turn_report, as Z1 did after
  every tool call), relabelled, marked when held, and outlined in the camera images the
  model receives, with each holding arm's grip offset.
- pick(arm, "o2") chooses the grasp from the measurements: across the width when the
  open jaws span it, else at the edge with the most room for the open jaws, about 1 cm
  below a wall's top; the tool straight down, or tilted forward where only that
  reaches. It travels above, re-measures the object from there (correcting by up to
  5 cm; a flat object is grasped at its height), comes down the last 3 cm in 1 cm
  motions, closes, settles, lifts, and checks the grip: whether the object rose with the
  jaws (SAM 3, two cameras), and the jaws' reading where the gripper is measured; the
  wrist camera's look (holding) only when the rise cannot be measured.
- release(arm, into=, onto= or x, y) sets the held object down into a container (its
  open middle, or the nearest point inside it this arm reaches straight down), onto an
  object's top, or at a point, with the grip offset applied (where the object sits
  relative to the jaws) and its lowest point 5 mm above the surface, the last 3 cm in
  1 cm motions; it opens, clears, rises and withdraws the arm over its own side.
- transfer(arm, "o2", ...) is pick, then release when the jaws hold something.
- push(arm, "o2", x, y) slides the object so its centre ends near (x, y): the jaws
  closed and side by side across the push, from behind the object, just clear of the
  table, in straight pieces of at most 3 cm.

Positions are the jaws' centre (the fingertips), in the env frame. These are the default
tools for handling objects; configs/ablations/analytic-manipulation.toml puts the
manipulation skill's pick, place and push in their place (the direct-control rung R1
and RoboTwin use those).
"""

import json
import math
import sys

from harness_policy.geometry import TABLE_Z, at_border, down_with_yaw, mask_to_plane, pixel_ray, pixel_to_plane, \
    tilted_toward, triangulate
from harness_policy.runtime.events import Escalation
from harness_policy.runtime.session import ARMS, CAMERAS


LOCATE_AGREE_M = 0.03      # rays from two cameras pass this close to their 3D point
MAX_INSTANCES = 4          # instances of one description that locate keeps
MAX_DESCRIPTIONS = 4       # descriptions one locate measures, separated by ';'
OVERLAP = 0.5              # masks of one camera sharing more of the smaller one's pixels are one instance
SAME_OBJECT_M = 0.025      # instances of different names closer than this are one object
SAME_PIXELS_IOU = 0.6      # masks of two names this alike in one camera are one object (merged)
FLAT_Z = TABLE_Z + 0.02    # one view only: the instance is assumed this high
CLOSEUP_SIZE, CLOSEUP_MAX = 192, 12
JAW_OPENING_M = 0.088      # the open jaws span this much
JAW_REACH_M = 0.044        # each open jaw is this far from the grasp point
JAW_ROOM_M = 0.02          # pick wants this much room between an open jaw and any other object
WALL_TOP_MIN = TABLE_Z + 0.04  # an edge grasp closes about 1 cm below a wall at least this high
MIN_GRASP_Z = 0.7725       # the lowest straight-down grasp (jaw tips on the table)
REACH_Z = 0.88             # height at which locate checks which arm reaches an object
TILT_DEG = 45.0            # the forward tilt checked when no arm reaches straight down
DEFAULT_LIFT_M, MAX_LIFT_M = 0.10, 0.30
PICK_HOVER_M = 0.06        # pick travels to this far above the grasp height
SLOW_LOWER_M = 0.03        # ... comes down at speed to this far above, then in 1 cm motions
ALIGN_MAX_M = 0.05         # the re-measurement from the hover moves the grasp by at most this
FLAT_TOP_Z, FLAT_MAX_WIDTH_M = TABLE_Z + 0.02, 0.05  # a flat object is grasped at its height
PICK_SETTLE_SECONDS = 0.24  # the closed jaws squeeze this long before the lift
SETUP_AFTER, SETUP_NEAR_M = 2, 0.05  # after two failed picks by one arm at one spot, a third is refused
HELD_NEAR_M = (0.12, 0.10)  # an object this close (horizontally, vertically) to closed jaws is held by them
DEFAULT_HANG_M = 0.05      # how far a held object hangs below the jaws when the pick did not measure it
CLEARANCE_M, PATH_NEAR_M = 0.08, 0.10  # travel passes this far above objects this near its path
MAX_OBJECT_Z = TABLE_Z + 0.25  # higher "objects" (an arm taken for one) are ignored for clearance
PLACE_HOVER_M = 0.10       # release travels at most this far above its opening height ...
RELEASE_GAP_M = 0.005      # ... and opens with the object's lowest point this far above the surface
CLEAR_M = 0.02             # after opening it first rises only this much
CONTAINER_MIN_M, CONTAINER_MAX_M = 0.10, 0.45  # a measured object this wide (and at most this long) is a container
CONTAINER_MARGIN_M = 0.04  # release into a container keeps the object's centre this far inside its walls
CONTAINER_FLOOR_M = 0.015  # a container's floor is taken as this far above the table
PARK_XY = {"left_arm": (-0.30, -0.25), "right_arm": (0.30, -0.25)}  # release withdraws each arm here
PARK_Z = 0.92
PUSH_GAP_M = 0.03          # push starts with the jaws this far behind the object's near side
PUSH_Z = TABLE_Z + 0.015   # the closed jaw tips just clear of the table
PUSH_SEGMENT_M = 0.03      # a push slides in straight pieces this long
PUSH_RANGE_M = (0.005, 0.40)
PUSH_RETRACT_M = 0.05
FREE_SPACE_SPEED = 0.6     # m/s of link6 through free space


def _logged(method):
    """Write each tool's result to the kernel log (stderr outside the program's output)."""
    def wrapper(self, *args, **kwargs):
        result = None
        try:
            result = method(self, *args, **kwargs)
            return result
        finally:
            try:
                print(json.dumps({"skill": method.__name__.removeprefix("eef_"), "step": self.state.get("step"),
                                  "result": result}, default=str), file=sys.__stderr__, flush=True)
            except Exception:
                pass
    wrapper.__name__, wrapper.__doc__ = method.__name__, method.__doc__
    return wrapper


def exports(session):
    """The program's names of the tools (the methods carry an eef_ prefix, so they do not
    shadow the manipulation skill's on the session class)."""
    return {"locate": session.eef_locate, "pick": session.eef_pick, "release": session.eef_release,
            "transfer": session.eef_transfer, "push": session.eef_push, "move_tip": session.eef_move_tip}


def turn_report(session):
    """The scene re-measured after every program (EpisodeCore.drain)."""
    return session._eef_turn_report()


class EefTools:
    # -- locate --------------------------------------------------------------------

    @_logged
    def eef_locate(self, object):
        """Measure the objects named by `object` (up to 4 names separated by ';') with SAM 3
        in every camera. Returns {"objects": [{"id", "name", "aliases", "position",
        "triangulated", "score", "length", "width", "angle_deg", "container", "held_by",
        "reach", "jaws"}], "text"}; the objects (o1, o2, ... left to right) are the scene
        that pick, release, transfer and push take ids from. Names SAM 3 matched to the
        same pixels are one object (the others in its aliases). A close-up of every object
        is attached to your next turn: check each is what you named (SAM 3 matches names
        loosely). The names stay tracked: after every program the scene is measured again
        and relabelled."""
        names = [d.strip() for d in str(object).split(";") if d.strip()][:MAX_DESCRIPTIONS]
        if not names:
            raise ValueError('locate needs a description, e.g. "gray tape dispenser" or "red mug; white basket"')
        observation = self.get_observation(list(CAMERAS))
        objects, outlines, notes = self._eef_scene_from(observation, names)
        # The names stay tracked: every turn re-measures them (turn_report).
        self.__dict__.update(_eef_names=names, _eef_measured={"step": self.state.get("step"), "outlines": outlines,
                                                               "observation": observation})
        try:
            self.show(_closeups(observation.images, outlines), "locate close-ups")
        except Exception:
            pass  # pictures are a help, never a reason to fail
        lines = [self._eef_line(item) for item in objects]
        text = "\n".join(lines + notes) if lines else "\n".join(notes)
        return {"objects": [{k: v for k, v in item.items()} for item in objects], "text": text}

    def _eef_scene_from(self, observation, names):
        """Measure `names` in `observation` and make them the scene: (objects labelled o1,
        o2, ... in the order measured, {camera: [(label, mask)]}, notes). An object measured
        next to jaws that hold something (a pick check passed) is marked as held by them."""
        objects, outlines, notes = [], {}, []
        for name in names:
            found, text = self._eef_measure(observation, name)
            if text:
                notes.append(text)
            for item in found:
                # Two names SAM 3 matched to the same pixels are one object (a locate of
                # "tiara; headband; crown" found one tiara three times, and pick kept its
                # open jaws clear of the other two copies).
                twin = next((kept for kept in objects if kept["name"] != name
                             and _same_pixels(kept["masks"], item["masks"])), None)
                if twin is None:
                    item["aliases"] = []
                    objects.append(item)
                    continue
                if (item["triangulated"], item["score"]) > (twin["triangulated"], twin["score"]):
                    aliases = twin["aliases"] + [twin["name"]]
                    twin.clear()
                    twin.update(item, aliases=aliases)
                else:
                    twin["aliases"].append(name)
        masks = {}
        for number, item in enumerate(objects, 1):
            item["id"] = f"o{number}"
            masks[item["id"]] = item.pop("masks")
            for camera, mask in masks[item["id"]].items():
                outlines.setdefault(camera, []).append((f"{item['id']} {item['name'][:12]}", mask))
        self.__dict__["_eef_masks"] = (observation, masks)  # pick's part grasps take footprints from them
        for i, a in enumerate(objects):
            for b in objects[i + 1:]:
                if a["name"] != b["name"] and math.dist(a["position"][:2], b["position"][:2]) < SAME_OBJECT_M:
                    notes.append(f"{a['id']} ('{a['name']}') and {b['id']} ('{b['name']}') are the same object: "
                                 "the names overlap, so decide from the close-ups which it is.")
        hands = {arm: self._eef_tip(arm) for arm in ARMS if self._gripper_commands.get(arm, 1.0) <= 0.5}
        for item in objects:
            item["held_by"] = next((arm for arm, (hx, hy, hz) in hands.items() if arm in self._eef_grips()
                                    and math.hypot(item["position"][0] - hx, item["position"][1] - hy) <= HELD_NEAR_M[0]
                                    and item["position"][2] > TABLE_Z + 0.04), None)
        self.__dict__["_eef_scene"] = objects
        return objects, outlines, notes

    def _tracked_positions(self):
        """{name: [positions]} of the names the last locate tracks, measured now in every
        camera (no images kept, nothing relabelled), positions triangulated from two
        cameras only (a one-camera position is assumed at a height and jumps as the arms
        hide and show the object): what run_policy's progress watch compares. None before
        a locate or when the cameras cannot be read."""
        names = self.__dict__.get("_eef_names")
        if not names:
            return None
        try:
            observation = self.get_observation(list(CAMERAS))
        except Exception:
            return None  # no measurement judges nothing: the policy goes on
        return {name: [[float(v) for v in item["position"]] for item in self._eef_measure(observation, name)[0]
                       if item.get("triangulated")] for name in names}

    def _eef_turn_report(self):
        """After every program, as Z1 did after every tool call: the objects of the last
        locate measured again now in every camera and relabelled o1, o2, ... (the labels
        the next program's pick, release, transfer and push take), held objects marked,
        the camera images outlined; the grip offset of each arm that holds something, and
        an object that seems to have slipped out. None before the first locate."""
        names = self.__dict__.get("_eef_names")
        if not names:
            return None
        before = {arm: next((o for o in self._eef_scene_objects() if o.get("held_by") == arm), None) for arm in ARMS}
        grips = self._eef_grips()
        for arm in list(grips):  # jaws commanded open since the pick hold nothing any more
            if self._gripper_commands.get(arm, 1.0) > 0.5:
                grips.pop(arm)
        measured = self.__dict__.get("_eef_measured") or {}
        if measured.get("step") == self.state.get("step") and measured.get("observation") is not None:
            observation, outlines = measured["observation"], measured["outlines"]  # nothing moved since locate
            objects = self._eef_scene_objects()
        else:
            observation = self.get_observation(list(CAMERAS))
            objects, outlines, notes = self._eef_scene_from(observation, names)
            self.__dict__["_eef_measured"] = {"step": self.state.get("step"), "outlines": outlines,
                                              "observation": observation}
        lines = [self._eef_line(item) for item in objects] or [f"none of {', '.join(repr(n) for n in names)} was found"]
        text = ["Scene this turn (outlined in the images; these labels are the ones your next program's pick, "
                "release, transfer and push take, and they can change from turn to turn):"] + lines
        for arm, grip in sorted(grips.items()):
            held = next((o for o in objects if o.get("held_by") == arm), None)
            name = (before.get(arm) or {}).get("name")
            # Only a measured height says it lies on the table: one camera's view assumes it
            # (the held scissors, seen by the holding arm's wrist camera only, "lay" there).
            if held is None and name and any(o["name"] == name and o["triangulated"]
                                             and o["position"][2] <= TABLE_Z + 0.04 for o in objects):
                text.append(f"Grip, {arm}: no '{name}' is measured next to its closed jaws, and one lies on the table: "
                            "it may have slipped out; check the images before you release.")
                continue
            text.append(self._eef_grip_text(arm, grip, held))
        images = {}
        try:
            for camera, image in _outlined(observation.images, outlines).items():
                images[camera] = _data_url(image)
        except Exception:
            pass  # pictures are a help, never a reason to fail
        return {"text": "\n".join(text), "images": images}

    def _eef_grip_text(self, arm, grip, held):
        """Where the held object's centre sits relative to the grasp point (measured now in
        the air when two cameras see it next to the jaws) and how far it hangs below them."""
        dx, dy, hang = grip
        hand = self._eef_tip(arm)
        air = held is not None and held.get("triangulated") and held["position"][2] > TABLE_Z + 0.04
        ax, ay = (held["position"][0] - hand[0], held["position"][1] - hand[1]) if air else (dx, dy)
        if air:
            where = (f"measured now in the air, the held object's centre is ({ax:+.3f}, {ay:+.3f}) from the grasp "
                     f"point (on the table before the pick it was ({dx:+.3f}, {dy:+.3f})"
                     + ("; it hangs tilted in the jaws, and lands where it hangs" if math.hypot(ax - dx, ay - dy) > 0.02
                        else "") + ")")
        else:
            where = (f"the held object's centre was ({dx:+.3f}, {dy:+.3f}) from the grasp point on the table before "
                     "the pick (not measured in the air this turn)")
        what = f"{held['id']} '{held['name']}'" if held else "what it picked"
        return (f"Grip, {arm} holds {what}: {where}; release applies this offset. Its lowest point hangs about "
                f"{hang:.3f} m below the grasp point.")

    def _eef_measure(self, observation, name):
        """([instance dicts with masks], note) of one description, left to right."""
        import numpy as np

        found, failures = {}, []
        for camera in CAMERAS:
            if camera not in observation.images:
                continue
            try:
                masks = _distinct([m for m in self.segmenter.segment(observation.images[camera], name)
                                   if m.centroid])[:MAX_INSTANCES]
            except Exception as error:  # a failed segmentation loses that view only
                masks = []
                failures.append(f"{camera}: {type(error).__name__}")
            if masks:
                found[camera] = masks
        if not found:
            if failures:
                return [], (f"SAM 3 failed on every camera ({', '.join(failures)}), so '{name}' could not be measured "
                            "this time; this says nothing about whether it is there.")
            return [], (f"'{name}' was not found in any camera. Describe it differently (a short noun phrase such "
                        "as 'gray tape dispenser'), or move a wrist camera to see it.")

        def spread(point, pixels):
            worst = 0.0
            for camera, (u, v) in pixels.items():
                origin, direction = pixel_ray(observation, camera, u, v)
                offset = np.asarray(point) - origin
                worst = max(worst, float(np.linalg.norm(offset - direction * float(offset @ direction))))
            return worst

        # One instance per mask of the primary camera, paired with the mask of another
        # camera whose ray agrees best; each mask is used once. A mask the image border
        # cuts is not triangulated: its centroid is not the object's, and the point slides
        # along the other ray (the arms' start pose cuts the near row of objects at the
        # bottom of the wrist images: 2-4 cm off, 1-3 cm low). A point higher than any
        # object or held one pairs two different instances of the name.
        primary = "cam_head" if "cam_head" in found else sorted(found)[0]
        ceiling = max(MAX_OBJECT_Z, max(self._eef_tip(arm)[2] for arm in ARMS) + HELD_NEAR_M[1])
        candidates = []
        for i, mask in enumerate(found[primary]):
            if at_border(mask):
                continue
            for camera in (c for c in found if c != primary):
                for j, other in enumerate(found[camera]):
                    if at_border(other):
                        continue
                    pair = {primary: mask.centroid, camera: other.centroid}
                    point = triangulate(observation, pair)
                    error = spread(point, pair)
                    if error <= LOCATE_AGREE_M and TABLE_Z - 0.01 <= point[2] <= ceiling:
                        candidates.append((error, i, camera, j, point))
        instances, used = {}, set()
        for error, i, camera, j, point in sorted(candidates, key=lambda c: c[0]):
            if i in instances or (camera, j) in used:
                continue
            instances[i] = {"point": point, "error": error, "masks": {primary: found[primary][i], camera: found[camera][j]}}
            used.add((camera, j))
        for i, mask in enumerate(found[primary]):
            if i not in instances:
                flat = pixel_to_plane(observation, primary, *mask.centroid, FLAT_Z)
                if flat is not None:
                    instances[i] = {"point": flat, "error": None, "masks": {primary: mask}}
        if not instances:
            return [], f"'{name}' was seen in {primary}, but its position could not be measured."
        items = []
        for item in sorted(instances.values(), key=lambda entry: float(entry["point"][0])):
            x, y, z = (float(v) for v in item["point"])
            shapes = [s for s in (_shape(observation, c, m, z) for c, m in item["masks"].items()) if s]
            shape = max(shapes, key=lambda s: s["length"] * s["width"]) if shapes else None
            items.append({"name": name, "position": [round(x, 4), round(y, 4), round(z, 4)],
                          "triangulated": item["error"] is not None,
                          "score": round(max(m.score for m in item["masks"].values()), 2),
                          "length": None if shape is None else round(shape["length"], 4),
                          "width": None if shape is None else round(shape["width"], 4),
                          "angle_deg": None if shape is None else round(shape["angle_deg"], 1),
                          "seen_in": sorted(item["masks"]), "masks": item["masks"]})
        for item in items:
            item["container"] = _is_container(item)
            item["reach"] = self._eef_reach(*item["position"][:2])
        return items, ""

    def _eef_line(self, item):
        x, y, z = item["position"]
        where = (f"seen in {' and '.join(item['seen_in'])}" if item["triangulated"] else
                 f"seen in {item['seen_in'][0]} only, so assumed {FLAT_Z - TABLE_Z:.2f} m above the table")
        size = ("" if item["width"] is None else
                f", {item['length'] * 100:.0f} x {item['width'] * 100:.0f} cm, long side at {item['angle_deg']:.0f} deg")
        reach = {"down": "straight down by", "tilted": f"tilted forward {TILT_DEG:.0f} deg by"}
        arms = [f"{reach[mode]} the {arm.split('_')[0]} arm" for arm, mode in item["reach"].items() if mode in reach]
        state = (f", held by the {item['held_by']}" if item.get("held_by") else ", a container" if item["container"] else "")
        wide = item["width"] is not None and item["width"] >= JAW_OPENING_M - 0.01
        item["jaws"] = (("too wide for the open jaws: pick takes it by an edge" if item["container"] else
                         "too wide for the open jaws: pick takes a thin part of it, else an edge")
                        if wide else "the jaws close across its width")
        also = (f" (also what {', '.join(repr(a) for a in item['aliases'])} matched: the same pixels)"
                if item.get("aliases") else "")
        return (f"{item['id']} '{item['name']}'{also} at ({x:.3f}, {y:.3f}, {z:.3f}) ({where}; SAM 3 match {item['score']:.2f})"
                f"{size}{state}; reached {', '.join(arms) if arms else 'by neither arm, even tilted'}; {item['jaws']}.")

    def _eef_reach(self, x, y):
        """{arm: "down" | "tilted" | None}: which arms reach (x, y) straight down at REACH_Z,
        or only tilted forward by TILT_DEG."""
        modes = {}
        for arm in ARMS:
            modes[arm] = None
            for mode, orientation in (("down", down_with_yaw(0.0)), ("tilted", tilted_toward(math.pi / 2, 90 - TILT_DEG))):
                try:
                    if self.ik(arm, self.link6_for_tip([x, y, REACH_Z], orientation)) is not None:
                        modes[arm] = mode
                        break
                except Exception:
                    modes[arm] = "unknown"
                    break
        return modes

    # -- pick ----------------------------------------------------------------------

    @_logged
    def eef_pick(self, arm, object, grasp="auto", lift=None):
        """Grasp the scene object `object` (an id from the last locate) with `arm` and lift
        it. grasp: "auto" (across the width if the open jaws span it; else across a thin part
        of it, a handle or a band, unless it is a container; else by the edge with the most
        room for them), "across", "part", "edge" or "tilted" (by the near side, the tool
        tilted forward). lift: how far to rise (at least 0.10 m, plus half the width for an
        edge grasp). Returns {"ok", "arm", "object", "grasp", "grasp_point", "check",
        "reason"}; ok is whether the jaws hold it after the lift."""
        arm = _arm_name(arm)
        obj = self._eef_object(object)
        if obj.get("held_by"):
            return {"ok": False, "arm": arm, "object": obj["id"], "reason": f"already held by the {obj['held_by']}"}
        if grasp not in ("auto", "across", "part", "edge", "tilted"):
            raise ValueError("grasp must be auto, across, part, edge or tilted")
        plan = self._eef_grasp(obj, arm, grasp)
        result = {"ok": False, "arm": arm, "object": obj["id"], "grasp": plan["how"]}
        if plan["pitch_deg"] is None:
            return {**result, "reason": f"beyond the {arm}'s reach, even tilted: use the other arm, or push it nearer"}
        failures = self.__dict__.setdefault("_eef_failed_spots", {}).setdefault(arm, [])
        if sum(math.hypot(sx - plan["x"], sy - plan["y"]) <= SETUP_NEAR_M for sx, sy in failures) >= SETUP_AFTER:
            return {**result, "reason": (f"two picks with the {arm} have already failed within {SETUP_NEAR_M * 100:.0f} "
                                         "cm of here: set up first (push it into the open, clear its neighbour, or use "
                                         "the other arm), then pick it where it is easy")}
        last = self.__dict__.get("_eef_last_failed", {}).get(arm)
        pose = (plan["x"], plan["y"], plan["yaw_deg"], plan["grasp_z"], plan["pitch_deg"])
        if last and all(abs(a - b) <= tol for a, b, tol in zip(pose, last, (0.003, 0.003, 3.0, 0.003, 3.0))):
            return {**result, "reason": "the same pick just failed with this arm: change what made it fail first"}
        need = DEFAULT_LIFT_M + ((obj["width"] or 0.0) / 2 if plan["how"].startswith("by its edge") else 0.0)
        lift = min(MAX_LIFT_M, max(need, float(lift) if lift is not None else 0.0))
        orientation = self._eef_orientation(plan["yaw_deg"], plan["pitch_deg"])
        if self._gripper_commands.get(arm, 1.0) < 0.999:
            self._eef_quiet(self.open_gripper, arm)
        here = self._eef_tip(arm)
        hover = max(plan["grasp_z"] + PICK_HOVER_M,
                    self._eef_clearance(here[:2], (plan["x"], plan["y"]), skip=obj))
        trace = self._eef_travel(arm, [plan["x"], plan["y"], hover], orientation)
        if not trace.ok:
            return {**result, "reason": f"could not get above it: {trace.event.kind} {trace.event.detail}"}
        grasp_z, note = self._eef_align(obj, arm, plan)
        x, y = plan["x"], plan["y"]
        steps = [grasp_z + SLOW_LOWER_M] + [round(grasp_z + d, 4) for d in (0.02, 0.01, 0.0)]
        for z in steps:
            trace = self._eef_quiet(self.move_eef, arm, self.link6_for_tip([x, y, z], orientation),
                                    speed=FREE_SPACE_SPEED if z == steps[0] else None)
            if not trace.ok and trace.event.kind != "BLOCKED":
                return {**result, "reason": f"the descent stopped: {trace.event.kind} {trace.event.detail}"}
            if trace.event.kind == "BLOCKED":  # contact: close where it is
                break
        self._eef_quiet(self.close_gripper, arm)
        self._eef_quiet(self.hold, PICK_SETTLE_SECONDS)
        before = obj["position"]
        self._eef_quiet(self.move_eef, arm, self.link6_for_tip([x, y, grasp_z + lift], orientation),
                        speed=FREE_SPACE_SPEED)
        held, check = self._eef_pick_check(obj, arm, lift)
        grips = self._eef_grips()
        if held:
            grips[arm] = (before[0] - x, before[1] - y, max(0.0, grasp_z - TABLE_Z))
            obj["held_by"] = arm
            self.__dict__.setdefault("_eef_last_failed", {}).pop(arm, None)
        else:
            grips.pop(arm, None)
            failures.append((x, y))
            self.__dict__.setdefault("_eef_last_failed", {})[arm] = pose
        return {**result, "ok": held, "grasp_point": [round(x, 4), round(y, 4), round(grasp_z, 4)],
                "yaw_deg": round(plan["yaw_deg"], 1), "pitch_deg": plan["pitch_deg"], "lift": round(lift, 3),
                "check": check, **({"note": note} if note else {}),
                **({} if held else {"reason": "not held after the lift"})}

    def _eef_pick_check(self, obj, arm, lift):
        """(held, check) after the lift. With a measured gripper: the jaws' reading, then
        whether the object rose with them. Without one (the XPolicyLab contract: the
        gripper value is the last command) the evidence is whether the object rose with the
        jaws, measured with SAM 3 from two cameras (Z1's pick check); the wrist camera's
        look (holding()) decides only when that cannot be measured. Asking the look first
        and letting go on its NO lost objects the jaws did hold."""
        name = obj["name"]
        if self._measured_gripper():
            held = self.holding(arm)
            check = "the jaws hold something" if held else "the jaws closed on nothing"
            rose = self._eef_rose(obj, arm, lift) if held else None
            if rose is False:
                return False, check + f", but no '{name}' rose with them: NOT held"
            return held, check + (f", and '{name}' rose with them" if rose else "")
        rose = self._eef_rose(obj, arm, lift)
        if rose is True:
            return True, f"'{name}' rose with the jaws (measured with SAM 3): held"
        if rose is False:
            return False, f"the '{name}' nearest the jaws did not rise with them (measured with SAM 3): NOT held"
        held = self.holding(arm)
        return held, (f"'{name}' could not be measured next to the jaws; the wrist camera's look says "
                      + ("they hold something" if held else "they closed on nothing"))

    def _eef_grasp(self, obj, arm, choice):
        """Grasp point, yaw, height and tilt from the object's measurements."""
        x, y, z = obj["position"]
        a = math.radians(obj["angle_deg"] or 0.0)
        short = (-math.sin(a), math.cos(a))
        wide = obj["width"] is not None and obj["width"] >= JAW_OPENING_M - 0.01
        part = (self._eef_part(obj, arm) if (wide and choice == "auto" and not obj.get("container"))
                or choice == "part" else None)
        if part is not None:
            # A flat spatula or bag by its edge closed above it (general_pickup layouts 3
            # and 4); the analytic pick took the spatula by its handle.
            x, y, yaw = part
            grasp_z, how = max(MIN_GRASP_Z, TABLE_Z + (z - TABLE_Z) / 2.0), "across a thin part of it (a handle, a band)"
        elif (wide and choice != "across") or choice in ("edge", "tilted", "part"):
            base = self.profile["arm_bases"][arm]
            half = (obj["width"] or 0.04) / 2
            yaw = (math.degrees(math.atan2(short[1], short[0])) + 90.0) % 180.0 - 90.0
            edges = [(x + s * half * short[0], y + s * half * short[1]) for s in (1.0, -1.0)]
            scored = [(min(self._eef_jaw_room(obj, ex, ey, yaw)[0], JAW_ROOM_M), -math.hypot(ex - base[0], ey - base[1]),
                       (ex, ey)) for ex, ey in edges]
            x, y = max(scored)[2]
            grasp_z, how = max(z, WALL_TOP_MIN) - 0.01, "by its edge with the most room for the jaws (too wide for them)"
            if choice == "part":
                how += "; no thin part of it fits the jaws"
        else:
            yaw = ((obj["angle_deg"] or 0.0) + 180.0) % 180.0 - 90.0
            grasp_z, how = max(MIN_GRASP_Z, TABLE_Z + (z - TABLE_Z) / 2.0), "across its width"
        room, nearest = self._eef_jaw_room(obj, x, y, yaw)
        if nearest is not None and room < JAW_ROOM_M:
            how += (f"; an open jaw comes within {max(room, 0.0) * 100:.0f} cm of {nearest['id']} '{nearest['name']}'"
                    + (" (it would land on it)" if room <= 0 else "") + ": if it fails, push one of them apart first")
        mode = self._eef_reach(x, y).get(arm)
        pitch = 0.0 if mode in ("down", "unknown") else (TILT_DEG if mode == "tilted" else None)
        if choice == "tilted":
            pitch, how = TILT_DEG, how + ", tilted"
        return {"x": x, "y": y, "yaw_deg": yaw, "grasp_z": grasp_z, "pitch_deg": pitch, "how": how}

    def _eef_part(self, obj, arm):
        """(x, y, yaw_deg) of a straight-down grasp across a thin part of the object (a
        handle, a band) whose jaws land clear of the rest of it, from the grasping skill's
        part grasps on its footprint, the most substantial part first; None without one the
        arm reaches straight down."""
        points = self._eef_footprint(obj)
        if points is None:
            return None
        base = self.profile["arm_bases"][arm][:2]
        for part in self._part_grasps(points, float(points[0][2]), 0.0, base):
            px, py = part["tip"][:2]
            if self._eef_reach(px, py).get(arm) == "down":
                return px, py, math.degrees(part["yaw"])
        return None

    def _eef_footprint(self, obj):
        """The object's footprint: its mask (the head camera's first) projected on the plane
        at its height, as points; None without a mask."""
        observation, masks = self.__dict__.get("_eef_masks") or (None, {})
        found = masks.get(obj["id"]) or {}
        camera = "cam_head" if "cam_head" in found else next(iter(sorted(found)), None)
        if observation is None or camera is None:
            return None
        points = mask_to_plane(observation, camera, found[camera], z=obj["position"][2])
        return points if len(points) >= 10 else None

    def _eef_jaw_room(self, obj, x, y, yaw):
        """(room, nearest): how far the nearer open jaw at (x, y) with this yaw is from the
        footprint of the nearest other scene object (negative: it lands on it)."""
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        room, nearest = 1.0, None
        for other in self._eef_scene_objects():
            if other is obj or other.get("held_by"):
                continue
            radius = max(other["length"] or 0.04, other["width"] or 0.04) / 2
            for side in (1.0, -1.0):
                gap = math.hypot(x + side * JAW_REACH_M * c - other["position"][0],
                                 y + side * JAW_REACH_M * s - other["position"][1]) - radius
                if gap < room:
                    room, nearest = gap, other
        return room, nearest

    def _eef_align(self, obj, arm, plan):
        """From above the grasp, measure the object again (the wrist camera now looks down on
        it) and move the grasp by how far it turned out to be, keeping the offset (not for
        a grasp across a thin part: the whole object's centre measured from another view
        says little about where its handle is, and a spatula's handle grasp moved 18 mm and
        missed it, general_pickup layout 4); a flat narrow object is grasped at its
        measured height. Returns (grasp_z, note)."""
        grasp_z, note = plan["grasp_z"], ""
        try:
            observation = self.get_observation([f"cam_{arm.split('_')[0]}_wrist", "cam_head"])
            found, _ = self._eef_measure(observation, obj["name"])
        except Exception:
            return grasp_z, note
        near = [item for item in found if item["position"][2] <= MAX_OBJECT_Z]
        if not near:
            return grasp_z, note
        new = min(near, key=lambda item: math.dist(item["position"][:2], obj["position"][:2]))
        dx, dy = new["position"][0] - obj["position"][0], new["position"][1] - obj["position"][1]
        if math.hypot(dx, dy) > ALIGN_MAX_M:
            return grasp_z, note
        part = plan["how"].startswith("across a thin part")
        if math.hypot(dx, dy) >= 0.003 and part:
            note = (f"re-measured from above, the object's centre was {dx * 1000:+.0f}, {dy * 1000:+.0f} mm off; "
                    "the grasp stays on the thin part as planned")
        elif math.hypot(dx, dy) >= 0.003:
            plan["x"], plan["y"] = plan["x"] + dx, plan["y"] + dy
            obj["position"] = [round(obj["position"][0] + dx, 4), round(obj["position"][1] + dy, 4), obj["position"][2]]
            note = f"re-measured from above, the object was {dx * 1000:+.0f}, {dy * 1000:+.0f} mm off; the grasp followed"
        top = new["position"][2]
        if new["triangulated"] and not obj.get("triangulated"):
            # Seen by one camera at locate, so its height was assumed: the wrist camera now
            # looks down on it, and with the head camera it is measured.
            obj["position"][2], obj["triangulated"] = round(top, 4), True
            grasp_z = (max(top, WALL_TOP_MIN) - 0.01 if plan["how"].startswith("by its edge")
                       else max(MIN_GRASP_Z, TABLE_Z + (top - TABLE_Z) / 2.0))
            note = ((note + "; " if note else "") + f"seen by one camera before, its height is measured from above "
                    f"now (z = {top:.3f}): the jaws close at z = {grasp_z:.3f}")
        narrow = new["width"] is not None and new["width"] < FLAT_MAX_WIDTH_M
        if new["triangulated"] and narrow and top < FLAT_TOP_Z and grasp_z > top - 0.002:
            grasp_z = max(MIN_GRASP_Z, top - 0.005)
            note = (note + "; " if note else "") + f"a flat object: closed at its height, z = {grasp_z:.3f}"
        return grasp_z, note

    def _eef_rose(self, obj, arm, lift):
        """Whether an instance of the object's name next to the jaws rose by half the lift;
        None when it cannot be told (not seen by two cameras near the jaws)."""
        try:
            found, _ = self._eef_measure(self.get_observation(list(CAMERAS)), obj["name"])
        except Exception:
            return None
        hand = self._eef_tip(arm)
        near = [item for item in found if item["triangulated"]
                and math.hypot(item["position"][0] - hand[0], item["position"][1] - hand[1]) <= HELD_NEAR_M[0]]
        if not near:
            return None
        nearest = min(near, key=lambda item: math.hypot(item["position"][0] - hand[0], item["position"][1] - hand[1]))
        return nearest["position"][2] - obj["position"][2] >= lift / 2

    # -- release -------------------------------------------------------------------

    @_logged
    def eef_release(self, arm, into=None, onto=None, x=None, y=None, withdraw=True):
        """Set down what `arm` holds: into a container (a scene id: its open middle, or the
        nearest spot inside it this arm reaches straight down; resting on its floor), onto
        an object (a scene id: centred on its top), or at the table point (x, y). The
        object's centre goes there (the grip offset from the pick applied) and it opens
        with the object's lowest point 5 mm above the surface. To cover an object with an
        upside-down container, give that object's x, y (onto sets the container on top of
        it). withdraw (default True) moves the arm over its own side afterwards. Returns
        {"ok", "arm", "where", "release_z", "reason"}."""
        arm = _arm_name(arm)
        landing = self._eef_landing(arm, into, onto, x, y)
        lx, ly, surface, words = landing
        result = {"ok": False, "arm": arm, "where": words}
        grips = self._eef_grips()
        grip = grips.get(arm)
        if grip is None:
            if not self.holding(arm):
                return {**result, "reason": f"the {arm} holds nothing"}
            grip = (0.0, 0.0, DEFAULT_HANG_M)  # held, but not picked by pick (a learned policy closed the jaws)
        dx, dy, hang = self._eef_air_offset(arm, grip)
        release_z = surface + hang + RELEASE_GAP_M
        gx, gy = lx - dx, ly - dy
        here = self._eef_tip(arm)
        orientation = list(self.state[f"{arm.split('_')[0]}_ee_pose"][3:])
        if self._eef_reach(gx, gy).get(arm) == "tilted":
            orientation = self._eef_orientation(0.0, TILT_DEG)
        held = next((o for o in self._eef_scene_objects() if o.get("held_by") == arm), None)
        extent = hang + ((held["width"] or 0.0) / 2 if held else 0.0)
        on_way = [o for o in self._eef_scene_objects() if math.dist(o["position"][:2], (lx, ly)) > PATH_NEAR_M]
        clear = self._eef_clearance(here[:2], (gx, gy), skip=held, objects=on_way)
        height = max(min(here[2], release_z + PLACE_HOVER_M), clear + extent if clear else 0.0)
        trace = None
        for hover in dict.fromkeys((height, max(release_z + 0.06, clear), max(release_z + 0.03, clear))):
            if hover > height:
                continue
            trace = self._eef_travel(arm, [gx, gy, hover], orientation)
            if trace.ok:
                break
        if trace is None or not trace.ok:
            return {**result, "reason": f"could not get above the spot: {trace.event.kind if trace else 'no height'}"}
        lower = [release_z + SLOW_LOWER_M] + [round(release_z + d, 4) for d in (0.02, 0.01, 0.0)]
        for z in lower:
            if z > self._eef_tip(arm)[2] + 0.003:
                continue
            trace = self._eef_quiet(self.move_eef, arm, self.link6_for_tip([gx, gy, z], orientation))
            if trace.event.kind == "BLOCKED":  # it touched down
                break
        self._eef_quiet(self.open_gripper, arm)
        top = self._eef_tip(arm)[2]
        self._eef_quiet(self.move_eef, arm, self.link6_for_tip([gx, gy, top + CLEAR_M], orientation))
        self._eef_quiet(self.move_eef, arm, self.link6_for_tip([gx, gy, top + DEFAULT_LIFT_M], orientation),
                        speed=FREE_SPACE_SPEED)
        if withdraw:
            park = PARK_XY[arm]
            self._eef_quiet(self.move_eef, arm, self.link6_for_tip([park[0], park[1], max(PARK_Z, top + DEFAULT_LIFT_M)],
                                                                   orientation), speed=FREE_SPACE_SPEED)
        grips.pop(arm, None)
        if held is not None:
            held.update({"held_by": None, "position": [round(lx, 4), round(ly, 4), round(surface + 0.01, 4)]})
        return {**result, "ok": True, "release_z": round(release_z, 4)}

    def _eef_landing(self, arm, into, onto, x, y):
        """(x, y, surface_z, words) where release sets the object down."""
        if into is not None or onto is not None:
            target = self._eef_object(into if into is not None else onto)
            if target.get("held_by"):
                raise ValueError(f"{target['id']} is being held")
            tx, ty = target["position"][:2]
            if into is not None:
                surface = TABLE_Z + CONTAINER_FLOOR_M
                if self._eef_reach(tx, ty).get(arm) not in ("down", "unknown") and target["width"] is not None:
                    base = self.profile["arm_bases"][arm]
                    span = math.hypot(tx - base[0], ty - base[1]) or 1.0
                    room = max(0.0, min(target["width"], target["length"]) / 2 - CONTAINER_MARGIN_M)
                    for back in [0.02 * k for k in range(1, int(room / 0.02) + 1)]:
                        k = (span - back) / span
                        spot = (base[0] + k * (tx - base[0]), base[1] + k * (ty - base[1]))
                        if self._eef_reach(*spot).get(arm) == "down":
                            tx, ty = spot
                            break
            else:
                surface = target["position"][2]
            return tx, ty, surface, f"{'into' if into is not None else 'onto'} {target['id']} '{target['name']}'"
        if x is None or y is None:
            raise ValueError("release needs into or onto (a scene id), or x and y")
        return float(x), float(y), TABLE_Z, f"at ({float(x):.3f}, {float(y):.3f})"

    def _eef_air_offset(self, arm, grip):
        """The grip offset measured now, in the air (a held object often hangs tilted, which
        moves its centre), when two cameras see it next to the jaws; else the pick's."""
        dx, dy, hang = grip
        held = next((o for o in self._eef_scene_objects() if o.get("held_by") == arm), None)
        if held is None:
            return dx, dy, hang
        try:
            found, _ = self._eef_measure(self.get_observation(list(CAMERAS)), held["name"])
        except Exception:
            return dx, dy, hang
        hand = self._eef_tip(arm)
        air = [item for item in found if item["triangulated"] and item["position"][2] > TABLE_Z + 0.04
               and math.hypot(item["position"][0] - hand[0], item["position"][1] - hand[1]) <= HELD_NEAR_M[0]]
        if not air:
            return dx, dy, hang
        near = min(air, key=lambda item: math.hypot(item["position"][0] - hand[0], item["position"][1] - hand[1]))
        return near["position"][0] - hand[0], near["position"][1] - hand[1], hang

    # -- transfer and push -----------------------------------------------------------

    @_logged
    def eef_transfer(self, arm, object, into=None, onto=None, x=None, y=None, grasp="auto", withdraw=True):
        """pick, then, if the jaws hold it, release: one call for moving an object into a
        container or onto an open spot. Returns {"ok", "pick", "release"}; it stops after a
        pick that holds nothing. Use pick and release apart where the placement must be
        precise, to see how the object hangs first."""
        pick = self.eef_pick(arm, object, grasp=grasp)
        if not pick["ok"]:
            return {"ok": False, "pick": pick, "release": None}
        release = self.eef_release(arm, into=into, onto=onto, x=x, y=y, withdraw=withdraw)
        return {"ok": release["ok"], "pick": pick, "release": release}

    @_logged
    def eef_push(self, arm, object, x, y):
        """Push the scene object `object` along the table so its centre ends near (x, y): the
        jaws close, go behind it on the far side from (x, y), come down just clear of the
        table and slide in straight pieces of at most 3 cm, then lift clear. Returns {"ok",
        "arm", "object", "start", "end", "reason"}; locate it again to see where it went."""
        arm = _arm_name(arm)
        obj = self._eef_object(object)
        tx, ty = float(x), float(y)
        ox, oy = obj["position"][:2]
        distance = math.hypot(tx - ox, ty - oy)
        result = {"ok": False, "arm": arm, "object": obj["id"]}
        if not PUSH_RANGE_M[0] <= distance <= PUSH_RANGE_M[1]:
            return {**result, "reason": f"a push moves an object {PUSH_RANGE_M[0] * 100:.1f} to "
                                        f"{PUSH_RANGE_M[1] * 100:.0f} cm; this one is {distance * 100:.1f} cm"}
        ux, uy = (tx - ox) / distance, (ty - oy) / distance
        radius = max(obj["length"] or 0.06, obj["width"] or 0.06) / 2
        start = (ox - ux * (radius + PUSH_GAP_M), oy - uy * (radius + PUSH_GAP_M))
        end = (tx - ux * radius, ty - uy * radius)
        orientation = down_with_yaw(math.atan2(uy, ux) + math.pi / 2)  # jaws side by side across the push
        if self._gripper_commands.get(arm, 1.0) > 0.001:
            self._eef_quiet(self.close_gripper, arm)
        here = self._eef_tip(arm)
        hover = max(here[2], TABLE_Z + 0.10)
        trace = self._eef_travel(arm, [start[0], start[1], hover], orientation)
        if not trace.ok:
            return {**result, "reason": f"could not get behind it: {trace.event.kind} {trace.event.detail}"}
        self._eef_quiet(self.move_eef, arm, self.link6_for_tip([start[0], start[1], PUSH_Z], orientation))
        pieces = max(1, math.ceil(math.hypot(end[0] - start[0], end[1] - start[1]) / PUSH_SEGMENT_M))
        reached = 0
        for i in range(1, pieces + 1):
            point = [start[0] + (end[0] - start[0]) * i / pieces, start[1] + (end[1] - start[1]) * i / pieces, PUSH_Z]
            trace = self._eef_quiet(self.move_eef, arm, self.link6_for_tip(point, orientation))
            if not trace.ok:
                break
            reached = i
        tip = self._eef_tip(arm)
        self._eef_quiet(self.move_eef, arm, self.link6_for_tip([tip[0], tip[1], PUSH_Z + PUSH_RETRACT_M], orientation))
        obj["position"] = [round(tx, 4), round(ty, 4), obj["position"][2]]
        return {**result, "ok": reached == pieces, "start": [round(v, 3) for v in start],
                "end": [round(v, 3) for v in end], "pieces": f"{reached}/{pieces}",
                **({} if reached == pieces else {"reason": f"stopped after {reached} of {pieces} pieces: "
                                                           f"{trace.event.kind}"})}

    # -- free motion -----------------------------------------------------------------

    @_logged
    def eef_move_tip(self, arm, x, y, z, yaw_deg=0.0, pitch_deg=0.0, gripper=None, speed=FREE_SPACE_SPEED):
        """Free motion in the tools' own terms (Z2's move_eef): move `arm`'s jaws' centre to
        (x, y, z), the tool straight down with the jaws closing along yaw_deg (0: along
        world x, 90: along world y), or tilted forward by pitch_deg (toward +y; 90 is
        horizontal); gripper (0 closed, 1 open) is set first when given. For free motion
        only (repositioning, a better wrist view, carrying high and clear); touch objects
        with pick, push and release. Returns {"ok", "arm", "tip", "arrival_m", "event"}:
        arrival_m is how far the jaws ended from the target (more than a centimetre near
        objects means something was touched)."""
        arm = _arm_name(arm)
        if gripper is not None:
            self._eef_quiet(self.open_gripper if float(gripper) >= 0.5 else self.close_gripper, arm)
        target = [float(x), float(y), float(z)]
        orientation = self._eef_orientation(float(yaw_deg), float(pitch_deg))
        trace = self._eef_quiet(self.move_eef, arm, self.link6_for_tip(target, orientation), speed=speed)
        tip = self._eef_tip(arm)
        return {"ok": trace.ok, "arm": arm, "tip": [round(v, 4) for v in tip],
                "arrival_m": round(math.dist(tip, target), 4), "event": trace.event.kind,
                **({} if trace.ok else {"detail": trace.event.detail})}

    # -- helpers ---------------------------------------------------------------------

    def _eef_scene_objects(self):
        return self.__dict__.get("_eef_scene", [])

    def _eef_grips(self):
        """arm -> (dx, dy, hang): the held object's centre minus the grasp point, and how far
        its lowest point hangs below the jaws, from the pick."""
        return self.__dict__.setdefault("_eef_grip", {})

    def _eef_object(self, ref):
        oid = ref.get("id") if isinstance(ref, dict) else str(ref).strip()
        found = next((o for o in self._eef_scene_objects() if o["id"] == oid), None)
        if found is None:
            raise ValueError(f"No object {oid!r} in the scene; locate first (ids: "
                             f"{', '.join(o['id'] for o in self._eef_scene_objects()) or 'none'})")
        return found

    def _eef_tip(self, arm):
        """The jaws' centre of `arm` now (x, y, z)."""
        return tuple(self.fingertip_of(self.state[f"{arm.split('_')[0]}_ee_pose"])[:3])

    @staticmethod
    def _eef_orientation(yaw_deg, pitch_deg):
        """Straight down with the jaws closing along yaw_deg, or tilted forward by pitch_deg."""
        if pitch_deg:
            return tilted_toward(math.pi / 2, 90.0 - pitch_deg)
        return down_with_yaw(math.radians(yaw_deg))

    def _eef_clearance(self, start, end, skip=None, objects=None):
        """Lowest height that passes CLEARANCE_M above every scene object within PATH_NEAR_M of
        the straight path from start to end (x, y), other than `skip`; 0 when none."""
        sx, sy = start
        ex, ey = end
        dx, dy = ex - sx, ey - sy
        length2 = dx * dx + dy * dy
        height = 0.0
        for other in self._eef_scene_objects() if objects is None else objects:
            px, py, pz = other["position"]
            if other is skip or other.get("held_by") or pz > MAX_OBJECT_Z:
                continue
            s = 0.0 if length2 == 0 else max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length2))
            if math.hypot(px - (sx + s * dx), py - (sy + s * dy)) <= PATH_NEAR_M:
                height = max(height, pz + CLEARANCE_M)
        return height

    def _eef_travel(self, arm, tip, orientation):
        """Move through free space to `tip`: rise first when the arm is lower, then across;
        once more by way of the parking spot when the planner refuses."""
        here = self._eef_tip(arm)
        if tip[2] - here[2] >= 0.003:
            self._eef_quiet(self.move_eef, arm, self.link6_for_tip([here[0], here[1], tip[2]],
                                                                   self.state[f"{arm.split('_')[0]}_ee_pose"][3:]),
                            speed=FREE_SPACE_SPEED)
        trace = self._eef_quiet(self.move_eef, arm, self.link6_for_tip(tip, orientation), speed=FREE_SPACE_SPEED)
        if trace.ok or trace.event.kind != "SKILL_ERROR":
            return trace
        park = PARK_XY[arm]
        self._eef_quiet(self.move_eef, arm, self.link6_for_tip([park[0], park[1], max(PARK_Z, tip[2])], down_with_yaw(0.0)),
                        speed=FREE_SPACE_SPEED)
        return self._eef_quiet(self.move_eef, arm, self.link6_for_tip(tip, orientation), speed=FREE_SPACE_SPEED)

    @staticmethod
    def _eef_quiet(skill, *args, **kwargs):
        """Run a skill inside a tool: an event no handler resolves becomes the returned trace
        instead of ending the program (EpisodeEnded still does)."""
        try:
            return skill(*args, **kwargs)
        except Escalation as escalation:
            return escalation.trace


def _arm_name(arm):
    name = str(arm)
    name = name if name.endswith("_arm") else f"{name}_arm"
    if name not in ARMS:
        raise ValueError(f"arm must be one of {list(ARMS)} (or left, right)")
    return name


def _is_container(item):
    return item["width"] is not None and item["width"] >= CONTAINER_MIN_M and item["length"] <= CONTAINER_MAX_M


def _same_pixels(a, b, iou=SAME_PIXELS_IOU):
    """Whether two instances' masks ({camera: Mask}) cover the same pixels in a camera
    both were seen in: intersection over union at least `iou` (an object in a holder or a
    basket covers a small part of it, so it stays a separate object)."""
    for camera in set(a) & set(b):
        x, y = a[camera], b[camera]
        if x.array.shape != y.array.shape:
            continue
        union = (x.array | y.array).sum()
        if union and (x.array & y.array).sum() >= iou * union:
            return True
    return False


def _distinct(masks, overlap=OVERLAP):
    """Masks by score, dropping any that shares more than `overlap` of the smaller one's
    pixels with a better one: SAM 3 often returns a pile and its top object, or one
    object twice."""
    kept = []
    for mask in sorted(masks, key=lambda m: -m.score):
        if all(mask.array.shape != k.array.shape
               or (mask.array & k.array).sum() <= overlap * min(mask.area, k.area) for k in kept):
            kept.append(mask)
    return kept


def _shape(observation, camera, mask, z):
    """Footprint of a mask projected on the plane at height z: length, width and the long
    side's angle (degrees from world x), or None."""
    import numpy as np

    flat = mask_to_plane(observation, camera, mask, z=z)
    if len(flat) < 10:
        return None
    xy = flat[:, :2] - flat[:, :2].mean(axis=0)
    _values, vectors = np.linalg.eigh(np.cov(xy.T))
    long_axis, short_axis = vectors[:, 1], vectors[:, 0]
    length = float(np.percentile(xy @ long_axis, 97) - np.percentile(xy @ long_axis, 3))
    width = float(np.percentile(xy @ short_axis, 97) - np.percentile(xy @ short_axis, 3))
    angle = (math.degrees(math.atan2(long_axis[1], long_axis[0])) + 90.0) % 180.0 - 90.0
    return {"length": length, "width": width, "angle_deg": angle}


def _outlined(images, outlines):
    """Copies of the camera images with each object's mask outlined, its centre marked and
    its label written next to it."""
    import cv2
    import numpy as np

    out = {}
    for camera, items in outlines.items():
        if camera not in images:
            continue
        image = np.ascontiguousarray(np.asarray(images[camera]).copy())
        for label, mask in items:
            if mask.array.shape != image.shape[:2]:
                continue
            contours, _ = cv2.findContours(mask.array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
            u, v = (int(round(c)) for c in mask.centroid)
            cv2.drawMarker(image, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
            cv2.putText(image, label, (max(0, u + 8), max(14, v - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        out[camera] = image
    return out


def _data_url(image):
    """An RGB array as a JPEG data URL (the form camera images reach the model in)."""
    import base64

    import cv2
    import numpy as np

    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Could not encode the image")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode()


def _closeups(images, outlines):
    """One image of close-up crops, each object from the camera that shows it largest,
    labelled, so the model can check what each detection is."""
    import cv2
    import numpy as np

    best = {}
    for camera, items in outlines.items():
        if camera not in images:
            continue
        for label, mask in items:
            if mask.area and (label not in best or mask.area > best[label][1].area):
                best[label] = (camera, mask)
    tiles = []
    for label, (camera, mask) in list(best.items())[:CLOSEUP_MAX]:
        image = np.asarray(images[camera])
        x0, y0, x1, y1 = mask.box
        side = max(x1 - x0, y1 - y0) * 1.6 + 24
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        crop = image[b:int(min(image.shape[0], cy + side / 2)), a:int(min(image.shape[1], cx + side / 2))]
        if crop.size == 0:
            continue
        tile = cv2.resize(np.ascontiguousarray(crop), (CLOSEUP_SIZE, CLOSEUP_SIZE))
        cv2.rectangle(tile, (0, 0), (CLOSEUP_SIZE - 1, 22), (0, 0, 0), -1)
        cv2.putText(tile, f"{label} ({camera.replace('cam_', '')})", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 255, 0), 1)
        tiles.append(tile)
    if not tiles:
        raise ValueError("no close-ups")
    columns = min(4, len(tiles))
    rows = -(-len(tiles) // columns)
    sheet = np.zeros((rows * CLOSEUP_SIZE, columns * CLOSEUP_SIZE, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, columns)
        sheet[r * CLOSEUP_SIZE:(r + 1) * CLOSEUP_SIZE, c * CLOSEUP_SIZE:(c + 1) * CLOSEUP_SIZE] = tile[..., :3]
    return sheet
