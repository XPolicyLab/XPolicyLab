"""``Env_ICL``, the object a cell sees next to ``robot``: one place for what this episode has learnt
about the scene (in-context learning from the environment).

The robot proxy writes the facts its results carry - where a move was BLOCKED, how each close of
a gripper went, every measured Thing and when a contact may have moved it.  The model keeps its
own part next to them - ``Env_ICL.plan``, ``Env_ICL.facts``, ``Env_ICL.note()`` - so that a later
cell starts from what earlier cells found instead of re-deriving it.  After a cell that changed it,
the cell's output ends with an ``Env_ICL Δ`` line (what is new), and a retry close to one that
already failed is pointed out in its result.

RoboDojo episodes run with this module as their proxy and with ``prompts/robodojo/07_env_icl.md``, except
in the Memory tasks (:data:`MEMORY_TASKS`): see :class:`hexashell.xpl.session.RoboDojoSession`.
"""

from __future__ import annotations

import math
import re
from typing import Any

from . import proxy
from .results import Grip, Move, Thing

#: RoboDojo's Memory dimension (prompts/ladders): these tasks run without Env_ICL
MEMORY_TASKS = frozenset({"cover_blocks", "imitate_sorting_sequence", "match_and_pick_from_conveyor",
                          "press_by_number", "swap_blocks", "swap_T"})
#: a BLOCKED end this close to an earlier one of the same arm counts as the same obstacle
OBSTACLE_NEAR_M = 0.03
#: a close this close (and turned this little) to an earlier one counts as the same grasp
GRASP_NEAR_M = 0.015
GRASP_YAW_DEG = 10.0
#: a contact or a drop this close to a measured Thing may have moved it
DISTURB_NEAR_M = 0.06

_AT = re.compile(r"Move (left|right)[^:]*: (BLOCKED|short of target) — at \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)"
                 r" pitch ([-\d]+) yaw ([-\d]+)[^,]*, [^f]*from \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)")
_LEFT = re.compile(r"\[(\d+) steps left\]")


def _d(a, b) -> float:
    return math.dist(a, b)


def _yaw_off(a, b) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _p(xyz) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in xyz) + ")"


class EnvICL:
    def __init__(self):
        #: the model's own: what it is doing next and why
        self.plan: Any = None
        #: the model's own: anything worth keeping (an axis, an offset, which side worked)
        self.facts: dict = {}
        self.notes: list[str] = []
        #: id -> dict(label, xyz, size, left, moved): each measured Thing, latest measurement
        self.things: dict[int, dict] = {}
        #: dict(arm, xyz, toward, n, left): where moves ended BLOCKED, one entry per place
        self.obstacles: list[dict] = []
        #: dict(arm, xyz, yaw, ok, fail, lost, left): each grasp place and how its closes went
        self.grasps: list[dict] = []
        self.left: int | None = None  # steps left at the latest robot result
        self._at: dict[str, tuple] = {}  # arm -> (grasp point, yaw) after its latest move
        self._held: dict[str, dict] = {}  # arm -> the grasp it took what it holds with
        self._delta: list[str] = []

    # -- the model's part ------------------------------------------------------------------------
    def note(self, text: str) -> None:
        """Keep a sentence for later cells (what failed and why, what to try next)."""
        self.notes.append(f"[{self.left} left] {text}" if self.left is not None else str(text))

    def blocked_near(self, xyz, arm=None, r=OBSTACLE_NEAR_M) -> list[dict]:
        """The recorded obstacles within r of xyz (of one arm, or of either)."""
        return [o for o in self.obstacles if (arm is None or o["arm"] == arm) and _d(o["xyz"], xyz) < r]

    def tried(self, arm, xyz, yaw) -> dict | None:
        """The recorded grasp place matching a close at xyz with yaw, if any."""
        for g in self.grasps:
            if g["arm"] == arm and _d(g["xyz"], xyz) < GRASP_NEAR_M and _yaw_off(g["yaw"], yaw) < GRASP_YAW_DEG:
                return g
        return None

    # -- what the proxy records ------------------------------------------------------------------
    def _stamp(self, res) -> None:
        left = getattr(res, "steps_left", None)
        if left is not None:
            self.left = left

    def on_move(self, mv, contact=False) -> list[str]:
        """A Move came back; returns notes for it.  contact: it was a goto(..., contact=True), a press,
        push or insertion, which is meant to end BLOCKED - not an obstacle."""
        self._stamp(mv)
        if getattr(mv, "dry_run", False) or not getattr(mv, "moved", True):
            return []
        self._at[mv.arm] = (tuple(mv.at), float(mv.angles[1]))
        if getattr(mv, "dropped", False):
            self._lost(mv.arm, tuple(mv.at))
        if mv.blocked and not contact:
            return self._blocked(mv.arm, tuple(mv.at), tuple(mv.target))
        return []

    def on_halt(self, text: str, dropped: bool = False) -> str:
        """A Blocked, or a Dropped (dropped=True), raised with this Move text; returns a note to add to
        it ('' if none)."""
        m = _AT.search(text)
        lm = _LEFT.search(text)
        if lm:
            self.left = int(lm.group(1))
        if not m:
            return ""
        arm = m.group(1)
        at = tuple(float(v) for v in m.group(3, 4, 5))
        target = tuple(float(v) for v in m.group(8, 9, 10))
        self._at[arm] = (at, float(m.group(7)))
        if dropped:
            self._lost(arm, at)
            return ""
        notes = self._blocked(arm, at, target)
        return "\n".join(f"  note: {n}" for n in notes)

    def on_grip(self, g) -> list[str]:
        self._stamp(g)
        if getattr(g, "refused", False) or g.arm not in self._at:
            return []
        if g.goal > 0.05:
            self._held.pop(g.arm, None)  # an open lets go of it
            return []
        xyz, yaw = self._at[g.arm]
        ok = bool(g.holding)
        e = self.tried(g.arm, xyz, yaw)
        if e is None:
            e = dict(arm=g.arm, xyz=xyz, yaw=yaw, ok=0, fail=0, lost=0, left=self.left)
            self.grasps.append(e)
        e["ok" if ok else "fail"] += 1
        e["left"] = self.left
        self._held[g.arm] = e if ok else None
        notes = []
        if ok:
            self._delta.append(f"{g.arm} close at {_p(xyz)} yaw {yaw:.0f}: holding")
        else:
            self._delta.append(f"{g.arm} close at {_p(xyz)} yaw {yaw:.0f}: empty")
            if e["fail"] >= 2:
                notes.append(f"Env_ICL: this grasp (within {GRASP_NEAR_M * 100:.1f} cm and {GRASP_YAW_DEG:.0f}° of "
                             f"yaw) has now come up empty {e['fail']} times - change how you take it (another yaw or "
                             "side, another part of it, tilt the hand, or move it first) instead of repeating it")
            moved = [t for t in self.things.values() if t["moved"] and _d(t["xyz"], xyz) < DISTURB_NEAR_M]
            for t in moved:
                notes.append(f"Env_ICL: #{t['id']} '{t['label']}' may have moved since it was measured "
                             f"({t['moved']}) - measure it again before the next try")
        return notes

    def on_thing(self, t) -> None:
        if not getattr(t, "found", False) or t.center_xyz is None:
            return
        old = self.things.get(t.id)
        self.things[t.id] = dict(id=t.id, label=t.label, xyz=tuple(t.center_xyz), size=t.size,
                                 left=self.left, moved=None)
        if old is None:
            self._delta.append(f"#{t.id} '{t.label}' at {_p(t.center_xyz)}")
        elif _d(old["xyz"], t.center_xyz) > 0.01:
            self._delta.append(f"#{t.id} '{t.label}' moved {100 * _d(old['xyz'], t.center_xyz):.1f} cm "
                               f"to {_p(t.center_xyz)}")

    def _blocked(self, arm, at, target) -> list[str]:
        toward = tuple(b - a for a, b in zip(at, target))
        near = self.blocked_near(at, arm)
        if near:
            o = near[0]
            o["n"] += 1
            o["left"] = self.left
        else:
            o = dict(arm=arm, xyz=at, toward=toward, n=1, left=self.left)
            self.obstacles.append(o)
        self._disturb(at, f"{arm} arm BLOCKED near it at {self.left} left")
        self._delta.append(f"{arm} BLOCKED at {_p(at)}" + (f" (time {o['n']} here)" if o["n"] > 1 else ""))
        if o["n"] < 2:
            return []
        return [f"Env_ICL: {arm} has been BLOCKED within {OBSTACLE_NEAR_M * 100:.0f} cm of here {o['n']} times - "
                "a small shift will meet the same thing: look what is in the way, then go round it (another "
                "side, higher, another approach) or move it; Env_ICL.obstacles lists these places"]

    def _lost(self, arm, at) -> None:
        e = self._held.pop(arm, None)
        if e is not None:
            e["lost"] += 1
        self._disturb(at, f"something dropped near it at {self.left} left")
        self._delta.append(f"{arm} lost what it held near {_p(at)}")

    def _disturb(self, at, why) -> None:
        for t in self.things.values():
            if _d(t["xyz"], at) < DISTURB_NEAR_M and not t["moved"]:
                t["moved"] = why
                self._delta.append(f"#{t['id']} '{t['label']}' may have moved ({why})")

    def flush(self) -> str:
        """What changed since the last flush, as one line ('' if nothing)."""
        if not self._delta:
            return ""
        s = "; ".join(self._delta[-8:])
        if len(self._delta) > 8:
            s = f"(+{len(self._delta) - 8} more) " + s
        self._delta = []
        return "Env_ICL Δ: " + s

    def __repr__(self) -> str:
        lines = ["Env_ICL" + ("" if self.left is None else f" (at {self.left} steps left)") + ":"]
        if self.plan is not None:
            lines.append(f"  plan: {self.plan}")
        if self.facts:
            lines.append(f"  facts: {self.facts}")
        for t in self.things.values():
            s = f"  #{t['id']} '{t['label']}' at {_p(t['xyz'])} (measured at {t['left']} left)"
            if t["moved"]:
                s += f"  may have moved: {t['moved']}"
            lines.append(s)
        for o in self.obstacles:
            lines.append(f"  obstacle: {o['arm']} BLOCKED at {_p(o['xyz'])} ×{o['n']} heading "
                         f"{_p(o['toward'])} (last at {o['left']} left)")
        for g in self.grasps:
            lines.append(f"  grasp: {g['arm']} at {_p(g['xyz'])} yaw {g['yaw']:.0f}: "
                         f"holding ×{g['ok']}, empty ×{g['fail']}, lost on the way ×{g['lost']}")
        for n in self.notes[-10:]:
            lines.append(f"  note: {n}")
        return "\n".join(lines)

    __str__ = __repr__


class Robot(proxy.Robot):
    """``robot`` with every result also written into an EnvICL (``Env_ICL`` in a cell), and its notes
    about a retry added to the result or to the Blocked raised."""

    def __init__(self, call):
        self._env_icl = EnvICL()
        super().__init__(self._recorded(call))

    def _recorded(self, call):
        env = self._env_icl

        def recorded(name, args, kwargs):
            try:
                res = call(name, args, kwargs)
            except (proxy.Blocked, proxy.Dropped) as exc:
                try:
                    note = env.on_halt(str(exc), dropped=isinstance(exc, proxy.Dropped))
                except Exception:  # noqa: BLE001 - Env_ICL never breaks a robot call
                    note = ""
                if note:
                    raise type(exc)(f"{exc}\n{note}") from None
                raise
            try:
                if isinstance(res, Move):
                    res.notes += env.on_move(res, contact=bool(kwargs.get("contact")))
                elif isinstance(res, Grip):
                    res.notes += env.on_grip(res)
                elif isinstance(res, Thing):
                    env.on_thing(res)
                elif name == "things":
                    for t in res or ():
                        if isinstance(t, Thing):
                            env.on_thing(t)
                else:
                    env._stamp(res)
            except Exception:  # noqa: BLE001
                pass
            return res

        return recorded
