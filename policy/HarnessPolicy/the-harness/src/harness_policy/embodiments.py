"""Embodiments: what a robot or a computer is, independent of how a run reaches it.

embodiments/<name>/embodiment.toml holds the facts a policy may know about the
embodiment without asking a simulator. Its `kind` (KINDS, default robot) decides the
session that drives it (runtime/episode.py), the skills that run on it (a skill's
`embodiments`) and the keys it must have.

A robot's file holds joint groups and limits, arm mounts, the fingertip offset, gripper
widths and readings, control rate, planner limits, camera calibration (head camera fixed
in the world, wrist cameras fixed on link6), the action keys of XPolicyLab's protocol,
and the shield's defaults. A simulator worker reports the same facts live in its
profile; the XPolicyLab connector and contract-restricted workers take them from here.
A computer's (desktop) file holds its screen and the rate of its input steps.
"""

from pathlib import Path
import tomllib


EMBODIMENTS_ROOT = Path(__file__).resolve().parents[2] / "embodiments"
# Embodiment kinds and the keys a file of each kind must have. A worker reports its kind
# in its profile (`kind`, default robot).
KINDS = {
    "robot": ("name", "control_hz", "groups", "arm_bases", "cameras"),
    "desktop": ("name", "control_hz", "screen"),
}


def embodiment_path(name) -> Path:
    """The embodiment file of a name (`embodiments/<name>/embodiment.toml`) or a path; an
    unknown name is an error."""
    path = Path(name)
    if path.suffix == ".toml" or path.exists():
        return path.resolve()
    candidate = EMBODIMENTS_ROOT / str(name) / "embodiment.toml"
    if not candidate.is_file():
        known = sorted(p.parent.name for p in EMBODIMENTS_ROOT.glob("*/embodiment.toml"))
        raise ValueError(f"Unknown embodiment {name!r}; known: {known}")
    return candidate


def embodiment_kind(data: dict) -> str:
    """The kind of an embodiment tree or a worker profile (default robot); an unknown kind
    is an error."""
    kind = data.get("kind", "robot")
    if kind not in KINDS:
        raise ValueError(f"Unknown embodiment kind {kind!r}; kinds are {sorted(KINDS)}")
    return kind


def load_embodiment(name) -> dict:
    """An embodiment's tree (a name or a path, see embodiment_path); it must have the keys
    of its kind (KINDS): a robot's name, control_hz, groups, arm_bases and cameras."""
    with embodiment_path(name).open("rb") as handle:
        data = tomllib.load(handle)
    for key in KINDS[embodiment_kind(data)]:
        if key not in data:
            raise ValueError(f"Embodiment {name}: missing {key}")
    return data


def profile(embodiment: dict) -> dict:
    """The worker profile a robot embodiment implies (what a simulator worker reports
    live), for connections without a simulator."""
    if embodiment_kind(embodiment) != "robot":
        raise ValueError(f"Embodiment {embodiment['name']}: a worker profile is built here for robots only")
    groups = [{"name": name, "joints": list(group["joints"]), "gripper": group["gripper"]}
              for name, group in embodiment["groups"].items()]
    keys = ("control_hz", "physics_steps_per_tick", "action_low", "action_high", "cameras", "planner_limits",
            "tip_offset_m", "gripper_empty_opening", "gripper_open_width_m", "table_z", "return_to_start",
            "shield", "planner_world")
    return {"groups": groups, "arm_bases": {k: list(v) for k, v in embodiment["arm_bases"].items()},
            "embodiment": embodiment["name"], **{key: embodiment[key] for key in keys if key in embodiment}}
