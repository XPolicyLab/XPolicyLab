"""Switch registry: every experimental knob, its module, default, values, compliance level and paper experiments.

A run is described by the fully expanded switch map (``resolve``), written into the run config at each
switch's ``config_path`` (``apply``) and into the run manifest verbatim.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from knowin_agent.knowledge import brief as knowledge

PROFILES_DIR = Path(__file__).with_name("profiles")
COMPLIANCE_LEVELS = ("clean", "review", "privileged")
MODULES = ("M1", "M2", "M3", "M4", "M5", "agent")
KINDS = ("enum", "flag", "int", "float", "str", "optional_str", "version")

EXPERIMENTS = {
    "E1": "main result, all 42 tasks",
    "E2": "B ablation: self-evolution components at equal practice budget",
    "E3": "learning curve: cumulative practice executions vs test-dev score",
    "E4": "cross-task transfer: cold start with/without the shared library",
    "E5": "shared library iteration v0 -> v1 -> v2",
    "E6": "A perception: locate backends, observe, depth source",
    "E7": "C execution: transit, bimanual async, budget governance",
    "E8": "knowledge sources: scoring notes, recipe, task source, object models",
    "E9": "cross-model transfer of learned experience",
    "E10": "cost and failure analysis",
    "E11": "two decision scales: Local Interaction Agent delegation on/off",
    "E12": "action mode: code vs tool calls vs hybrid",
}

_VERSION = re.compile(r"off|packaged|latest|v\d+")
_MISSING = object()


class SwitchError(ValueError):
    pass


@dataclass(frozen=True)
class Switch:
    name: str
    module: str
    default: Any
    choices: Optional[tuple]
    compliance: str
    experiments: tuple[str, ...]
    config_path: str
    kind: str = "enum"
    description: str = ""
    levels: Optional[dict] = None
    encode: Optional[dict] = None
    also: tuple = ()
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def normalize(self, value: Any) -> Any:
        """The canonical value (flags become bools); raises SwitchError when out of range."""
        problem = None
        if self.kind == "flag":
            if value in ("on", "off"):
                value = value == "on"
            if not isinstance(value, bool):
                problem = "must be on/off (or a bool)"
        elif self.kind == "enum":
            if value not in self.choices:
                problem = f"must be one of {list(self.choices)}"
        elif self.kind in ("int", "float"):
            numeric = (int,) if self.kind == "int" else (int, float)
            if isinstance(value, bool) or not isinstance(value, numeric):
                problem = f"must be {'an integer' if self.kind == 'int' else 'a number'}"
            elif (self.minimum is not None and value < self.minimum) or (self.maximum is not None and value > self.maximum):
                problem = f"must be within [{self.minimum}, {self.maximum}]"
            elif self.kind == "float":
                value = float(value)
        elif self.kind == "str":
            if not isinstance(value, str) or not value:
                problem = "must be a non-empty string"
        elif self.kind == "optional_str":
            if value is not None and (not isinstance(value, str) or not value):
                problem = "must be null or a non-empty string"
        elif self.kind == "version":
            if not isinstance(value, str) or not _VERSION.fullmatch(value):
                problem = "must be 'off', 'packaged', 'latest' or 'v<n>'"
        if problem:
            raise SwitchError(f"{self.name}={value!r}: {problem}")
        return value

    def compliance_of(self, value: Any) -> str:
        if self.levels is None:
            return self.compliance
        return self.levels.get(value, "clean")

    def values_text(self) -> str:
        if self.kind == "flag":
            return "on/off"
        if self.kind == "enum":
            return " / ".join(str(c) for c in self.choices)
        if self.kind in ("int", "float"):
            low = "" if self.minimum is None else f"{self.minimum:g}"
            high = "" if self.maximum is None else f"{self.maximum:g}"
            return f"{self.kind} [{low}, {high}]"
        if self.kind == "version":
            return "off / packaged / latest / v<n>"
        return "null / string" if self.kind == "optional_str" else "string"


def _s(name, module, default, choices, compliance, experiments, *, config_path=None, kind=None, **extra) -> Switch:
    if kind is None:
        kind = "flag" if choices == "flag" else "enum"
    return Switch(name=name, module=module, default=default, choices=(False, True) if choices == "flag" else choices,
                  compliance=compliance, experiments=tuple(experiments), config_path=config_path or name, kind=kind,
                  **extra)


REGISTRY: dict[str, Switch] = {s.name: s for s in (
    # M1 perception
    _s("perception.locate", "M1", "depth", ("depth", "full", "fusion"), "clean", ["E1", "E6"],
       description="text-prompted locate backend; fusion = depth locate + top-down orientation, falls back"),
    _s("perception.observe", "M1", True, "flag", "clean", ["E6"],
       description="observe / look_down as an action (re-measure from a wrist view)"),
    _s("perception.depth", "M1", "service", ("service", "native", "none"), "privileged", ["E1", "E6"],
       levels={"service": "review", "native": "privileged"},
       also=(("backend.depth", {"service": "none", "native": "native", "none": "none"}),),
       description="depth source: learned depth service (state its training data), simulator depth, or none"),
    _s("perception.calibration", "M1", "sim", ("sim", "constant"), "privileged", ["E1", "E6"],
       levels={"sim": "privileged"}, also=(("backend.calibration", {"sim": True, "constant": "constant"}),),
       description="camera calibration read from the simulator, or constants (head fixed, wrist from arm pose)"),
    # M2 execution and budget
    _s("motion.path", "M2", "auto", ("auto", "line", "joint", "ee"), "clean", ["E7"],
       description="motion planning mode: auto picks joint planning when kinematics allow it"),
    _s("motion.transit_step_m", "M2", 0.04, None, "clean", ["E7"], kind="float", minimum=0.005, maximum=0.10,
       description="metres per native action in free space with an open gripper (H5 0.04; holding uses at most "
                   "0.03; H4 used 0.02 for both)"),
    _s("motion.bimanual_async", "M2", True, "flag", "clean", ["E7"],
       description="asynchronous per-arm tracks merged on the same tick (wait=False)"),
    _s("budget.mode", "M2", "enforce", ("off", "panel", "enforce"), "clean", ["E7", "E10"],
       description="budget panel in context only, or also refuse non-homing actions that do not fit"),
    # M3 task knowledge
    _s("knowledge.scoring", "M3", knowledge.DEFAULTS["knowledge.scoring"], knowledge.SCORING_LEVELS, "review",
       ["E1", "E8"], levels={"code": "review"},
       description="scoring notes: off, summarised notes (docs), or notes plus the official task source (code)"),
    _s("knowledge.recipe", "M3", knowledge.DEFAULTS["knowledge.recipe"], "flag", "clean", ["E8"],
       description="task recipe; the scoring notes win on disagreement"),
    _s("knowledge.task_source", "M3", knowledge.DEFAULTS["knowledge.task_source"], "flag", "review", ["E1", "E8"],
       levels={True: "review"}, description="official task source copy (run_reward / get_score)"),
    _s("knowledge.object_models", "M3", knowledge.DEFAULTS["knowledge.object_models"], "flag", "privileged",
       ["E1", "E8"], levels={True: "privileged"}, description="object model catalogue (asset sizes); default off"),
    # Agent
    _s("agent.action_mode", "agent", "hybrid", ("code", "tools", "hybrid", "program"), "clean", ["E12"],
       description="hybrid (default, scoring) = exec + locate/inspect/check_reach + remember/interact/finish; "
                   "code = exec + all non-motion tools; tools = robot APIs only; program = exec + notes/"
                   "delegation/finish"),
    _s("agent.delegation", "agent", True, "flag", "clean", ["E11"],
       description="Local Interaction Agent available for fine alignment under a contract"),
    _s("agent.model", "agent", "gpt-6-astra", None, "clean", ["E9"], kind="str",
       description="agent model; one fixed model for every task in a submission"),
    _s("agent.effort", "agent", "high", ("low", "medium", "high", "xhigh"), "clean", ["E9", "E10"],
       description="reasoning effort"),
    # M4 self-evolution
    _s("evolution.practice_scenes", "M4", 3, None, "clean", ["E2", "E3"], kind="int", minimum=1, maximum=64,
       description="parallel practice scenes per round (1 = single-scene baseline)"),
    _s("evolution.review_merge", "M4", True, "flag", "clean", ["E2"],
       description="reviewer merges the practice runs' experience between rounds"),
    _s("evolution.validation", "M4", True, "flag", "clean", ["E2", "E5"],
       description="validate candidate knowledge on validate scenes and keep the best version"),
    _s("evolution.stop_when_perfect", "M4", True, "flag", "clean", ["E2", "E3"],
       description="skip the remaining continuation rounds once a round's validation scored 100 on every scene"),
    _s("evolution.shared_library", "M4", "off", None, "clean", ["E2", "E4", "E5"], kind="version",
       description="cross-task shared library (off, packaged, latest, or an external registry version)"),
    _s("evolution.seed_knowledge", "M4", None, None, "clean", ["E4", "E9"], kind="optional_str",
       description="frozen knowledge to start from (frozen dir or run id), null = cold start"),
    _s("evolution.practice_budget", "M4", 80, None, "clean", ["E2", "E3", "E4", "E5"], kind="int", minimum=1,
       maximum=10_000, description="total practice executions per task (equal across compared profiles)"),
    _s("evolution.stall_rule", "M4", 15, None, "clean", ["E10"], kind="int", minimum=0, maximum=1000,
       description="executions without score gain before the stall prompt; 0 disables"),
    _s("evolution.stall_mode", "M4", "advise", ("advise", "enforce"), "clean", ["E10"],
       description="advise: the stall prompt asks for a diagnosis and keeps progress; enforce: submissions are "
                   "refused until lessons change and the program resets the scene"),
    _s("evolution.practice_feedback", "M4", True, "flag", "clean", ["E2", "E3"],
       description="practice receipts carry the native Score; off = blind practice as in a test (the reviewer "
                   "still gets the final Score), used to validate a merge before it is adopted"),
    _s("evolution.trajectory_mining", "M4", True, "flag", "clean", ["E2", "E5"],
       description="mine repeated action sequences from trajectories as library skill candidates"),
)}


def get(name: str) -> Switch:
    try:
        return REGISTRY[name]
    except KeyError:
        raise SwitchError(f"unknown switch {name!r}") from None


def defaults() -> dict[str, Any]:
    return {name: copy.deepcopy(s.default) for name, s in REGISTRY.items()}


def _flatten(values: Mapping[str, Any]) -> dict[str, Any]:
    """Accept dotted keys or one level of nesting ({'knowledge': {'recipe': False}})."""
    flat: dict[str, Any] = {}
    for key, value in values.items():
        if key not in REGISTRY and isinstance(value, Mapping):
            for sub, inner in value.items():
                flat[f"{key}.{sub}"] = inner
        else:
            flat[key] = value
    return flat


def validate(values: Mapping[str, Any]) -> dict[str, Any]:
    """Normalized copy of ``values`` (any subset of switches); raises SwitchError listing every problem."""
    problems, out = [], {}
    for name, value in _flatten(values).items():
        try:
            out[name] = get(name).normalize(value)
        except SwitchError as exc:
            problems.append(str(exc))
    if problems:
        raise SwitchError("; ".join(problems))
    return out


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    experiments: tuple[str, ...]
    switches: dict[str, Any] = field(default_factory=dict)
    path: Optional[str] = None


def _profile_from(data: Mapping[str, Any], *, name: Optional[str] = None, path: Optional[Path] = None) -> Profile:
    if "switches" not in data:
        data = {"name": name or "custom", "description": "", "experiments": [], "switches": dict(data)}
    unknown = set(data) - {"name", "description", "experiments", "switches"}
    if unknown:
        raise SwitchError(f"profile {name or data.get('name')!r}: unknown keys {sorted(unknown)}")
    experiments = tuple(data.get("experiments", ()))
    bad = [e for e in experiments if e not in EXPERIMENTS]
    if bad:
        raise SwitchError(f"profile {data.get('name')!r}: unknown experiments {bad}")
    return Profile(name=str(data.get("name") or name or "custom"), description=str(data.get("description", "")),
                   experiments=experiments, switches=validate(data["switches"]),
                   path=None if path is None else str(path))


def load_profile(name: str, profiles_dir: Optional[Path] = None) -> Profile:
    directory = Path(profiles_dir) if profiles_dir else PROFILES_DIR
    path = directory / f"{name}.json"
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) or not path.is_file():
        known = ", ".join(list_profiles(directory))
        raise SwitchError(f"unknown profile {name!r} (known: {known})")
    profile = _profile_from(json.loads(path.read_text(encoding="utf-8")), name=name, path=path)
    if profile.name != name:
        raise SwitchError(f"profile file {path.name} declares name {profile.name!r}")
    return profile


def list_profiles(profiles_dir: Optional[Path] = None) -> list[str]:
    directory = Path(profiles_dir) if profiles_dir else PROFILES_DIR
    return sorted(p.stem for p in directory.glob("*.json"))


ProfileArg = Union[None, str, Mapping[str, Any], Profile]


def profile_of(profile: ProfileArg, profiles_dir: Optional[Path] = None) -> Profile:
    if isinstance(profile, Profile):
        return profile
    if profile is None:
        return Profile(name="defaults", description="registry defaults", experiments=())
    if isinstance(profile, str):
        return load_profile(profile, profiles_dir)
    return _profile_from(profile)


def resolve(profile: ProfileArg = None, overrides: Optional[Mapping[str, Any]] = None, *,
            profiles_dir: Optional[Path] = None) -> dict[str, Any]:
    """Every switch: registry defaults, then the profile's overrides, then ``overrides``."""
    values = defaults()
    values.update(profile_of(profile, profiles_dir).switches)
    values.update(validate(overrides or {}))
    return values


def _set(config: dict, dotted: str, value: Any) -> None:
    node = config
    keys = dotted.split(".")
    for key in keys[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            raise SwitchError(f"config key {key!r} on the path {dotted!r} is not a mapping")
        node = child
    node[keys[-1]] = copy.deepcopy(value)


def apply(config: Mapping[str, Any], switches: Mapping[str, Any]) -> dict:
    """A copy of ``config`` with each given switch written at its config path (and coupled paths)."""
    out = copy.deepcopy(dict(config))
    for name, value in validate(switches).items():
        switch = REGISTRY[name]
        _set(out, switch.config_path, switch.encode[value] if switch.encode else value)
        for path, mapping in switch.also:
            _set(out, path, mapping[value])
    return out


def diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, tuple]:
    """``{name: (value_in_a, value_in_b)}`` for every switch whose value differs; absent = registry default."""
    a, b = validate(a), validate(b)
    out = {}
    for name, switch in REGISTRY.items():
        left, right = a.get(name, switch.default), b.get(name, switch.default)
        if left != right:
            out[name] = (left, right)
    return out


def compliance_report(switches: Mapping[str, Any]) -> dict:
    """Worst compliance level of an (expanded) switch map and the switches responsible for it."""
    values = {**defaults(), **validate(switches)}
    flagged = {}
    for name, value in values.items():
        level = REGISTRY[name].compliance_of(value)
        if level != "clean":
            flagged[name] = {"value": value, "level": level}
    worst = max([COMPLIANCE_LEVELS.index(f["level"]) for f in flagged.values()], default=0)
    return {"level": COMPLIANCE_LEVELS[worst], "flagged": flagged}


def to_env(switches: Mapping[str, Any]) -> dict[str, str]:
    """``KNOWIN_AGENT_<SWITCH>`` environment variables for subprocesses (flags as 1/0, null as empty)."""
    out = {}
    for name, value in validate(switches).items():
        if isinstance(value, bool):
            text = "1" if value else "0"
        else:
            text = "" if value is None else str(value)
        out["KNOWIN_AGENT_" + name.upper().replace(".", "_")] = text
    return out


def table_rows() -> list[dict]:
    """One row per switch for documentation (values, default, module, compliance, experiments, config path)."""
    rows = []
    for s in REGISTRY.values():
        default = ("on" if s.default else "off") if s.kind == "flag" else s.default
        rows.append({"switch": s.name, "values": s.values_text(), "default": default, "module": s.module,
                     "compliance": s.compliance, "experiments": list(s.experiments), "config_path": s.config_path})
    return rows


def _check_registry() -> None:
    for s in REGISTRY.values():
        assert s.module in MODULES and s.compliance in COMPLIANCE_LEVELS and s.kind in KINDS, s.name
        assert all(e in EXPERIMENTS for e in s.experiments), s.name
        assert s.normalize(s.default) == s.default, s.name
        if s.levels:
            assert all(level in COMPLIANCE_LEVELS for level in s.levels.values()), s.name
        for mapping in [s.encode] + [m for _, m in s.also]:
            if mapping is not None:
                assert set(mapping) == set(s.choices), s.name


_check_registry()
