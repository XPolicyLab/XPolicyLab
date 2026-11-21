"""Skills: the API a model's programs call.

A skill is a package `skills/<name>/` in the source checkout with

    skill.toml      name, summary, layer, the names it exports, what it needs
    __init__.py     a mixin class with the methods (and optional exports(session) and
                    turn_report(session): what the skill adds to the model's next turn
                    after every program)

A skill runs on the embodiment kinds its `embodiments` names (default robot). The
kernel's session class is the embodiment's session (runtime/episode.py) plus the mixin of
every skill for that embodiment, so skills can call each other's methods. Which skills a
program sees is decided per run: the system names them (or the interface layers do), and
a skill whose services are missing, or that does not run on the run's embodiment, is
dropped, from the kernel namespace and from the prompt alike (runtime/prompts.py), so the
model never reads about an API it cannot call.
"""

from dataclasses import dataclass, field
import importlib
from pathlib import Path
import tomllib

from harness_policy.embodiments import KINDS


SKILLS_ROOT = Path(__file__).resolve().parents[3] / "skills"
# Layers of the interface ladder (R0-R4); every other skill belongs to "base".
LAYERS = ("learned", "reactive", "handlers")
# Service kinds a skill can need (services/<name>/service.toml `kind`).
SERVICE_KINDS = ("segmenter", "pointer", "vlm", "verifier", "policy", "world_model", "mcp")
# The keys of skills/<name>/skill.toml.
SKILL_KEYS = {
    "name": "the skill's name, the same as its directory",
    "summary": "one line: what the skill gives programs",
    "layer": "base, or the interface layer that turns it on: learned, reactive or handlers",
    "mixin": "the class with its methods, as \"module:Class\" (e.g. \"skills.motion:Motion\")",
    "exports": "the names programs see: the mixin's methods, or names its exports(session) returns",
    "required": "never dropped: the session cannot run without it (core)",
    "needs": "service kinds that must all be available",
    "needs_any": "service kinds of which at least one must be available",
    "requires": "skills whose API this one's results or documentation refer to; dropped with them",
    "conflicts": "skills it cannot be enabled with: they export some of the same names",
    "optional": "not among the skills of its layer: only a configuration that enables it by name has it",
    "order": "position in the session's method resolution and in the prompt (lower first)",
    "embodiments": "the embodiment kinds it runs on, among robot and desktop (default [\"robot\"])",
}


@dataclass(frozen=True)
class Skill:
    """A skill as its skill.toml declares it (SKILL_KEYS); `path` is its directory."""

    name: str
    summary: str
    layer: str
    mixin: str
    exports: tuple[str, ...]
    # Never dropped: the session cannot run without it.
    required: bool = False
    # Every service kind here must be available, and at least one of needs_any.
    needs: tuple[str, ...] = ()
    needs_any: tuple[str, ...] = ()
    # Skills whose API this one's documentation and results refer to.
    requires: tuple[str, ...] = ()
    # Skills that export some of the same names; never enabled together.
    conflicts: tuple[str, ...] = ()
    # Only when a configuration names it (its layer alone does not bring it).
    optional: bool = False
    order: int = 100
    # The embodiment kinds it runs on (embodiments.KINDS).
    embodiments: tuple[str, ...] = ("robot",)
    path: Path = field(default=None, compare=False)

    @property
    def public(self) -> tuple[str, ...]:
        """The exports without a leading underscore: the API programs are told about (the
        driver's hook `__wrap_up__` is left out)."""
        return tuple(name for name in self.exports if not name.startswith("_"))


def _skill(path: Path) -> Skill:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    unknown = set(data) - set(SKILL_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    if data.get("name") != path.parent.name:
        raise ValueError(f"{path}: name must be the directory name {path.parent.name!r}")
    layer = data.get("layer", "base")
    if layer not in ("base", *LAYERS):
        raise ValueError(f"{path}: layer must be base or one of {LAYERS}")
    for key in ("needs", "needs_any"):
        bad = set(data.get(key, ())) - set(SERVICE_KINDS)
        if bad:
            raise ValueError(f"{path}: {key} names unknown service kinds {sorted(bad)}")
    embodiments = tuple(data.get("embodiments", ("robot",)))
    if not embodiments or set(embodiments) - set(KINDS):
        raise ValueError(f"{path}: embodiments must name kinds among {sorted(KINDS)}")
    return Skill(name=data["name"], summary=data.get("summary", ""), layer=layer, mixin=data["mixin"],
                 exports=tuple(data["exports"]), required=bool(data.get("required", False)),
                 needs=tuple(data.get("needs", ())), needs_any=tuple(data.get("needs_any", ())),
                 requires=tuple(data.get("requires", ())), conflicts=tuple(data.get("conflicts", ())),
                 optional=bool(data.get("optional", False)), order=int(data.get("order", 100)),
                 embodiments=embodiments, path=path.parent)


_CACHE: dict[Path, dict[str, Skill]] = {}


def load_skills(root: Path = SKILLS_ROOT) -> dict[str, Skill]:
    """Every skill under `root`, in prompt order."""
    root = Path(root).resolve()
    if root not in _CACHE:
        skills = [_skill(path) for path in sorted(root.glob("*/skill.toml"))]
        if not skills:
            raise ValueError(f"No skills under {root}")
        names = [skill.name for skill in skills]
        by_name = {skill.name: skill for skill in skills}

        def conflicting(a, b):
            # Never enabled together: declared conflicts, and skills for different embodiments.
            return (b in by_name[a].conflicts or a in by_name[b].conflicts
                    or not set(by_name[a].embodiments) & set(by_name[b].embodiments))

        owners = {}
        for skill in skills:
            for name in skill.exports:
                owners.setdefault(name, []).append(skill.name)
        duplicates = sorted(name for name, of in owners.items()
                            if any(not conflicting(a, b) for i, a in enumerate(of) for b in of[i + 1:]))
        if duplicates:
            raise ValueError(f"Names exported by more than one skill that may be enabled together: {duplicates}")
        for skill in skills:
            missing = (set(skill.requires) | set(skill.conflicts)) - set(names)
            if missing:
                raise ValueError(f"Skill {skill.name} names unknown skills {sorted(missing)}")
        _CACHE[root] = {skill.name: skill for skill in sorted(skills, key=lambda s: (s.order, s.name))}
    return _CACHE[root]


def skills_for_layers(layers: dict | None, root: Path = SKILLS_ROOT, *, embodiment: str = "robot") -> list[str]:
    """The skills of an interface rung given as layer switches ({"learned": bool, ...};
    defaults: learned off, reactive and handlers on) for an embodiment kind, without the
    optional ones."""
    layers = {"learned": False, "reactive": True, "handlers": True, **(layers or {})}
    unknown = set(layers) - set(LAYERS)
    if unknown:
        raise ValueError(f"Unknown runtime layers: {sorted(unknown)}")
    return [name for name, skill in load_skills(root).items()
            if embodiment in skill.embodiments and not skill.optional
            and (skill.layer == "base" or layers[skill.layer])]


def available_services(options: dict) -> set[str]:
    """Service kinds a kernel's options provide: the segmenter, pointer, VLM and
    verifier configs, and at least one policy, world model or MCP server."""
    kinds = {kind for kind in ("segmenter", "pointer", "vlm", "verifier") if options.get(kind)}
    if options.get("policies"):
        kinds.add("policy")
    if options.get("world_models"):
        kinds.add("world_model")
    if options.get("mcp"):
        kinds.add("mcp")
    return kinds


@dataclass
class Resolution:
    """The outcome of `resolve`: the enabled skills in prompt order, and each dropped
    skill with the reason."""

    enabled: list[str]
    dropped: dict[str, str]

    def to_dict(self) -> dict:
        """The resolution as a plain dict."""
        return {"enabled": list(self.enabled), "dropped": dict(self.dropped)}


def resolve(requested, services: set[str], *, strict: bool = False, root: Path = SKILLS_ROOT,
            embodiment: str = "robot") -> Resolution:
    """The requested skills that can run on this embodiment kind with these services. A
    skill is dropped when it does not run on the embodiment, a service it needs is missing
    or a skill it requires is not enabled (repeated until nothing changes); the
    embodiment's required skills are always enabled. strict: dropping is an error
    (evaluation runs must not run a smaller system than the one they name)."""
    if embodiment not in KINDS:
        raise ValueError(f"Unknown embodiment kind {embodiment!r}; kinds are {sorted(KINDS)}")
    known = load_skills(root)
    requested = list(dict.fromkeys(requested))
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise ValueError(f"Unknown skills {unknown}; known: {sorted(known)}")
    dropped = {name: f"does not run on the {embodiment} embodiment" for name in requested
               if embodiment not in known[name].embodiments}
    wanted = ({name for name in requested if name not in dropped}
              | {name for name, skill in known.items() if skill.required and embodiment in skill.embodiments})
    clashes = sorted({tuple(sorted((name, other))) for name in wanted for other in known[name].conflicts
                      if other in wanted})
    if clashes:
        raise ValueError("Skills that cannot be enabled together (they export the same names): "
                         + "; ".join(f"{a} and {b}" for a, b in clashes))
    changed = True
    while changed:
        changed = False
        for name in [n for n in known if n in wanted]:
            skill = known[name]
            reason = None
            missing = [kind for kind in skill.needs if kind not in services]
            if missing:
                reason = f"no {' or '.join(missing)} service"
            elif skill.needs_any and not set(skill.needs_any) & services:
                reason = f"needs one of the services {', '.join(skill.needs_any)}"
            else:
                off = [other for other in skill.requires if other not in wanted]
                if off:
                    reason = f"requires the skills {', '.join(off)}"
            if reason is not None:
                if skill.required:
                    raise ValueError(f"Required skill {name} cannot run: {reason}")
                wanted.discard(name)
                dropped[name] = reason
                changed = True
    if strict and dropped:
        raise ValueError("Skills the system names cannot run here: "
                         + "; ".join(f"{name} ({reason})" for name, reason in dropped.items()))
    return Resolution([name for name in known if name in wanted], dropped)


def _import(spec: str):
    module, _, attribute = spec.partition(":")
    value = importlib.import_module(module)
    return getattr(value, attribute) if attribute else value


def mixins(root: Path = SKILLS_ROOT, *, embodiment: str = "robot") -> list[type]:
    """The mixin class of every skill for an embodiment kind, imported, in the skills'
    order (the session's method resolution order)."""
    return [_import(skill.mixin) for skill in load_skills(root).values() if embodiment in skill.embodiments]


def namespace(session, names, root: Path = SKILLS_ROOT) -> dict:
    """The kernel namespace of a session with the skills `names` enabled: each skill's
    exports(session) when its module defines one, else the session attributes it
    exports."""
    known = load_skills(root)
    result = {}
    for name in names:
        skill = known[name]
        module = _import(skill.mixin.partition(":")[0])
        values = module.exports(session) if hasattr(module, "exports") else {
            export: getattr(session, export) for export in skill.exports}
        if set(values) != set(skill.exports):
            raise ValueError(f"Skill {name}: exports() returned {sorted(values)}, skill.toml lists "
                             f"{sorted(skill.exports)}")
        result.update(values)
    return result


def reporters(names, root: Path = SKILLS_ROOT) -> list:
    """The turn reports of the skills `names`, in skill order: each skill module's
    turn_report(session) -> {"text", "images"} or None, which the session calls after
    every program (EpisodeCore.drain)."""
    known = load_skills(root)
    found = []
    for name in names:
        module = _import(known[name].mixin.partition(":")[0])
        if hasattr(module, "turn_report"):
            found.append(module.turn_report)
    return found


def exported_names(root: Path = SKILLS_ROOT) -> dict[str, tuple[str, ...]]:
    """Every exported name and the skills that export it (more than one only for
    skills that conflict)."""
    names = {}
    for skill in load_skills(root).values():
        for export in skill.exports:
            names[export] = (*names.get(export, ()), skill.name)
    return names
