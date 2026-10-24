"""The agent configuration: what is evaluated, resolved for one machine.

There is one agent, configs/default.toml. It names the prompt pack, the skills (by
interface layers or by name), the services skills use (segmenter, policies, world
models), runtime settings, the budget, and its default model roles. An ablation
(configs/ablations/<name>.toml) includes it and changes what the paper's ablations
change; an experiment may override any key.

Model roles come from a model file, configs/models/<name>.toml (the agent's `model`,
an experiment's `model`, or `--model NAME`):

    [main]                          the model that writes programs
    backend = "openrouter"
    model = "x-ai/grok-4.7"
    [vlm]                           questions about images (and pointing without a pointer)
    service = "qwen3-vl-8b"         a role may be a service of the deploy ...
    timeout_seconds = 300           ... with its own settings on top
    [pointer]
    service = "molmo2"
    [verifier]                      overrides on top of main (the success verifier)
    [runtime]                       settings the model needs (context window, ...)

Resolving the agent against a deploy gives the driver's main model, runtime and
prompt, and a lock: everything that decides what a run computes and nothing about
where it runs (URLs, GPUs, ports). The lock records the agent's VERSION, and its hash
is the configuration's identity; a leaderboard entry, a paper table and an exported
XPolicyLab policy each name one.
"""

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path

from . import deep_merge, load_layered, read_toml, route_openrouter
from .. import AGENT_VERSION
from ..embodiments import KINDS
from ..runtime.prompts import assemble_prompt, features_for, select
from ..runtime.skills import available_services, resolve as resolve_skills, skills_for_layers
from ..services.deploy import client_config, load_deploy


REPO = Path(__file__).resolve().parents[3]
# The agent `harness` runs unless told otherwise, and where model files live.
DEFAULT_AGENT = REPO / "configs" / "default.toml"
MODELS_ROOT = REPO / "configs" / "models"
ROLES = ("main", "vlm", "pointer", "verifier")
SERVICE_KEYS = ("segmenter", "policies", "world_models", "mcp")
# What one entry of each list of services is, in messages.
SERVICE_ENTRY = {"policies": "policy", "world_models": "world model", "mcp": "MCP server"}
# The keys of configs/default.toml (and of an ablation, which includes it).
AGENT_KEYS = {
    "description": "one line, shown by harness config and the run viewer",
    "prompt": "the prompt: a pack directory (assembled for the enabled skills) or files, in order; or a table "
              "of them by embodiment kind (robot = ..., desktop = ...)",
    "mode": "program (default: the model writes programs) or direct (R0/R1: one action per call)",
    "roles": "the model roles the agent uses, among main, vlm, pointer, verifier (default all)",
    "skills": "layers = { learned, reactive, handlers }, or names = [...]; enable = [...] adds optional skills, "
              "disable = [...] removes skills",
    "services": "segmenter, policies = [...], world_models = [...], mcp = [...] (MCP tool servers for the mcp "
                "skill): services the skills use",
    "model": "the default model roles: a model file's name (configs/models/<name>.toml), or the roles inline",
    "runtime": "runtime settings: the [runtime] keys",
    "budget": "max_model_calls per episode (default 30)",
}
# The model roles of a model file (configs/models/<name>.toml), besides `runtime`
# (settings the model needs) and `description`.
ROLE_DOCS = {
    "main": "writes the programs (or, in direct mode, the actions)",
    "vlm": "answers questions about images inside programs (query_vlm), and points without a pointer",
    "pointer": "open-vocabulary pointing (point_to, localize)",
    "verifier": "overrides on top of main for the success verifier (verify, <verify>)",
}
# Settings of the agent's [runtime] (also of a model file's [runtime], and of an
# experiment's overrides). Documentation only: the resolution writes some of them
# itself (skills, skills_dropped, layers, the service configs).
RUNTIME_KEYS = {
    "bootstrap": "the kernel's session factory, \"module:function\" (default the program runtime's)",
    "cameras": "cameras whose images reach the model (default the three RoboDojo cameras)",
    "cell_timeout_seconds": "wall time one program cell may run before the kernel is stopped. A cell that hands the task to the learned policy uses the episode's remaining wall time instead",
    "contract": "the observation contract: native (default) or xpolicylab (core/contract.py)",
    "default_max_seconds": "max_seconds of a bounded execution that names none",
    "fixed_skills": "direct mode: locate, pick and place in the action vocabulary (R1)",
    "isolation": "run programs isolated (default true; \"auto\": where user namespaces work)",
    "keep_images": "images of the most recent turns kept in the history (default 2)",
    "label": "a name the run viewer shows instead of the interface rung",
    "layers": "written by the resolution from [skills] layers",
    "library": "a self-improvement library: [runtime.library] path = ...; frozen during evaluation",
    "max_history_turns": "keep only this many exchanges after the first message",
    "mode": "written by the resolution from the agent's mode",
    "pointer": "written by the resolution: the pointer role's client config",
    "policies": "written by the resolution: the learned policies' client configs",
    "python": "the kernel's interpreter (default the driver's)",
    "reactive_budget_ms": "time a guard or controller may take per tick (default 5)",
    "segmenter": "written by the resolution: the segmenter's client config",
    "skills": "written by the resolution: the enabled skills",
    "skills_dropped": "written by the resolution: skills left out, and why",
    "post_policy_calls": "with learned policies: model calls after the first run of a handed-over task, then the policy runs on to the end of the episode (0: it runs on at once)",
    "policy_progress_seconds": "a run of a learned policy also ends when no object the last locate tracks has moved 3 cm for this many seconds, measured every half of it (default none: only when it stops)",
    "strategy_first": "with learned policies: nothing moves before start(strategy, kind) (default false; the agent sets it)",
    "strict_skills": "driver-built configurations: fail instead of dropping a skill",
    "task_descriptions": "a directory of task descriptions (<task>.md) added to the first message, or false",
    "verifier": "check success with the verifier before finishing (default true)",
    "vlm": "written by the resolution: the vlm role's client config",
    "world_models": "written by the resolution: the world models' client configs",
    "mcp": "written by the resolution: the MCP servers' client configs",
}
# Files whose content decides what a run computes (the code part of the lock).
CODE_ROOTS = ("src/harness_policy", "skills", "benches", "prompts", "services")


def canonical(value) -> str:
    """`value` as compact JSON with sorted keys (other objects as their str), so equal
    trees give the same text."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def sha256(value) -> str:
    """The hex sha256 of `value`'s canonical JSON."""
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def code_fingerprint(roots=CODE_ROOTS) -> dict:
    """sha256 of every source file of the agent (Python, prompts, manifests), and of
    them all."""
    files = {}
    for root in roots:
        for path in sorted((REPO / root).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".md", ".toml", ".json", ".yml", ".yaml") \
                    and "__pycache__" not in path.parts:
                files[path.relative_to(REPO).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"sha256": sha256(files), "files": len(files)}


def _relative(value):
    """Paths inside the repository written relative to it (locks are machine independent)."""
    if isinstance(value, str) and value.startswith(str(REPO)):
        return Path(value).relative_to(REPO).as_posix()
    if isinstance(value, list):
        return [_relative(item) for item in value]
    if isinstance(value, dict):
        return {key: _relative(item) for key, item in value.items()}
    return value


@dataclass
class ResolvedAgent:
    """The agent resolved against one deploy: the driver's main model, verifier, runtime
    and prompt, the episode's model-call budget, and the lock with its identity (the
    first 16 hex digits of the lock's sha256). `base_prompt` is the prompt without the
    self-improvement library's additions."""

    name: str
    path: Path | None
    main: dict
    verifier: dict
    runtime: dict
    system_prompt: str
    base_prompt: str
    max_model_calls: int
    lock: dict
    identity: str
    # Where each service the agent uses runs on this deploy (not part of the lock).
    placement: dict = field(default_factory=dict)
    # What it was resolved from (resolve_agent's arguments, trees): an exported
    # policy resolves these again on the machine it lands on.
    sources: dict = field(default_factory=dict)
    # The embodiment kind its skills and prompt were chosen for (embodiments.KINDS).
    embodiment: str = "robot"


def _role(spec, deploy, role: str, strict: bool):
    """A role's model config: its own keys, on top of its service's client config."""
    if not spec:
        return None, None
    spec = dict(spec)
    service = spec.pop("service", None)
    if service is None:
        # The URL is where this machine calls the model. The lock keeps base_url_env,
        # not the value, so another machine's endpoint is the same agent.
        named = spec.get("base_url_env")
        from_env = os.environ.get(named, "").strip() if named else ""
        if from_env:
            spec["base_url"] = from_env
        return route_openrouter(spec), None
    base = client_config(deploy, service)
    if base is None:
        message = f"role {role} uses service {service}, which deploy {deploy.name} does not run"
        if strict or role == "main":
            raise ValueError(message)
        return None, {"role": role, "service": service, "dropped": message}
    return {**base, **spec}, {"role": role, "service": service}


def _service(entry, deploy, strict: bool, kind: str):
    """(name, client config or None) of one service entry: a name or {service = ...}."""
    spec = {"service": entry} if isinstance(entry, str) else dict(entry)
    name = spec.pop("service")
    base = client_config(deploy, name)
    if base is None:
        if strict:
            raise ValueError(f"The agent's {kind} {name} is not run by deploy {deploy.name}")
        return name, None
    return name, {**base, **spec}


def model_names() -> list[str]:
    """The model files of configs/models/, by name."""
    return sorted(path.stem for path in MODELS_ROOT.glob("*.toml"))


def model_path(name) -> Path:
    """The file of a model: a name of configs/models/, or a path to a .toml file."""
    text = str(name)
    if text.endswith(".toml") or "/" in text or "\\" in text:
        path = Path(text).resolve()
    else:
        path = MODELS_ROOT / f"{text}.toml"
    if not path.is_file():
        raise ValueError(f"No model file {path}; models: {', '.join(model_names())}")
    return path


def load_model(source) -> dict:
    """Model roles: a model file's name or path (with its includes), a tree as it is,
    or {} for None."""
    if source is None:
        return {}
    if isinstance(source, dict):
        return source
    return load_layered(model_path(source))


def resolve_agent(agent=None, *, model=None, deploy=None, strict: bool = False, name: str | None = None,
                  runtime_overrides: dict | None = None, embodiment: str = "robot") -> ResolvedAgent:
    """Resolve the agent (a path or a merged tree; default configs/default.toml) with
    model roles from `model` (a model name, path or tree; default the agent's own
    `model`) against `deploy` (a Deploy, a name, or None for $HARNESS_DEPLOY), for an
    embodiment kind (the bench's: its skills and prompt). strict: a service or skill the
    agent names that cannot run here is an error instead of being left out.
    runtime_overrides (an experiment's changes to the agent) apply after the model file's
    runtime settings."""
    if embodiment not in KINDS:
        raise ValueError(f"Unknown embodiment kind {embodiment!r}; kinds are {sorted(KINDS)}")
    path = None
    if agent is None:
        agent = DEFAULT_AGENT
    if not isinstance(agent, dict):
        path = Path(agent).resolve()
        agent = load_layered(path)
    deploy = deploy if deploy is not None and not isinstance(deploy, (str, Path)) else load_deploy(deploy)
    tree = dict(agent)
    label = name or (path.stem if path else None)
    unknown = set(tree) - set(AGENT_KEYS)
    if unknown:
        raise ValueError(f"Agent {path or label}: unknown keys {sorted(unknown)}")
    roles_tree = load_model(model if model is not None else tree.get("model"))
    unknown = set(roles_tree) - {*ROLES, "runtime", "description"}
    if unknown:
        raise ValueError(f"Model: unknown roles {sorted(unknown)}; roles are {ROLES} (and runtime)")
    # The roles this agent uses (direct control without perception: main only).
    used = tree.get("roles", list(ROLES))
    if "main" not in used or set(used) - set(ROLES):
        raise ValueError(f"Agent {path or label}: roles must include main and be among {ROLES}")
    roles_tree = {key: value for key, value in roles_tree.items() if key not in ROLES or key in used}
    runtime = deep_merge(deep_merge(tree.get("runtime", {}), roles_tree.get("runtime", {})), runtime_overrides or {})
    if runtime.get("task_descriptions") is False:
        runtime.pop("task_descriptions")
    if tree.get("mode", "program") == "direct":
        runtime["mode"] = "direct"
    from ..core.contract import CONTRACTS

    if runtime.get("contract", "native") not in CONTRACTS:
        raise ValueError(f"runtime.contract must be one of {CONTRACTS}")
    placement = {}
    main, record = _role(roles_tree.get("main"), deploy, "main", strict)
    if main is None:
        raise ValueError(f"Agent {path or label}: no main model role")
    if record:
        placement["main"] = record
    for role in ("vlm", "pointer"):
        config, record = _role(roles_tree.get(role), deploy, role, strict)
        if config is not None:
            runtime[role] = config
        if record:
            placement[role] = record
    verifier = dict(roles_tree.get("verifier", {}))
    services = tree.get("services", {})
    unknown = set(services) - set(SERVICE_KEYS)
    if unknown:
        raise ValueError(f"Agent {path or label}: services are {SERVICE_KEYS}, not {sorted(unknown)}")
    if services.get("segmenter"):
        service, config = _service(services["segmenter"], deploy, strict, "segmenter")
        placement["segmenter"] = {"service": service, **({} if config else {"dropped": "not run here"})}
        if config is not None:
            runtime["segmenter"] = config
    for key in ("policies", "world_models", "mcp"):
        entries = services.get(key, [])
        chosen = {}
        for entry in entries:
            service, config = _service(entry, deploy, strict, SERVICE_ENTRY[key])
            placement.setdefault(key, {})[service] = {} if config else {"dropped": "not run here"}
            if config is not None:
                chosen[service] = config
        if entries:
            runtime[key] = chosen
    # Skills: the interface layers' (R0-R4) or a list, without the disabled ones.
    skills = tree.get("skills", {})
    unknown = set(skills) - {"layers", "names", "enable", "disable"}
    if unknown:
        raise ValueError(f"Agent {path or label}: skills takes layers, names, enable, disable; not {sorted(unknown)}")
    runtime["layers"] = {"learned": False, "reactive": True, "handlers": True, **skills.get("layers", {})}
    requested = (list(skills.get("names") or skills_for_layers(runtime["layers"], embodiment=embodiment))
                 + list(skills.get("enable", [])))
    requested = [skill for skill in dict.fromkeys(requested) if skill not in set(skills.get("disable", []))]
    from ..runtime.driver import model_services

    # A skill the agent leaves out by design (R0 has no vision roles) is dropped here on
    # every machine. Strictness is about the machine: the services the agent names were
    # checked above; a skill named in skills.names that cannot run is the one more error.
    resolution = resolve_skills(requested, available_services(model_services(runtime, main, verifier)),
                                embodiment=embodiment)
    named = {skill: why for skill, why in resolution.dropped.items()
             if skill in set(skills.get("names") or ()) | set(skills.get("enable", ()))}
    if strict and named:
        raise ValueError(f"Agent {path or label}: skills it names cannot run: {named}")
    runtime["skills"] = resolution.enabled
    if resolution.dropped:
        runtime["skills_dropped"] = resolution.dropped
    # The prompt: a pack directory assembled for the skills, or files as they are; a
    # table of them gives the embodiment's.
    sources = tree.get("prompt", [])
    if isinstance(sources, dict):
        unknown = set(sources) - set(KINDS)
        if unknown or embodiment not in sources:
            raise ValueError(f"Agent {path or label}: a prompt table is keyed by embodiment kinds {sorted(KINDS)} "
                             f"and needs one for {embodiment}")
        sources = sources[embodiment]
    sources = [sources] if isinstance(sources, str) else list(sources)
    texts = []
    for source in sources:
        source = Path(source)
        features = features_for(runtime)
        texts.append(assemble_prompt(source, runtime["skills"], features=features) if source.is_dir()
                     else select(source.read_text(), set(runtime["skills"]) | features))
    system_prompt = "".join(text if text.endswith("\n") else text + "\n" for text in texts)
    base_prompt = system_prompt
    if runtime.get("library"):
        from ..runtime.driver import _library

        library = _library(runtime, Path("/"))
        runtime["library"] = {**runtime["library"], "path": str(library.directory), "manifest": library.manifest()}
        system_prompt += library.api_summary("reactive" in runtime["skills"]) + library.notes()
    budget = tree.get("budget", {})
    calls = int(budget.get("max_model_calls", 30))
    if calls < 1:
        raise ValueError("budget.max_model_calls must be a positive integer")
    lock = {
        "schema": 2,
        "agent": label,
        "agent_version": AGENT_VERSION,
        "mode": runtime.get("mode", "program"),
        "prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "prompt_sources": _relative([str(s) for s in sources]),
        "skills": runtime["skills"],
        "skills_dropped": runtime.get("skills_dropped", {}),
        # Roles by what they are, not where they run: a service-backed role is its service.
        "roles": {role: _lock_role(roles_tree.get(role)) for role in ROLES if roles_tree.get(role)},
        "services": {key: services.get(key) for key in SERVICE_KEYS if services.get(key)},
        "service_manifests": {service: sha256(read_toml(deploy.services[service].path))
                              for service in _service_names(roles_tree, services) if service in deploy.services},
        "runtime": _relative({key: value for key, value in runtime.items()
                              if key not in ("vlm", "pointer", "segmenter", "policies", "world_models", "mcp")}),
        "budget": {"max_model_calls": calls},
    }
    # Robot is the default kind and left out, so robot identities are as before.
    if embodiment != "robot":
        lock["embodiment"] = embodiment
    return ResolvedAgent(name=label, path=path, main=main, verifier=verifier, runtime=runtime,
                         system_prompt=system_prompt, base_prompt=base_prompt, max_model_calls=calls, lock=lock,
                         identity=sha256(lock)[:16], placement=placement,
                         sources={"agent": {key: value for key, value in tree.items() if key != "model"},
                                  "model": roles_tree, "runtime_overrides": runtime_overrides or {}},
                         embodiment=embodiment)


def _lock_role(spec):
    spec = dict(spec)
    return {key: value for key, value in spec.items() if key not in ("api_key", "base_url")} \
        if "service" in spec else {key: value for key, value in spec.items() if key != "api_key"}


def _service_names(roles_tree, services):
    names = [spec["service"] for spec in roles_tree.values() if isinstance(spec, dict) and "service" in spec]
    if services.get("segmenter"):
        entry = services["segmenter"]
        names.append(entry if isinstance(entry, str) else entry["service"])
    for key in ("policies", "world_models", "mcp"):
        names += [entry if isinstance(entry, str) else entry["service"] for entry in services.get(key, [])]
    return sorted(set(names))
