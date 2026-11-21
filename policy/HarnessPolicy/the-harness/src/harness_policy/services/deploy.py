"""Deploy files: where the services of one machine run.

configs/deploy/<machine>.toml says which GPUs this machine gives the project, where
its weights live, and for each service how it runs (git ignores machine files there;
only the example-*.toml ones are tracked):

    weights = "/data/weights"         root of the services' weight directories
    state = "/data/harness-services"  pid files and logs of started services
    gpus = [2, 3, 4, 5]               GPUs services and simulators may use
    gpu_memory_gb = 24

    [sim]                             simulator workers of `harness run`
    gpus = [5]
    workers_per_gpu = 2

    [services.sam3]
    gpu = 3                           a GPU index, or "auto" (fit by vram_gb)
    [services.molmo2]
    mode = "external"                 someone else runs it
    url = "http://10.0.0.7:8830"
    [services.openwam]
    mode = "off"

    [services.g05]
    gpu = 1
    mode = "on_demand"                started when a skill first calls it ...
    swap = "big"                      ... and sharing GPU 1 with the other services of
    [services.openwam]                its swap group, one loaded at a time
    gpu = 1
    mode = "on_demand"
    swap = "big"

A service the deploy file does not list is off. Modes: resident (started by
`harness up`, kept running), on_demand (started by the broker when a skill needs it,
stopped after idle_seconds or when a peer of its swap group is needed; see broker.py),
external (only a URL), off. Placement never changes what a run computes, only where
and when; systems decide which services a run uses.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
import tomllib

from .registry import SERVICES_ROOT, ServiceSpec, load_services


REPO = Path(__file__).resolve().parents[3]
DEPLOY_ROOT = REPO / "configs" / "deploy"
MODES = ("resident", "on_demand", "external", "off")
# Modes in which a service runs on this machine.
LOCAL = ("resident", "on_demand")
# The keys of configs/deploy/<machine>.toml.
DEPLOY_KEYS = {
    "weights": "root of the services' weight directories (relative to the file, or absolute)",
    "state": "pid files and logs of the services it starts",
    "robodojo": "the RoboDojo checkout (default $HARNESS_ROBODOJO_ROOT, else third_party/RoboDojo)",
    "xpolicylab": "the XPolicyLab checkout (default <robodojo>/XPolicyLab)",
    "gpus": "the GPUs services and simulators may use",
    "gpu_memory_gb": "memory of each of those GPUs (default 24); gpu = \"auto\" plans with it",
    "sim": "simulator workers of `harness run`: gpus, workers_per_gpu, base_port",
    "services": "[services.<name>] for each service that runs here; a service not listed is off",
    "broker_port": "where the broker of on-demand services listens (default 8899)",
}
# The keys of one [services.<name>] table of a deploy.
PLACEMENT_KEYS = {
    "mode": f"one of {', '.join(MODES)} (default resident)",
    "gpu": "a GPU index, or \"auto\": the GPU with the most room for its vram_gb",
    "port": "a port other than the manifest's",
    "url": "for mode external: where it answers",
    "python": "another interpreter for it",
    "env": "extra environment variables",
    "swap": "on_demand: the swap group; services of one group share a GPU, one loaded at a time",
    "idle_seconds": "on_demand: stop it after this long without a request",
}
DEPLOY_ENV = "HARNESS_DEPLOY"


@dataclass
class Placement:
    """How one service runs on a deploy (PLACEMENT_KEYS). A service the deploy file does
    not list has mode off; loading the deploy replaces gpu = "auto" with a GPU index."""

    service: str
    mode: str = "off"
    gpu: int | str | None = None
    port: int | None = None
    url: str | None = None
    python: str | None = None
    env: dict = field(default_factory=dict)
    # on_demand: services of one swap group share a GPU, one loaded at a time.
    swap: str | None = None
    idle_seconds: float | None = None


@dataclass
class Deploy:
    """A loaded deploy file: the machine's weights, state and checkout paths, its GPUs, the
    simulator settings, and the placement of each service it lists. Without a deploy file
    (name "none") every service is off."""

    name: str
    path: Path | None
    weights: Path
    state: Path
    robodojo: Path
    xpolicylab: Path
    gpus: list[int] | None
    gpu_memory_gb: float
    sim: dict
    placements: dict[str, Placement]
    services_root: Path = SERVICES_ROOT
    # The broker of on-demand services listens here (services/broker.py).
    broker_port: int = 8899

    @property
    def services(self) -> dict[str, ServiceSpec]:
        """Every service manifest (under `services_root`) by name, whether it runs here or not."""
        return load_services(self.services_root)

    def placement(self, name: str) -> Placement:
        """How service `name` runs here (mode off when the file does not list it); an
        unknown service is an error."""
        if name not in self.services:
            raise ValueError(f"Unknown service {name!r}; known: {sorted(self.services)}")
        return self.placements.get(name) or Placement(name)

    def variables(self) -> dict:
        """The placeholders of service commands, settings and needed files: python (this
        process's interpreter), repo, weights, robodojo, xpolicylab, xpolicylab_parent, state."""
        return {"python": sys.executable, "repo": str(REPO), "weights": str(self.weights),
                "robodojo": str(self.robodojo), "xpolicylab": str(self.xpolicylab),
                "xpolicylab_parent": str(self.xpolicylab.parent), "state": str(self.state)}


def _resolve(value, base: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else (base / path).resolve()


def deploy_path(name: str | os.PathLike | None) -> Path | None:
    """The deploy file for a name (configs/deploy/<name>.toml) or a path; None names
    $HARNESS_DEPLOY, and without it there is no deploy file."""
    name = name or os.environ.get(DEPLOY_ENV)
    if not name:
        return None
    path = Path(name)
    if path.suffix == ".toml" or path.exists():
        return path.resolve()
    candidate = DEPLOY_ROOT / f"{name}.toml"
    if not candidate.is_file():
        known = sorted(p.stem for p in DEPLOY_ROOT.glob("*.toml"))
        raise ValueError(f"No deploy file {candidate}; known: {known}")
    return candidate


def load_deploy(name: str | os.PathLike | None = None, *, services_root: Path = SERVICES_ROOT) -> Deploy:
    """The deploy of `name` (see deploy_path); without one, every service is off and
    only hosted models are available."""
    path = deploy_path(name)
    data = {}
    if path is not None:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    unknown = set(data) - set(DEPLOY_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    base = path.parent if path else Path.cwd()
    robodojo = _resolve(data.get("robodojo") or os.environ.get("HARNESS_ROBODOJO_ROOT")
                        or REPO / "third_party" / "RoboDojo", base)
    deploy = Deploy(
        name=path.stem if path else "none", path=path,
        weights=_resolve(data.get("weights", REPO / "weights"), base),
        state=_resolve(data.get("state", Path.home() / ".cache" / "harness-policy" / "services"), base),
        robodojo=robodojo, xpolicylab=_resolve(data.get("xpolicylab", robodojo / "XPolicyLab"), base),
        gpus=[int(g) for g in data["gpus"]] if "gpus" in data else None,
        gpu_memory_gb=float(data.get("gpu_memory_gb", 24.0)), sim=dict(data.get("sim", {})),
        placements={}, services_root=services_root, broker_port=int(data.get("broker_port", 8899)))
    services = deploy.services
    for name, entry in (data.get("services") or {}).items():
        if name not in services:
            raise ValueError(f"{path}: unknown service {name!r}; known: {sorted(services)}")
        if not isinstance(entry, dict):
            entry = {"mode": str(entry)}
        unknown = set(entry) - set(PLACEMENT_KEYS)
        if unknown:
            raise ValueError(f"{path}: services.{name} has unknown keys {sorted(unknown)}")
        mode = entry.get("mode", "resident")
        if mode not in MODES:
            raise ValueError(f"{path}: services.{name}.mode must be one of {MODES}")
        if mode == "external" and not entry.get("url"):
            raise ValueError(f"{path}: services.{name} is external and needs a url")
        if mode in LOCAL and "gpu" not in entry and services[name].vram_gb > 0:
            raise ValueError(f"{path}: services.{name} runs here and needs a gpu (an index or \"auto\")")
        if entry.get("swap") and mode != "on_demand":
            raise ValueError(f"{path}: services.{name}: only on_demand services swap")
        deploy.placements[name] = Placement(name, mode, entry.get("gpu"), entry.get("port"), entry.get("url"),
                                            entry.get("python"), dict(entry.get("env", {})), entry.get("swap"),
                                            entry.get("idle_seconds"))
    _place(deploy)
    return deploy


def _place(deploy: Deploy) -> None:
    """Give services with gpu = "auto" a GPU of the deploy that still has room for
    them (largest first; the members of a swap group share one GPU and need the room of
    the largest), and check that no two services share a port."""
    services = deploy.services
    used = dict(planned_memory(deploy))
    units: dict[tuple, list] = {}
    fixed: dict[str, int] = {}
    for placement in deploy.placements.values():
        swap = placement.swap if placement.mode == "on_demand" else None
        if placement.mode in LOCAL and placement.gpu == "auto":
            units.setdefault(("swap", swap) if swap else ("service", placement.service), []).append(placement)
        elif swap and isinstance(placement.gpu, int):
            fixed[swap] = placement.gpu

    def need(members):
        return max(services[p.service].vram_gb for p in members)

    for key, members in sorted(units.items(), key=lambda item: -need(item[1])):
        amount = need(members)
        names = ", ".join(p.service for p in members)
        if key[0] == "swap" and key[1] in fixed:
            # A member was placed by hand: the group is on its GPU, and grows to the largest.
            gpu = fixed[key[1]]
            placed = max(services[p.service].vram_gb for p in deploy.placements.values()
                         if p.mode == "on_demand" and p.swap == key[1] and isinstance(p.gpu, int))
            extra = max(0.0, amount - placed)
            if used.get(gpu, 0.0) + extra > deploy.gpu_memory_gb:
                raise ValueError(f"GPU {gpu} of swap group {key[1]} has no room for {names} (planned: {used})")
        else:
            extra = amount
            candidates = [g for g in (deploy.gpus or []) if used.get(g, 0.0) + amount <= deploy.gpu_memory_gb]
            if not candidates:
                raise ValueError(f"No GPU of {deploy.gpus} has {amount:g} GB left for {names} (planned: {used})")
            gpu = min(candidates, key=lambda g: used.get(g, 0.0))
        used[gpu] = used.get(gpu, 0.0) + extra
        for placement in members:
            placement.gpu = gpu
    ports = {}
    for placement in deploy.placements.values():
        if placement.mode in LOCAL:
            port = placement.port or services[placement.service].port
            if port in ports:
                raise ValueError(f"Services {ports[port]} and {placement.service} both use port {port}")
            ports[port] = placement.service


def planned_memory(deploy: Deploy) -> dict[int, float]:
    """GPU memory the deploy plans per GPU: resident services add up, a swap group
    counts as its largest member (one is loaded at a time)."""
    services = deploy.services
    used: dict[int, float] = {}
    groups: dict[tuple, float] = {}
    for placement in deploy.placements.values():
        if placement.mode not in LOCAL or not isinstance(placement.gpu, int):
            continue
        need = services[placement.service].vram_gb
        if placement.mode == "on_demand" and placement.swap:
            key = (placement.gpu, placement.swap)
            groups[key] = max(groups.get(key, 0.0), need)
        else:
            used[placement.gpu] = used.get(placement.gpu, 0.0) + need
    for (gpu, _group), need in groups.items():
        used[gpu] = used.get(gpu, 0.0) + need
    return used


def render(value, variables: dict):
    """Fill {name} placeholders in strings (recursively in lists and dicts)."""
    if isinstance(value, str):
        out = value
        for key, item in variables.items():
            out = out.replace("{" + key + "}", str(item))
        return out
    if isinstance(value, (list, tuple)):
        return [render(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: render(item, variables) for key, item in value.items()}
    return value


def port(deploy: Deploy, name: str) -> int:
    """The port a service listens on here: its placement's, else its manifest's."""
    placement = deploy.placement(name)
    return int(placement.port or deploy.services[name].port)


def url(deploy: Deploy, name: str) -> str | None:
    """Where clients reach a service, or None when it is off."""
    placement = deploy.placement(name)
    if placement.mode == "off":
        return None
    if placement.mode == "external":
        return placement.url.rstrip("/")
    scheme = "ws" if deploy.services[name].protocol == "xpolicylab" else "http"
    return f"{scheme}://127.0.0.1:{port(deploy, name)}"


def client_config(deploy: Deploy, name: str) -> dict | None:
    """The config a skill or agent uses for this service (its manifest's client table
    with the URL filled in), or None when the deploy does not run it."""
    address = url(deploy, name)
    if address is None:
        return None
    spec = deploy.services[name]
    host_port = address.split("://", 1)[-1]
    variables = {"url": address, "host": host_port.rsplit(":", 1)[0],
                 "port": host_port.rsplit(":", 1)[-1] if ":" in host_port else ""}
    config = render(spec.client or {"url": "{url}"}, variables)
    if deploy.placement(name).mode == "on_demand":
        # Clients ask the broker to start it before each call (services/broker.py).
        config["on_demand"] = {"url": f"http://127.0.0.1:{deploy.broker_port}", "service": name}
    return config


def launch(deploy: Deploy, name: str) -> tuple[list[str], dict, str]:
    """(argv, environment, working directory) that start a service of this deploy."""
    spec = deploy.services[name]
    placement = deploy.placement(name)
    variables = {**deploy.variables(), "port": port(deploy, name)}
    python = render(placement.python or spec.python, variables)
    variables["python"] = python
    # An xpolicylab service runs XPolicyLab's own server for its policy folder, unless it
    # brings a command that serves the same protocol (services/openwam-policy.toml).
    if spec.protocol == "xpolicylab" and not spec.command:
        xpl = spec.xpolicylab
        policy = xpl["policy"]
        overrides = {"host": "127.0.0.1", "port": variables["port"], "bench_name": xpl.get("bench_name", "RoboDojo"),
                     "task_name": "any", "ckpt_name": xpl.get("ckpt_name", "default"),
                     "env_cfg_type": xpl.get("env_cfg_type", "arx_x5"), "seed": 0, "policy_name": policy,
                     "action_type": xpl.get("action_type", "joint"), "action_dim": xpl.get("action_dim", 14),
                     **xpl.get("overrides", {})}
        argv = [python, "-u", "{xpolicylab}/setup_policy_server.py", "--config_path",
                f"{{xpolicylab}}/policy/{policy}/deploy.yml", "--overrides",
                *[f"{key}={value}" for key, value in overrides.items()]]
        pythonpath = xpl.get("pythonpath", ["{xpolicylab_parent}", "{xpolicylab}"])
        environment = {"PYTHONPATH": os.pathsep.join(render(pythonpath, variables)),
                       "PYTHONWARNINGS": "ignore::UserWarning"}
        cwd = xpl.get("cwd", "{xpolicylab}")
    else:
        argv = list(spec.command)
        environment = {"PYTHONPATH": os.pathsep.join([str(REPO), str(REPO / "src")])}
        cwd = spec.cwd
    environment.update({"PYTHONUNBUFFERED": "1", **spec.env, **placement.env})
    if isinstance(placement.gpu, int):
        environment["CUDA_VISIBLE_DEVICES"] = str(placement.gpu)
    elif placement.gpu is None:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    return render(argv, variables), render(environment, variables), render(cwd, variables)


def weight_dirs(deploy: Deploy, name: str) -> list[Path]:
    """The service's weight directories under the deploy's weights root, one per
    `[[weights]]` entry."""
    return [deploy.weights / entry["dir"] for entry in deploy.services[name].weights]


def missing_files(deploy: Deploy, name: str) -> list[str]:
    """Files the service needs that are not on this machine."""
    spec = deploy.services[name]
    variables = deploy.variables()
    return [path for path in render(list(spec.needs_files), variables) if not Path(path).exists()]
