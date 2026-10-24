"""Service manifests: the model processes skills and agents call.

A service is one deployable model server, described by services/<name>.toml:

    kind = "segmenter"            segmenter | pointer | vlm | llm | policy | world_model | kinematics
    summary = "..."
    protocol = "http"             http | openai_chat | xpolicylab
    command = ["{python}", "services/servers/serve_sam3.py", "--checkpoint", "{weights}/sam3/sam3.pt",
               "--port", "{port}"]
    port = 8810                   default port
    vram_gb = 7.2                 GPU memory it holds once loaded (measured)
    load_seconds = 60             start until it answers
    client = { url = "{url}" }    the kernel's or agent's config for it ({url}, {host}, {port})
    [[weights]]                   where `harness fetch` gets its files
    dir = "sam3"
    modelscope_model = "facebook/sam3"

An XPolicyLab policy needs no command: its [xpolicylab] table names the policy folder,
action type and overrides, and the server is XPolicyLab's own setup_policy_server.py.
Where a service runs (GPU, port, resident or on demand, or not at all) is the deploy
file's business (deploy.py), not the manifest's.
"""

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


SERVICES_ROOT = Path(__file__).resolve().parents[3] / "services"
KINDS = ("segmenter", "pointer", "vlm", "llm", "policy", "world_model", "kinematics", "mcp")
# mcp: the Model Context Protocol over streamable HTTP (services/mcp.py).
PROTOCOLS = ("http", "openai_chat", "xpolicylab", "mcp")
WEIGHT_SOURCES = ("modelscope_model", "modelscope_dataset", "hf_model", "hf_dataset", "local")
# The keys of services/<name>.toml.
MANIFEST_KEYS = {
    "kind": f"what the service is for: one of {', '.join(KINDS)}",
    "summary": "one line: the model and where it comes from",
    "protocol": f"how clients talk to it: one of {', '.join(PROTOCOLS)} (default http)",
    "port": "its default port (a deploy may give another)",
    "vram_gb": "GPU memory it takes, measured; placement and swapping use it",
    "load_seconds": "how long it takes to answer after a start (waits allow three times this)",
    "command": "argv that starts it, with {python} {port} {weights} {repo} {robodojo} {xpolicylab} ... "
               "(protocol xpolicylab: optional, in place of XPolicyLab's setup_policy_server.py)",
    "xpolicylab": "for protocol xpolicylab: policy folder, ckpt_name, action_type (joint, or ee: link6 poses "
                  "that run_policy solves by IK), action_dim, cwd, pythonpath, overrides",
    "env": "environment variables of the process",
    "python": "interpreter (default {python}: the project environment's)",
    "cwd": "working directory (default {repo})",
    "client": "the config its users get: url = \"{url}\" and what the model is told about it (description)",
    "weights": "[[weights]] entries: dir below the deploy's weights root and one source "
               f"({', '.join(WEIGHT_SOURCES)}); `harness fetch` downloads them",
    "needs_files": "paths that must exist before it can start (checked by harness doctor and up)",
    "notes": "anything an operator should know",
}


@dataclass(frozen=True)
class ServiceSpec:
    """A service manifest (MANIFEST_KEYS) as loaded: `name` is the file's stem and `path`
    the file. Placeholders such as {python} or {port} are left for the deploy to fill."""

    name: str
    kind: str
    summary: str
    protocol: str
    port: int
    vram_gb: float
    load_seconds: float = 120.0
    command: tuple = ()
    xpolicylab: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    python: str = "{python}"
    cwd: str = "{repo}"
    client: dict = field(default_factory=dict)
    weights: tuple = ()
    # Paths that must exist before the service can start, with placeholders ({weights}/...).
    needs_files: tuple = ()
    notes: str = ""
    path: Path = field(default=None, compare=False)


def _spec(path: Path) -> ServiceSpec:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    unknown = set(data) - set(MANIFEST_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    if data.get("kind") not in KINDS:
        raise ValueError(f"{path}: kind must be one of {KINDS}")
    protocol = data.get("protocol", "http")
    if protocol not in PROTOCOLS:
        raise ValueError(f"{path}: protocol must be one of {PROTOCOLS}")
    if protocol == "xpolicylab":
        if not data.get("xpolicylab", {}).get("policy"):
            raise ValueError(f"{path}: an xpolicylab service names its policy folder in [xpolicylab] policy")
    elif not data.get("command"):
        raise ValueError(f"{path}: needs a command")
    for entry in data.get("weights", []):
        sources = [key for key in WEIGHT_SOURCES if key in entry]
        if "dir" not in entry or len(sources) != 1:
            raise ValueError(f"{path}: each [[weights]] has a dir and exactly one of {WEIGHT_SOURCES}")
    return ServiceSpec(
        name=path.stem, kind=data["kind"], summary=data.get("summary", ""), protocol=protocol,
        port=int(data["port"]), vram_gb=float(data.get("vram_gb", 0.0)),
        load_seconds=float(data.get("load_seconds", 120.0)), command=tuple(data.get("command", ())),
        xpolicylab=dict(data.get("xpolicylab", {})), env=dict(data.get("env", {})),
        python=data.get("python", "{python}"), cwd=data.get("cwd", "{repo}"), client=dict(data.get("client", {})),
        weights=tuple(dict(entry) for entry in data.get("weights", [])),
        needs_files=tuple(data.get("needs_files", ())), notes=data.get("notes", ""), path=path)


_CACHE: dict[Path, dict[str, ServiceSpec]] = {}


def load_services(root: Path = SERVICES_ROOT) -> dict[str, ServiceSpec]:
    """Every service manifest under `root` (its *.toml files) by name, validated; read
    once per root."""
    root = Path(root).resolve()
    if root not in _CACHE:
        specs = [_spec(path) for path in sorted(root.glob("*.toml"))]
        _CACHE[root] = {spec.name: spec for spec in specs}
    return _CACHE[root]


def service(name: str, root: Path = SERVICES_ROOT) -> ServiceSpec:
    """The manifest of service `name`; an unknown name is an error."""
    services = load_services(root)
    if name not in services:
        raise ValueError(f"Unknown service {name!r}; known: {sorted(services)}")
    return services[name]
