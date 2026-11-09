"""Configuration: TOML files that include others, with paths relative to the file
that names them, and `--set key.path=value` overrides.

Layers, later ones winning (configs/README.md):

    code defaults            the values the code uses when a file names none
    default.toml             the agent: prompt, skills, services, runtime, budget, and
                             its default model (ablations/<name>.toml include it)
    models/<name>.toml       model roles: main, vlm, pointer, verifier (an API model or a
                             service), plus the runtime settings a model needs
    experiments/<name>.toml  the agent with a model on a benchmark: tasks, seeds,
                             layouts, budgets, worker options, overrides
    --model, --set           the command line

and configs/deploy/<machine>.toml says where services run (services/deploy.py). Keys
live in environment variables, never in these files.
"""

import copy
import os
from pathlib import Path
import tomllib


def read_toml(path: Path) -> dict:
    """One TOML file as a tree, without following its includes."""
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def deep_merge(base: dict, override: dict) -> dict:
    """`override` on top of `base`: tables merge key by key, anything else replaces."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# Keys whose values are paths, as dotted paths into a file's tree. They are made
# absolute against the directory of the file that sets them, before files merge.
PATH_KEYS = ("prompt", "runtime.task_descriptions", "runtime.library.path", "agent",
             "overrides.agent.prompt", "overrides.agent.runtime.task_descriptions",
             "overrides.agent.runtime.library.path")


def _absolutize(tree: dict, base: Path) -> dict:
    tree = copy.deepcopy(tree)
    for dotted in PATH_KEYS:
        *parents, last = dotted.split(".")
        node = tree
        for part in parents:
            node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                break
        if not isinstance(node, dict) or last not in node:
            continue
        value = node[last]
        if isinstance(value, str):
            node[last] = str((base / value).resolve())
        elif isinstance(value, list):
            node[last] = [str((base / item).resolve()) if isinstance(item, str) else item for item in value]
    return tree


def load_layered(path, _seen=()) -> dict:
    """A TOML file with its `include = [...]` files merged under it (in order; the
    file's own keys win), path keys made absolute."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"Include cycle through {path}")
    data = read_toml(path)
    includes = data.pop("include", [])
    includes = [includes] if isinstance(includes, str) else list(includes)
    merged = {}
    for include in includes:
        merged = deep_merge(merged, load_layered(path.parent / include, (*_seen, path)))
    return deep_merge(merged, _absolutize(data, path.parent))


def parse_set(text: str) -> tuple[list[str], object]:
    """`a.b.c=value` as (["a", "b", "c"], value); the value is read as TOML (a number,
    true, "text", [1, 2], {k = 1}) and as a plain string when it is not valid TOML."""
    if "=" not in text:
        raise ValueError(f"--set needs key.path=value, got {text!r}")
    key, raw = text.split("=", 1)
    try:
        value = tomllib.loads(f"value = {raw}")["value"]
    except tomllib.TOMLDecodeError:
        value = raw
    return [part for part in key.strip().split(".") if part], value


def apply_sets(tree: dict, sets) -> dict:
    """A copy of `tree` with `--set` items written into it, in order. An item is a
    `key.path=value` string or a parsed (keys, value) pair; missing tables on the path
    are created, and a path through a value that is not a table is an error."""
    tree = copy.deepcopy(tree)
    for item in sets or ():
        keys, value = parse_set(item) if isinstance(item, str) else item
        node = tree
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f"--set {'.'.join(keys)}: {key} is not a table")
        node[keys[-1]] = value
    return tree


BASE_URL_ENV = "OPENROUTER_BASE_URL"
RELAY_URL_ENV = "OPENROUTER_RELAY_URL"
RELAY_TOKEN_ENV = "OPENROUTER_RELAY_TOKEN"


def route_openrouter(tree, environ=os.environ):
    """Point every OpenRouter model in a harness at another base URL, when asked.

    Applies to each model config with `backend = "openrouter"` and no `base_url` (so
    the openrouter.ai default) or an openrouter.ai `base_url`:

    - OPENROUTER_BASE_URL (e.g. https://my-proxy.example/api/v1) replaces the URL;
      the key is still read from the model's `api_key_env` (OPENROUTER_API_KEY).
    - OPENROUTER_RELAY_URL replaces the URL with scripts/openrouter_relay.py and
      reads the key from OPENROUTER_RELAY_TOKEN.

    Setting both is an error. Models elsewhere (self-hosted servers, other providers)
    are left alone. The rewritten configs go into the run snapshot, so a run records
    the URL it called.
    """
    base = environ.get(BASE_URL_ENV, "").strip()
    relay = environ.get(RELAY_URL_ENV, "").strip()
    if base and relay:
        raise ValueError(f"Set {BASE_URL_ENV} or {RELAY_URL_ENV}, not both")
    if not (base or relay):
        return tree
    if isinstance(tree, list):
        return [route_openrouter(item, environ) for item in tree]
    if not isinstance(tree, dict):
        return tree
    tree = {key: route_openrouter(value, environ) for key, value in tree.items()}
    base_url = tree.get("base_url")
    direct = isinstance(base_url, str) and "openrouter.ai" in base_url
    if tree.get("backend") == "openrouter" and (base_url is None or direct):
        tree["base_url"] = relay or base
        if relay:
            tree["api_key_env"] = RELAY_TOKEN_ENV
    return tree
