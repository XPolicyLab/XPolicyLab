#!/usr/bin/env python3
"""The frozen system of a measurement -- manifest, verification, import closure and tamper test.

A frozen measurement runs one fixed system. This module writes a manifest of everything that decides what runs and
compares the current state with it, entry by entry:

    /usr/bin/python3 -B agp_core/freeze.py write   [--manifest PATH]                 the manifest of the current state
    /usr/bin/python3 -B agp_core/freeze.py verify  [--manifest PATH] [--seal SEAL]   exit 0 when nothing differs
    /usr/bin/python3 -B agp_core/freeze.py tamper  [--manifest PATH] [--out JSON]    every entry changed alone is refused
    /usr/bin/python3 -B agp_core/freeze.py closure [--manifest PATH] [--out JSON]    the runtime's imports are all listed

Stdlib and PyYAML only (run it with /usr/bin/python3 -B or the .venv-sim python, never the shell's bare python3).
`closure` starts the two interpreters of the runtime (.venv-agent, and .venv-sim under activate_eval.sh) for import
probes; nothing else starts a process except `git` (read-only, --no-optional-locks) and the pinned codex binary's
`--version`.

Who verifies:
  * agp_core/official_deploy.py at the start of every episode (each layout is a new evaluator process), before the worker
    or the policy starts: any difference is an infrastructure failure of kind freeze_mismatch, which makes
    agp_core/run_official.py stop the batch instead of retrying;
  * agp_core/run_official.py --frozen-measurement before the batch and before every attempt (and once more at the end);
  * agp_core/run_agent.py (official conditions) re-checks the pinned codex binary and the prompt rendering against the
    pins the adapter wrote into the session (frozen_policy.json) right before codex starts.

Packaging (scripts/package_policy.py): the implementation runs from the packaged copy
POLICY/agp_runtime that deploy.yml agp.runtime_root names, and its interpreters and everything it writes are in AGP_HOME
(agp_core/paths.py; W on this machine, passed by agp_core/run_official.py). The state below therefore does not depend on where
this module itself was loaded from (W's copy in the launcher and the tools, the packaged copy in the adapter): Layout
takes the code root from deploy.yml and AGP_HOME from the environment.

The manifest (CLI default AGP_HOME/frozen/freeze.json; the adapter and the launchers use the fixed path agp_core/paths.py
FREEZE_MANIFEST, relative to AGP_HOME; never inside the policy folder or the checkout):
    format, created_utc, created_by   what the file is, when and by what it was written (metadata, not state)
    seal       sha256 of the canonical JSON of everything else: an edit anywhere, metadata included, is refused
    state      compared with the current state entry by entry; a missing or an additional entry is a difference too:
      roots         the implementation (deploy.yml runtime_root, the packaged runtime), the eval folder, the checkout, the
                    policy folder and AGP_HOME (realpath)
      deploy        deploy.yml: sha256 and every value (agp: model, effort, service_tier, episode limits,
                    runtime_root, codex_binary as written; the two paths may be relative, see agp_core/paths.py
                    deploy_pins, and the sections below use them resolved)
      run_controls  the run controls a frozen measurement pins (OURS_ROOT, AGP_EPISODE_TIMEOUT_S, AGP_END_GRACE_S), the
                    test-only controls it refuses (TEST_ONLY), and whether the pinned implementation is in the policy
                    folder
      flags         every switch of agp_core/features.py as the policy sees it (the built-in set agp_core/features.py SYSTEM,
                    which agp_core/official_deploy.py gives the policy; an A/B arm's overlay applied)
      prompt        sha256 of agp_core/PROMPT.md and of its rendering with those flags: the text agp_core/run_agent.py sends
                    to codex ahead of the per-episode launch message (task and first observation, preload switch)
      codex         the pinned release binary (path, not a link, sha256, `--version`), every file of its release
                    folder (every episode has a fresh CODEX_HOME, agp_core/run_agent.py: the system skills codex lists
                    to the model are the ones the pinned binary writes there)
      files         sha256 of every file of the policy folder, recursively (caches skipped): its own files and the
                    packaged runtime agp_runtime/ (agp_core/, migration/ without preview_assets, the control runtime modules
                    CONTROL_RUNTIME -- `closure` shows that they are exactly the ones loaded from source/agent -- and
                    the launch scripts scripts/activate_sim.sh and scripts/activate_eval.sh)
      constants     the public files our constants are derived from, by use: public camera (agp_core/public_camera.py),
                    joint limits (ARX.usd, X5A.urdf, curobo.yml, agp_core/policy_planner_wrist.json and the ARX.usd sha256
                    recorded in agp_core/policy_planner.py), visual gap (agp_core/visual_gap.py), the X5 robot asset folder and
                    the benchmark configuration (env_cfg/**, task/RoboDojo/config/_task.yml)
      checkout      HEAD, git status (superproject and submodules) and the submodule commits of the clean checkout
      python        the interpreters (sim: AGP_HOME/.venv-sim, agent: AGP_HOME/.venv-agent, policy: AGP_HOME/.venv-policy,
                    the stub policy server's): interpreter binary, pyvenv.cfg, the files at the top of site-packages
                    (.pth, editable finders) and every installed distribution (its RECORD)

Not frozen here (recorded per session instead, episode.json freeze.not_frozen): codex's credentials, its model cache
fetched from the provider, the provider-side model itself, and the list of the local user's skills (~/.agents/skills,
~/.codex/skills) that agp_core/run_agent.py disables on the codex command line (codex_experiment_config.py, a frozen file,
disables every one it finds at launch, so none reaches the model whatever the list holds). The account pool
(AGP_HOME/private/accounts, agp_core/accounts.py; logins renew themselves, so they can never be frozen) is recorded as a
summary only: labels and account hashes per slot, whether the local user's own login exists (account_pool_record).

Development mode and A/B arms (agp_core/official_deploy.py, agp_core/run_official.py, agp_core/evaluation/ab.py):
  * a frozen measurement is unchanged: agp_core/paths.py FREEZE_MANIFEST, verified as above, refused on any difference;
    a manifest with an `arm` section is refused there (it is an A/B arm's manifest, not a frozen system's);
  * an A/B arm's manifest is the same manifest with one more state section, `arm` (experiment, arm name, the arm's
    `overlay` -- ARM_KEYS: deploy.yml's model, effort, service_tier, and switches over the built-in set -- and the
    `effective` values); `flags` and `prompt` are then computed with the effective switches (what that arm's policy
    runs).
    write_arm() seals it at the start of an A/B batch; verify() of such a manifest collects the current state with
    the manifest's own overlay, so an arm verifies only while nothing but its configuration differs from the frozen
    system and nothing at all has changed since the batch started. A frozen measurement refuses the arm's run controls
    (ARM_CONTROLS: AGP_ARM_MANIFEST, AGP_ARM_SEAL) whatever its manifest lists (run_control_problems);
  * development runs (neither of the two) verify no manifest: the adapter records the digest of the current state
    (state_digest) and whether it still equals the frozen system's manifest, for the record only.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

import yaml

FORMAT = "agp-freeze/1"
W = Path(__file__).resolve().parents[1]       # the tree this module lives in (W, or the packaged runtime)
# the roots (agp_core/paths.py, by file path: this module is also loaded by file path, with W off sys.path)
_paths_spec = importlib.util.spec_from_file_location("agp_paths", W / "agp_core" / "paths.py")
paths = importlib.util.module_from_spec(_paths_spec)
_paths_spec.loader.exec_module(paths)
EVAL_ROOT = paths.EVAL                  # AGP_EVAL_ROOT, else W/../robodojo-eval (packaged: the folder of the checkout)
POLICY_DIR = paths.POLICY_DIR
# the CLI default when --manifest is not given (a neutral location in AGP_HOME); the adapter and the launchers use
# agp_core/paths.py FREEZE_MANIFEST (freeze_manifest()), never this default
DEFAULT_MANIFEST = paths.AGP_HOME / "frozen" / "freeze.json"
META = ("format", "created_utc", "created_by", "seal")
SUBMODULES = ("XPolicyLab", "third_party/IsaacLab", "third_party/curobo")
VENVS = {"sim": ".venv-sim", "agent": ".venv-agent", "policy": ".venv-policy"}
CODEX_HOME = "private/codex_episodes/<run>-<session>-<pid> (fresh per episode, agp_core/run_agent.py)"
# what the runtime writes or runs from in AGP_HOME: never an implementation file (closure, audit_loaded)
HOME_RUNTIME_DIRS = (".venv-sim", ".venv-agent", ".venv-policy", "cache", "config", "data", "logs", "tmp", "private",
                     "runs")
# The control runtime modules of W/source/agent that the policy imports (the MCP server: ControlRuntime, the tool
# models; agp_core/run_agent.py runs codex_experiment_config.py). `closure` checks this list against the imports.
CONTROL_RUNTIME = (
    "source/agent/agp/experiments/control_runtime/adapter.py",
    "source/agent/agp/experiments/control_runtime/errors.py",
    "source/agent/agp/experiments/control_runtime/mcp_server.py",
    "source/agent/agp/experiments/control_runtime/runtime.py",
    "source/agent/agp/tools/codex_experiment_config.py",
)
SKILL_CONFIG = "source/agent/agp/tools/codex_experiment_config.py"
# the launch scripts of the runtime (packaged with it; the client's default eval_env is scripts/activate_eval.sh)
LAUNCH_SCRIPTS = ("scripts/activate_sim.sh", "scripts/activate_eval.sh")
# run controls of the simulator process (environment): test-only ones a frozen measurement refuses, and the ones it pins
TEST_ONLY = ("AGP_SCRIPTED", "AGP_FAULT", "AGP_FREEZE_MANIFEST")
PINNED = ("OURS_ROOT", "AGP_EPISODE_TIMEOUT_S", "AGP_END_GRACE_S")
# an A/B arm (agp_core/evaluation/ab.py): the keys an arm may set (deploy.yml's model settings, and switches over the
# built-in set), and the arm's run controls (refused in a frozen measurement, independently of the TEST_ONLY list a
# manifest records)
ARM_KEYS = ("model", "effort", "service_tier", "switches")
ARM_CONTROLS = ("AGP_ARM_MANIFEST", "AGP_ARM_SEAL")
ARM_SECTION = "arm"
ARM_NAME = re.compile(r"[A-Za-z0-9_-]{1,40}")
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
CHUNK = 1 << 22
SHA = re.compile(r"[0-9a-f]{64}")


# ----------------------------------------------------------------------------- helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(CHUNK)
            if not block:
                return h.hexdigest()
            h.update(block)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def seal_of(manifest):
    """sha256 of the canonical JSON of the manifest without its seal."""
    return hashlib.sha256(canonical({k: v for k, v in manifest.items() if k != "seal"})).hexdigest()


def jsonable(data):
    return json.loads(json.dumps(data, default=str, allow_nan=False))


def file_entry(path):
    """sha256 of a regular file; 'symlink:<target>', 'directory' or 'missing' otherwise."""
    path = Path(path)
    if path.is_symlink():
        return "symlink:" + os.readlink(path)
    if path.is_file():
        return sha256_file(path)
    return "directory" if path.is_dir() else "missing"


def tree_entries(root, prefix):
    """{prefix + relative path: file_entry} of every file below root (caches skipped)."""
    root = Path(root)
    out = {}
    if not root.is_dir():
        return {prefix: file_entry(root)}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not Path(dirpath, d).is_symlink())
        for name in sorted(filenames) + sorted(d for d in os.listdir(dirpath)
                                               if Path(dirpath, d).is_symlink() and Path(dirpath, d).is_dir()):
            path = Path(dirpath, name)
            out[prefix + path.relative_to(root).as_posix()] = file_entry(path)
    return out


def git(repo, *args):
    env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"), "LANG": "C", "LC_ALL": "C"}
    out = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), *args], capture_output=True, text=True,
                         timeout=120, env=env)
    return out.stdout if out.returncode == 0 else f"<git {' '.join(args)} failed: {out.stderr.strip()[:200]}>"


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pinned_code_root(policy_dir):
    """deploy.yml agp.runtime_root of the policy folder, resolved (agp_core/paths.py deploy_pins); the packaged runtime
    POLICY/agp_runtime when deploy.yml cannot be read or has none."""
    policy_dir = Path(policy_dir)
    try:
        agp = (yaml.safe_load((policy_dir / "deploy.yml").read_text()) or {}).get("agp") or {}
        root = paths.deploy_pins(agp, policy_dir)["runtime_root"]
    except (OSError, yaml.YAMLError, AttributeError):
        root = None
    return Path(root) if isinstance(root, str) and root else policy_dir / paths.PACKAGE_DIRNAME


class Layout:
    """Where everything is, given the policy folder (the checkout and eval folder are its ancestors): the code root
    (`code`, default deploy.yml agp.runtime_root: the implementation the adapter loads, wherever this module was loaded
    from), AGP_HOME (`home`, default agp_core/paths.py agp_home(): the interpreters and what the runtime writes) and W, the
    tree this module lives in (only to name files loaded from it)."""

    def __init__(self, policy_dir=None, code=None, home=None):
        self.policy_dir = Path(policy_dir or POLICY_DIR).absolute()
        self.checkout = self.policy_dir.parents[2]
        self.eval_root = self.checkout.parent
        self.home = Path(home).absolute() if home is not None else paths.agp_home()
        self.code = Path(code).absolute() if code is not None else pinned_code_root(self.policy_dir)
        self.w = self.code              # the implementation (older name)
        self.dev = W
        # label order: the policy folder first (the packaged runtime is in it), then this module's tree
        self.roots = {"POLICY": self.policy_dir, "W": self.dev, "AGP_HOME": self.home, "CHECKOUT": self.checkout,
                      "EVAL": self.eval_root}

    def path(self, label):
        root, _, rel = label.partition(":")
        return self.roots[root] / rel if rel else self.roots[root]

    def label(self, path):
        """ROOT:relative of a path (the first root of `roots` holding it, by realpath), else the path itself."""
        real = Path(os.path.realpath(path))
        for root, base in self.roots.items():
            base = Path(os.path.realpath(base))
            if real.is_relative_to(base):
                return f"{root}:{real.relative_to(base).as_posix()}"
        return str(path)

    def runtime_state_file(self, path):
        """Whether a path is something the runtime keeps in AGP_HOME (interpreters, caches, sessions), not code."""
        real = Path(os.path.realpath(path))
        home = Path(os.path.realpath(self.home))
        return any(real.is_relative_to(home / name) for name in HOME_RUNTIME_DIRS)

    def code_inside_policy(self):
        return paths.inside(self.code, self.policy_dir)


# ----------------------------------------------------------------------------- the state
def _section(build, *args):
    try:
        return build(*args)
    except Exception as exc:            # recorded as the section's value: it then differs from the manifest
        return {"error": f"{type(exc).__name__}: {exc}"}


def deploy_state(lay):
    path = lay.policy_dir / "deploy.yml"
    return {"sha256": sha256_file(path), "values": jsonable(yaml.safe_load(path.read_text()) or {})}


def _agp(deploy):
    values = deploy.get("values") if isinstance(deploy, dict) else None
    return (values or {}).get("agp") or {}


def system_switches(code=None):
    """The built-in switch set (agp_core/features.py SYSTEM of the implementation at `code`, default the tree this module
    lives in) as environment strings."""
    features = load_by_path("_agp_freeze_features", Path(code if code is not None else W) / "agp_core" / "features.py")
    return features.system_switches()


def frozen_switches(agp, code=None):
    """The switch set a policy runs with, as environment strings (as agp_core/official_deploy.py load_config): the
    built-in set (system_switches of `code`) updated by agp's `switches` (an A/B arm's overlay, effective_agp)."""
    switches = system_switches(code)
    switches.update({str(k): str(int(v)) if isinstance(v, bool) else str(v)
                     for k, v in (agp.get("switches") or {}).items()})
    return switches


def run_controls_state(lay, agp):
    root = paths.deploy_pins(agp, lay.policy_dir)["runtime_root"]
    return {"pinned": {"OURS_ROOT": root, "AGP_EPISODE_TIMEOUT_S": float(agp.get("episode_timeout_s", 14400)),
                       "AGP_END_GRACE_S": float(agp.get("end_grace_s", 180))},
            "test_only": list(TEST_ONLY),
            # packaging: the pinned implementation is the policy folder's own copy
            "implementation_in_policy_folder": (isinstance(root, str) and bool(root)
                                                and paths.inside(root, lay.policy_dir)
                                                and Path(root, "agp_core", "official_deploy.py").is_file())}


def flags_and_prompt(lay, agp):
    features = load_by_path("_agp_freeze_features", lay.code / "agp_core" / "features.py")
    flags = features.current(frozen_switches(agp, lay.code))
    template = (lay.code / "agp_core" / "PROMPT.md").read_text()
    rendered = features.render(template, flags)
    prompt = {"template": lay.label(lay.code / "agp_core" / "PROMPT.md"),
              "template_sha256": sha256_file(lay.code / "agp_core" / "PROMPT.md"),
              "rendered_sha256": sha256_text(rendered), "rendered_bytes": len(rendered.encode("utf-8"))}
    return dict(sorted(flags.items())), prompt


def rendered_prompt(lay, agp):
    """The rendering of agp_core/PROMPT.md with the frozen flags (as agp_core/run_agent.py renders it)."""
    features = load_by_path("_agp_freeze_features", lay.code / "agp_core" / "features.py")
    return features.render((lay.code / "agp_core" / "PROMPT.md").read_text(),
                           features.current(frozen_switches(agp, lay.code)))


# ----------------------------------------------------------------------------- A/B arms
def validate_overlay(overlay):
    """An arm's overlay (deploy.yml's model settings, switches over the built-in set), normalised (model settings as
    strings, switches as environment strings); ValueError when it is empty, names another key than ARM_KEYS, or holds
    a malformed value."""
    if not isinstance(overlay, dict) or not overlay:
        raise ValueError(f"an arm overlay is a non-empty mapping of {ARM_KEYS}, got {overlay!r}")
    unknown = sorted(str(k) for k in set(overlay) - set(ARM_KEYS))
    if unknown:
        raise ValueError(f"an arm may set only {ARM_KEYS}, not {unknown}")
    out = {}
    for key in ("model", "effort", "service_tier"):
        if key in overlay:
            value = overlay[key]
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"arm overlay {key} must be a non-empty string, got {value!r}")
            out[key] = value
    if "switches" in overlay:
        switches = overlay["switches"]
        if not isinstance(switches, dict) or not switches:
            raise ValueError(f"arm overlay switches must be a non-empty mapping, got {switches!r}")
        out["switches"] = {}
        for name, value in sorted(switches.items()):
            if not isinstance(name, str) or not re.fullmatch(r"OURS_[A-Z0-9_]+", name):
                raise ValueError(f"arm overlay switch {name!r} is not an OURS_* name")
            if isinstance(value, bool):
                value = int(value)
            if not isinstance(value, (int, str)) or str(value).strip() != str(value) or str(value) == "":
                raise ValueError(f"arm overlay switch {name} needs a plain value, got {value!r}")
            if name == "OURS_RGB_ONLY" and str(value) == "0":
                raise ValueError("arm overlay OURS_RGB_ONLY=0 is refused: the policy observes and measures from RGB "
                                 "images only")
            out["switches"][name] = str(value)
    return out


def effective_agp(agp, overlay):
    """deploy.yml agp values with an arm's (validated) overlay applied: model settings replaced, switches updated (the
    overlay's switches; frozen_switches puts them over the built-in set)."""
    effective = copy.deepcopy(agp or {})
    for key in ("model", "effort", "service_tier"):
        if key in overlay:
            effective[key] = overlay[key]
    if "switches" in overlay:
        switches = dict(effective.get("switches") or {})
        switches.update(overlay["switches"])
        effective["switches"] = switches
    return effective


def arm_record(agp, arm, code=None):
    """The `arm` state section: {experiment, arm, overlay, effective} (effective = the model settings and the switch
    set the arm's policy runs with; the built-in set of the implementation at `code`, frozen_switches)."""
    if not isinstance(arm, dict):
        raise ValueError(f"an arm is {{experiment, arm, overlay}}, got {arm!r}")
    for label in ("experiment", "arm"):
        value = arm.get(label)
        if not isinstance(value, str) or not ARM_NAME.fullmatch(value):
            raise ValueError(f"arm {label} must match {ARM_NAME.pattern}, got {value!r}")
    overlay = validate_overlay(arm.get("overlay"))
    effective = effective_agp(agp, overlay)
    return {"experiment": arm["experiment"], "arm": arm["arm"], "overlay": overlay,
            "effective": {"model": effective.get("model"), "effort": effective.get("effort"),
                          "service_tier": effective.get("service_tier"),
                          "switches": frozen_switches(effective, code)}}


def arm_of(manifest):
    """{experiment, arm, overlay} of an A/B arm's manifest, None for any other manifest."""
    state = manifest.get("state") if isinstance(manifest, dict) else None
    section = state.get(ARM_SECTION) if isinstance(state, dict) else None
    if not isinstance(section, dict):
        return None
    return {"experiment": section.get("experiment"), "arm": section.get("arm"), "overlay": section.get("overlay")}


def state_digest(state):
    """sha256 of the canonical JSON of a collected state (development runs record it instead of verifying)."""
    return hashlib.sha256(canonical(state)).hexdigest()


def codex_state(lay, agp):
    binary = paths.deploy_pins(agp, lay.policy_dir)["codex_binary"]
    if not isinstance(binary, str) or not binary:
        return {"error": "deploy.yml agp.codex_binary is not set"}
    path = Path(binary)
    release = path.parent.parent
    try:
        package = json.loads((release / "codex-package.json").read_text())
    except (OSError, ValueError) as exc:
        package = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=60,
                             env={"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp")})
        version_output = (out.stdout or out.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        version_output = f"<{type(exc).__name__}: {exc}>"
    return {"binary": binary, "binary_is_symlink": path.is_symlink(), "binary_realpath": os.path.realpath(path),
            "through_auto_update_link": "/standalone/current/" in binary or binary.endswith("/.local/bin/codex"),
            "binary_sha256": file_entry(path), "version_output": version_output,
            "version": package.get("version") if isinstance(package, dict) else None,
            "release_dir": str(release), "package": package, "release_files": tree_entries(release, ""),
            "home": f"AGP_HOME:{CODEX_HOME}"}


def skill_overrides_record(policy_dir=None):
    """Not frozen: the local user's skills agp_core/run_agent.py disables on the codex command line (count, sha256)."""
    lay = Layout(policy_dir)
    try:
        text = load_by_path("_agp_freeze_skill_config", lay.code / SKILL_CONFIG).skill_overrides()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"disabled": text.count("enabled=false"), "sha256": sha256_text(text)}


def account_pool_record(home=None):
    """Not frozen: the account pool's summary (agp_core/accounts.py pool_summary: slot labels, account hashes; no
    credential)."""
    try:
        module = load_by_path("_agp_freeze_accounts", W / "agp_core" / "accounts.py")
        return module.pool_summary(home if home is not None else paths.AGP_HOME)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def files_state(lay):
    """Every file of the policy folder, recursively (caches skipped): its own files and the packaged runtime. An
    implementation outside the policy folder (only a development override; the adapter refuses it) adds the files the
    runtime loads from it (agp_core/*.py, agp_core/*.json, agp_core/PROMPT.md, migration/*.py,
    migration/*.json, migration/preview_assets/**, CONTROL_RUNTIME, the launch scripts)."""
    out = tree_entries(lay.policy_dir, "POLICY:")
    if not lay.code_inside_policy():
        code = lay.code
        found = []
        for pattern in ("agp_core/*.py", "agp_core/*.json", "migration/*.py", "migration/*.json"):
            found += sorted(code.glob(pattern))
        found += [code / "agp_core" / "PROMPT.md", *[code / rel for rel in CONTROL_RUNTIME + LAUNCH_SCRIPTS]]
        for path in found:
            out[lay.label(path)] = file_entry(path)
        base = lay.label(code / "migration" / "preview_assets")
        out.update(tree_entries(code / "migration" / "preview_assets", base + "/"))
    return dict(sorted(out.items()))


def _robot_files(lay):
    r = lay.checkout
    top = yaml.safe_load((r / "env_cfg" / "arx_x5.yml").read_text())["config"]
    robots = [x for x in yaml.safe_load((r / "env_cfg" / "robot" / f"{top['robot']}.yml").read_text())["robots"]
              if x.get("robot_type") == "arm"]
    robot = robots[0]["robot_name"]
    robot_dir = f"Assets/Robots/{robot}"
    config = yaml.safe_load((r / robot_dir / "robot_config.yml").read_text())
    urdf = os.path.normpath(f"{robot_dir}/{config['urdf_path']}")
    return top, robot_dir, urdf


def constants_state(lay):
    r = lay.checkout
    top, robot_dir, urdf = _robot_files(lay)

    def group(rels):
        return {f"CHECKOUT:{rel}": file_entry(r / rel) for rel in rels}
    public_camera = group(["env_cfg/arx_x5.yml", f"env_cfg/camera/{top['camera']}.yml", "env_cfg/camera/template.py",
                           "env/camera_manager/camera_manager.py", "utils/rotations.py", f"env_cfg/robot/{top['robot']}.yml",
                           f"env_cfg/scene/{top['scene']}.yml", f"{robot_dir}/robot_config.yml", urdf])
    joint_limits = group([f"{robot_dir}/ARX.usd", urdf, f"{robot_dir}/curobo.yml", f"{robot_dir}/robot_config.yml"])
    joint_limits[lay.label(lay.code / "agp_core" / "policy_planner_wrist.json")] = file_entry(
        lay.code / "agp_core" / "policy_planner_wrist.json")
    source = (lay.code / "agp_core" / "policy_planner.py").read_text()
    match = re.search(r'"file":\s*"Assets/Robots/x5/ARX\.usd",\s*"sha256":\s*"([0-9a-f]{64})"', source)
    joint_limits["recorded_ARX_usd_sha256_in_policy_planner"] = match.group(1) if match else None
    visual_gap = group([urdf, f"{robot_dir}/meshes/link7.STL", f"{robot_dir}/meshes/link8.STL",
                        f"{robot_dir}/robot_config.yml"])
    config = tree_entries(r / "env_cfg", "CHECKOUT:env_cfg/")
    config["CHECKOUT:task/RoboDojo/config/_task.yml"] = file_entry(r / "task" / "RoboDojo" / "config" / "_task.yml")
    return {"public_camera": public_camera, "joint_limits": joint_limits, "visual_gap": visual_gap,
            "robot_assets": tree_entries(r / robot_dir, f"CHECKOUT:{robot_dir}/"), "benchmark_config": config}


def checkout_state(lay):
    r = lay.checkout
    gitlinks = {}
    for line in git(r, "ls-tree", "HEAD", *SUBMODULES).splitlines():
        parts = line.split()
        if len(parts) >= 4:
            gitlinks[line.split("\t", 1)[1]] = parts[2]
    return {"head": git(r, "rev-parse", "HEAD").strip(),
            "status": git(r, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=untracked"),
            "submodules": {sm: {"head": git(r / sm, "rev-parse", "HEAD").strip(), "gitlink": gitlinks.get(sm),
                                "status": git(r / sm, "status", "--porcelain=v1", "--untracked-files=all")}
                           for sm in SUBMODULES}}


def venv_state(lay, name):
    venv = lay.home / VENVS[name]
    python = venv / "bin" / "python"
    sites = sorted(venv.glob("lib/python*/site-packages"))
    if len(sites) != 1:
        return {"error": f"expected one site-packages in {venv}, found {[str(s) for s in sites]}"}
    site = sites[0]
    top = {p.name: file_entry(p) for p in sorted(site.iterdir()) if p.is_file() and not p.is_symlink()}
    dists = {p.name: file_entry(p / "RECORD") for p in sorted(site.glob("*.dist-info"))}
    real = os.path.realpath(python)
    return {"python": f"AGP_HOME:{VENVS[name]}/bin/python", "realpath": real, "interpreter_sha256": file_entry(real),
            "pyvenv_cfg_sha256": file_entry(venv / "pyvenv.cfg"), "site_packages": site.relative_to(venv).as_posix(),
            "site_top_level": top, "dists": dists}


def collect(policy_dir=None, arm=None):
    """The current state, in the manifest's `state` layout (JSON values only). With `arm` ({experiment, arm, overlay})
    the state of that A/B arm: an `arm` section, and flags and prompt from the arm's effective switches."""
    lay = Layout(policy_dir)
    deploy = _section(deploy_state, lay)
    agp = _agp(deploy)
    used = agp
    if arm is not None:
        record = _section(arm_record, agp, arm, lay.code)
        if "error" not in record:
            used = effective_agp(agp, record["overlay"])
    flags_prompt = _section(flags_and_prompt, lay, used)
    flags, prompt = flags_prompt if isinstance(flags_prompt, tuple) else (flags_prompt, flags_prompt)
    state = {
        "roots": {"ours_root": str(lay.code), "eval_root": str(lay.eval_root), "checkout": str(lay.checkout),
                  "policy_dir": str(lay.policy_dir), "agp_home": os.path.realpath(lay.home)},
        "deploy": deploy,
        "run_controls": _section(run_controls_state, lay, agp),
        "flags": flags,
        "prompt": prompt,
        "codex": _section(codex_state, lay, agp),
        "files": _section(files_state, lay),
        "constants": _section(constants_state, lay),
        "checkout": _section(checkout_state, lay),
        "python": {name: _section(venv_state, lay, name) for name in VENVS},
    }
    if arm is not None:
        state[ARM_SECTION] = record
    return jsonable(state)


def build_manifest(policy_dir=None, created_by=None, state=None):
    manifest = {"format": FORMAT,
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "created_by": created_by or f"agp_core/freeze.py write (pid {os.getpid()})",
                "state": state if state is not None else collect(policy_dir)}
    manifest["seal"] = seal_of(manifest)
    return manifest


def _write_json(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _refuse_hashed_location(path, policy_dir=None):
    """A manifest inside the policy folder or the checkout would change the state it records (self-reference)."""
    lay = Layout(policy_dir)
    for root in (lay.policy_dir, lay.checkout):
        if paths.inside(Path(path).parent, root):
            raise ValueError(f"{path} lies in {root}: a manifest must be outside the policy folder and the checkout")


def write(path=None, policy_dir=None, created_by=None):
    path = Path(path or DEFAULT_MANIFEST)
    _refuse_hashed_location(path, policy_dir)
    manifest = build_manifest(policy_dir, created_by)
    _write_json(path, manifest)
    return manifest


def write_arm(path, experiment, arm, overlay, policy_dir=None, created_by=None):
    """Seal the manifest of one A/B arm (the current state with the arm's overlay) at `path`; ValueError when the
    overlay is not a valid arm configuration. Never writes the frozen system's manifest."""
    path = Path(path)
    lay = Layout(policy_dir)
    agp = _agp(_section(deploy_state, lay))
    frozen = paths.freeze_manifest(lay.home)
    if path.resolve() in {Path(DEFAULT_MANIFEST).resolve(), Path(frozen).resolve()}:
        raise ValueError(f"{path} is the frozen system's manifest, not an arm's")
    _refuse_hashed_location(path, policy_dir)
    spec = {"experiment": experiment, "arm": arm, "overlay": overlay}
    arm_record(agp, spec)            # raises on an invalid arm before anything is written
    manifest = build_manifest(policy_dir, created_by or f"agp_core/freeze.py write_arm ({experiment}/{arm})",
                              state=collect(policy_dir, arm=spec))
    _write_json(path, manifest)
    return manifest


# ----------------------------------------------------------------------------- comparison
def _show(value, limit=160):
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= limit else text[:limit] + f"... ({len(text)} chars)"


def entry_name(path):
    return "::".join(str(p) for p in path)


def compare(expected, found, path=()):
    """[{entry, path, expected, found}] for every entry that differs; missing and additional entries included."""
    if isinstance(expected, dict) and isinstance(found, dict):
        problems = []
        for key in sorted(set(expected) | set(found), key=str):
            if key not in found:
                problems.append(_problem(path + (key,), expected[key], "<not in the current state>"))
            elif key not in expected:
                problems.append(_problem(path + (key,), "<not in the manifest>", found[key]))
            else:
                problems.extend(compare(expected[key], found[key], path + (key,)))
        return problems
    return [] if expected == found else [_problem(path, expected, found)]


def _problem(path, expected, found):
    return {"entry": entry_name(path), "path": list(path), "expected": _show(expected), "found": _show(found)}


def check_manifest(manifest, current, expected_seal=None):
    """Problems of a manifest (dict) against the current state (collect()); [] when it verifies."""
    if not isinstance(manifest, dict):
        return [_problem(("manifest",), "a JSON object", type(manifest).__name__)]
    problems = []
    if manifest.get("format") != FORMAT:
        problems.append(_problem(("format",), FORMAT, manifest.get("format")))
    seal = manifest.get("seal")
    if seal != seal_of(manifest):
        problems.append(_problem(("seal",), "the sha256 of the manifest's content", seal))
    if expected_seal is not None and seal != expected_seal:
        problems.append(_problem(("seal",), f"{expected_seal} (the seal the launcher verified)", seal))
    extra = sorted(set(manifest) - set(META) - {"state"})
    for key in extra:
        problems.append(_problem((key,), "<not a manifest entry>", manifest[key]))
    if "state" not in manifest:
        problems.append(_problem(("state",), "the frozen state", "<missing>"))
    else:
        problems.extend(compare(manifest["state"], current, ("state",)))
    return problems


def load_manifest(path):
    return json.loads(Path(path).read_text())


def verify(manifest_path=None, policy_dir=None, *, expected_seal=None, current=None):
    """Verdict of the manifest at manifest_path against the current state:
    {ok, manifest, seal, problems, entries, seconds}."""
    started = time.monotonic()
    path = Path(manifest_path or DEFAULT_MANIFEST)
    verdict = {"ok": False, "manifest": str(path), "seal": None, "problems": [], "entries": 0}
    try:
        manifest = load_manifest(path)
    except (OSError, ValueError) as exc:
        verdict["problems"] = [_problem(("manifest",), "a readable manifest", f"{type(exc).__name__}: {exc}")]
        verdict["seconds"] = round(time.monotonic() - started, 3)
        return verdict
    if current is None:
        current = collect(policy_dir, arm=arm_of(manifest))      # an arm's manifest with its own overlay
    verdict["seal"] = manifest.get("seal") if isinstance(manifest, dict) else None
    verdict["arm"] = arm_of(manifest)
    verdict["created_utc"] = manifest.get("created_utc") if isinstance(manifest, dict) else None
    verdict["problems"] = check_manifest(manifest, current, expected_seal)
    verdict["entries"] = sum(1 for _ in leaves(manifest)) if isinstance(manifest, dict) else 0
    verdict["ok"] = not verdict["problems"]
    verdict["seconds"] = round(time.monotonic() - started, 3)
    return verdict


def run_control_problems(environ, state):
    """A frozen measurement's run controls (environment of the simulator process) against the manifest state:
    a test-only control set at all, or a pinned control set to another value than the manifest's."""
    controls = (state or {}).get("run_controls") or {}
    pinned = controls.get("pinned") or {}
    problems = []
    for name in ARM_CONTROLS:                   # an A/B arm's configuration never enters a frozen measurement
        if environ.get(name):
            problems.append({"entry": f"run control {name}", "path": ["run_controls", "arm", name],
                             "expected": "unset (an A/B arm's configuration: refused in a frozen measurement)",
                             "found": _show(environ[name])})
    if isinstance(state, dict) and ARM_SECTION in state:
        problems.append({"entry": "manifest arm section", "path": ["state", ARM_SECTION],
                         "expected": "a frozen system's manifest (no A/B arm section)",
                         "found": _show(state.get(ARM_SECTION))})
    for name in controls.get("test_only") or TEST_ONLY:
        if environ.get(name):
            problems.append({"entry": f"run control {name}", "path": ["run_controls", "test_only", name],
                             "expected": "unset (test-only: refused in a frozen measurement)",
                             "found": _show(environ[name])})
    for name in PINNED:
        value = environ.get(name)
        if not value:
            continue
        want = pinned.get(name)
        if name == "OURS_ROOT":
            same = isinstance(want, str) and os.path.realpath(value) == os.path.realpath(want)
        else:
            try:
                same = float(value) == float(want)
            except (TypeError, ValueError):
                same = False
        if not same:
            problems.append({"entry": f"run control {name}", "path": ["run_controls", "pinned", name],
                             "expected": _show(want), "found": _show(value)})
    return problems


def summary_lines(verdict, limit=12):
    lines = [f"frozen system {'VERIFIED' if verdict['ok'] else 'MISMATCH'}: {verdict['manifest']} (seal "
             f"{str(verdict.get('seal'))[:16]}, {verdict.get('entries')} entries, {verdict.get('seconds')} s)"]
    for problem in verdict["problems"][:limit]:
        lines.append(f"  {problem['entry']}: expected {problem['expected']}, found {problem['found']}")
    if len(verdict["problems"]) > limit:
        lines.append(f"  ... {len(verdict['problems']) - limit} more")
    return lines


# ----------------------------------------------------------------------------- tamper test
def leaves(data, path=()):
    """Paths of every leaf (a non-dict value, or an empty dict) of a manifest."""
    if isinstance(data, dict) and data:
        for key in sorted(data, key=str):
            yield from leaves(data[key], path + (key,))
    else:
        yield path


def get_at(data, path):
    for key in path:
        data = data[key]
    return data


def set_at(data, path, value):
    for key in path[:-1]:
        data = data[key]
    data[path[-1]] = value


def mutate(value):
    """A different value of the same kind."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, str):
        if SHA.fullmatch(value):
            return value[:-1] + ("0" if value[-1] != "0" else "1")
        return value + "~tampered"
    if isinstance(value, list):
        return value + ["~tampered"]
    if isinstance(value, dict):
        return {"~tampered": True}
    return "~tampered"


def reseal(manifest):
    manifest = copy.deepcopy(manifest)
    manifest["seal"] = seal_of(manifest)
    return manifest


def tamper_test(manifest, current):
    """Every entry of the manifest changed alone, in a copy (the real files are never touched):
      plain     the copy as edited -> must be refused (the seal no longer matches; a state entry also differs)
      resealed  the copy with a recomputed seal -> must be refused for every state entry, and the refusal must name
                that entry; metadata (created_utc, created_by) is then accepted (it is not state), a changed seal
                is restored by resealing, a changed format is refused
    plus structural copies (resealed): each state section removed, an additional file entry, an unknown top-level key."""
    base = check_manifest(manifest, current)
    report = {"base_problems": base, "entries": 0, "plain_refused": 0, "resealed_refused": 0,
              "resealed_named_entry": 0, "metadata_accepted_when_resealed": [], "failures": [], "structural": []}
    if base:
        report["failures"].append("the manifest does not verify against the current state; the tamper test needs an "
                                  "intact manifest")
        return report
    for path in leaves(manifest):
        report["entries"] += 1
        changed = copy.deepcopy(manifest)
        set_at(changed, path, mutate(get_at(manifest, path)))
        plain = check_manifest(changed, current)
        if plain:
            report["plain_refused"] += 1
        else:
            report["failures"].append(f"plain edit of {entry_name(path)} was accepted")
        if path == ("seal",):
            continue
        resealed = check_manifest(reseal(changed), current)
        if path[0] == "state":
            if resealed:
                report["resealed_refused"] += 1
            else:
                report["failures"].append(f"resealed edit of {entry_name(path)} was accepted")
            if any(tuple(p["path"]) == path or tuple(p["path"]) == path[:len(p["path"])] for p in resealed):
                report["resealed_named_entry"] += 1
            else:
                report["failures"].append(f"resealed edit of {entry_name(path)}: refusal does not name it "
                                          f"({[p['entry'] for p in resealed][:3]})")
        elif path == ("format",):
            if resealed:
                report["resealed_refused"] += 1
            else:
                report["failures"].append("resealed edit of format was accepted")
        elif resealed:
            report["failures"].append(f"resealed edit of metadata {entry_name(path)} was refused: {resealed[:2]}")
        else:
            report["metadata_accepted_when_resealed"].append(entry_name(path))
    for section in sorted(manifest["state"]):
        changed = copy.deepcopy(manifest)
        del changed["state"][section]
        problems = check_manifest(reseal(changed), current)
        report["structural"].append({"edit": f"state section {section} removed", "refused": bool(problems)})
    changed = copy.deepcopy(manifest)
    changed["state"]["files"]["W:agp_core/unlisted_module.py"] = "0" * 64
    report["structural"].append({"edit": "additional file entry W:agp_core/unlisted_module.py",
                                 "refused": bool(check_manifest(reseal(changed), current))})
    changed = copy.deepcopy(manifest)
    changed["extra"] = 1
    report["structural"].append({"edit": "unknown top-level key", "refused": bool(check_manifest(reseal(changed),
                                                                                                 current))})
    for item in report["structural"]:
        if not item["refused"]:
            report["failures"].append(f"structural edit accepted: {item['edit']}")
    report["ok"] = not report["failures"]
    return report


# ----------------------------------------------------------------------------- the runtime's imports
AGENT_ENTRIES = ("agp_core.server", "agp_core.preload", "agp_core.assess_watch", "agp_core.features")     # MCP server, run_agent.py
SIM_ENTRY = "XPolicyLab.policy.AGP.deploy"                                                 # what main.py imports
SCRIPT_ENTRIES = ("agp_core/run_agent.py", SKILL_CONFIG)                                       # run as scripts
# started by path or by `-m` (a string, invisible to the import closure). Not the test-only crash server
# agp_core/mcp_crash_server.py (AGP_FAULT=mcp_crash): `closure` probes the packaged runtime, which leaves it out
# (scripts/package_policy.py), and the packaged adapter refuses that fault
STARTED_ENTRIES = ("agp_core/run_agent.py", "agp_core/official_scripted.py", "agp_core/server.py", SKILL_CONFIG)


def module_file(name, lay, importer=None):
    """The workspace file of an importable module name, or None when it is not ours (third party, stdlib, benchmark).
    `importer`: the importing file, for the control runtime's own sibling imports (`from adapter import ...`)."""
    parts = name.split(".")
    candidates = []
    if parts[0] in ("agp_core", "migration"):
        candidates.append(lay.code.joinpath(*parts))
    elif parts[0] == "agp":
        candidates.append(lay.code.joinpath("source", "agent", *parts))
    elif importer is not None and Path(importer).is_relative_to(lay.code / "source" / "agent"):
        candidates.append(Path(importer).parent.joinpath(*parts))
    for base in candidates:
        if base.with_suffix(".py").is_file():
            return base.with_suffix(".py")
        if (base / "__init__.py").is_file():
            return base / "__init__.py"
    return None


def static_closure(lay, entries):
    """Implementation files reachable through any import statement (module level or inside functions) from the entry
    files: a superset of what the runtime can load from the code root."""
    todo = [Path(p) for p in entries]
    seen = set()
    while todo:
        path = todo.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for name in names:
                found = module_file(name, lay, importer=path)
                if found is not None and found not in seen:
                    todo.append(found)
    return seen


PROBE = r"""
import importlib, json, sys
errors = {}
for name in sys.argv[1].split(","):
    try:
        importlib.import_module(name)
    except BaseException as exc:
        errors[name] = f"{type(exc).__name__}: {exc}"
extra = sys.argv[2] if len(sys.argv) > 2 else ""
if extra == "deploy":
    import XPolicyLab.policy.AGP.deploy as shim
    impl = shim._implementation()
    for name in ("agp_core.official_worker", "agp_core.policy_planner", "agp_core.approx_home", "agp_core.planner_injection",
                 "agp_core.rgb_observe", "agp_core.visual_gap", "agp_core.public_camera", "agp_core.freeze",
                 "migration.worker", "migration.kinematics", "migration.robot_geometry"):
        try:
            importlib.import_module(name)
        except BaseException as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
files = sorted({getattr(m, "__file__", None) or "" for m in list(sys.modules.values())} - {""})
print("@@PROBE@@" + json.dumps({"files": files, "errors": errors, "executable": sys.executable}))
"""


def _run_probe(cmd, env, cwd):
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env, cwd=cwd)
    marker = [line for line in out.stdout.splitlines() if line.startswith("@@PROBE@@")]
    if not marker:
        return {"error": f"exit {out.returncode}", "stderr": out.stderr[-3000:], "stdout": out.stdout[-2000:]}
    return json.loads(marker[-1][len("@@PROBE@@"):])


def from_patched_copy(name):
    """Whether a label names a file of a patched benchmark copy (W/source/RoboDojo, in whichever root)."""
    rel = name.partition(":")[2]
    return rel.startswith("source/RoboDojo/") or "/source/RoboDojo/" in rel


def implementation_label(lay, path):
    """The manifest label of an implementation file (the policy folder, W, or AGP_HOME outside what the runtime keeps
    there), None for anything else (third party, stdlib, benchmark, interpreters, caches)."""
    if not os.path.isabs(path):           # e.g. torch's _classes.py / _ops.py pseudo modules
        return None
    if lay.runtime_state_file(path):
        return None
    name = lay.label(path)
    return name if name.startswith(("POLICY:", "W:", "AGP_HOME:")) else None


def closure(manifest_path=None, policy_dir=None):
    """Import probes of the runtime's two interpreters (AGP_HOME's .venv-agent; .venv-sim under the packaged
    scripts/activate_eval.sh, the client's default eval_env) and the static import closure, against the manifest's
    files. ok when every implementation file the runtime loads or can import is listed (packaged: all of them in the
    policy folder; a file loaded from W or AGP_HOME is never listed), nothing is loaded from the patched copy
    W/source/RoboDojo, and the files loaded from source/agent are exactly CONTROL_RUNTIME."""
    lay = Layout(policy_dir)
    manifest = load_manifest(manifest_path or DEFAULT_MANIFEST)
    listed = set(manifest["state"]["files"])
    home = os.environ.get("HOME", "/tmp")
    base_env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": home, "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(lay.home / "tmp"), "AGP_HOME": str(lay.home),
                "LANG": os.environ.get("LANG", "C.UTF-8")}
    agent = _run_probe([str(lay.home / ".venv-agent" / "bin" / "python"), "-B", "-c", PROBE, ",".join(AGENT_ENTRIES)],
                       dict(base_env, PYTHONPATH=f"{lay.code}/source/agent:{lay.code}"), str(lay.code))
    sim_cmd = (f"source {shlex.quote(str(lay.code / 'scripts' / 'activate_eval.sh'))} && "
               f"cd {shlex.quote(str(lay.checkout))} && "
               f'exec python -B -c "$AGP_FREEZE_PROBE" XPolicyLab.policy.AGP.deploy deploy')
    sim = _run_probe(["bash", "-c", sim_cmd], dict(base_env, ROBODOJO_ROOT=str(lay.checkout), AGP_FREEZE_PROBE=PROBE),
                     str(lay.checkout))
    static = static_closure(lay, [lay.policy_dir / "deploy.py", lay.code / "agp_core" / "official_deploy.py",
                                  *[lay.code / rel for rel in STARTED_ENTRIES]])
    agent_root = lay.label(lay.code / "source" / "agent") + "/"

    def ours(paths):
        return {name for name in (implementation_label(lay, p) for p in paths) if name is not None}
    report = {"listed": len(listed)}
    problems = []
    for name, probe in (("agent", agent), ("sim", sim)):
        if "error" in probe:
            problems.append(f"{name} probe failed: {probe['error']} {probe.get('stderr', '')[-500:]}")
            report[name] = probe
            continue
        loaded = ours(probe["files"])
        report[name] = {"executable": probe["executable"], "import_errors": probe["errors"],
                        "workspace_files_loaded": sorted(loaded), "unlisted": sorted(loaded - listed),
                        "from_patched_copy": sorted(n for n in loaded if from_patched_copy(n)),
                        "from_source_agent": sorted(n for n in loaded if n.startswith(agent_root))}
        if probe["errors"]:
            problems.append(f"{name} probe import errors: {probe['errors']}")
        if loaded - listed:
            problems.append(f"{name}: loaded but not in the manifest: {sorted(loaded - listed)}")
        if report[name]["from_patched_copy"]:
            problems.append(f"{name}: loaded from a patched benchmark copy (source/RoboDojo)")
    static_labels = ours(str(p) for p in static)
    report["static"] = {"files": sorted(static_labels), "unlisted": sorted(static_labels - listed)}
    if static_labels - listed:
        problems.append(f"static closure: importable but not in the manifest: {sorted(static_labels - listed)}")
    runtime_agent = {lay.label(lay.code / rel) for rel in CONTROL_RUNTIME}
    agent_side = set(report.get("agent", {}).get("from_source_agent", [])) | {lay.label(lay.code / SKILL_CONFIG)}
    static_agent = {n for n in static_labels if n.startswith(agent_root)}
    report["control_runtime"] = {"listed": sorted(runtime_agent), "loaded_by_agent_probe_plus_script": sorted(agent_side),
                                 "static": sorted(static_agent)}
    if agent_side != runtime_agent:
        problems.append(f"control runtime: loaded {sorted(agent_side)} != CONTROL_RUNTIME {sorted(runtime_agent)}")
    if not static_agent <= runtime_agent:
        problems.append(f"control runtime: statically importable {sorted(static_agent - runtime_agent)} not listed")
    report["problems"] = problems
    report["ok"] = not problems
    return report


def loaded_implementation_files(policy_dir=None, modules=None):
    """Implementation files of the modules loaded in this process (the policy folder, W, AGP_HOME outside the
    interpreters, caches and sessions; implementation_label)."""
    lay = Layout(policy_dir)
    out = set()
    for module in list((modules if modules is not None else sys.modules).values()):
        path = getattr(module, "__file__", None)
        if not path:
            continue
        name = implementation_label(lay, path)
        if name is not None:
            out.add(name)
    return sorted(out)


def audit_loaded(manifest_state, policy_dir=None, modules=None):
    """This process's loaded workspace files against the manifest (runtime audit, recorded per episode)."""
    loaded = loaded_implementation_files(policy_dir, modules)
    listed = set(((manifest_state or {}).get("files") or {}))
    return {"loaded": len(loaded), "unlisted": [n for n in loaded if n not in listed],
            "from_patched_copy": [n for n in loaded if from_patched_copy(n)]}


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("write", "verify", "tamper", "closure", "show"))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--policy-dir", default=str(POLICY_DIR))
    parser.add_argument("--seal", help="verify: the seal the manifest must have")
    parser.add_argument("--out", help="tamper / closure: write the report as JSON here")
    parser.add_argument("--home", help="AGP_HOME (default: the environment's, else agp_core/paths.py's default)")
    args = parser.parse_args(argv)
    if args.home:
        os.environ["AGP_HOME"] = os.path.abspath(os.path.expanduser(args.home))
    if args.command == "write":
        manifest = write(args.manifest, args.policy_dir, created_by="agp_core/freeze.py write")
        state = manifest["state"]
        print(f"wrote {args.manifest}: seal {manifest['seal']}, {sum(1 for _ in leaves(manifest))} entries, "
              f"{len(state['files'])} files, codex {state['codex'].get('version_output')}")
        verdict = verify(args.manifest, args.policy_dir)
        print("\n".join(summary_lines(verdict)))
        return 0 if verdict["ok"] else 1
    if args.command == "verify":
        verdict = verify(args.manifest, args.policy_dir, expected_seal=args.seal)
        print("\n".join(summary_lines(verdict, limit=50)))
        return 0 if verdict["ok"] else 1
    if args.command == "show":
        print(json.dumps(load_manifest(args.manifest), indent=1, sort_keys=True))
        return 0
    if args.command == "tamper":
        started = time.monotonic()
        current = collect(args.policy_dir)
        report = tamper_test(load_manifest(args.manifest), current)
        report["seconds"] = round(time.monotonic() - started, 2)
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
        print(f"tamper test of {args.manifest}: {report['entries']} entries; refused as edited "
              f"{report['plain_refused']}/{report['entries']}; refused after resealing {report['resealed_refused']} "
              f"(state entries and format), refusal naming the entry {report['resealed_named_entry']}; metadata "
              f"accepted after resealing: {report['metadata_accepted_when_resealed']}; structural "
              f"{sum(i['refused'] for i in report['structural'])}/{len(report['structural'])} refused; "
              f"failures {len(report['failures'])}")
        for failure in report["failures"][:20]:
            print("  FAIL", failure)
        return 0 if report.get("ok") else 1
    report = closure(args.manifest, args.policy_dir)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    for name in ("agent", "sim"):
        part = report.get(name, {})
        print(f"{name} probe: {len(part.get('workspace_files_loaded', []))} workspace files loaded, unlisted "
              f"{part.get('unlisted')}, from source/agent {part.get('from_source_agent')}")
    print(f"static closure: {len(report['static']['files'])} files, unlisted {report['static']['unlisted']}")
    print(f"control runtime: {report['control_runtime']}")
    print("closure OK" if report["ok"] else "closure PROBLEMS:\n  " + "\n  ".join(report["problems"]))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
