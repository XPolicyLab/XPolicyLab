"""Where the AGP workspace finds what lies outside it, with no machine-specific absolute path (nothing depends on a
local absolute path).

The implementation lives in two places with the same relative layout (agp_core/, migration/, source/agent/..., scripts/,
env/):

    the development tree W        the source of truth (git), tooling, tests, runs
    the packaged runtime          POLICY_DIR/agp_runtime, an exact copy of the runtime files made by
                                  scripts/package_policy.py (the leaderboard needs a self-contained adapter);
                                  deploy.yml agp.runtime_root names it, and the evaluator, the policy and the frozen
                                  measurement load only this copy

This file tells the two apart by its location (PACKAGED: the folder above agp_core/ is agp_runtime in a policy folder,
next to a deploy.yml) and derives every root from an environment variable or, when it is unset, from its location
(the default layout: W and the eval folder side by side, the clean checkout inside the eval folder):

    CODE          the tree this file lives in (W, or POLICY_DIR/agp_runtime); W is kept as its name
    AGP_HOME      AGP_HOME, else CODE in the development tree, else DEFAULT_HOME (~/.agp-robodojo) when packaged: what
                  the runtime writes and the interpreters it uses -- .venv-sim, .venv-agent, private/codex_episodes,
                  tmp, cache, config, data, logs and the default runs/official; never inside the checkout. The local
                  launcher passes AGP_HOME = W, so everything stays in the development tree.
    EVAL          packaged: the folder holding the checkout; else AGP_EVAL_ROOT, else W/../robodojo-eval
                  (activate_eval.sh, tools/ and the clean checkout)
    CHECKOUT      packaged: the checkout the policy folder is in; else EVAL/RoboDojo (the clean official checkout)
    POLICY_DIR    packaged: the folder above CODE; else CHECKOUT/XPolicyLab/policy/AGP (deploy.py, deploy.yml)
    PACKAGE       POLICY_DIR/agp_runtime, the packaged runtime
    upstream()    AGP_UPSTREAM_ROOT, else the folder the checkout's Assets link points into (CHECKOUT/Assets ->
                  UPSTREAM/Assets; without a link, the checkout itself): the upstream tree the zero-patch checks compare
    RUNTIME_LIBS  AGP_RUNTIME_LIBS, else AGP_HOME/../robodojo/runtime-libs: system libraries unpacked without root
                  (scripts/activate_sim.sh prepends RUNTIME_LIBS/usr/lib/x86_64-linux-gnu when the folder exists)
    HOME          the user's home (Path.home()): ~/.codex (login, pinned release), ~/.local/bin/codex, ~/.agents

deploy.yml's path pins (agp.runtime_root, codex_binary) are written relative (deploy_pins below). The frozen system's
manifest is not a deploy.yml entry: its path is fixed here (FREEZE_MANIFEST, relative to AGP_HOME; freeze_manifest()).

Stdlib only. Modules that W's package is imported from use `from agp_core import paths`; modules that are also loaded by
file path with W off sys.path (agp_core/freeze.py, agp_core/run_official.py, agp_core/evaluation/ab.py, the checks, the viewer)
load this file by its path next to them (importlib.util.spec_from_file_location), so the roots always belong to the
tree the caller itself lives in.
"""
import os
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
W = CODE
PACKAGE_DIRNAME = "agp_runtime"
# packaged: CODE is POLICY_DIR/agp_runtime (a policy folder holds deploy.yml)
PACKAGED = CODE.name == PACKAGE_DIRNAME and (CODE.parent / "deploy.yml").is_file()
DEFAULT_HOME = "~/.agp-robodojo"           # the shell scripts of the policy folder use the same default


def agp_home(environ=None):
    """AGP_HOME (absolute, ~ expanded), else CODE in the development tree, else DEFAULT_HOME when packaged."""
    environ = os.environ if environ is None else environ
    value = environ.get("AGP_HOME") or (DEFAULT_HOME if PACKAGED else str(CODE))
    return Path(os.path.abspath(os.path.expanduser(value)))


AGP_HOME = agp_home()
if PACKAGED:
    POLICY_DIR = CODE.parent
    CHECKOUT = POLICY_DIR.parents[2]
    EVAL = CHECKOUT.parent
else:
    EVAL = Path(os.environ.get("AGP_EVAL_ROOT") or W.parent / "robodojo-eval")
    CHECKOUT = EVAL / "RoboDojo"
    POLICY_DIR = CHECKOUT / "XPolicyLab" / "policy" / "AGP"
PACKAGE = POLICY_DIR / PACKAGE_DIRNAME
RUNTIME_LIBS = Path(os.environ.get("AGP_RUNTIME_LIBS") or AGP_HOME.parent / "robodojo" / "runtime-libs")
HOME = Path.home()
# the environment variables of the roots above (agp_core/checks/run.py passes those that are set on to every check)
ROOT_VARIABLES = ("AGP_EVAL_ROOT", "AGP_UPSTREAM_ROOT", "AGP_RUNTIME_LIBS", "AGP_HOME")
PIN_KEYS = ("runtime_root", "codex_binary")
# The frozen system's manifest (agp_core/freeze.py), relative to AGP_HOME: a manifest is machine state and lies outside
# the policy folder whose files it hashes. Fixed in code (not a deploy.yml entry: agp_core/official_deploy.py load_config
# refuses agp.freeze_manifest); a frozen measurement verifies it before every episode (agp_core/official_deploy.py),
# a development run only records whether the current state still matches it. Test only: AGP_FREEZE_MANIFEST.
FREEZE_MANIFEST = "agp_core/frozen/os_ev/freeze.json"
# The official checkout this system runs on (RoboDojo main 09bd5f5e, which includes upstream PR #48, the articulation
# reset fix that lets swap_blocks' button rest at 100% with one environment). The zero-patch check
# (agp_core/checks/zero_patch.py) and scripts/setup_node.sh / env/sim/versions.json hold the same commits.
CHECKOUT_COMMIT = "09bd5f5e762b949f5858ece31347ce62057b3a5e"
CHECKOUT_SUBMODULES = {"XPolicyLab": "53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2",
                       "third_party/IsaacLab": "afca7b09d60d8beb9c1cb28b43066499940b969b",
                       "third_party/curobo": "d17b54ce32cba095c0b000c4c58777075d11de0e"}
# the policy folder: these files and the packaged runtime folder (a README.md there is allowed, not required;
# install.sh: the XPolicyLab file standard's environment setup)
POLICY_FILES = ("__init__.py", "deploy.py", "deploy.yml", "eval.sh", "install.sh", "model.py",
                "setup_eval_env_client.sh", "setup_eval_policy_server.sh", PACKAGE_DIRNAME)


def upstream():
    """The upstream RoboDojo tree: AGP_UPSTREAM_ROOT, else where the checkout's Assets link points (its parent)."""
    if os.environ.get("AGP_UPSTREAM_ROOT"):
        return Path(os.environ["AGP_UPSTREAM_ROOT"])
    return Path(os.path.realpath(CHECKOUT / "Assets")).parent


def inside(path, root):
    """Whether path lies in root, root included (both resolved)."""
    return Path(os.path.realpath(path)).is_relative_to(os.path.realpath(root))


def deploy_pins(agp, policy_dir):
    """deploy.yml agp.runtime_root / codex_binary as absolute path strings. A leading ~ is the user's home; a relative
    value is relative to the policy folder (the folder of deploy.yml, its realpath: a link to it does not move the
    anchor). An absolute value is taken as it is; a missing or non-string value is returned unchanged (the caller
    refuses it)."""
    out = {}
    for key in PIN_KEYS:
        value = (agp or {}).get(key)
        if isinstance(value, str) and value:
            value = os.path.expanduser(value)
            anchor = os.path.realpath(policy_dir)
            if not os.path.isabs(value) and isinstance(anchor, str) and os.path.isabs(anchor):
                value = os.path.normpath(os.path.join(anchor, value))
        out[key] = value
    return out


def freeze_manifest(home=None):
    """The frozen system's manifest (FREEZE_MANIFEST) as an absolute path string, relative to AGP_HOME (`home`,
    default agp_home())."""
    home = str(home if home is not None else agp_home())
    return os.path.normpath(os.path.join(home, FREEZE_MANIFEST))
