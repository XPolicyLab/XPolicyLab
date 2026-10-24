"""Export the agent as an XPolicyLab policy folder (`harness export xpolicylab`).

    harness export xpolicylab --name HarnessPolicy \
        --out <XPolicyLab>/policy/HarnessPolicy [--vendor] [--check --gpu 5 --env harness-policy]

The folder follows XPolicyLab's adapter standard (CONTRIBUTING.md): model.py (a thin
Model that delegates to harness_policy.connectors.package), deploy.py and the eval.sh /
setup_eval_policy_server.sh / setup_eval_env_client.sh trio copied unchanged from
demo_policy, deploy.yml with the standard keys, install.sh, a checkpoint download
script, and a README in their template. It adds the frozen agent (agent.json: what
the agent was resolved from, with repository paths relative to the checkout; the lock
and its identity; the code fingerprint and commit) and the deploy template; the policy
refuses to run when the machine it lands on resolves a different identity.
Evaluation only: no data processing or training (declared in the README).
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


from ..connectors.package import relative_tree

REPO = Path(__file__).resolve().parents[3]
STANDARD_SCRIPTS = ("deploy.py", "eval.sh", "setup_eval_policy_server.sh", "setup_eval_env_client.sh", "__init__.py")
# Not copied by --vendor: the external checkouts.
VENDOR_SKIP = ("third_party",)


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


MODEL_PY = '''"""Harness Policy as an XPolicyLab policy (exported by `harness export xpolicylab`).

The policy is the frozen Harness Policy agent (agent.json): a model writes programs that
call perception, motion and learned skills; this Model hands XPolicyLab's observations to
those programs and their actions back (harness_policy/connectors/xpolicylab.py). model.py
never decodes images: the policy server has decoded every observation it forwards.
"""

from pathlib import Path
import sys

from XPolicyLab.model_template import ModelTemplate

PACKAGE = Path(__file__).resolve().parent
_HARNESS = PACKAGE / "the-harness"
for _path in (_HARNESS / "src", _HARNESS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from harness_policy.connectors.package import HarnessModel  # noqa: E402


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.harness = HarnessModel(model_cfg, PACKAGE)

    def update_obs(self, obs):
        self.harness.update_obs(obs)

    def update_obs_batch(self, obs_list):
        self.harness.update_obs_batch(obs_list)

    def get_action(self):
        return self.harness.get_action()

    def get_action_batch(self, env_idx_list=None):
        return self.harness.get_action_batch(env_idx_list)

    def reset(self):
        self.harness.reset()
'''

DEPLOY_YML = '''policy_name: {name}
protocol: ws
host: localhost
port: null
bench_name: null
task_name: null
ckpt_name: null
env_cfg_type: null
seed: null
action_type: joint
gpu_id: null
eval_batch: true

# Harness Policy. While the model writes a program, or a model loads, get_action returns
# an empty chunk after poll_seconds (below the client's 120 s request limit); the
# standard episode loop then takes no action and asks again.
poll_seconds: 90
# Traces of every episode (default: results/ in this folder).
result_dir: null
max_wall_seconds: 5400
'''

INSTALL_SH = '''#!/bin/bash
# Installs Harness Policy for this XPolicyLab policy into the policy environment.
#   bash install.sh [python of the policy environment]          (default: python)
# Tested with the harness-policy conda environment (the-harness/environment/setup.sh),
# which also runs RoboDojo; any environment with RoboDojo's (torch 2.7, cuRobo) and
# XPolicyLab's packages works after this script.
set -euo pipefail
PY=${{1:-python}}
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
XPL_ROOT="$(cd "$HERE/../.." && pwd)"
if [ ! -d "$HERE/the-harness" ]; then
  git clone {remote} "$HERE/the-harness"
  git -C "$HERE/the-harness" checkout {commit}
fi
"$PY" -m pip install -e "$XPL_ROOT"
"$PY" -m pip install --constraint "$HERE/the-harness/environment/constraints.txt" \\
    -r "$HERE/the-harness/environment/requirements-skills.txt"
"$PY" -c "import sys; sys.path[:0] = ['$HERE/the-harness/src', '$HERE/the-harness']; import harness_policy.connectors.package"
echo "Harness Policy installed for $PY"
'''

DOWNLOAD_SH = '''#!/bin/bash
# Downloads the weights of the services this policy uses into checkpoints/
# (ModelScope and hf-mirror, sha256-checked): {services}.
#   bash download_checkpoints.sh [python]
set -euo pipefail
PY=${{1:-python}}
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$HERE"
"$PY" the-harness/scripts/harness fetch {services} --deploy "$HERE/deploy.toml"
'''

DEPLOY_TOML = """# Where this policy's services run. At start the policy server writes a deploy of its
# own (results/<task>-<time>-<pid>/deploy.toml): every service on the GPUs it sees
# (policy_gpu_id; several, comma-separated, give each model room), placed by their free
# memory, on ports of its own. Paths here are relative to this folder.
weights = "checkpoints"          # download_checkpoints.sh puts the weights here
# robodojo = "../../.."          # the RoboDojo checkout (default: the one around XPolicyLab)

# A service may run elsewhere, or with another python, as in a deploy file:
# [services.molmo2]
# mode = "external"
# url = "http://10.0.0.7:8830"
"""

README = '''# {name}

**Contributor:** {contributor} | **Paper:** Harness Policy | **arXiv:** TBD | **Original code:** {remote}

{name} is the Harness Policy agent: a language model ({model}) reads the learned
policy's description and hands it whole tasks it was trained on ({policies}), from the
start pose; for what the policy cannot do (picking the one object a name, a label or a
picture identifies) it writes Python programs that perceive ({perception}), plan and move
the arms through bounded skills. A reactive layer runs guards every control tick and a
fixed shield checks every action.
The agent is frozen in `agent.json` (identity `{identity}`): the policy refuses to run
if the machine it lands on would resolve a different one.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/{name}
bash install.sh <python of the policy environment>
bash download_checkpoints.sh <python of the policy environment>
{environment}
```

{vendor_note}

## Data Processing

Not supported: this is an evaluation-only adapter (no `process_data.sh`). The learned
policies it calls are RoboDojo's own released checkpoints.

## Training

Not supported (no `train.sh`): the agent has no trained component of its own.

## Evaluation

```bash
bash eval.sh RoboDojo <task_name> {name} {env_cfg} {action_type} <seed> <policy_gpu_ids> <env_gpu_id> <policy_env> <eval_env>

# Offline wiring check with scripted model replies (no simulator, no model calls)
HARNESS_SCRIPT=$PWD/the-harness/configs/scripts/xpolicylab_check.json EVAL_ENV_TYPE=debug \\
  bash eval.sh RoboDojo stack_bowls {name} {env_cfg} {action_type} 0 0 0 <policy_env> <eval_env>
```

`policy_gpu_ids` may list several GPUs (`2,3`): the services ({service_list}) take
about {vram:.0f} GB in all; when they do not fit at once, the learned policies and the
world model load on demand, one at a time (the simulator waits while a model loads, so
only wall time changes). `action_type` must be `joint`; `ckpt_name` is not read.

## Configuration

| deploy.yml key | Meaning |
| --- | --- |
| `poll_seconds` | while the model thinks, `get_action` returns an empty chunk after this long (default 90, below the client's 120 s request limit) |
| `result_dir` | where each episode's trace (model requests, programs, events) goes; default `results/` |
| `max_wall_seconds` | wall-time limit of one episode |

Environment: {environment_note}. `HARNESS_SCRIPT` replays model replies
instead of calling the model (wiring checks).

## Notes

- A batch of environments (`eval_batch: true`): one program per environment. While any
  of them is still waiting on the model, every chunk comes back empty and the
  simulator does not step. The learned policy's server still serves one environment
  at a time; the lock in `services/policy_server.py` keeps those calls from mixing.
- One fixed agent for every task: the policy uses only the observation XPolicyLab
  hands it (RGB cameras, measured arm joints, link6 poses, the instruction, the last
  gripper commands); it is told neither the task nor its step limit (`task_name` only
  labels the traces). Camera calibration and kinematics come from the robot's
  description (`the-harness/embodiments/{embodiment}`, {kinematics_note}).
- The policy never reads the outcome: an episode that continues after the program
  finished is not taken as a failure; the arms hold until the client ends it.
'''


def hosted_environment(agent) -> tuple[str, str]:
    """Shell lines and a short note for the servers the agent calls.

    The URL and the key are the evaluator's. The model id is the one the agent asks for.
    """
    exports, notes, seen = [], [], set()
    configs = [agent.main]
    vlm = agent.runtime.get("vlm")
    if isinstance(vlm, dict):
        configs.append(vlm)
    for config in configs:
        if not isinstance(config, dict) or config.get("service"):
            continue
        url_env = config.get("base_url_env")
        if url_env and url_env not in seen:
            seen.add(url_env)
            exports.append(f"export {url_env}=http://HOST:PORT/v1   # model id {config.get('model')}")
            notes.append(f"`{url_env}` (the server for `{config.get('model')}`)")
        key = config.get("api_key_env")
        if key and key not in seen:
            seen.add(key)
            if url_env:
                exports.append(f"# export {key}=...   # only if that server requires a key")
                notes.append(f"`{key}` if the server requires a key")
            else:
                exports.append(f"export {key}=...")
                notes.append(f"`{key}`")
        elif config.get("backend", "openrouter") == "openrouter" and "OPENROUTER_API_KEY" not in seen:
            seen.add("OPENROUTER_API_KEY")
            exports.append("export OPENROUTER_API_KEY=...        # the main model and the VLM are hosted")
            notes.append("`OPENROUTER_API_KEY` (hosted models)")
    if not exports:
        return "# no hosted model", "none"
    return "\n".join(exports), "; ".join(notes)


def resolve_source(source=None, *, deploy=None, model=None, sets=()):
    """(agent, bench) of an experiment file or an agent file (default configs/default.toml),
    resolved strictly."""
    from ..config import apply_sets, load_layered, parse_set
    from ..config.agent import DEFAULT_AGENT, resolve_agent
    from ..config.experiments import is_experiment, load_experiment

    source = Path(source or DEFAULT_AGENT).resolve()
    tree = load_layered(source)
    if is_experiment(tree):
        experiment = load_experiment(source, deploy=deploy, sets=sets, strict=True, model=model)
        return experiment.agent, experiment.bench
    tree = apply_sets(tree, [parse_set(item) for item in sets])
    return resolve_agent(tree, model=model, deploy=deploy, strict=True, name=source.stem), "robodojo"


# The kinematics service (services/kinematics.toml: fk, ik and plan_path through RoboDojo's
# cuRobo planner) serves this embodiment; the shield's workspace and Cartesian checks and the
# tools by pose use it, so another robot needs its own.
KINEMATICS_EMBODIMENT = "arx_x5"


def check_embodiment(embodiment: str) -> dict:
    """The embodiment a policy folder is exported for, refused unless calibrated on its rig
    (a profile without the key, such as RoboDojo's simulated arx_x5, counts as calibrated)
    and served by a kinematics service of its own (its kinematics_service; arx_x5's is
    services/kinematics.toml)."""
    from ..embodiments import load_embodiment

    data = load_embodiment(embodiment)
    if data.get("calibrated", True) is not True:
        raise ValueError(f"Embodiment {embodiment} is not calibrated: measure its rig and set calibrated = true "
                         "(docs/how-to/real-robot.md)")
    if data["name"] != KINEMATICS_EMBODIMENT and not data.get("kinematics_service"):
        raise ValueError(f"Embodiment {embodiment} has no kinematics service: services/kinematics.toml is the "
                         f"{KINEMATICS_EMBODIMENT}'s, and the shield and the tools by pose would run its fk and ik "
                         "on this robot's joints; give the robot its own (kind kinematics) and name it in "
                         "kinematics_service (docs/how-to/real-robot.md)")
    return data


def export(source=None, *, name: str, out, deploy=None, model=None, sets=(), vendor: bool = False,
           xpolicylab=None, contributor: str = "Harness Policy team", allow_dirty: bool = False,
           embodiment: str = "arx_x5") -> dict:
    """Write the policy folder for the agent of `source` (an experiment or an agent file;
    default configs/default.toml), for `embodiment` (embodiments/<name>; RoboDojo's ARX X5
    by default, another robot once its profile is calibrated)."""
    from ..config.agent import CODE_ROOTS, _service_names, code_fingerprint
    from ..services.deploy import load_deploy
    from ..services.registry import load_services

    if not name.isidentifier():
        raise ValueError(f"The policy name {name!r} must be a Python identifier (XPolicyLab imports policy.<name>)")
    deploy = load_deploy(deploy)
    agent, bench = resolve_source(source, deploy=deploy, model=model, sets=sets)
    if bench != "robodojo":
        raise ValueError("Only RoboDojo agents export to XPolicyLab so far")
    robot = check_embodiment(embodiment)
    kinematics = robot.get("kinematics_service", "kinematics")
    if agent.runtime.get("contract", "native") != "xpolicylab":
        raise ValueError("Export an agent that runs under the XPolicyLab contract (runtime.contract = \"xpolicylab\"), "
                         "so development runs saw what the evaluated policy will see")
    if agent.runtime.get("task_descriptions"):
        raise ValueError("RoboDojo's official client tells a policy no task name, so an exported agent cannot pick "
                         "a task description; export one without runtime.task_descriptions")
    if subprocess.run(["git", "-C", str(REPO), "rev-parse", "--git-dir"], capture_output=True).returncode != 0:
        raise ValueError(f"{REPO} is not a git checkout: an exported policy pins the commit it runs")
    dirty = _git("status", "--porcelain", "--", *CODE_ROOTS, "configs", "embodiments")
    if dirty and not allow_dirty:
        raise ValueError(f"Uncommitted changes in the code a policy runs; commit them first:\n{dirty}")
    registry = load_services()
    sources = agent.sources
    services = sorted(set(_service_names(sources["model"], sources["agent"].get("services", {}))) | {kinematics})
    unknown = [service for service in services if service not in registry]
    if unknown:
        raise ValueError(f"Unknown services {unknown}")
    demo = Path(xpolicylab or deploy.xpolicylab) / "policy" / "demo_policy"
    if not demo.is_dir():
        raise ValueError(f"XPolicyLab's demo_policy not found at {demo} (give --xpolicylab)")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    commit = _git("rev-parse", "HEAD")
    remote = "https://github.com/EvolvingLMMs-Lab/the-harness.git"
    document = {"name": agent.name, "identity": agent.identity, "lock": agent.lock, "bench": bench,
                "embodiment": embodiment, "services": services, "commit": commit, "dirty": bool(dirty),
                "code": code_fingerprint(),
                "sources": relative_tree(sources, REPO)}
    (out / "agent.json").write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    (out / "model.py").write_text(MODEL_PY)
    (out / "deploy.yml").write_text(DEPLOY_YML.format(name=name))
    (out / "deploy.toml").write_text(DEPLOY_TOML)
    (out / "install.sh").write_text(INSTALL_SH.format(remote=remote, commit=commit))
    downloads = [s for s in services if registry[s].weights and not all(w.get("local") for w in registry[s].weights)]
    (out / "download_checkpoints.sh").write_text(DOWNLOAD_SH.format(services=" ".join(downloads)))
    for script in STANDARD_SCRIPTS:
        shutil.copyfile(demo / script, out / script)
    for script in ("eval.sh", "setup_eval_policy_server.sh", "setup_eval_env_client.sh", "install.sh",
                   "download_checkpoints.sh"):
        (out / script).chmod(0o755)
    vram = sum(registry[s].vram_gb for s in services)
    policies = ", ".join(_service_names({}, {"policies": sources["agent"].get("services", {}).get("policies", [])}))
    perception = "SAM 3 segmentation, " + ("Molmo2 pointing" if "molmo2" in services
                                           else "the model's own look at the images")
    vendor_note = ("`the-harness/` holds the-harness at the exported commit "
                   f"({commit[:12]}); install.sh uses it as it is." if vendor else
                   f"install.sh clones the-harness at the exported commit ({commit[:12]}) into `the-harness/`.")
    environment, environment_note = hosted_environment(agent)
    (out / "README.md").write_text(README.format(
        name=name, contributor=contributor, remote=remote.removesuffix(".git"), model=agent.main.get("model"),
        policies=policies or "none", identity=agent.identity, vendor_note=vendor_note,
        service_list=", ".join(services), vram=vram, perception=perception, embodiment=embodiment,
        env_cfg=robot["xpolicylab"]["env_cfg_type"], action_type=robot["xpolicylab"]["action_type"],
        environment=environment, environment_note=environment_note,
        kinematics_note=("RoboDojo's own cuRobo planner" if kinematics == "kinematics"
                         else f"its kinematics service `{kinematics}`")))
    if vendor:
        # The files the agent was resolved from: tracked and untracked, not ignored.
        target = out / "the-harness"
        if target.exists():
            shutil.rmtree(target)
        listed = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", "-c", "-o", "--exclude-standard"],
                                capture_output=True, check=True).stdout.decode().split("\0")
        for relative in filter(None, listed):
            source = REPO / relative
            if relative.split("/")[0] in VENDOR_SKIP or not source.is_file() or out in source.parents:
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return {"folder": str(out), "identity": agent.identity, "services": services, "commit": commit,
            "vram_gb": vram, "vendored": vendor, "dirty": bool(dirty)}


def check(folder, *, gpu: str, env: str, eval_env: str | None = None, task: str = "stack_bowls",
          script=None, timeout: float = 1800.0, log=print) -> dict:
    """XPolicyLab's PR checks on the folder: bash -n, py_compile, the image greps, and the
    debug closed loop (EVAL_ENV_TYPE=debug) with scripted model replies."""
    folder = Path(folder).resolve()
    results = {}
    results["bash -n"] = all(subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
                             for path in sorted(folder.glob("*.sh")))
    results["py_compile"] = subprocess.run([sys.executable, "-m", "py_compile", str(folder / "model.py"),
                                            str(folder / "deploy.py")], capture_output=True).returncode == 0
    grep = subprocess.run(["grep", "-rnE", r"cv2\.imdecode|np\.frombuffer|Image\.open", str(folder / "model.py"),
                           str(folder / "deploy.py")], capture_output=True, text=True)
    results["no decode in model.py/deploy.py"] = grep.returncode == 1
    script = Path(script or REPO / "configs" / "scripts" / "xpolicylab_check.json").resolve()
    env_vars = {**os.environ, "EVAL_ENV_TYPE": "debug", "HARNESS_SCRIPT": str(script)}
    # The robot the folder was exported for (agent.json), as its README's commands name it.
    from ..embodiments import load_embodiment

    robot = load_embodiment(json.loads((folder / "agent.json").read_text()).get("embodiment", "arx_x5"))
    command = ["bash", "eval.sh", "RoboDojo", task, folder.name, robot["xpolicylab"]["env_cfg_type"],
               robot["xpolicylab"]["action_type"], "0", str(gpu), str(gpu), env, eval_env or env]
    log(f"check: {' '.join(command)} (in {folder})")
    try:
        run = subprocess.run(command, cwd=folder, env=env_vars, capture_output=True, text=True, timeout=timeout)
        output = run.stdout + run.stderr
        results["debug closed loop"] = run.returncode == 0 and "[MAIN] eval finished" in output and \
            "Traceback" not in output
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or b"").decode(errors="replace") if isinstance(error.stdout, bytes) else str(error.stdout)
        results["debug closed loop"] = False
    (folder / "results").mkdir(exist_ok=True)
    (folder / "results" / "check.log").write_text(output or "")
    return {"ok": all(results.values()), "checks": results, "log": str(folder / "results" / "check.log")}
