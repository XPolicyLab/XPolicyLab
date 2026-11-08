"""Read-only preflight for the machine WALL rerun script; no CUDA/model imports.

Physical GPU placement is a launch plan, NOT an instruction/task router.
"""

import argparse
import ast
import json
from pathlib import Path
import socket
import subprocess
import sys
import tomllib

ISAAC_GPUS = frozenset({"1", "2", "3", "4", "5", "7"})
TASKS = (
    "make_kong", "pour_by_language", "classify_objects_by_language", "fasten_screws",
    "fill_egg_holder", "pack_objects_into_box", "pack_objects_into_box_random",
    "build_tower", "play_stacking_toy", "fill_pen_holder", "store_tools_in_toolbox",
    "make_toast", "make_toast_random", "insert_tubes", "align_blocks", "swap_T",
    "stack_blocks_by_language", "pick_from_conveyor_by_image", "play_tic_tac_toe",
    "arrange_largest_number_random",
)
PROTECTED_TASKS = frozenset({"cover_blocks", "classify_objects", "organize_table",
                             "imitate_sorting_sequence"})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_gpus(csv):
    gpus = csv.split(",")
    require(len(gpus) == len(set(gpus)) and set(gpus) <= ISAAC_GPUS,
            "Isaac GPUs must be unique and among 1,2,3,4,5,7; never 0/6")
    return gpus


def full_tasks(robodojo):
    """Frozen runner plan from native dimension membership (not runtime routing).

    Parse literals without importing simulator/scoring code or reading any results.
    A full candidate run includes keepers afresh; old results stay untouched.
    """
    path = Path(robodojo) / "scripts/internal/summarize_result.py"
    tree = ast.parse(path.read_text())
    dimensions = next(ast.literal_eval(node.value) for node in tree.body
                      if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                         t.id == "DIMENSIONS" for t in node.targets))
    base = [name for names in dimensions.values() for name in names]
    gen = dimensions["Generalization"]
    require(len(base) == len(set(base)) == 42 and len(gen) == 12, "Expected native 42 tasks / 12 Gen pairs")
    tasks = tuple(base + [name + "_random" for name in gen])
    require(len(tasks) == len(set(tasks)) == 54, "Expected full 54 task halves")
    return tasks


def validate_package(package, expected_version="1.11.0"):
    require(package.is_dir() and not package.is_symlink(), "Export must be a real directory")
    agent = json.loads((package / "agent.json").read_text())
    require(agent["lock"]["agent_version"] == expected_version, "Expected frozen agent " + expected_version)
    if expected_version == "1.12.0":
        require("complementary" in agent["sources"]["agent"]["skills"].get("enable", []),
                "Expected explicitly enabled complementary skill")
    runtime = agent["sources"]["agent"]["runtime"]
    require(runtime["policy_action_mode"] == "ee_passthrough" and runtime["policy_span"] == "cell",
            "Expected EE passthrough/cell")
    require(not runtime.get("task_routing") and not runtime.get("task_descriptions"),
            "No task-name routing/descriptions allowed")
    require(agent["sources"]["agent"]["services"]["policies"] == ["wall-oss"], "Expected WALL only")
    for role in ("vlm", "verifier"):
        require(agent["lock"]["roles"][role].get("temperature") == 1, role + " needs temperature=1")
    require(agent["lock"]["roles"]["main"]["model"] == "gt-6.1-sol-a", "Expected Codex model")
    yml = (package / "deploy.yml").read_text()
    require("\naction_type: ee\n" in yml, "deploy.yml must declare EE")
    require(yml.startswith("policy_name: " + package.name + "\n"), "Wrong adapter name")
    deploy = tomllib.loads((package / "deploy.toml").read_text())
    for name, url in (("wall-oss", "ws://127.0.0.1:19012"), ("sam3", "http://127.0.0.1:8810")):
        entry = deploy["services"][name]
        require(entry.get("mode") == "external" and entry.get("url") == url, name + " must stay external")
    # Re-resolve from the vendor, without HarnessModel (which starts processes).
    # The template contains external services only: no GPU detection or startup.
    check = """
import json, sys
from pathlib import Path
package = Path(sys.argv[1])
sys.path[:0] = [str(package / 'the-harness/src'), str(package / 'the-harness')]
from harness_policy.config.agent import code_fingerprint, resolve_agent
from harness_policy.connectors.package import relocate
from harness_policy.services.deploy import load_deploy
document = json.loads((package / 'agent.json').read_text())
assert code_fingerprint() == document['code'], 'Vendored code fingerprint mismatch'
sources = relocate(document['sources'], package / 'the-harness')
deploy = load_deploy(package / 'deploy.toml')
agent = resolve_agent(sources['agent'], model=sources['model'], deploy=deploy,
                      strict=True, name=document['name'], runtime_overrides=sources['runtime_overrides'])
assert agent.identity == document['identity'], 'Vendored identity mismatch'
print('vendor identity/fingerprint verified')
"""
    result = subprocess.run([sys.executable, "-c", check, str(package)], capture_output=True, text=True,
                            timeout=30)
    require(result.returncode == 0, "Vendor validation failed: " + result.stderr.strip())
    return agent


def gpu_problems(gpus, query=subprocess.check_output):
    """Query each allowed card separately: DEVICE_LOST on GPU0/6 cannot poison this.

    Any process/memory/query error refuses launch. No process is stopped or retried.
    """
    problems = []
    for gpu in gpus:
        try:
            row = query(["nvidia-smi", "-i", gpu, "--query-gpu=index,uuid,memory.used",
                         "--format=csv,noheader,nounits"], text=True, stderr=subprocess.PIPE, timeout=10)
            index, uuid, memory = [v.strip() for v in row.strip().split(",")]
            require(index == gpu and uuid.startswith("GPU-"), "Invalid GPU metadata")
            apps = query(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid",
                          "--format=csv,noheader,nounits"], text=True, stderr=subprocess.PIPE, timeout=10)
            occupied = any(line.strip().isdigit() for line in apps.splitlines())
            if occupied or int(memory) >= 100:
                problems.append(f"GPU{gpu} busy ({memory} MiB); wait, never kill")
            else:
                print(f"[hack] physical GPU{gpu}: idle ({memory} MiB)")
        except (OSError, ValueError, subprocess.SubprocessError):
            problems.append(f"GPU{gpu} health query failed; refuse launch")
    return problems


def claimed_gpus(argv):
    """Read only launch placement args, never process auth/environment."""
    markers = {"smoke_all_tasks.sh", "setup_eval_env_client.sh", "eval_policy.sh"}
    script = next((i for i, arg in enumerate(argv) if Path(arg).name in markers
                   or arg.endswith(("eval_client/run.py", "eval_client/main.py"))), None)
    if script is None:
        return set()
    claimed = set()
    for i, arg in enumerate(argv[:-1]):
        if arg in {"--env-gpu", "--env-gpu-ids", "--device_id"}:
            claimed.update(argv[i + 1].split(","))
    if Path(argv[script]).name == "setup_eval_env_client.sh" and len(argv) > script + 7:
        claimed.add(argv[script + 7])
    return claimed


def process_problems(gpus, proc=Path("/proc")):
    problems = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if claimed_gpus(argv) & set(gpus):
                problems.append(f"Launcher PID {entry.name} still claims selected GPUs; wait, never kill")
        except (OSError, PermissionError):
            continue
    return problems


def check(args):
    gpus = parse_gpus(args.gpus)
    full = bool(getattr(args, "all_tasks", False))
    agent = validate_package(args.package, expected_version="1.12.0") if full else validate_package(args.package)
    tasks = full_tasks(args.robodojo) if full else TASKS
    run_id = "wall-oss-codex-skills-seed0" if full else "wall-oss-codex-hack-seed0"
    if not full:
        require(not PROTECTED_TASKS.intersection(tasks), "Protected task in rerun plan")
    require(not args.out.exists() and not args.out.is_symlink(), "Independent OUT already exists; no overwrite/retry")
    # All links must be NEW, including dangling links; --check-only creates none.
    links = [args.package / "results", args.robodojo / "smoke_results" / run_id]
    links += [args.robodojo / "eval_result/RoboDojo" / task / args.package.name for task in tasks]
    require(all(not p.exists() and not p.is_symlink() for p in links), "New result namespace already exists")
    require(args.auth.is_file(), "Local auth file missing")
    print(f"[hack] agent={agent['lock']['agent_version']} identity={agent['identity']}")
    print("[wall] seed=0; native; tasks=" + ",".join(tasks))
    print("[hack] Isaac/local kinematics=" + args.gpus + "; GPU6=WALL external only")
    print("[hack] OUT=" + str(args.out) + "; old outputs untouched")
    problems = []
    for name, port in (("WALL GPU6", 19012), ("SAM3", 8810), ("Codex relay", 18087)):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                pass
            print(f"[hack] {name} :{port} reachable")
        except OSError:
            problems.append(f"{name} :{port} unavailable; no service launch/restart")
    problems += gpu_problems(gpus) + process_problems(gpus)
    require(not problems, "\n".join(problems))
    print("[hack] PRECHECK PASS (read-only; no evaluation started)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--robodojo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--auth", type=Path, required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--all-tasks", action="store_true", help="Full frozen 1.12.0 candidate, 42 tasks/54 halves")
    args = parser.parse_args()
    try:
        check(args)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print("[hack] REFUSED: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
