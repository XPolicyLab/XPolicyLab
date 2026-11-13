"""Finite local launcher for specified weights and cached RoboTwin inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import time

from fixed_runtime import PROFILES, cache_manifest, load_cases

POLICY = Path(__file__).resolve().parents[1]


def identity(pid):
    p = Path('/proc') / str(pid)
    row = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start=int(row[19]), uid=p.stat().st_uid,
                boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def append_native(path, value):
    with path.open('a') as f:
        f.write(json.dumps(dict(value, at=time.time())) + '\n')
        f.flush()
        os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=[*PROFILES, "both"], default="both")
    parser.add_argument("--task", default="all", help="Cached task name, or all")
    parser.add_argument("--plan-file", type=Path, help="Ordered list of complete cached profile/task pairs")
    parser.add_argument("--cache-root", type=Path, default=POLICY / "evaluation/cache")
    parser.add_argument("--check-cache", action="store_true", help="CPU-only cache validation")
    parser.add_argument("--robotwin-root", type=Path)
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument("--checkpoint-root", type=Path)
    checkpoints.add_argument("--weights", type=Path, help="Compatible tensor-only BF16 safetensors")
    parser.add_argument("--assets", type=Path, help="Third-party assets when --weights is used")
    parser.add_argument("--policy-python", default="policy")
    parser.add_argument("--eval-python", default="sim")
    parser.add_argument("--policy-device", default="0")
    parser.add_argument("--env-device", default="1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cache, manifest, _ = cache_manifest(args.cache_root)
    profiles = list(PROFILES) if args.profile == "both" else [args.profile]
    tasks = manifest["task_names"] if args.task == "all" else [args.task]
    plan = []
    for profile in profiles:
        rows = load_cases(profile, root=cache)
        for task in tasks:
            selected = [r for r in rows if r["task"] == task]
            if len(selected) != 100:
                raise ValueError("Unknown or incomplete cached task: " + task)
            plan.append(dict(profile=profile, task=task, task_config=selected[0]["task_config"],
                             instruction_type=selected[0]["instruction_type"], trials=100))
    if args.plan_file:
        requested = json.loads(args.plan_file.read_text())
        keys = [(p['profile'], p['task']) for p in requested]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError('Empty or duplicate worker task plan')
        available = {(p['profile'], p['task']): p for p in plan}
        plan = [available[key] for key in keys]
    info = dict(protocol="fixed_seed_prompt_replay", dataset_id=manifest["dataset_id"],
                cache_manifest_sha256=hashlib.sha256((cache / "evaluation_10000.json").read_bytes()).hexdigest(),
                trials=100 * len(plan), expert_check=False, stability_check=False, plan=plan)
    if args.check_cache:
        print(json.dumps(dict(info, cache_valid=True, models_loaded=False), indent=2))
        return
    if args.robotwin_root is None or args.output is None or not (args.weights or args.checkpoint_root):
        parser.error("Set --robotwin-root, a new --output, and --weights or --checkpoint-root")
    sdk = args.robotwin_root.expanduser().resolve()
    xpl = sdk / "XPolicyLab"
    installed = xpl / "policy" / POLICY.name
    if not installed.exists() or installed.resolve() != POLICY:
        raise ValueError("Install this directory at " + str(installed) + " (a symlink is supported)")
    binding = json.loads((POLICY / "evaluation/SDK_BINDING.json").read_text())
    for relative, key in [("scripts/eval_policy_xpolicylab.py", "official_client_sha256"),
                          ("envs/_base_task.py", "base_task_sha256")]:
        if hashlib.sha256((sdk / relative).read_bytes()).hexdigest() != binding[key]:
            raise ValueError("RoboTwin SDK version differs: " + relative)
    for item in plan:
        if not (sdk / "env_cfg/task_config" / (item["task_config"] + ".yml")).is_file():
            raise FileNotFoundError("Missing cached task configuration: " + item["task_config"])
    output = args.output.expanduser().absolute()
    if output.exists():
        raise FileExistsError("Use a new output directory; existing trials are never replayed")
    env = dict(os.environ, CLOUDROBO_ROBOTWIN_ROOT=str(sdk), CLOUDROBO_XPOLICYLAB_ROOT=str(xpl),
               CLOUDROBO_FIXED_CACHE_ROOT=str(cache), PYTHONDONTWRITEBYTECODE="1")
    for name in ["ROBOTWIN_EVAL_ARGS_FILE", "CLOUDROBO_FIXED_WEIGHTS", "CLOUDROBO_FIXED_ASSETS"]:
        env.pop(name, None)
    if args.weights:
        weights = args.weights.expanduser().resolve()
        if weights.suffix != ".safetensors" or not weights.is_file() or args.assets is None:
            raise ValueError("--weights needs a safetensors file and --assets")
        assets = args.assets.expanduser().resolve()
        env.update(CLOUDROBO_FIXED_WEIGHTS=str(weights), CLOUDROBO_FIXED_ASSETS=str(assets))
        checkpoint = str(weights)
    else:
        root = args.checkpoint_root.expanduser().resolve()
        weights = root / "weights/cloudrobo_wm_bf16.safetensors"
        assets = root / "weights/third_party"
        if not weights.is_file():
            raise FileNotFoundError(weights)
        checkpoint = str(root)
    for name in ["text_encoder.safetensors", "vae.safetensors", "dino.safetensors", "tokenizer/tokenizer_config.json"]:
        if not (assets / name).is_file():
            raise FileNotFoundError(assets / name)
    port = args.port
    if port is not None and not 1024 <= port <= 65535:
        raise ValueError("Invalid server port")
    if not args.dry_run and port is None:
        port = int(subprocess.check_output(["bash", str(xpl / "utils/get_free_port.sh")], env=env, text=True))
    port_value = str(port) if port else "<free-port>"
    server_command = ["bash", str(POLICY / "setup_eval_policy_server.sh"), "RoboTwin", plan[0]["task"],
                      checkpoint, "arx_x5", "joint", "0", args.policy_device, args.policy_python, port_value, "127.0.0.1"]
    commands = [["bash", str(POLICY / "setup_fixed_eval_env_client.sh"), "RoboTwin", p["task"], checkpoint,
                 "arx_x5", "joint", "0", args.env_device, args.eval_python,
                 "ckpt_name=" + checkpoint + ",action_type=joint", port_value, "127.0.0.1"] for p in plan]
    if args.dry_run:
        print(json.dumps(dict(info, dry_run=True, models_loaded=False, server_command=server_command,
                             client_commands=commands, output=str(output)), indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    with (output / "RUN.json").open("x") as f:
        json.dump(dict(info, weights=str(weights), assets=str(assets)), f, indent=2)
    server = None
    try:
        with (output / "policy_server.log").open("x") as log:
            server = subprocess.Popen(server_command, env=env, cwd=sdk, stdout=log, stderr=subprocess.STDOUT)
            append_native(output / 'NATIVE_EVENTS.jsonl', dict(event='server_launch', identity=identity(server.pid)))
            subprocess.run(["bash", str(xpl / "utils/wait_for_policy_server.sh"), "127.0.0.1", str(port),
                            str(server.pid), "Fixed-cache policy server", "1200"], env=env, check=True)
            for item, cmd in zip(plan, commands):
                target = output / item["profile"] / item["task"]
                task_env = dict(env, CLOUDROBO_FIXED_OUTPUT=str(target),
                                CLOUDROBO_ZERO_DEPTH_GUARD_OUTPUT=str(target / 'simulator_witness'),
                                CLOUDROBO_TASK_CONFIG=item["task_config"],
                                CLOUDROBO_INSTRUCTION_TYPE=item["instruction_type"])
                with (output / (item["profile"] + "__" + item["task"] + ".log")).open("x") as log:
                    client = subprocess.Popen(cmd, cwd=sdk, env=task_env, stdout=log, stderr=subprocess.STDOUT)
                    original = identity(client.pid)
                    append_native(output / 'NATIVE_EVENTS.jsonl', dict(event='client_launch', task=item, identity=original))
                    code = client.wait()
                    append_native(output / 'NATIVE_EVENTS.jsonl', dict(event='client_exit', task=item,
                                  identity=original, native_exit=code, actual_wait=True))
                    if code != 0:
                        raise subprocess.CalledProcessError(code, cmd)
            results = [json.loads((output / p["profile"] / p["task"] / "FIXED_RESULT.json").read_text()) for p in plan]
            if any(not r["complete"] or r["trials"] != 100 for r in results):
                raise RuntimeError("Incomplete cached task result")
            trials, success = sum(r["trials"] for r in results), sum(r["successes"] for r in results)
            summary = dict(protocol=info["protocol"], complete=True, tasks=len(results), trials=trials,
                           successes=success, success_percent=100 * success / trials,
                           expert_check=False, stability_check=False, task_results=results)
            summary["profiles"] = {}
            for profile in profiles:
                selected = [r for r in results if r["profile"] == profile]
                if not selected:
                    continue
                count = sum(r["trials"] for r in selected)
                wins = sum(r["successes"] for r in selected)
                summary["profiles"][profile] = dict(trials=count, successes=wins, success_percent=100 * wins / count)
            with (output / "FIXED_SUMMARY.json").open("x") as f:
                json.dump(summary, f, indent=2)
            print(json.dumps(summary, indent=2))
    finally:
        if server is not None:
            if server.poll() is None:
                server.terminate()
            try:
                code = server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
                code = server.wait()
            (output / "POLICY_SERVER_EXIT.json").write_text(json.dumps({"returncode": code}) + "\n")


if __name__ == "__main__":
    main()
