"""Read cached inputs and run their finite, ordered RoboTwin rollouts."""
from pathlib import Path
import hashlib
import json
import os
import re
import sys
import time
import types

ROOT = Path(__file__).resolve().parent
PROFILES = {"clean_seen": ("seen", {"demo_clean", "cr_wam_demo_clean"}),
            "random_unseen": ("unseen", {"demo_randomized", "cr_wam_demo_randomized"})}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_manifest(root=None):
    root = Path(root).expanduser().resolve() if root else ROOT / "cache"
    checks = json.loads((root / "CHECKSUMS.json").read_text())["files"]
    manifest = root / "evaluation_10000.json"
    if sha(manifest) != checks["evaluation_10000.json"]["sha256"]:
        raise ValueError("Cache manifest checksum differs")
    if root == (ROOT / "cache").resolve():
        expected = json.loads((ROOT / "SDK_BINDING.json").read_text())["default_cache_manifest_sha256"]
        if sha(manifest) != expected:
            raise ValueError("Bundled cache identity differs")
    value = json.loads(manifest.read_text())
    if value.get("schema") not in {"cloudrobo.fixed-seed-prompt-inputs.v1", "cloudrobo.fixed-evaluation-inputs.v1"}:
        raise ValueError("Unsupported fixed-input cache")
    tasks = value["task_names"]
    if (value["total_episodes"] != 10000 or len(set(tasks)) != 50
            or any(not re.fullmatch(r"[a-z0-9_]+", t) for t in tasks)
            or {p["id"] for p in value["profiles"]} != set(PROFILES)):
        raise ValueError("Require both original 50-task, 10000-input profiles")
    return root, value, checks


def load_cases(profile, task=None, root=None):
    root, manifest, checks = cache_manifest(root)
    if profile not in PROFILES or (task is not None and task not in manifest["task_names"]):
        raise ValueError("Unknown cached profile or task")
    item = next(p for p in manifest["profiles"] if p["id"] == profile)
    name = item["inputs_file"]
    if name != f"inputs/{profile}.jsonl" or item["task_config"] not in PROFILES[profile][1]:
        raise ValueError("Unexpected cache profile/configuration")
    if sha(root / name) != checks[name]["sha256"]:
        raise ValueError("Cached input bytes differ")
    rows = [json.loads(s) for s in (root / name).read_text().splitlines()]
    if len(rows) != 5000 or {r["task"] for r in rows} != set(manifest["task_names"]):
        raise ValueError("Require all 5000 original profile inputs")
    by_task = {t: [] for t in manifest["task_names"]}
    seen = set()
    for r in rows:
        if (r["profile_id"] != profile or r["instruction_type"] != PROFILES[profile][0]
                or r["task_config"] != item["task_config"] or r["case_id"] in seen):
            raise ValueError("Cache input identity differs")
        seen.add(r["case_id"])
        if (type(r["seed"]) is not int or r["seed"] < 0
                or type(r["now_ep_num"]) is not int or not 0 <= r["now_ep_num"] < 100
                or not isinstance(r["prompt"], str) or not r["prompt"].strip()
                or hashlib.sha256(r["prompt"].encode()).hexdigest() != r["prompt_sha256"]
                or (r.get("payload") is not None and not isinstance(r["payload"], dict))):
            raise ValueError("Invalid cached seed, prompt, ordinal or payload")
        by_task[r["task"]].append(r)
    for values in by_task.values():
        if ([r["ordinal"] for r in values] != list(range(100))
                or len({r["seed"] for r in values}) != 100):
            raise ValueError("Require each original task's 100 inputs, in order")
    return by_task[task] if task is not None else rows


def initialize_rollout_metadata(task, env):
    """Initialize success-check metadata otherwise assigned by expert motion."""
    module = sys.modules[type(env).__module__]
    def tag(value):
        return module.ArmTag(value)
    if task == "open_laptop":
        face = module.get_face_prod(env.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
        env.arm_tag = tag("left" if face > 0 else "right")
    if task in {"place_object_scale", "put_object_cabinet"}:
        env.arm_tag = tag("right" if env.object.get_pose().p[0] > 0 else "left")
    if task == "put_object_cabinet":
        env.origin_z = env.object.get_pose().p[2]


def run_fixed_task(official, task_name, task_env, args, model_client, st_seed,
                   *, usr_args, test_num=100, video_size=None, instruction_type=None):
    if test_num != 100 or int(usr_args.get("num_workers", 1)) != 1:
        raise ValueError("Use all 100 cached inputs and one environment per server")
    if official.parse_bool(usr_args.get("expert_check", True)):
        raise ValueError("Fixed evaluation requires expert_check=false")
    profile = {"seen": "clean_seen", "unseen": "random_unseen"}.get(instruction_type)
    cases = load_cases(profile, task_name, os.environ.get("CLOUDROBO_FIXED_CACHE_ROOT"))
    if args["task_config"] != cases[0]["task_config"]:
        raise ValueError("Task configuration differs from this cache")
    output = Path(args.get("eval_video_save_dir") or os.environ["CLOUDROBO_FIXED_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    current = {}
    original = task_env.check_stable
    inherited = "check_stable" not in vars(task_env)
    bypasses = 0
    def bypass_stability(self):
        nonlocal bypasses
        bypasses += 1
        return True, []
    def setup(env, **kwargs):
        before = bypasses
        env.setup_demo(**kwargs)
        if bypasses != before + 1:
            raise RuntimeError("Expected exactly one bypassed setup stability check")
    with (output / "fixed_cases.jsonl").open("x", encoding="utf-8") as stream:
        def record(event, case, **extra):
            current.clear()
            current.update(case)
            stream.write(json.dumps(dict(event=event, input=case, expert_executed=False,
                stability_check_executed=False, stability_check_bypassed=True,
                at=time.time(), **extra), ensure_ascii=False) + "\n")
            stream.flush()
        namespace = dict(vars(official))
        namespace.update(fixed_cases=cases, initialize_rollout_metadata=initialize_rollout_metadata,
                         fixed_record=record, fixed_setup=setup)
        source = ROOT / "fixed_rollout.py"
        exec(compile(source.read_bytes(), str(source), "exec"), namespace)
        task_env.check_stable = types.MethodType(bypass_stability, task_env)
        try:
            last_seed, success = namespace["eval_remote_policy"](task_name, task_env, args, model_client,
                st_seed, usr_args=usr_args, test_num=test_num, video_size=video_size,
                instruction_type=instruction_type)
            result = dict(protocol="fixed_seed_prompt_replay", profile=profile, task=task_name,
                complete=True, trials=100, successes=success, expert_check=False,
                stability_check=False, stability_bypass_calls=bypasses,
                payloads_missing=sum(c.get("payload") is None for c in cases),
                video_enabled=bool(args.get("eval_video_save_dir")))
            with (output / "FIXED_RESULT.json").open("x") as f:
                json.dump(result, f, indent=2)
            return last_seed, success
        except BaseException as error:
            record("fixed_input_error_hold", dict(current), error_type=type(error).__name__,
                   error=str(error), success=None, replacement_seed_selected=False)
            official.safe_close_env(task_env)
            raise
        finally:
            if inherited:
                del task_env.check_stable
            else:
                task_env.check_stable = original
