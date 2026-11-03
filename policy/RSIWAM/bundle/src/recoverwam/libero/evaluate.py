"""Portable coordinator and strict audit for the three LIBERO protocols."""

import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from omegaconf import OmegaConf
from recoverwam.preflight import write_json_atomic
from .protocols import cells, validate_config, REVISIONS


def validate_checkpoint(cfg):
    import torch
    from recoverwam.checkpoint_compat import action_config
    from .cli import validate_training_config

    action = torch.load(cfg.action_checkpoint, map_location="cpu", mmap=True, weights_only=True)
    video = torch.load(cfg.video_checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if action.get("step") != 30000 or video.get("step") != 10000 or not video.get("dit"):
        raise ValueError("Expected compact Action 30k and standalone Video 10k")
    legacy = OmegaConf.create(action_config(action))
    validate_training_config(legacy, require_portable_target=False)
    if int(legacy.batch_size) != 32 or int(legacy.max_steps) != 30000:
        raise ValueError("Checkpoint is not the large-batch horizontal Action-30k recipe")
    if action["action_expert"]["action_encoder.weight"].shape != (512, 7):
        raise ValueError("Expected LIBERO Action expert hidden512/action7")
    if action["proprio_encoder"]["weight"].shape != (4096, 8):
        raise ValueError("Expected LIBERO proprio8 -> text4096")
    return dict(action_step=action["step"], video_step=video["step"])


def runtime_environment(cfg, gpu):
    root = Path(cfg.benchmark_root).resolve()
    core = root / ("liberopro/liberopro" if cfg.benchmark == "pro" else "libero/libero")
    for rel in ("__init__.py", "bddl_files", "init_files", "assets"):
        if not (core / rel).exists():
            raise FileNotFoundError(core / rel)
    runtime = Path(cfg.runtime_config_dir).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    paths = dict(
        benchmark_root=str(core),
        bddl_files=str(core / "bddl_files"),
        init_states=str(core / "init_files"),
        assets=str(core / "assets"),
        datasets=str(core.parent / "datasets"),
    )
    runtime_file = runtime / "config.yaml"
    if runtime_file.exists() and OmegaConf.to_container(OmegaConf.load(runtime_file)) != paths:
        raise ValueError("Runtime configuration belongs to a different benchmark checkout")
    if not runtime_file.exists():
        temporary = runtime_file.with_name(f"config.yaml.tmp.{os.getpid()}")
        OmegaConf.save(OmegaConf.create(paths), temporary)
        temporary.replace(runtime_file)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        MUJOCO_EGL_DEVICE_ID=str(gpu),
        LIBERO_CONFIG_PATH=str(runtime),
        TOKENIZERS_PARALLELISM="false",
        PYTHONPATH=os.pathsep.join([str(root), str(Path(__file__).resolve().parents[2])]),
    )
    if cfg.benchmark == "pro":
        env["LIBERO_TYPE"] = "pro"
    else:
        env.pop("LIBERO_TYPE", None)
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != REVISIONS[cfg.benchmark]:
            raise ValueError(f"Wrong {cfg.benchmark} checkout revision: {revision}")
    return env


def worker_config(cfg, output, manifest, gpu, seed_id):
    p = validate_config(cfg)
    c = OmegaConf.load(cfg.training_config)
    from .cli import validate_training_config

    validate_training_config(c)
    c.model.model_id = str(Path(cfg.wan_model_dir).resolve())
    c.model.tokenizer_model_id = str(Path(cfg.wan_model_dir).resolve() / "google/umt5-xxl")
    c.model.dit_checkpoint_path = str(Path(cfg.video_checkpoint).resolve())
    c.model.load_text_encoder = True
    c.model.tokenizer_max_len = 128
    c.model.skip_dit_load_from_pretrain = True
    c.model.action_dit_pretrained_path = None
    c.ckpt = str(Path(cfg.action_checkpoint).resolve())
    c.gpu_id = int(gpu)
    c.seed = p["seed"]
    c.EVALUATION = dict(
        task_manifest=str(manifest),
        output_dir=str(output),
        env_num=1,
        device="cuda",
        num_trials=p["trials"],
        num_steps_wait=p["settle"],
        replan_steps=10,
        action_horizon=32,
        num_inference_steps=10,
        sigma_shift=5.0,
        camera_preprocess_mode="training_exact",
        video_seed_offset=1,
        binarize_gripper=True,
        use_action_ensembler=False,
        visualize_future_video=False,
        save_one_step_video=False,
        save_rollout_videos=bool(cfg.save_rollout_videos),
        text_cfg_scale=1.0,
        negative_prompt="",
        rand_device="cpu",
        tiled=False,
        dataset_stats_path=str(Path(cfg.dataset_stats).resolve()),
        video_checkpoint_override=str(Path(cfg.video_checkpoint).resolve()),
        reset_mode=p["reset"],
        trial_seed_mode=p["seed_mode"],
        trial_seed_id=seed_id,
        init_state_start=p["init_start"],
        trial_seed_start=1,
        fresh_environment_per_episode=(cfg.benchmark == "pro"),
        policy_backend="recoverwam_native",
    )
    return c


def check_idle_gpus(gpus):
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    memory = {int(row.split(",")[0]): int(row.split(",")[1]) for row in rows.splitlines()}
    subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv",
        ],
        check=True,
    )
    try:
        subprocess.run(["tmux", "list-sessions"], check=False)
    except FileNotFoundError:
        pass
    for gpu in gpus:
        if gpu not in memory or memory[gpu] > 2048:
            raise RuntimeError(f"GPU {gpu} absent or busy; coordinate existing jobs first")


def audit(cfg, output=None, require_status=True):
    from .eval_helpers import hash_libero_trial_seed, resolve_max_steps

    cfg = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
    p = validate_config(cfg)
    root = Path(output or cfg.output_dir)
    expected = set(cells(cfg.benchmark))
    seen = set()
    issues = []
    totals = {}
    successes = 0
    episodes = 0
    for path in sorted((root / "results").rglob("*_results.json")):
        try:
            r = json.loads(path.read_text())
            key = (r.get("trial_seed_id"), r["task_suite"], int(r["task_id"]))
            if key not in expected or key in seen:
                raise ValueError(f"unexpected/duplicate cell {key}")
            seen.add(key)
            good, bad = r["success_episodes"], r["failure_episodes"]
            if len(set(good + bad)) != p["trials"] or set(good + bad) != set(range(p["trials"])):
                raise ValueError("episode IDs must partition the complete trial range")
            expected_seeds = (
                [hash_libero_trial_seed(key[1], key[2], i, key[0]) for i in range(p["trials"])]
                if cfg.benchmark == "ood"
                else list(range(1, 11))
                if cfg.benchmark == "pro"
                else [None] * 50
            )
            wanted = dict(
                total_episodes=p["trials"],
                successes=len(good),
                reset_mode=p["reset"],
                trial_seed_mode=p["seed_mode"],
                policy_seed=p["seed"],
                max_steps_per_episode=resolve_max_steps(key[1]),
                init_state_indices=list(range(p["init_start"], p["init_start"] + p["trials"])),
                trial_seeds=expected_seeds,
            )
            for k, v in wanted.items():
                if r.get(k) != v:
                    raise ValueError(f"{k} mismatch")
            successes += len(good)
            episodes += p["trials"]
            cell = totals.setdefault(key[1], dict(successes=0, episodes=0))
            cell["successes"] += len(good)
            cell["episodes"] += p["trials"]
        except (ValueError, KeyError, TypeError) as e:
            issues.append(f"{path}: {e}")
    missing = sorted(expected - seen, key=str)
    if require_status:
        plan_path = root / "RUN_CONFIG.json"
        if not plan_path.is_file():
            issues.append("Missing immutable RUN_CONFIG.json")
        else:
            contract = json.loads(plan_path.read_text())
            if contract.get("config") != OmegaConf.to_container(cfg, resolve=True):
                issues.append("Audit configuration differs from immutable RUN_CONFIG.json")
            for job in contract["jobs"]:
                status = root / "status" / f"{job}.status"
                if not status.is_file() or status.read_text().strip() != "0":
                    issues.append(f"Incomplete worker: {job}")
    complete = not issues and not missing and episodes == p["episodes"]
    return dict(
        benchmark=cfg.benchmark,
        protocol=p["name"],
        complete=complete,
        successes=successes,
        episodes=episodes,
        expected_episodes=p["episodes"],
        success_rate=(successes / episodes if episodes else None),
        suites=totals,
        missing=[list(x) for x in missing],
        issues=issues,
    )


def run(cfg, gpus, dry_run=False, resume=False):
    p = validate_config(cfg)
    if not gpus or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
        raise ValueError("Provide unique non-negative GPU IDs")
    print(
        json.dumps(
            dict(
                benchmark=cfg.benchmark,
                protocol=p,
                cells=len(cells(cfg.benchmark)),
                action_step=30000,
                video_step=10000,
                shift=5,
                steps=10,
                replan=10,
                gpus=gpus,
            ),
            indent=2,
        )
    )
    if dry_run:
        return
    validate_checkpoint(cfg)
    check_idle_gpus(gpus)
    root = Path(cfg.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    import fcntl

    lock = (root / ".coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = []
    for seed in range(3) if cfg.benchmark == "ood" else [None]:
        group = [(s, t) for se, s, t in cells(cfg.benchmark) if se == seed]
        for index, gpu in enumerate(gpus):
            tasks = group[index :: len(gpus)]
            if tasks:
                jobs.append((f"seed{seed}_gpu{gpu}", gpu, seed, tasks))
    config = OmegaConf.to_container(cfg, resolve=True)
    contract = dict(
        config=config,
        gpus=gpus,
        jobs=[j[0] for j in jobs],
        artifacts={
            k: dict(
                path=str(Path(config[k]).resolve()),
                size_bytes=Path(config[k]).stat().st_size,
                mtime_ns=Path(config[k]).stat().st_mtime_ns,
            )
            for k in ("action_checkpoint", "video_checkpoint", "dataset_stats", "training_config")
        },
    )
    previous = root / "RUN_CONFIG.json"
    if previous.exists():
        if not resume or json.loads(previous.read_text()) != contract:
            raise ValueError(
                "Output already exists or resume contract differs; choose a fresh output_dir"
            )
    elif resume:
        raise ValueError("Cannot resume without RUN_CONFIG.json")
    write_json_atomic(contract, previous)
    for d in ("manifests", "logs", "status", "results"):
        (root / d).mkdir(exist_ok=True)
    (root / "full_run.status").write_text("RUNNING\n")
    try:
        environments = {gpu: runtime_environment(cfg, gpu) for gpu in gpus}

        def gpu_worker(gpu):
            for name, g, seed, tasks in jobs:
                if g != gpu:
                    continue
                manifest = root / "manifests" / f"{name}.txt"
                manifest.write_text("".join(f"{s},{t}\n" for s, t in tasks))
                output = root / "results" / f"seed{seed}"
                worker = worker_config(cfg, output, manifest, gpu, seed)
                worker.release_benchmark = cfg.benchmark
                worker.release_benchmark_root = str(Path(cfg.benchmark_root).resolve())
                worker_path = root / "manifests" / f"{name}.yaml"
                OmegaConf.save(worker, worker_path)
                status = root / "status" / f"{name}.status"
                status.write_text("RUNNING\n")
                with (root / "logs" / f"{name}.log").open("w") as log:
                    result = subprocess.run(
                        [sys.executable, "-m", "recoverwam.libero.worker", "--config", str(worker_path)],
                        env=environments[gpu],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                status.write_text(str(result.returncode) + "\n")
                if result.returncode:
                    return result.returncode
            return 0

        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            codes = list(pool.map(gpu_worker, gpus))
        report = audit(cfg, root)
        write_json_atomic(report, root / "summary.json")
        complete = report["complete"] and not any(codes)
        (root / "full_run.status").write_text("0\n" if complete else "1\n")
        if not complete:
            raise RuntimeError("Evaluation incomplete; see summary.json and worker logs")
    except BaseException:
        (root / "full_run.status").write_text("1\n")
        raise
