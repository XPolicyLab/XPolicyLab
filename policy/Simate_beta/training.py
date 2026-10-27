import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

POLICY_DIR = Path(__file__).resolve().parent
XPL_ROOT = POLICY_DIR.parents[1]
sys.path.insert(0, str(XPL_ROOT.parent))
from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name

REVISION = "8ae595cda0eea1595afdb240749c7b595872f5c0"


def path(value):
    return Path(value).expanduser().resolve()


def required_file(value):
    value = path(value)
    if not value.is_file():
        raise ValueError(f"Required file is missing: {value}")
    return value


def package_assets(root):
    assets = json.loads(required_file(root / "training_assets.json").read_text())
    for key in ("manifest", "target_overrides", "conveyor_audit", "phase_ranges"):
        required_file(root / assets[key])
    for name in ("metadata.json", "phases.npy", "strengths.npy"):
        required_file(root / assets["phase_index"] / name)
    for source in ("robodojo", "play_tic_tac_toe"):
        required_file(root / source / "schema.json")
    return assets


def prepare(args):
    root = path(os.environ.get("SIMATE_DATA_ROOT", POLICY_DIR / "data/simate-beta-training-data"))
    package_assets(root)
    if args.verify:
        for line in required_file(root / "SHA256SUMS").read_text().splitlines():
            digest, relative = line.split(maxsplit=1)
            source = required_file(root / relative.lstrip("*"))
            source.relative_to(root)
            h = hashlib.sha256()
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != digest:
                raise ValueError(f"Checksum mismatch: {relative}")
    name = build_run_dir_name(vars(args), include_seed=False)
    destination = POLICY_DIR / "data" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        if destination.resolve() != root:
            raise ValueError(f"Refusing to replace {destination}; choose a different ckpt_name.")
    else:
        destination.symlink_to(root, target_is_directory=True)
    print(f"Prepared native SiMate dataset: {destination} -> {root}")


def train(args):
    from omegaconf import OmegaConf

    source = path(os.environ.get("SIMATE_TRAIN_ROOT", POLICY_DIR / "training_source"))
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"Expected training source {REVISION}, got {revision}")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip():
        raise ValueError("Training source has tracked modifications.")
    if int(os.environ["SIMATE_ROBOT_ACTION_DIM"]) != 14:
        raise ValueError("This recipe requires the arx_x5 joint layout (14 values, padded internally).")
    name = build_run_dir_name(vars(args))
    data_name = build_run_dir_name(vars(args), include_seed=False)
    root = path(POLICY_DIR / "data" / data_name)
    assets = package_assets(root)
    init = path(os.environ.get("SIMATE_INIT_ROOT", POLICY_DIR / "pretrained/simate-beta-init"))
    env = dict(os.environ)
    values = {
        "SIMATE_DATA_ROOT": root,
        "SIMATE_ROBODOJO_ROOT": root / "robodojo",
        "SIMATE_PLAY_TIC_TAC_TOE_ROOT": root / "play_tic_tac_toe",
        "SIMATE_MANIFEST": root / assets["manifest"],
        "SIMATE_PHASE_INDEX": root / assets["phase_index"],
        "SIMATE_TARGET_OVERRIDES": root / assets["target_overrides"],
        "SIMATE_CONVEYOR_AUDIT": root / assets["conveyor_audit"],
        "SIMATE_PHASE_RANGES": root / assets["phase_ranges"],
        "SIMATE_STATS_SOURCE_PHASE": assets["source_phase_fingerprint"],
        "SIMATE_STATS_TARGET_PHASE": assets["target_phase_fingerprint"],
        "SIMATE_INIT_CHECKPOINT": required_file(init / "model.safetensors"),
        "SIMATE_TOKENIZER": required_file(init / "tokenizer.model"),
        "SIMATE_STATISTICS": required_file(init / "state_action_statistics.json"),
    }
    env.update({key: str(value) for key, value in values.items()})
    os.environ.update({key: str(value) for key, value in values.items()})
    cfg = OmegaConf.load(source / "configs/simate-beta_mem20_ae_80k.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    output_root = path(os.environ.get("SIMATE_CHECKPOINT_ROOT", POLICY_DIR / "checkpoints"))
    cfg.experiment.name = name
    cfg.experiment.work_dir = str(output_root)
    cfg.experiment.seed = args.seed
    nnodes = int(cfg.engine.nnodes)
    if nnodes > 1 and ("NODE_RANK" not in env or "MASTER_ADDR" not in env):
        raise ValueError("Set NODE_RANK and MASTER_ADDR on every node for multi-node training.")
    rank = int(env.get("NODE_RANK", "0")) if nnodes > 1 else 0
    if not 0 <= rank < nnodes:
        raise ValueError("NODE_RANK is outside engine.nnodes.")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    run_dir = output_root / name
    if not args.dry_run and (run_dir / "model.safetensors").exists():
        raise ValueError("A finished checkpoint already exists; choose a new ckpt_name.")
    if not cfg.train_cfg.save_final_checkpoint or not cfg.train_cfg.export_final_checkpoint:
        raise ValueError("The adapter requires final checkpoint saving and export.")
    launch_dir = run_dir / "launch"
    launch_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", prefix=f"rank{rank}-", dir=launch_dir, delete=False) as f:
        OmegaConf.save(cfg, f)
        config_path = f.name
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    env["PYTHON"] = sys.executable
    env["PYTHONPATH"] = str(source)
    env["LOG_DIR"] = str(run_dir / "logs")
    env.setdefault("CACHE_ROOT", str(run_dir / "runtime_cache" / f"rank{rank}"))
    command = ["bash", str(source / "tools/run_distributed_train.sh"), "--config", config_path]
    if args.dry_run:
        command.append("--dry-run")
    print(f"Training source: {source} @ {revision}", flush=True)
    print(f"Output directory: {run_dir}", flush=True)
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=source, env=env, check=True)
    if not args.dry_run and rank == 0:
        final = run_dir / "exported_ckpts" / f"{name}-{int(cfg.train_cfg.max_steps)}.safetensors"
        required_file(final)
        target = run_dir / "model.safetensors"
        if target.exists() or target.is_symlink():
            raise ValueError(f"Refusing to replace {target}; use a new ckpt_name for each training run.")
        target.symlink_to(final.relative_to(run_dir))
        shutil.copyfile(values["SIMATE_TOKENIZER"], run_dir / "tokenizer.model")
        print(f"Evaluation checkpoint ready: {run_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("prepare", "train"):
        p = sub.add_parser(mode)
        p.add_argument("bench_name", choices=["RoboDojo"])
        p.add_argument("ckpt_name")
        p.add_argument("env_cfg_type", choices=["arx_x5"])
        p.add_argument("action_type", choices=["joint"])
        if mode == "prepare":
            p.add_argument("--verify", action="store_true", help="Verify every file in SHA256SUMS.")
        else:
            p.add_argument("seed", type=int)
            p.add_argument("gpu_id", help="Comma-separated local GPU indices.")
            p.add_argument("--dry-run", action="store_true")
    argv = sys.argv[1:]
    overrides = []
    if "--" in argv:
        split = argv.index("--")
        argv, overrides = argv[:split], argv[split + 1:]
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.ckpt_name):
        parser.error("ckpt_name must be a single alphanumeric run label, not a path.")
    if args.mode == "prepare":
        if overrides:
            parser.error("Data preparation does not accept training overrides.")
        prepare(args)
    else:
        if args.seed < 0 or not re.fullmatch(r"\d+(,\d+)*", args.gpu_id):
            parser.error("Use a nonnegative seed and comma-separated GPU indices.")
        args.overrides = overrides
        train(args)


if __name__ == "__main__":
    main()
