"""Launch the compact Bagel-Robotics-1-Preview policy through XPolicyLab's standard server."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


POLICY_DIR = Path(__file__).resolve().parent
REMOVED_VIDEO_PARAMETERS = 6003498177


def command(args):
    if (args.bench, args.env_cfg, args.action_type) != ("RoboDojo", "arx_x5", "ee"):
        raise ValueError("Bagel-Robotics-1-Preview supports RoboDojo/arx_x5/EE16 only")
    xpl = Path(args.xpl_root).resolve() if args.xpl_root else POLICY_DIR.parent.parent
    if not (xpl / "setup_policy_server.py").is_file():
        raise FileNotFoundError(f"XPolicyLab server not found under {xpl}")
    if str(xpl.parent) not in sys.path:
        sys.path.insert(0, str(xpl.parent))
    from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root

    bundle = resolve_checkpoint_root(
        dict(ckpt_name=args.ckpt, model_path=args.bundle, bench_name=args.bench,
             env_cfg_type=args.env_cfg, action_type=args.action_type, seed=args.seed),
        POLICY_DIR / "checkpoints", policy_dir=POLICY_DIR,
    )
    if bundle.is_file():
        bundle = bundle.parent
    bundle = bundle.resolve()
    wan = (Path(args.wan).expanduser().resolve() if args.wan else
           POLICY_DIR / "checkpoints/Wan2.1-I2V-14B-480P")
    from download_checkpoint import verify

    verify(bundle, wan)
    manifest = json.loads((bundle / "manifest.json").read_text())
    architecture = json.loads((bundle / "pruning_architecture.json").read_text())
    if (manifest.get("format") != "vpp2-robodojo-bundle-v1" or
            manifest.get("step") != 100000 or
            manifest.get("pruning", {}).get("removed_video_parameters") !=
            REMOVED_VIDEO_PARAMETERS or
            architecture.get("removed_video_parameters") != REMOVED_VIDEO_PARAMETERS):
        raise ValueError("Select the recovered 6.003B Bagel-Robotics-1-Preview compact bundle")
    overrides = dict(
        policy_name=POLICY_DIR.name, bench_name=args.bench, task_name=args.task,
        ckpt_name=args.ckpt, env_cfg_type=args.env_cfg, action_type="ee", seed=args.seed,
        host=args.host, port=args.port, checkpoint_path=str(bundle / "action.pt"),
        video_checkpoint_path=str(bundle / "video.pt"),
        dataset_stats_path=str(bundle / "dataset_stats.json"), wan_model_dir=str(wan),
        num_inference_steps=10, sigma_shift=1.0, replan_steps=16,
        action_horizon=32, history_stride=25, video_seed_offset=1000003,
        temporal_ensemble_mode="fixed90",
    )
    cmd = [sys.executable, "-u", str(xpl / "setup_policy_server.py"),
           "--config_path", str(POLICY_DIR / "deploy.yml"), "--overrides",
           *(f"{key}={value}" for key, value in overrides.items())]
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES=args.gpu, VPP2_TORCH_LOAD_MMAP="1",
        VPP2_ACTION_SAMPLER="euler", VPP2_TEMPORAL_ENSEMBLE_MODE="fixed90",
        PYTHONPATH=os.pathsep.join((str(POLICY_DIR / "runtime"), str(xpl.parent),
                                    str(xpl), os.environ.get("PYTHONPATH", ""))),
    )
    return cmd, env, xpl.parent, overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", default="RoboDojo")
    parser.add_argument("--task", default="insert_tubes")
    parser.add_argument("--ckpt", default="bagel_robotics_1_preview")
    parser.add_argument("--env-cfg", default="arx_x5")
    parser.add_argument("--action-type", default="ee")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--bundle")
    parser.add_argument("--wan")
    parser.add_argument("--xpl-root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cmd, env, cwd, settings = command(args)
    print(json.dumps({"policy": POLICY_DIR.name, "settings": settings,
                      "bundle": str(Path(settings["checkpoint_path"]).parent)}, indent=2),
          flush=True)
    if not args.dry_run:
        os.chdir(cwd)
        os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
