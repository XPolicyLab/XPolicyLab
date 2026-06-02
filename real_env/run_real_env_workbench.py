from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .constants import XPOLICYLAB_ROOT, XONE_ROOT
from .helpers import load_yaml, str_to_bool

def default_deploy_config(policy_name: str) -> Path:
    return XPOLICYLAB_ROOT / "policy" / policy_name / "deploy.yml"


def cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    keys = [
        "dataset_name",
        "task_name",
        "env_cfg_type",
        "expert_data_num",
        "action_type",
        "ckpt_setting",
        "policy_name",
        "base_cfg",
        "host",
        "port",
        "seed",
        "eval_batch",
        "force_reach_mode",
    ]
    return {key: getattr(args, key) for key in keys if getattr(args, key) is not None}


def build_deploy_cfg(args: argparse.Namespace) -> dict[str, Any]:
    policy_name = args.policy_name or "ACT"
    deploy_config = args.deploy_config or default_deploy_config(policy_name)

    deploy_cfg: dict[str, Any] = {}
    if deploy_config is not None:
        deploy_config = Path(deploy_config).expanduser()
        if deploy_config.exists():
            deploy_cfg.update(load_yaml(deploy_config))
        elif args.deploy_config is not None:
            raise FileNotFoundError(f"deploy config not found: {deploy_config}")

    deploy_cfg.update(cli_overrides(args))
    deploy_cfg.setdefault("policy_name", policy_name)
    deploy_cfg.setdefault("host", "localhost")
    deploy_cfg.setdefault("eval_batch", False)
    return deploy_cfg


def validate_deploy_cfg(deploy_cfg: dict[str, Any]) -> None:
    required_keys = ["base_cfg", "task_name", "policy_name", "host", "port", "ckpt_setting"]
    missing_keys = [key for key in required_keys if deploy_cfg.get(key) is None]
    if missing_keys:
        raise ValueError(f"missing required deploy config keys: {missing_keys}")


def print_launch_info(deploy_cfg: dict[str, Any], eval_episode_num: int) -> None:
    task_name = deploy_cfg["task_name"]
    policy_name = deploy_cfg["policy_name"]
    ckpt_setting = str(deploy_cfg["ckpt_setting"])
    print("[RealEnvWorkbench] launch config")
    print(f"  policy:          {policy_name}")
    print(f"  task:            {task_name}")
    print(f"  base_cfg:        {deploy_cfg['base_cfg']}")
    print(f"  ckpt_setting:    {ckpt_setting}")
    print(f"  server:          {deploy_cfg['host']}:{deploy_cfg['port']}")
    print(f"  eval_batch:      {deploy_cfg.get('eval_batch', False)}")
    print(f"  eval episodes:   {eval_episode_num}")
    print(f"  layout root:     {XONE_ROOT / 'layouts' / task_name}")
    print(f"  result root:     {XONE_ROOT / 'eval_results' / policy_name / ckpt_setting / task_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the RealEnvWorkbench on the robot client side. "
            "Start the policy server separately, then connect to it with --host/--port."
        )
    )
    parser.add_argument("--deploy_config", type=Path, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--env_cfg_type", type=str, default=None)
    parser.add_argument("--expert_data_num", type=str, default=None)
    parser.add_argument("--action_type", type=str, default=None)
    parser.add_argument("--ckpt_setting", type=str, default=None)
    parser.add_argument("--policy_name", type=str, default=None)
    parser.add_argument("--base_cfg", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval_batch", type=str_to_bool, default=None)
    parser.add_argument("--eval_episode_num", type=int, default=1)
    parser.add_argument("--force_reach_mode", type=str_to_bool, default=None)
    parser.add_argument("--poll_hz", type=float, default=30.0)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--record_trajectory", action="store_true")
    parser.add_argument("--record_fps", type=float, default=30.0)
    parser.add_argument("--record_crf", type=int, default=0)
    parser.add_argument("--record_camera", action="append", default=None)
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--print_config_only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deploy_cfg = build_deploy_cfg(args)
    validate_deploy_cfg(deploy_cfg)

    print_launch_info(deploy_cfg, args.eval_episode_num)

    if args.print_config_only:
        return 0

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from .real_env_client import RealEnv
    from .recorder import build_episode_recorder
    from .workbench import RealEnvWorkbench

    real_env = RealEnv(deploy_cfg)
    recorder = None
    if args.record_video or args.record_trajectory:
        recorder = build_episode_recorder(
            real_env,
            fps=args.record_fps,
            record_video=args.record_video,
            record_trajectory=args.record_trajectory,
            camera_names=args.record_camera,
            crf=args.record_crf,
        )
        print(
            "[RealEnvWorkbench] recorder enabled: "
            f"video={args.record_video}, trajectory={args.record_trajectory}, "
            f"fps={args.record_fps}, crf={args.record_crf}"
        )

    gui = RealEnvWorkbench(
        env=real_env,
        eval_episode_num=args.eval_episode_num,
        poll_hz=args.poll_hz,
        recorder=recorder,
    )

    gui.start()
    gui.prepare_task()
    assert gui.m_app is not None
    return int(gui.m_app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
