from __future__ import annotations

import argparse
import builtins
import io
import json
import os
import sys
import types
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


@contextmanager
def task_config_video_override(
    robotwin_eval: types.ModuleType,
    robotwin_root: Path,
    task_config: str,
    enabled: bool,
):
    """Override eval_video_log for one eval call without editing RoboTwin."""
    config_path = (robotwin_root / "task_config" / f"{task_config}.yml").resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"RoboTwin task config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"RoboTwin task config must be a mapping: {config_path}")
    config["eval_video_log"] = enabled
    rendered = yaml.safe_dump(config, sort_keys=False)

    had_module_open = hasattr(robotwin_eval, "open")
    previous_open = getattr(robotwin_eval, "open", None)

    def patched_open(file, mode="r", *args, **kwargs):
        try:
            candidate = Path(file)
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            candidate = candidate.resolve()
        except (TypeError, OSError):
            candidate = None
        if candidate == config_path and "r" in mode and "b" not in mode:
            return io.StringIO(rendered)
        return builtins.open(file, mode, *args, **kwargs)

    robotwin_eval.open = patched_open
    try:
        yield
    finally:
        if had_module_open:
            robotwin_eval.open = previous_open
        else:
            delattr(robotwin_eval, "open")


def parse_result_file(path: Path, expected_episodes: int) -> dict[str, Any]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [line for line in lines if line]
    if not values:
        raise ValueError(f"Empty RoboTwin result file: {path}")
    try:
        success_rate = float(values[-1])
    except ValueError as exc:
        raise ValueError(f"Invalid success rate in {path}: {values[-1]!r}") from exc
    if not 0.0 <= success_rate <= 1.0:
        raise ValueError(f"Success rate outside [0, 1] in {path}: {success_rate}")
    success_count = int(round(success_rate * expected_episodes))
    return {
        "episodes": expected_episodes,
        "success_count": success_count,
        "success_rate": success_rate,
    }


def native_to_standard(observation: dict[str, Any], instruction: str) -> dict[str, Any]:
    source = observation.get("observation", observation)
    try:
        qpos = np.asarray(observation["joint_action"]["vector"], dtype=np.float32).reshape(-1)
    except KeyError as exc:
        raise KeyError("RoboTwin observation is missing joint_action.vector") from exc
    if qpos.size != 14:
        raise ValueError(f"Expected RoboTwin 14-D qpos, got {qpos.shape}")
    cameras = {
        "cam_head": "head_camera",
        "cam_left_wrist": "left_camera",
        "cam_right_wrist": "right_camera",
    }
    vision = {}
    for standard_name, native_name in cameras.items():
        try:
            color = np.asarray(source[native_name]["rgb"])
        except KeyError as exc:
            raise KeyError(f"RoboTwin observation is missing {native_name}.rgb") from exc
        vision[standard_name] = {"color": color}
    return {
        "data_format_version": "v1.0",
        "instruction": instruction,
        "vision": vision,
        "state": {
            "left_arm_joint_state": qpos[0:6],
            "left_ee_joint_state": qpos[6:7],
            "right_arm_joint_state": qpos[7:13],
            "right_ee_joint_state": qpos[13:14],
        },
    }


def standard_action_to_qpos(action: dict[str, Any]) -> np.ndarray:
    parts = [
        np.asarray(action["left_arm_joint_state"], dtype=np.float32).reshape(-1),
        np.asarray(action["left_ee_joint_state"], dtype=np.float32).reshape(-1),
        np.asarray(action["right_arm_joint_state"], dtype=np.float32).reshape(-1),
        np.asarray(action["right_ee_joint_state"], dtype=np.float32).reshape(-1),
    ]
    qpos = np.concatenate(parts)
    if qpos.size != 14:
        raise ValueError(f"Expected XPolicyLab 14-D action, got {qpos.shape}")
    return qpos


def build_policy_shim(model_client):
    shim = types.ModuleType("OLA_SEM_XPL_BRIDGE")

    def get_model(_usr_args):
        return model_client

    def reset_model(client):
        client.call(func_name="reset")

    def eval_policy(TASK_ENV, client, observation):
        instruction = TASK_ENV.get_instruction()
        client.call(
            func_name="update_obs",
            obs=native_to_standard(observation, instruction),
        )
        actions = client.call(func_name="get_action")
        for action_idx, action in enumerate(actions):
            if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
                break
            TASK_ENV.take_action(standard_action_to_qpos(action), action_type="qpos")
            if action_idx + 1 < len(actions) and not TASK_ENV.eval_success:
                feedback = TASK_ENV.get_obs()
                client.call(
                    func_name="update_obs",
                    obs=native_to_standard(feedback, instruction),
                )

    shim.get_model = get_model
    shim.reset_model = reset_model
    shim.eval = eval_policy
    return shim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--test-num", default=1, type=int)
    parser.add_argument("--result-path-tag", default="xpolicylab_ola_sem")
    parser.add_argument("--eval-video-log", default=True, type=str2bool)
    parser.add_argument("--summary-path")
    parser.add_argument("--checkpoint-root")
    args = parser.parse_args()

    if args.test_num <= 0:
        raise ValueError(f"--test-num must be positive, got {args.test_num}")

    robotwin_root = Path(args.robotwin_root).resolve()
    if not (robotwin_root / "script" / "eval_policy.py").is_file():
        raise FileNotFoundError(f"Invalid RoboTwin root: {robotwin_root}")
    xpl_root = Path(__file__).resolve().parents[2]
    workspace_root = xpl_root.parent
    for path in (workspace_root, xpl_root, robotwin_root, robotwin_root / "script"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    os.chdir(robotwin_root)

    from client_server.ws import WsModelClient

    client = WsModelClient(
        url=f"ws://{args.host}:{args.port}",
        evaluation_id=f"RoboTwin-{args.task_name}-{args.seed}",
        trial_id=f"{args.task_name}-{args.seed}",
        request_timeout_s=1200,
        max_connect_seconds=1200,
        ws_ping_timeout_s=120,
    )
    module_name = "OLA_SEM_XPL_BRIDGE"
    sys.modules[module_name] = build_policy_shim(client)
    result_root = (
        robotwin_root
        / "eval_result"
        / args.task_name
        / module_name
        / args.task_config
        / args.result_path_tag
    )
    old_results = set(result_root.glob("*/_result.txt"))
    try:
        from script import eval_policy as robotwin_eval

        with task_config_video_override(
            robotwin_eval,
            robotwin_root,
            args.task_config,
            enabled=args.eval_video_log,
        ):
            robotwin_eval.main(
                {
                    "policy_name": module_name,
                    "task_name": args.task_name,
                    "task_config": args.task_config,
                    "ckpt_setting": "xpolicylab",
                    "seed": args.seed,
                    "instruction_type": "unseen",
                    "inference_mode": "history_flow",
                    "test_num": args.test_num,
                    "num_inference_timesteps": 4,
                    "history_action_noise_std": 0.02,
                    "future_video_denoise_fraction": 1.0,
                    "result_path_tag": args.result_path_tag,
                    "model_config": "robotwin.yml",
                }
            )

        new_results = set(result_root.glob("*/_result.txt")) - old_results
        if len(new_results) != 1:
            raise RuntimeError(
                f"Expected exactly one new RoboTwin result under {result_root}, "
                f"found {len(new_results)}"
            )
        result_path = new_results.pop()
        result = parse_result_file(result_path, args.test_num)
        if not args.eval_video_log:
            videos = list(result_path.parent.glob("*.mp4"))
            if videos:
                raise RuntimeError(
                    f"Video logging was disabled but {len(videos)} MP4 files were created"
                )
        summary = {
            "status": "completed",
            "task": args.task_name,
            "task_config": args.task_config,
            "seed": args.seed,
            "instruction_type": "unseen",
            "checkpoint_root": args.checkpoint_root,
            "result_path_tag": args.result_path_tag,
            "result_file": str(result_path.resolve()),
            "video_enabled": args.eval_video_log,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "completed_at": datetime.now().astimezone().isoformat(),
            **result,
        }
        print("[OLA_SEM] RESULT " + json.dumps(summary, sort_keys=True))
        if args.summary_path:
            summary_path = Path(args.summary_path)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        client.close()
        sys.modules.pop(module_name, None)


if __name__ == "__main__":
    main()
