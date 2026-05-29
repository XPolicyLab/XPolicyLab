import argparse
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

from task_config import get_robot_type


ROOT_DIR = Path(__file__).resolve().parent
ENV_SETUP_PREFIX = ()
def build_shell_cmd(args: list[str], extra_env: dict[str, str] | None = None) -> list[str]:
    env_prefix = ""
    if extra_env:
        env_parts = [f"{key}={shlex.quote(value)}" for key, value in extra_env.items()]
        env_prefix = " ".join(env_parts) + " "
    shell_cmd = f"{env_prefix}{shlex.join(args)}"
    return ["bash", "-lc", shell_cmd]


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RoboChallenge task with correct robot + script based on task_name and test_type."
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Task name, e.g. arrange_flowers, clean_dining_table. Must exist in task_config.ROBO_CHALLENGE_TASKS.",
    )
    parser.add_argument(
        "--test_type",
        type=str,
        choices=["mock", "real"],
        required=True,
        help="Test type: 'mock' uses *_test.py with mock=True; 'real' uses *_demo.py with real robot interface.",
    )
    # Common
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Inference server URL, e.g. http://localhost:7778",
    )
    parser.add_argument(
        "--action_nums",
        type=int,
        default=20,
        help="Number of actions returned per inference step (used by real demo scripts).",
    )
    # Real-robot only
    parser.add_argument(
        "--user_token",
        type=str,
        help="User token for real robot evaluation (required when --test_type real).",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        help="Run ID for real robot evaluation (required when --test_type real).",
    )

    args = parser.parse_args()
    if args.action_nums <= 0:
        raise SystemExit("--action_nums must be a positive integer.")

    robot_type = get_robot_type(args.task_name)
    if not robot_type:
        raise SystemExit(
            f"Unknown task_name '{args.task_name}'. "
            "Please check task_config.ROBO_CHALLENGE_TASKS."
        )

    robot_type = robot_type.lower()
    test_type = args.test_type.lower()

    # Decide which script to run based on robot_type + test_type
    script = None
    cmd: list[str] = []

    server_proc = None

    if test_type == "mock":
        mock_port = find_free_port()
        mock_url = f"http://127.0.0.1:{mock_port}"
        print(f"[run_task] selected free mock server port: {mock_port}")

        # Start mock server: run mock_robot_server.py --task_name <task_name> in mock_server directory
        server_cmd = build_shell_cmd([
            "python",
            "mock_robot_server.py",
            "--server_port",
            str(mock_port),
            "--task_name",
            args.task_name,
        ])
        mock_cwd = ROOT_DIR / "mock_server"
        print(f"[run_task] starting mock server (cwd={mock_cwd}): {' '.join(server_cmd)}")
        server_proc = subprocess.Popen(server_cmd, cwd=str(mock_cwd))

        # 给 mock server 一点时间完成启动
        time.sleep(3.0)

        # Mock uses *_test.py with --url --task_name
        if robot_type == "arx5":
            script = "arx5_test.py"
        elif robot_type == "aloha":
            script = "aloha_test.py"
        elif robot_type == "franka":
            script = "franka_test.py"
        elif robot_type == "ur5":
            script = "ur5_test.py"
        else:
            raise SystemExit(f"Unsupported robot_type '{robot_type}' for mock.")

        cmd = build_shell_cmd([
            "python",
            str(ROOT_DIR / script),
            "--url",
            args.url,
            "--robot-url",
            mock_url,
            "--task_name",
            args.task_name,
        ])

    else:  # real
        # Real uses *_demo.py with --user_token --run_id --url --task_name
        if not args.user_token or not args.run_id:
            raise SystemExit(
                "For --test_type real, both --user_token and --run_id must be provided."
            )

        if robot_type == "arx5":
            script = "arx5_demo.py"
        elif robot_type == "aloha":
            script = "aloha_demo.py"
        elif robot_type == "franka":
            script = "franka_demo.py"
        elif robot_type == "ur5":
            script = "ur5_demo.py"
        else:
            raise SystemExit(f"Unsupported robot_type '{robot_type}' for real.")

        cmd = build_shell_cmd([
            "python",
            str(ROOT_DIR / script),
            "--user_token",
            args.user_token,
            "--run_id",
            args.run_id,
            "--url",
            args.url,
            "--action_nums",
            str(args.action_nums),
            "--task_name",
            args.task_name,
        ])

    print(f"[run_task] robot_type={robot_type}, test_type={test_type}, script={script}")

    try:
        subprocess.run(cmd, check=True, cwd=str(ROOT_DIR))
    finally:
        # 仅在 mock 模式下需要关闭 mock server
        if server_proc is not None and server_proc.poll() is None:
            print("[run_task] stopping mock server")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()



if __name__ == "__main__":
    main()

