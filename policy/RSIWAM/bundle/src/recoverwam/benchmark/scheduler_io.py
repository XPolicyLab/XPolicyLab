from __future__ import annotations
import json, os, math, time, shlex, subprocess, socket, sys
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    last_error: Exception | None = None
    for _ in range(5):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"JSON root is not an object: {path}")
            return payload
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            time.sleep(0.1)
    assert last_error is not None
    raise last_error


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def load_tasks(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def details_by_layout(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    raw = payload.get("details") or {}
    if not isinstance(raw, dict):
        raise ValueError("result details must be an object")
    result: dict[int, dict[str, Any]] = {}
    for key, detail in raw.items():
        if not isinstance(detail, dict):
            raise ValueError(f"malformed detail row: {key}")
        layout_id = int(detail.get("layout_id", key))
        if layout_id in result:
            raise ValueError(f"duplicate layout_id={layout_id}")
        result[layout_id] = detail
    return result


def native_target(task: str, official_tasks: set[str]) -> int:
    return 25 if task.endswith("_random") or f"{task}_random" in official_tasks else 50


def result_parent(config: dict[str, Any], task: str) -> Path:
    return (
        Path(config["robodojo_root"])
        / "eval_result"
        / "RoboDojo"
        / task
        / config.get("policy_name", "RecoverWAM")
        / config.get("env_cfg", "arx_x5")
        / (
            f"{int(config.get('seed', 1))}_ckpt_name={config['ckpt_name']},"
            f"action_type={config['action_type']}"
        )
    )


def canonical_result_path(config: dict[str, Any], task: str) -> Path:
    return result_parent(config, task) / f"{config['run_id']}_{task}" / "_result.json"


def layout_count(config: dict[str, Any], task: str) -> int:
    root = (
        Path(config["robodojo_root"])
        / "Assets"
        / "Eval_Layout"
        / "RoboDojo"
        / config.get("env_cfg", "arx_x5")
        / str(int(config.get("seed", 0)))
    )
    return len(list(root.glob(f"{task}_[0-9]*.json")))


def reference_seconds(config: dict[str, Any]) -> dict[str, float]:
    path_value = config.get("reference_summary")
    if not path_value or not Path(path_value).is_file():
        return {}
    payload = read_json(Path(path_value))
    result: dict[str, float] = {}
    for row in payload.get("results", []):
        if not isinstance(row, dict):
            continue
        try:
            seconds = float(row.get("elapsed_sec", 0))
        except (TypeError, ValueError):
            continue
        task = str(row.get("task", ""))
        if task and math.isfinite(seconds) and seconds > 0:
            result[task] = seconds
    return result


def split_even(values: list[int], groups: int) -> list[list[int]]:
    if groups < 1 or groups > len(values):
        raise ValueError(f"invalid group count {groups} for {len(values)} values")
    result: list[list[int]] = []
    cursor = 0
    for index in range(groups):
        size = len(values) // groups + (1 if index < len(values) % groups else 0)
        result.append(values[cursor : cursor + size])
        cursor += size
    return result


def client_environment(config, run_id):
    """Build the simulator environment using explicitly configured tool locations."""
    python = Path(config["sim_python"])
    extra_dirs = config.get("sim_bin_dirs", [])
    if not isinstance(extra_dirs, list) or any(not isinstance(p, str) for p in extra_dirs):
        raise ValueError("sim_bin_dirs must be a list of directories on the simulator host")
    return {
        "PATH": ":".join(
            [
                *extra_dirs,
                str(python.parent),
                "/usr/local/cuda/bin",
                "/usr/local/bin",
                "/usr/bin",
                "/bin",
            ]
        ),
        "ROBODOJO_RUN_ID": run_id,
        "EVAL_NUM": "native",
        "ROBODOJO_SAVE_EVAL_VIDEOS": "true" if config.get("save_eval_videos") else "false",
        "ROBODOJO_SAVE_EVAL_VIDEOS_MAX_LAYOUTS": str(config.get("save_eval_videos_max_layouts", 5)),
        "ROBODOJO_MAX_BASH_RETRIES": "10",
        "SETUPTOOLS_SCM_PRETEND_VERSION": "0.8.0",
    }


def launch_shard(config, shard):
    """Launch an isolated client; local or SSH, shared filesystem required."""
    root = Path(config["robodojo_root"])
    python = Path(config["sim_python"])
    status = Path(shard["status"])
    status.parent.mkdir(parents=True, exist_ok=True)
    Path(shard["log"]).parent.mkdir(parents=True, exist_ok=True)
    env_values = client_environment(config, shard["run_id"])
    command = [
        "bash",
        "scripts/robodojo.sh",
        "client",
        "--dataset",
        "RoboDojo",
        "--task",
        shard["task"],
        "--env-cfg",
        config.get("env_cfg", "arx_x5"),
        "--policy-name",
        config.get("policy_name", "RecoverWAM"),
        "--policy-host",
        "127.0.0.1",
        "--policy-port",
        str(shard["policy_port"]),
        "--seed",
        str(config.get("seed", 1)),
        "--env-gpu",
        str(shard["gpu"]),
        "--ckpt",
        config["ckpt_name"],
        "--action-type",
        "ee",
        "--connect-timeout",
        "30",
    ]
    probe = (
        "import socket; s=socket.create_connection(('127.0.0.1',%d),5); s.close()"
        % shard["policy_port"]
    )
    finish = (
        'rc=$?; printf "%s\n" "$rc" > '
        + shlex.quote(str(status) + ".tmp")
        + "; mv "
        + shlex.quote(str(status) + ".tmp")
        + " "
        + shlex.quote(str(status))
    )
    lines = ["set -e", "trap " + shlex.quote(finish) + " EXIT", "cd " + shlex.quote(str(root))]
    lines += ["export " + k + "=" + shlex.quote(v) for k, v in env_values.items()]
    if config.get("save_eval_videos"):
        lines += [
            "command -v ffmpeg >/dev/null || { echo 'Video saving requires ffmpeg on PATH; configure sim_bin_dirs.' >&2; exit 127; }",
            "ffmpeg -version >/dev/null",
        ]
    lines += [shlex.join([str(python), "-c", probe]), shlex.join(command)]
    payload = "\n".join(lines)
    host = shard.get("ssh_host")
    if host in (None, "", "local", "localhost"):
        with open(shard["log"], "ab") as log:
            subprocess.Popen(
                ["bash", "-c", payload],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    else:
        # A fresh session name must be unused. Never terminate another job.
        remote = shlex.join(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                shard["session"],
                "bash",
                "-c",
                payload + " >" + shlex.quote(shard["log"]) + " 2>&1",
            ]
        )
        subprocess.run(
            ["ssh", "-p", str(shard["ssh_port"]), "-o", "BatchMode=yes", host, remote], check=True
        )
