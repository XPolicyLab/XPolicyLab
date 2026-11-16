"""Archive and replay an episode's submitted code after its agent has stopped."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import select
import signal
import subprocess
import time
import traceback
from pathlib import Path

from environments import execute
from orchestrator.config import CODE_ROOT, REPO, config_from_dict
from orchestrator.files import append_jsonl, write_json
from orchestrator.resources import gpu_lease, sim_env
from orchestrator.scoring import capture as capture_scoring
from orchestrator.session import ExecutionTimeout, deadline
from simulators.observations import export_observation


def assemble_history(run, *, phase, seed):
    """Keep exact source bytes and boundaries, including failed requests and resets."""
    run = Path(run)
    output = run / "final"
    output.mkdir()
    history = run / "workspace/observations"
    paths = sorted(
        (p for p in history.glob("*/code.py") if p.parent.name.isdecimal()),
        key=lambda p: int(p.parent.name),
    )
    segments = []
    with (output / "solution.py").open("wb") as stream:
        stream.write(b"# Recorded submissions; manifest.json preserves execution boundaries.\n")
        for path in paths:
            code = path.read_bytes()
            number = int(path.parent.name)
            source = str(path.relative_to(run))
            stream.write(f"\n\n# Execution {number}: {source}\n".encode())
            start = stream.tell()
            stream.write(code)
            segments.append(
                {
                    "execution": number,
                    "source": source,
                    "start_byte": start,
                    "end_byte": stream.tell(),
                    "sha256": hashlib.sha256(code).hexdigest(),
                }
            )
            stream.write(b"\n")
    manifest = {"phase": phase, "seed": seed, "segments": segments}
    write_json(output / "manifest.json", manifest)
    return manifest


class TraceVideos:
    """Stream all submissions into one video per camera, without retaining the episode."""

    def __init__(self, output, fps, stack):
        self.output, self.fps, self.stack = output, fps, stack
        self.writers, self.paths, self.counts = {}, {}, {}

    def append(self, frames):
        import imageio.v2 as iio

        ranges = {}
        for camera, images in frames.items():
            if not images:
                continue
            if not camera.replace("_", "").isalnum():
                raise ValueError("Unsafe camera filename")
            if camera not in self.writers:
                name = f"trace_{camera}.mp4"
                writer = iio.get_writer(self.output / name, fps=self.fps, codec="libx264")
                self.stack.callback(writer.close)
                self.writers[camera] = writer
                self.paths[camera] = name
                self.counts[camera] = 0
            start = self.counts[camera]
            for frame in images:
                self.writers[camera].append_data(frame)
                self.counts[camera] += 1
            ranges[camera] = [start, self.counts[camera]]
        return ranges


def replay_history(cfg, run, *, event=lambda value: None):
    """Run in a fresh simulator process, preserving the original session's boundaries."""
    from simulators.backend import build

    output = Path(run) / "final"
    manifest = json.loads((output / "manifest.json").read_text())
    program = (output / "solution.py").read_bytes()
    phase, seed = manifest["phase"], manifest["seed"]
    timeout = (
        cfg.playground.submission_timeout_sec
        if phase == "playground"
        else cfg.test.submission_timeout_sec
    )
    started = time.monotonic()
    state = {
        "phase": phase,
        "seed": seed,
        "status": "running",
        "complete": False,
        "ok": True,
        "success": False,
        "segments_total": len(manifest["segments"]),
        "segments_replayed": 0,
        "native_steps_used": 0,
        "program": "solution.py",
        "manifest": "manifest.json",
        "trace": "trace.jsonl",
        "videos": {},
        "failure": None,
    }
    write_json(output / "result.json", state)
    env = None
    try:
        with contextlib.ExitStack() as stack:
            env = build(cfg, seed=seed)
            env.execution_phase = phase
            env.reset(seed=seed)
            capture_scoring(env, output, execution=0)
            export_observation(env, output / "observations/000000", workspace=output)
            videos = TraceVideos(output, cfg.feedback_fps, stack)
            state["videos"] = videos.paths
            for segment in manifest["segments"]:
                number = segment["execution"]
                folder = output / "observations" / f"{number:06d}"
                folder.mkdir(parents=True)
                code = program[segment["start_byte"] : segment["end_byte"]]
                if hashlib.sha256(code).hexdigest() != segment["sha256"]:
                    raise ValueError(f"Archived source changed for execution {number}")
                state["current_execution"] = number
                write_json(output / "result.json", state)
                event({"timeout": timeout + 10})
                step_started = time.monotonic()
                try:
                    with deadline(timeout):
                        result = execute(env, code.decode("utf-8"))
                except ExecutionTimeout:
                    # The simulator is no longer safe to continue. Preserve the
                    # available observation and frame prefix, then mark incomplete.
                    event({"timeout": timeout + 10})
                    export_observation(env, folder, workspace=output)
                    frame_ranges = videos.append(env._frames)
                    error = traceback.format_exc()
                    (folder / "stderr.txt").write_text(error)
                    record = {
                        "execution": number,
                        "source": segment["source"],
                        "ok": False,
                        "failure": "execution_timeout",
                        "success": bool(env.success),
                        "native_steps_used": env.control_steps,
                        "observation": str(folder.relative_to(output)),
                        "frame_ranges": frame_ranges,
                    }
                    write_json(folder / "result.json", record)
                    append_jsonl(output / "trace.jsonl", record)
                    state.update(success=bool(env.success), native_steps_used=env.control_steps)
                    raise
                event({"timeout": timeout + 10})
                scoring = capture_scoring(env, output, execution=number)
                export_observation(env, folder, workspace=output)
                frame_ranges = videos.append(result.frames)
                (folder / "stdout.txt").write_text(result.stdout)
                (folder / "stderr.txt").write_text(result.stderr)
                record = {
                    "execution": number,
                    "source": segment["source"],
                    "ok": result.ok,
                    "success": result.success,
                    "terminated": result.terminated,
                    "truncated": result.truncated,
                    "native_steps_used": env.control_steps,
                    "observation": str(folder.relative_to(output)),
                    "frame_ranges": frame_ranges,
                    "elapsed_sec": round(time.monotonic() - step_started, 6),
                }
                write_json(folder / "result.json", record)
                append_jsonl(output / "trace.jsonl", record)
                state.update(
                    segments_replayed=state["segments_replayed"] + 1,
                    ok=state["ok"] and result.ok,
                    success=result.success,
                    native_steps_used=env.control_steps,
                    **(scoring or {}),
                )
                write_json(output / "result.json", state)
                # A program error, native success or truncation does not discard
                # later submissions: the original agent could continue or reset.
            state.update(status="complete", complete=True)
            scoring = capture_scoring(
                env, output, execution=state["segments_replayed"], final=True
            )
            state.update(scoring or {})
    except (Exception, ExecutionTimeout) as exc:
        state.update(
            status="failed",
            complete=False,
            ok=False,
            failure=f"{type(exc).__name__}: {exc}",
        )
        (output / "error.txt").write_text(traceback.format_exc())
    state["elapsed_sec"] = round(time.monotonic() - started, 6)
    write_json(output / "result.json", state)
    # Isaac's application shutdown can exit the interpreter with code zero.
    # Close all video writers and persist the terminal result before that call.
    if env is not None:
        env.close()
    return state


def run_replay(cfg, run, experiment, *, phase, seed, gpu=None):
    """Launch an archival replay independently of agent budgets and scoring."""
    run = Path(run)
    output = run / "final"
    started = time.monotonic()
    failure = None
    process = None
    try:
        manifest = assemble_history(run, phase=phase, seed=seed)
        if not manifest["segments"]:
            state = {
                "status": "empty",
                "complete": False,
                "ok": False,
                "failure": "no_submissions",
                "segments_total": 0,
                "segments_replayed": 0,
            }
            write_json(output / "result.json", state)
            return {**state, "result_path": "final/result.json"}
        read_fd, write_fd = os.pipe()
        try:
            with gpu_lease(gpu):
                process = subprocess.Popen(
                    [
                            str(cfg.python),
                            "-m",
                            "orchestrator.replay",
                            "--config",
                            str(
                                run / "episode-config.json"
                                if (run / "episode-config.json").exists()
                                else experiment / "resolved.json"
                            ),
                            "--run-dir",
                            str(run),
                            "--ready-fd",
                            str(write_fd),
                        ],
                    cwd=CODE_ROOT,
                    env=sim_env(gpu, cfg.simulator.backend),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    pass_fds=(write_fd,),
                    start_new_session=True,
                )
                os.close(write_fd)
                write_fd = None
                expires = time.monotonic() + 1200
                events = bytearray()
                try:
                    while process.poll() is None:
                        if select.select([read_fd], [], [], 0.2)[0]:
                            chunk = os.read(read_fd, 65536)
                            if not chunk:
                                process.wait(timeout=max(0.01, expires - time.monotonic()))
                                break
                            events.extend(chunk)
                            while b"\n" in events:
                                line, _, rest = events.partition(b"\n")
                                events = bytearray(rest)
                                expires = time.monotonic() + json.loads(line)["timeout"]
                        if time.monotonic() >= expires:
                            raise TimeoutError("Replay simulator timed out")
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=5)
        finally:
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)
        if process.returncode != 0:
            failure = f"replay_process_exited: {process.returncode}"
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    path = output / "result.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    if failure or state.get("status") not in {"complete", "failed"}:
        state.update(
            status="failed",
            complete=False,
            ok=False,
            failure=state.get("failure") or failure or "missing_replay_result",
        )
    state["process_sec"] = round(time.monotonic() - started, 6)
    write_json(path, state)
    return {**state, "result_path": "final/result.json"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--ready-fd", required=True, type=int)
    args = parser.parse_args()
    cfg = config_from_dict(json.loads(args.config.read_text()))
    try:
        state = replay_history(
            cfg,
            args.run_dir,
            event=lambda value: os.write(args.ready_fd, (json.dumps(value) + "\n").encode()),
        )
        return 0 if state["complete"] else 1
    finally:
        os.close(args.ready_fd)


if __name__ == "__main__":
    raise SystemExit(main())
