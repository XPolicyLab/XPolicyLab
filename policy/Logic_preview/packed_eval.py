"""Run the shared 746 layout queue with one seed-matched policy server per worker group."""
import argparse
import asyncio
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from uuid import uuid4
import runpy

import yaml

from packed_watchdog import update_job_monitor


@dataclass(frozen=True)
class Task:
    name: str
    target: int
    count: int
    parent: Path


def inventory(seed, ckpt):
    names = subprocess.check_output(
        [sys.executable, "scripts/internal/task_inventory.py", "--only-runnable"], text=True
    ).splitlines()
    config = yaml.safe_load(Path("task/RoboDojo/config/_task.yml").read_text())
    layout_dir = Path("Assets/Eval_Layout/RoboDojo/arx_x5") / str(seed)
    files = [path.name for path in layout_dir.iterdir()]
    tasks = []
    for name in names:
        target = int(config["tasks"].get(name, {}).get("eval_nums", config["common"]["eval_nums"]))
        pattern = re.compile(rf"{re.escape(name)}_\d+\.json")
        ids = {int(filename[len(name)+1:-5]) for filename in files if pattern.fullmatch(filename)}
        count = len(ids)
        if ids != set(range(count)):
            raise ValueError(f"{name}: expected contiguous zero-based layout IDs")
        if not 0 < target <= count:
            raise ValueError(f"{name}: {count} layouts for {target} requested episodes")
        parent = Path("eval_result/RoboDojo") / name / "Logic_preview/arx_x5" / f"{seed}_ckpt_name={ckpt},action_type=ee"
        tasks.append(Task(name, target, count, parent))
    if not tasks or len(names) != len(set(names)):
        raise ValueError("Expected a nonempty, unique official runnable task inventory")
    return tasks


async def stop_client(process):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with suppress(ProcessLookupError):
            os.killpg(process.pid, sig)
        try:
            await asyncio.wait_for(process.wait(), 10)
            return
        except asyncio.TimeoutError:
            continue
    raise RuntimeError(f"Simulator process group {process.pid} did not exit")


async def wait_client(process, directory, manifest, args):
    started = time.time()
    monitor = dict(started_at=started, last_progress_at=started, log=str(directory / "client.log"))
    while process.returncode is None:
        try:
            return await asyncio.wait_for(process.wait(), 5)
        except asyncio.TimeoutError:
            pass
        now = time.time()
        try:
            stamp = manifest.stat().st_mtime_ns
        except FileNotFoundError:
            stamp = None
        reason = update_job_monitor(monitor, now, stamp)
        if args.timeout and now - started >= args.timeout:
            reason = "wall time limit"
        if args.idle_timeout and now - monitor["last_progress_at"] >= args.idle_timeout:
            reason = "no simulator progress"
        if reason:
            (directory / "watchdog.txt").write_text(reason + "\n")
            await stop_client(process)
            return 124
    return process.returncode


async def run_client(args, task, gpu, directory, assigned, lease, pipeline):
    run_id = directory.name
    directory.mkdir()
    task.parent.mkdir(parents=True, exist_ok=True)
    manifest = task.parent / f"_resume_{run_id}.json"
    # Reserve the default result path. A checkout that ignores save_dir must
    # fail before its partial shard can become visible to the summarizer.
    reserved = task.parent / run_id
    reserved.touch(exist_ok=False)
    # The official client's resume interface filters layout ids and redirects
    # output outside the scored directory. Unstable layouts are never scored.
    excluded = sorted(set(range(task.count)).difference(assigned))
    command = ["bash", "scripts/robodojo.sh", "client", "--dataset", "RoboDojo",
               "--policy-name", "Logic_preview", "--env-cfg", "arx_x5", "--action-type", "ee",
               "--task", task.name, "--policy-host", args.policy_host, "--policy-port", str(args.port),
               "--env-gpu", gpu, "--ckpt", args.ckpt, "--seed", str(args.seed), "--eval-num", "native"]
    try:
        state = dict(run_id=run_id, save_dir=str(directory), abandoned_layout_ids=excluded, details={})
        env = dict(os.environ, ROBODOJO_RUN_ID=run_id, EVAL_NUM="native",
                   ROBODOJO_MAX_BASH_RETRIES="1", ROBODOJO_SAVE_EVAL_VIDEOS="true",
                   ROBODOJO_SAVE_EVAL_VIDEOS_MAX_LAYOUTS=str(task.count))
        env.setdefault("ROBODOJO_POLICY_CONNECT_SECONDS", "60")
        if pipeline is not None:
            state["eval_pipeline"] = pipeline
            env.update(ROBODOJO_EVAL_PIPELINE=json.dumps(pipeline), ROBODOJO_SIM_DEVICE=pipeline["sim_device"])
        write_json(manifest, state)
        with (directory / "client.log").open("w") as log:
            process = await asyncio.create_subprocess_exec(
                *command, env=env, pass_fds=(lease.fileno(),),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                status = await wait_client(process, directory, manifest, args)
            except asyncio.CancelledError:
                await stop_client(process)
                raise
        write_json(directory / "exit.json", status)
        return status
    finally:
        # The native client deletes the manifest on normal exit. Retain any
        # crash checkpoint for recovery by any group on the shared filesystem.
        if manifest.exists():
            manifest.replace(directory / "native-resume.json")
        reserved.unlink(missing_ok=True)


def read_result(directory, assigned, payload=None, incomplete=None):
    path = directory / "_result.json"
    if payload is None:
        if not path.exists():
            return {}, {}
        payload = json.loads(path.read_text())
    details = payload["details"]
    if not isinstance(details, dict) or set(details) != {str(i) for i in range(payload["eval_time"])}:
        raise ValueError(f"Inconsistent episode count in {path}")
    rows = {}
    seen = set()
    for index, record in details.items():
        layout = record["layout_id"]
        if type(layout) is not int or layout not in assigned or layout in seen:
            raise ValueError(f"Unexpected or duplicate layout {layout} in {path}")
        seen.add(layout)
        success, score = record["success"], record["score"]
        if type(success) is not bool or type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError(f"Malformed outcome for layout {layout} in {path}")
        prefix = f"episode_{int(index):07d}_"
        tag = "success" if success else "fail"
        videos = {p.name[len(prefix):]: p for p in directory.glob(prefix + "*.mp4")}
        for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            video = videos.get(f"{camera}_{tag}.mp4")
            if video is None or not video.stat().st_size:
                if incomplete is None:
                    raise ValueError(f"Missing {camera} video for layout {layout} in {path}")
                incomplete.add(layout)
                break
        else:
            rows[layout] = record, videos
    metadata = {k: v for k, v in payload.items() if k not in ("details", "score", "success_rate", "eval_time")}
    return rows, metadata


def publish(task, rows, metadata, work, run_id):
    if len(rows) < task.target:
        raise ValueError(f"{task.name}: only {len(rows)}/{task.target} stable layouts")
    # Publish scores and videos together so the official summarizer only sees
    # complete tasks. Preserve native row fields and the camera filenames.
    staging = Path(tempfile.mkdtemp(dir=work, prefix=f"{task.name}-publish-"))
    details = {}
    total_score = 0.0
    for index, (layout, (record, videos)) in enumerate(sorted(rows.items())[:task.target]):
        details[str(index)] = record
        total_score += record["score"]
        for suffix, src in videos.items():
            os.link(src, staging / f"episode_{index:07d}_{suffix}")
    payload = dict(metadata, eval_time=len(details), details=details,
                   success_rate=sum(r["success"] for r in details.values()) / len(details),
                   score=total_score / len(details) * 100)
    (staging / "_result.json").write_text(json.dumps(payload, indent=2) + "\n")
    target = task.parent / run_id
    if target.exists():
        # The previous coordinator may have published just before losing its lock.
        if json.loads((target / "_result.json").read_text()) != payload:
            raise ValueError(f"Existing result disagrees with recovered shards: {target}")
        shutil.rmtree(staging)
    else:
        staging.rename(target)


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, sort_keys=True) + "\n")
    temp.replace(path)


@contextmanager
def locked(path):
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def attempt_result(directory, assigned, task, pipeline):
    """Recover verified scores and positively identified non-scoring layouts."""
    incomplete = set()
    intact = True
    manifest = directory / "native-resume.json"
    if not manifest.exists():
        manifest = task.parent / f"_resume_{directory.name}.json"
    native = json.loads(manifest.read_text()) if manifest.exists() else {}
    try:
        rows, extra = read_result(directory, assigned, incomplete=incomplete)
    except json.JSONDecodeError:
        # A killed native JSON writer may leave a partial result file. The
        # client's atomic resume manifest is the last committed snapshot.
        rows, extra = {}, {}
        intact = False
    if pipeline is not None:
        if rows and extra.get("eval_pipeline") != pipeline:
            raise ValueError(f"Missing or mismatched pipeline in {directory}/_result.json")
        if native and native.get("eval_pipeline") != pipeline:
            raise ValueError(f"Missing or mismatched pipeline in {manifest}")
    saved = native.get("details") or {}
    if saved:
        recovered, _ = read_result(directory, assigned, {"details": saved, "eval_time": len(saved)}, incomplete)
        for layout, item in recovered.items():
            if layout in rows and item[0] != rows[layout][0]:
                raise ValueError(f"Conflicting native checkpoint for layout {layout}")
            rows.setdefault(layout, item)
    if pipeline is not None:
        extra["eval_pipeline"] = pipeline
    abandoned = set(native.get("abandoned_layout_ids", ())).intersection(assigned)
    log = directory / "client.log"
    # Some RoboDojo versions remove the final manifest. In non-vectorized
    # runs they report an instability count and explicit broken layout IDs.
    # Never infer rejection from a short result or a successful exit alone.
    status_file = directory / "exit.json"
    status = json.loads(status_file.read_text()) if status_file.exists() else None
    if log.exists():
        unstable = 0
        exhausted = fault = False
        with log.open(errors="replace") as stream:
            for line in stream:
                if "[PhysX] broken envs=" in line:
                    match = re.search(r"abandon seeds=\[([\d, ]*)\]", line)
                    if match:
                        abandoned.update(int(i) for i in re.findall(r"\d+", match[1]) if int(i) in assigned)
                if "Unstable nums:" in line:
                    match = re.search(r"Unstable nums: (\d+)", line)
                    if match:
                        unstable = int(match[1])
                exhausted |= "No more seeds to run, exiting." in line
                fault |= "[Eval] unhandled exception" in line
        missing = set(assigned) - rows.keys() - abandoned
        if intact and not incomplete and status == 0 and exhausted and not fault and unstable == len(missing):
            abandoned.update(missing)
    if abandoned & (rows.keys() | incomplete):
        raise ValueError(f"Layout marked both scored and abandoned in {directory}")
    return rows, extra, abandoned


def collect(job, task, work, pipeline):
    rows, metadata, abandoned = {}, {}, set()
    for attempt in job["attempts"]:
        directory = work / attempt["name"]
        if not directory.is_dir():
            continue
        records, extra, rejected = attempt_result(directory, attempt["layouts"], task, pipeline)
        if (rows.keys() | abandoned) & (records.keys() | rejected):
            raise ValueError(f"Duplicate layout outcome for {task.name}")
        rows.update(records)
        abandoned.update(rejected)
        if records:
            if metadata and metadata != extra:
                raise ValueError(f"Conflicting result metadata for {task.name}")
            metadata = extra
    return rows, metadata, abandoned


def add_job(plan, task, start, stop, costs):
    plan["jobs"].append(dict(task=task.name, start=start, stop=stop, attempts=[], done=False,
                             estimate=costs.get(task.name, task.target) * (stop-start) / task.target))


class Queue:
    def __init__(self, args, tasks, work, identity):
        self.args, self.work = args, work
        self.by_name = {t.name: t for t in tasks}
        self.costs = json.loads(Path(__file__).with_name("reference_task_seconds.json").read_text())
        self.pipelines = identity["pipelines"]
        self.path = work / "plan.json"
        self.lock = work / "plan.lock"
        with locked(self.lock):
            if self.path.exists():
                plan = self.load()
                if plan["identity"] != identity:
                    raise ValueError(f"Run configuration changed; use a new run ID: {work}")
                retry = os.getenv("LOGIC_PREVIEW_RETRY_ID")
                if retry and retry != plan.get("retry_id"):
                    for job in plan["jobs"]:
                        if job.pop("failed", False):
                            job["attempt_base"] = len(job["attempts"])
                            del plan["failed"][job["task"]]
                    plan["retry_id"] = retry
                    write_json(self.path, plan)
                return
            plan = dict(identity=identity, publish_id=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        + "_" + uuid4().hex[:8], jobs=[], published=[], failed={})
            for task in tasks:
                groups = math.ceil(task.target / args.chunk)
                size, extra = divmod(task.target, groups)
                start = 0
                for group in range(groups):
                    stop = start + size + (group < extra)
                    add_job(plan, task, start, stop, self.costs)
                    start = stop
            write_json(self.path, plan)

    def load(self):
        return json.loads(self.path.read_text())

    def settle_task(self, plan, task):
        jobs = [j for j in plan["jobs"] if j["task"] == task.name]
        if not all(j["done"] for j in jobs):
            return
        rows, metadata = {}, None
        for job in jobs:
            records, extra, _ = collect(job, task, self.work, self.pipelines.get(task.name))
            if rows.keys() & records.keys() or (records and metadata is not None and extra != metadata):
                raise ValueError(f"Shard overlap or inconsistent metadata for {task.name}")
            rows.update(records)
            if records:
                metadata = extra
        if len(rows) >= task.target:
            publish(task, rows, metadata, self.work, plan["publish_id"])
            plan["published"].append(task.name)
            print(f"Completed {task.name}: {task.target} episodes", flush=True)
        else:
            start = max(j["stop"] for j in jobs)
            stop = min(task.count, start + task.target - len(rows) + 2)
            if start == stop:
                plan["failed"][task.name] = f"exhausted layouts at {len(rows)}/{task.target}"
                return
            add_job(plan, task, start, stop, self.costs)
            print(f"{task.name}: adding unused layouts {start}..{stop-1}", flush=True)

    def claim(self, gpu):
        """The short plan lock serializes edits; the child inherits its shard lease."""
        with locked(self.lock):
            plan = self.load()
            if len(plan["published"]) + len(plan["failed"]) == len(self.by_name):
                if plan["failed"]:
                    raise RuntimeError(f"Unfinished tasks: {plan['failed']}")
                return True
            pending = sorted((j for j in plan["jobs"] if not j["done"] and j["task"] not in plan["failed"]),
                             key=lambda j: (-j["estimate"], j["task"], j["start"]))
            for job in pending:
                task = self.by_name[job["task"]]
                lease = (self.work / f"{task.name}-{job['start']}.lock").open("a")
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
                try:
                    rows, _, abandoned = collect(job, task, self.work, self.pipelines.get(task.name))
                    missing = sorted(set(range(job["start"], job["stop"])) - rows.keys() - abandoned)
                    if not missing:
                        job["done"] = True
                        self.settle_task(plan, task)
                        write_json(self.path, plan)
                        lease.close()
                        return False
                    attempts = job["attempts"]
                    if attempts:
                        exit_file = self.work / attempts[-1]["name"] / "exit.json"
                        code = json.loads(exit_file.read_text()) if exit_file.exists() else None
                        used = len(attempts) - job.get("attempt_base", 0)
                        if used and (code in (2, 78) or used >= self.args.max_attempts):
                            reason = f"layouts {missing} unresolved after {len(attempts)} attempts (exit {code}); see {exit_file.parent}"
                            job["failed"] = True
                            plan["failed"][task.name] = reason
                            print(f"Failed {task.name}: {reason}", flush=True)
                            write_json(self.path, plan)
                            lease.close()
                            return False
                    name = f"{plan['publish_id']}_{task.name}_{job['start']}_{len(attempts)+1}"
                    attempts.append(dict(name=name, layouts=missing, host=socket.gethostname(),
                                         gpu=gpu, policy_host=self.args.policy_host, port=self.args.port))
                    write_json(self.path, plan)
                    return task, missing, self.work / name, lease
                except BaseException:
                    lease.close()
                    raise
            return None


async def evaluate(args, queue):
    async def worker(gpu):
        while True:
            claim = queue.claim(gpu)
            if claim is True:
                return
            if claim is None:
                await asyncio.sleep(0.5)
                continue
            if claim is False:
                await asyncio.sleep(0)
                continue
            task, layouts, directory, lease = claim
            with lease:
                await run_client(args, task, gpu, directory, layouts, lease, queue.pipelines.get(task.name))

    workers = [asyncio.create_task(worker(gpu)) for gpu in args.gpus.split(",")]
    try:
        await asyncio.gather(*workers)
    finally:
        for worker_task in workers:
            worker_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


def run_identity(args, tasks):
    resolver = Path("src/eval_client/eval_pipeline.py")
    pipelines = {}
    if resolver.exists():
        resolve = runpy.run_path(str(resolver))["resolve_eval_pipeline"]
        for task in tasks:
            cfg = yaml.safe_load(Path(f"task/RoboDojo/config/{task.name}.yml").read_text()) or {}
            pipelines[task.name] = resolve(task.name, os.getenv("ROBODOJO_SIM_DEVICE", "auto"),
                                           bool(cfg.get("Fluid")), cfg.get("fluid_usd_sync", False))
    fingerprint = hashlib.sha256()
    policy = Path(__file__).resolve().parent
    sources = [policy / f for f in ("packed_eval.py", "packed_watchdog.py", "model.py", "launch_policy.py", "deploy.py",
                                    "deploy.yml", "setup_eval_policy_server.sh", "SHA256SUMS")]
    sources += sorted((policy / "model_code/src").rglob("*.py"))
    sources += sorted(Path("src/eval_client").glob("*.py"))
    sources += sorted(Path("task/RoboDojo/config").glob("*.yml"))
    for path in sources:
        fingerprint.update(path.read_bytes())
    return dict(seed=args.seed, checkpoint=args.ckpt, chunk=args.chunk, policy="Logic_preview",
                action_type="ee", robot="arx_x5", code=fingerprint.hexdigest(), pipelines=pipelines,
                policy_steps=int(os.getenv("LOGIC_PREVIEW_NUM_INFERENCE_STEPS", "10")),
                sigma_shift=float(os.getenv("LOGIC_PREVIEW_SIGMA_SHIFT", "1.0")),
                replan_steps=int(os.getenv("LOGIC_PREVIEW_REPLAN_STEPS", "24")),
                device_map=os.getenv("LOGIC_PREVIEW_INFERENCE_DEVICE_MAP", ""),
                tasks={t.name: [t.target, t.count] for t in tasks},
                max_attempts=args.max_attempts, timeout=args.timeout, idle_timeout=args.idle_timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--run-id", default=os.getenv("LOGIC_PREVIEW_RUN_ID") or uuid4().hex)
    parser.add_argument("--timeout", type=float, default=os.getenv("SHARD_TIMEOUT_SECONDS", 0))
    parser.add_argument("--idle-timeout", type=float, default=os.getenv("SHARD_IDLE_TIMEOUT_SECONDS", 0))
    parser.add_argument("--max-attempts", type=int, default=5)
    args = parser.parse_args()
    if any(not math.isfinite(v) or v < 0 for v in (args.timeout, args.idle_timeout)):
        parser.error("timeouts must be finite non-negative seconds")
    if args.chunk < 1 or args.max_attempts < 1 or not re.fullmatch(r"\d+(,\d+)*", args.gpus):
        parser.error("expected positive chunk and attempt limits and comma-separated simulator GPU ids")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", args.run_id):
        parser.error("invalid run ID")
    tasks = inventory(args.seed, args.ckpt)
    work = Path("eval_result/.logic-preview").resolve() / args.run_id / f"seed-{args.seed}"
    work.mkdir(parents=True, exist_ok=True)
    seed_order = os.getenv("LOGIC_PREVIEW_SEED_ORDER")
    if seed_order:
        with locked(work.parent / "rounds.lock"):
            rounds = work.parent / "rounds.json"
            if rounds.exists() and json.loads(rounds.read_text()) != seed_order.split():
                parser.error("The run ID already has a different seed order")
            write_json(rounds, seed_order.split())
    print(f"Run {args.run_id}, seed {args.seed}, shared recovery directory: {work}", flush=True)
    queue = Queue(args, tasks, work, run_identity(args, tasks))

    async def run():
        loop = asyncio.get_running_loop()
        main_task = asyncio.current_task()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, main_task.cancel)
        await evaluate(args, queue)

    asyncio.run(run())


if __name__ == "__main__":
    main()
