"""Bounded independent process jobs with durable outcomes and owned cleanup."""

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

from .processes import ManagedProcess
from .storage import atomic_json, identifier


@dataclass(frozen=True)
class ProcessJob:
    id: str
    command: tuple[str, ...]
    cwd: Path
    environment: dict = field(default_factory=dict)
    timeout_s: float = 3600


def run_batch(jobs, output, *, workers, cancelled=None):
    """Commands own their services; no shell or reused output paths.

    A process exit is infrastructure evidence, not a task success judgment.
    Callers must validate native rollout evidence separately.
    """
    jobs = list(jobs)
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive")
    if not jobs or len({job.id for job in jobs}) != len(jobs):
        raise ValueError("Nonempty jobs with unique identities required")
    for job in jobs:
        identifier(job.id)
        if (
            not job.command
            or not math.isfinite(job.timeout_s)
            or job.timeout_s <= 0
            or not job.cwd.is_dir()
        ):
            raise ValueError("Job requires a command, cwd and positive timeout")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cancel = cancelled or Event()

    def run(job):
        folder = output / job.id
        folder.mkdir()
        record = dict(id=job.id, state="queued", returncode=None)
        receipt = folder / "process.json"
        atomic_json(receipt, record)
        if cancel.is_set():
            record["state"] = "cancelled"
            atomic_json(receipt, record)
            return record
        process = ManagedProcess(
            job.id,
            job.command,
            cwd=job.cwd,
            log_path=folder / "process.log",
            env_overrides=dict(job.environment, PHYSICALRSI_JOB_DIR=str(folder)),
        )
        try:
            process.start()
            record.update(state="running", pid=process.pid)
            atomic_json(receipt, record)
            deadline = time.monotonic() + job.timeout_s
            while process.poll() is None:
                if cancel.is_set():
                    record["state"] = "cancelled"
                    break
                if time.monotonic() >= deadline:
                    record["state"] = "timed_out"
                    break
                cancel.wait(0.02)
            else:
                record.update(
                    state="completed" if process.poll() == 0 else "failed",
                    returncode=process.poll(),
                )
        except Exception as error:
            record.update(
                state="failed", error=type(error).__name__ + ": " + str(error)
            )
        finally:
            try:
                process.stop()
            except Exception as error:
                record.update(state="cleanup_failed", error=str(error))
                cancel.set()
            atomic_json(receipt, record)
        return record

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        results = list(pool.map(run, jobs))
    except BaseException:
        cancel.set()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    atomic_json(output / "batch.json", dict(workers=workers, results=results))
    return results
