"""Fresh config-derived layouts, with explicit split and complete batch receipts.

Geometric generation is not physics validation. Neither generation nor its
input guard may read benchmark layouts or historical trajectories. The guard
protects this trusted Python sampler; it is not an arbitrary-code sandbox.
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

from PhysicalRSI_core.infra.batch import ProcessJob, run_batch
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.self_harness import now

TASKS = tuple(read_json(Path(__file__).with_name("configs") / "tasks.json")["tasks"])
FORBIDDEN_INPUTS = {"Eval_Layout", "Train_Layout", "Traj", "eval_result"}
ELIGIBILITY = {
    "pick_from_conveyor_by_image": "photo/basket fixture placement and conveyor scene require native validation",
    "imitate_sorting_sequence": "requires a fresh support-arm demonstration bound to this layout",
    "match_and_pick_from_conveyor": "conveyor scene distribution requires native validation",
    "cover_blocks": "finite canonical layouts do not establish unseen semantic coverage",
}


def implementation():
    root = Path(__file__).parent
    paths = [Path(__file__), *root.joinpath("layout_sampling").glob("*.py")]
    return {str(path.relative_to(root)): file_digest(path) for path in paths}


def _input_guard(roots, output):
    """Install once in a disposable generation process, after importing libraries."""
    roots = tuple(Path(root).resolve() for root in roots)
    output = Path(output).resolve()
    accessed = set()

    def guard(event, arguments):
        if event not in {"open", "os.listdir", "os.scandir"}:
            return
        name = arguments[0]
        if isinstance(name, int):
            name = os.readlink(f"/proc/self/fd/{name}")
        if not isinstance(name, (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(name)).resolve()
        if path.is_relative_to(output):
            return
        if FORBIDDEN_INPUTS.intersection(path.parts):
            raise PermissionError(
                "Benchmark layouts and historical trajectories are forbidden inputs"
            )
        if not any(path.is_relative_to(root) for root in roots):
            raise PermissionError(
                "Sampler read outside public configuration/object inputs: " + str(path)
            )
        if event == "open":
            mode, flags = arguments[1:3]
            if (isinstance(mode, str) and any(c in mode for c in "wax+")) or flags & (
                os.O_WRONLY | os.O_RDWR
            ):
                raise PermissionError("Sampler input trees are read-only")
            accessed.add(str(path))

    sys.addaudithook(guard)
    return accessed


def _worker(request_path):
    # Load numerical libraries before restricting filesystem reads. The worker
    # executes only our sampler, never model-produced code or user plugins.
    import numpy as np

    from .layout_sampling import extended
    from .layout_sampling.catalog import generate_train_layout

    np.random.default_rng(0).random()
    request = read_json(request_path)
    if request["implementation"] != implementation():
        raise ValueError("Layout sampler changed after batch creation")
    if request["task"] in extended.SUPPORTED_TASKS:
        generate_train_layout = extended.generate_train_layout
    source, assets, output = (
        Path(request[k]).resolve() for k in ("source", "assets", "output")
    )
    accesses = _input_guard(
        [
            source / "task/RoboDojo/config",
            source / "env_cfg/scene",
            assets / "Object",
            Path(__file__).with_name("layout_sampling"),
        ],
        output,
    )
    result = dict(
        task=request["task"], index=request["index"], state="generation_failed"
    )
    try:
        layout = generate_train_layout(
            source_root=source,
            assets_root=assets,
            task=request["task"],
            generation_seed=request["seed"],
            layout_id=request["index"],
        )
        provenance = layout.pop("_train_layout")
        path = output / "layout.json"
        atomic_json(path, layout)
        result.update(
            state="generated",
            sha256=file_digest(path),
            content_sha256=digest(layout),
            provenance=provenance,
            eligibility_blocker=ELIGIBILITY.get(request["task"]),
        )
    except Exception as error:
        result.update(error=type(error).__name__ + ": " + str(error))
    result["inputs"] = {name: file_digest(Path(name)) for name in sorted(accesses)}
    atomic_json(output / "receipt.json", result)


def generate_batch(
    source,
    assets,
    output,
    *,
    tasks=None,
    count=1,
    seed=0,
    split="development",
    comparison_sha256=None,
    timeout_s=180,
    workers=1,
):
    tasks = list(TASKS if tasks is None else tasks)
    if not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= set(TASKS):
        raise ValueError("Expected unique, declared RoboDojo tasks")
    if type(count) is not int or count < 1 or type(seed) is not int or seed < 0:
        raise ValueError("count must be positive and seed nonnegative integers")
    if type(workers) is not int or not 1 <= workers <= 10:
        raise ValueError("workers must be between one and ten")
    if split not in {"development", "validation"}:
        raise ValueError("Generation cannot consume or write a benchmark test split")
    if split == "validation" and (
        not isinstance(comparison_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", comparison_sha256)
    ):
        raise ValueError("Validation requires the frozen comparison identity")
    if split == "development" and comparison_sha256 is not None:
        raise ValueError("Development must not receive validation comparison state")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Worker timeout must be finite and positive")
    source, assets, output = (Path(p).resolve() for p in (source, assets, output))
    if any(FORBIDDEN_INPUTS.intersection(p.parts) for p in (source, assets, output)):
        raise ValueError(
            "Fresh layouts require independent source, assets and output paths"
        )
    output.mkdir(parents=True, exist_ok=False)
    # Distinct streams even when a user repeats a seed across split labels.
    stream_seed = int(
        digest(dict(seed=seed, split=split, comparison=comparison_sha256))[:16], 16
    )
    manifest = dict(
        schema_version=1,
        state="running",
        split=split,
        seed=seed,
        tasks=tasks,
        count_per_task=count,
        comparison_sha256=comparison_sha256,
        physics_status="not_validated",
        distribution_status="public_config_and_task_semantics_not_qualified",
        implementation=implementation(),
        generated_at=now(),
        entries=[],
    )
    seen = set()
    jobs = []
    requests = []
    for task in tasks:
        for index in range(count):
            folder = output / task / str(index)
            folder.mkdir(parents=True)
            request = dict(
                source=str(source),
                assets=str(assets),
                output=str(folder),
                task=task,
                index=index,
                seed=stream_seed,
                implementation=manifest["implementation"],
            )
            request_path = folder / "request.json"
            atomic_json(request_path, request)
            job_id = f"generator-{len(requests):05d}"
            jobs.append(
                ProcessJob(
                    job_id,
                    (
                        sys.executable,
                        "-m",
                        "PhysicalRSI_baselines.robodojo.layouts",
                        "--worker",
                        str(request_path),
                    ),
                    Path.cwd(),
                    timeout_s=timeout_s,
                )
            )
            requests.append((job_id, task, index, folder))
    processes = run_batch(
        jobs,
        output / "processes",
        workers=min(workers, len(jobs)),
    )
    process_by_id = {row["id"]: row for row in processes}
    for job_id, task, index, folder in requests:
        try:
            process = process_by_id[job_id]
            if process["state"] != "completed":
                raise RuntimeError(
                    f"Layout worker {process['state']}; see processes/{job_id}/process.log"
                )
            row = read_json(folder / "receipt.json")
            if implementation() != manifest["implementation"]:
                raise ValueError("Sampler changed while generation was running")
            if row["state"] == "generated":
                identity = (task, row["content_sha256"])
                if identity in seen:
                    row.update(
                        state="duplicate",
                        error="Identical geometry within task batch",
                    )
                seen.add(identity)
                row["file"] = str((folder / "layout.json").relative_to(output))
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            row = dict(task=task, index=index, state="generation_failed", error=str(error))
        manifest["entries"].append(row)
        atomic_json(output / "manifest.json", manifest)
    manifest["state"] = (
        "generated"
        if all(r["state"] == "generated" for r in manifest["entries"])
        else "incomplete"
    )
    atomic_json(output / "manifest.json", manifest)
    return manifest


def verify_batch(path, *, split, comparison_sha256=None):
    """Evaluator-side reader: reject missing, tampered, duplicate or wrong-split inputs."""
    path = Path(path).resolve()
    manifest = read_json(path)
    if (
        manifest["schema_version"] != 1
        or manifest["split"] != split
        or manifest["state"] != "generated"
    ):
        raise ValueError("Wrong split or incomplete layout batch")
    if manifest["comparison_sha256"] != comparison_sha256:
        raise ValueError("Layout cohort belongs to a different comparison")
    if (
        type(manifest["count_per_task"]) is not int
        or manifest["count_per_task"] < 1
        or not manifest["tasks"]
        or len(set(manifest["tasks"])) != len(manifest["tasks"])
        or not set(manifest["tasks"]) <= set(TASKS)
    ):
        raise ValueError("Invalid task/count coverage")
    expected = {
        (task, i)
        for task in manifest["tasks"]
        for i in range(manifest["count_per_task"])
    }
    actual = [(row["task"], row["index"]) for row in manifest["entries"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Layout coverage differs")
    seen = set()
    for row in manifest["entries"]:
        layout = (path.parent / row["file"]).resolve()
        if (
            not layout.is_relative_to(path.parent)
            or row["state"] != "generated"
            or file_digest(layout) != row["sha256"]
        ):
            raise ValueError("Changed or invalid layout")
        content = digest(read_json(layout))
        key = (row["task"], content)
        if content != row["content_sha256"] or key in seen:
            raise ValueError("Changed or duplicate layout content")
        seen.add(key)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=TASKS)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split", choices=["development", "validation"], default="development"
    )
    parser.add_argument("--comparison-sha256")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.worker:
        _worker(args.worker)
        return 0
    if not all((args.source, args.assets, args.output)):
        parser.error("--source, --assets and --output are required")
    result = generate_batch(
        args.source,
        args.assets,
        args.output,
        tasks=args.tasks,
        count=args.count,
        seed=args.seed,
        split=args.split,
        comparison_sha256=args.comparison_sha256,
        workers=args.workers,
    )
    print(
        json.dumps(
            dict(
                state=result["state"],
                generated=sum(r["state"] == "generated" for r in result["entries"]),
                requested=len(result["entries"]),
                workers=args.workers,
            )
        )
    )
    return 0 if result["state"] == "generated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
