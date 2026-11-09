#!/usr/bin/env python3
"""Grow a harness library on development seeds, then freeze it (paper Sec. 3.3).

    python scripts/self_improve.py --experiment configs/experiments/dev.toml --deploy mine \
        --tasks stack_bowls,classify_objects --explore-seeds 0-3 --validate-seeds 10-13 \
        --library libraries/r4 --gpu 0 --output runs/self-improve

For each task: run the explore seeds with the current library; for each
explored episode ask the reflection model for one candidate entry; run the
validate seeds with and without the candidate; keep it only if the mean score
rises by at least --min-gain. The seeds are layouts of the experiment's kind:
development layouts (procedural) by default, or an official pack's when the
experiment says `layouts = "eval"` (then evaluate the frozen library on layouts it
never saw, such as another pack's); explore and validate seeds are disjoint from
each other. The final library version is written to OUTPUT/library.json.
Run inside the harness-policy environment.
"""

import argparse
from pathlib import Path
import json
import shutil
import sys
import tempfile
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from harness_policy.config.experiments import load_experiment  # noqa: E402
from harness_policy.runtime import suite as run_suite  # noqa: E402  (worker start/stop helpers)
from harness_policy.runtime.suite import BENCH_EMBODIMENT  # noqa: E402
from harness_policy.runtime.chat import ChatModel  # noqa: E402
from harness_policy.runtime.driver import attach_library, load_task_config, run_program_episode  # noqa: E402
from harness_policy.runtime.library import Library  # noqa: E402
from harness_policy.runtime.reflect import propose  # noqa: E402


def log(output, message):
    run_suite.log(output, message)


# RoboDojo refuses an official layout that does not settle after reset; its own evaluation
# loop takes the next layout instead (as harness run does with spares).
UNSTABLE = "unstable after reset"


def run_seeds(config, seeds, directory, spares=(), restart=None):
    """{layout: score} of one episode per seed. A layout that does not settle is set aside
    (seed-<n>.unstable) and the next of `spares` runs in its place; with no spare left it
    scores None (left out of comparisons). An episode the worker fails before its first
    step (its simulator hung) runs once more on a restarted worker (`restart`), else scores
    None; other errors score 0."""
    scores, spares, retried = {}, iter(spares), {}
    for seed in seeds:
        while seed is not None:
            run = directory / f"seed-{seed}"
            summary = run_program_episode(config, seed, run)
            if summary["status"] == "error" and UNSTABLE in str(summary.get("error")):
                run.rename(directory / f"seed-{seed}.unstable-{int(time.time())}")
                scores[seed] = None
                seed = next(spares, None)
                continue
            if summary["status"] == "error" and not summary.get("steps"):
                # The worker failed before the first step (a reset that timed out): not the
                # agent's score. Run the layout once more, else leave it out.
                run.rename(directory / f"seed-{seed}.failed-{int(time.time())}")
                if retried.get(seed):
                    scores[seed] = None
                    break
                retried[seed] = True
                if restart is not None:
                    restart()
                continue
            scores[seed] = 0.0 if summary["status"] == "error" else float(summary["reward"] or 0.0)
            break
    return scores


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def paired_gain(baseline, scores):
    """Mean score rise over the layouts both runs scored."""
    common = [seed for seed in baseline if baseline[seed] is not None and scores.get(seed) is not None]
    return mean([scores[s] for s in common]) - mean([baseline[s] for s in common]), common


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True,
                        help="experiment file (configs/experiments/*.toml): the agent, its model, budgets")
    parser.add_argument("--deploy", help="deploy name or path (default $HARNESS_DEPLOY)")
    parser.add_argument("--set", dest="sets", action="append", default=[], help="experiment override key=value")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--explore-seeds", required=True)
    parser.add_argument("--validate-seeds", required=True)
    parser.add_argument("--explore-spares", default="", help="layouts that replace unstable explore layouts")
    parser.add_argument("--validate-spares", default="", help="layouts that replace unstable validate layouts")
    parser.add_argument("--library", required=True, help="library directory (created if missing)")
    parser.add_argument("--min-gain", type=float, default=0.05, help="required rise of the mean score")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--port", type=int, default=8795)
    parser.add_argument("--dev-layout-dir", help="development layouts (default the RoboDojo checkout's)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    explore = run_suite.expand_seeds(args.explore_seeds)
    validate = run_suite.expand_seeds(args.validate_seeds)
    explore_spares = run_suite.expand_seeds(args.explore_spares) if args.explore_spares else []
    validate_spares = run_suite.expand_seeds(args.validate_spares) if args.validate_spares else []
    if set(explore + explore_spares) & set(validate + validate_spares):
        raise SystemExit("explore and validate seeds (and their spares) must be disjoint")
    output = Path(args.output).resolve()
    (output / "workers").mkdir(parents=True, exist_ok=True)
    library = Path(args.library).resolve()
    for section in ("notes", "memory", "skills", "reactive"):
        (library / section).mkdir(parents=True, exist_ok=True)
    experiment = load_experiment(args.experiment, deploy=args.deploy, sets=args.sets, tasks=args.tasks)
    worker_args = types.SimpleNamespace(benchmark="robodojo", layout_source=experiment.layouts,
                                        dev_layout_dir=args.dev_layout_dir,
                                        layout_pack_seed=experiment.layout_pack_seed, boot_timeout=600,
                                        record_video=False, ground_truth=False, shield=None)
    decisions = []
    for round_index in range(args.rounds):
        for task in run_suite.expand_tasks(args.tasks):
            base = load_task_config(args.experiment, task=task, seeds=explore, deploy=args.deploy, sets=args.sets)
            # The worker hands programs what the agent's contract allows (the leaderboard
            # protocol's observations for the default agent), as harness run does.
            worker_args.contract = base.runtime.get("contract", "native")
            worker_args.embodiment = BENCH_EMBODIMENT["robodojo"]
            log(output, f"round {round_index} {task}: starting worker")
            worker = {"process": run_suite.start_worker(worker_args, task, args.gpu, args.port, output)}

            def restart():
                log(output, f"{task}: worker failed an episode before its first step; restarting it")
                run_suite.stop_worker(worker["process"])
                worker["process"] = run_suite.start_worker(worker_args, task, args.gpu, args.port, output)
            try:
                base.environment = {**base.environment, "endpoint": f"http://127.0.0.1:{args.port}"}
                reflection = ChatModel({**base.main, "temperature": 0.3})
                stamp = f"r{round_index}-{task}-{int(time.time())}"
                current = attach_library(base, library)
                explored = output / stamp / "explore"
                ran = run_seeds(current, explore, explored, explore_spares, restart)
                baseline = None
                # Only episodes that ran are reflected on (an error before the first step
                # has nothing to learn from).
                for seed in [s for s, score in ran.items() if score is not None]:
                    run = explored / f"seed-{seed}"
                    if json.loads((run / "summary.json").read_text()).get("steps", 0) == 0:
                        continue
                    candidate, raw = propose(reflection, run, library)
                    record = {"round": round_index, "task": task, "from_seed": seed, "candidate": None,
                              "reply": raw[-2000:]}
                    if candidate is None:
                        decisions.append(record)
                        continue
                    section, name, text = candidate
                    record["candidate"] = f"{section}/{name}"
                    if (library / section / name).exists():
                        record["decision"] = "duplicate name"
                        decisions.append(record)
                        continue
                    if baseline is None:
                        baseline = run_seeds(attach_library(base, library), validate,
                                             output / stamp / f"validate-baseline-{Library(library).version()[:10]}",
                                             validate_spares, restart)
                    trial = Path(tempfile.mkdtemp(prefix="library-trial-", dir=output))
                    shutil.copytree(library, trial, dirs_exist_ok=True)
                    (trial / section / name).write_text(text)
                    # The candidate runs on the layouts the baseline settled on.
                    scores = run_seeds(attach_library(base, trial), [s for s, v in baseline.items() if v is not None],
                                       output / stamp / f"validate-{section}-{name}", restart=restart)
                    gain, common = paired_gain(baseline, scores)
                    record.update(baseline=baseline, scores=scores, layouts=common, gain=gain,
                                  decision="accepted" if gain >= args.min_gain else "rejected")
                    if gain >= args.min_gain:
                        (library / section / name).write_text(text)
                        baseline = None  # the library changed; re-measure before the next candidate
                    shutil.rmtree(trial, ignore_errors=True)
                    decisions.append(record)
                    log(output, f"{task} seed {seed}: {section}/{name} gain {gain:+.3f} -> {record['decision']}")
            finally:
                run_suite.stop_worker(worker["process"])
            (output / "decisions.json").write_text(json.dumps(decisions, indent=2))
    manifest = Library(library).manifest()
    (output / "library.json").write_text(json.dumps(manifest, indent=2))
    log(output, f"frozen library version {manifest['version'][:12]} with {sum(len(v) for v in manifest['entries'].values())} entries")


if __name__ == "__main__":
    main()
