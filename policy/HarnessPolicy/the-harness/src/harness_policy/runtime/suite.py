"""Run an experiment over many tasks and seeds: `harness run EXPERIMENT.toml`.

One RoboDojo (or RoboTwin) worker serves one task, so each slot (a GPU of the
deploy's [sim] x workers_per_gpu) starts a worker for its next task, runs that
task's seeds, then stops the worker. Finished episodes (summary.json present) are
skipped, so a rerun resumes. Results go to OUTPUT/<task>/seed-<n>/; summarize with
`harness report OUTPUT`. Run inside the harness-policy environment.
"""

import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from ..core.robot import RemoteEnvironment
from .chat import scripted_models
from .driver import run_program_episode
from .report import DIMENSIONS, MOVING_TARGETS

ROOT = Path(__file__).resolve().parents[3]
LOCK = threading.Lock()
# Model and verifier cost (USD) of the episodes this suite has run, all slots.
SPENT = {"usd": 0.0}


def log(output, message):
    """Print `message` with the time and append it to OUTPUT/suite.log (one slot at a time)."""
    line = f"{time.strftime('%H:%M:%S')} {message}"
    with LOCK:
        print(line, flush=True)
        with open(output / "suite.log", "a") as handle:
            handle.write(line + "\n")


def robotwin_tasks():
    """The 50 RoboTwin 2.0 tasks, from the official per-task step limits."""
    import yaml

    root = Path(os.environ.get("HARNESS_ROBOTWIN_ROOT", ROOT / "third_party" / "RoboTwin"))
    return sorted(yaml.safe_load((root / "env_cfg" / "task_config" / "_eval_step_limit.yml").read_text()))


def expand_tasks(spec, benchmark="robodojo"):
    """The task names of a comma-separated spec, in order and without repeats. On RoboDojo an
    item is a task, a dimension name, "moving" (the moving-target tasks) or "all"; on
    RoboTwin a task or "all" (its 50 tasks)."""
    if benchmark == "robotwin":
        return list(dict.fromkeys(t for item in spec.split(",") if item.strip()
                                  for t in (robotwin_tasks() if item.strip() == "all" else [item.strip()])))
    tasks = []
    for item in spec.split(","):
        item = item.strip()
        if item == "all":
            tasks += [t for group in DIMENSIONS.values() for t in group]
        elif item == "moving":
            tasks += MOVING_TARGETS
        elif item in DIMENSIONS:
            tasks += DIMENSIONS[item]
        elif item:
            tasks.append(item)
    return list(dict.fromkeys(tasks))


def expand_seeds(spec):
    """The seeds of a comma-separated spec of seeds and inclusive ranges ("0-4,7")."""
    seeds = []
    for item in spec.split(","):
        if "-" in item:
            low, high = item.split("-")
            seeds += list(range(int(low), int(high) + 1))
        elif item:
            seeds.append(int(item))
    return seeds


def _toml_inline(value):
    """A JSON value as a TOML inline value (tables, arrays, numbers, strings, booleans)."""
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{key} = {_toml_inline(item)}" for key, item in value.items()) + " }"
    if isinstance(value, list):
        return "[" + ", ".join(_toml_inline(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    return repr(float(value)) if isinstance(value, float) else str(value)


def worker_options(args, task, directory):
    """Write the worker's options for `task` to `<directory>/worker-<task>.toml` and return
    the path: RoboTwin's task config, split and seed group, or RoboDojo's video, ground
    truth, shield and layout settings."""
    if args.benchmark == "robotwin":
        # Evaluation seeds are the official ones (expert-checked from 100000 * (1 + group));
        # development seeds start at 10**7.
        lines = [f'task = "{task}"', f'task_config = "{args.robotwin_task_config}"',
                 f'split = "{"dev" if args.layout_source == "procedural" else "eval"}"',
                 f"seed_group = {args.layout_pack_seed}"]
        if args.robotwin_seed_dir:
            lines.append(f'seed_cache_dir = "{args.robotwin_seed_dir}"')
        path = directory / f"worker-{task}.toml"
        path.write_text("\n".join(lines) + "\n")
        return path
    lines = [f'task = "{task}"', f"record_video = {'true' if args.record_video else 'false'}"]
    if args.ground_truth:
        lines.append(f'ground_truth_log = "{directory.parent / "truth" / f"{task}.jsonl"}"')
    if args.shield:
        lines.append(f"shield = {_toml_inline(json.loads(args.shield))}")
    if args.layout_source == "procedural":
        lines.append('layout_source = "procedural"')
        if args.dev_layout_dir:
            lines.append(f'dev_layout_dir = "{args.dev_layout_dir}"')
    else:
        lines += [f"layout_pack_seed = {args.layout_pack_seed}"]
    path = directory / f"worker-{task}.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def start_worker(args, task, gpu, port, output):
    """Start a simulator worker for `task` on `gpu`, serving on `port` in its own process
    group (log: `workers/worker-<task>.log`), and return the process once it answers. Raises
    RuntimeError when it exits, or stops it and raises when it is not ready in time."""
    options = worker_options(args, task, output / "workers")
    # Isaac Lab logs under $TMPDIR/isaaclab; a shared /tmp may belong to another user.
    tmpdir = output / "tmp" / f"gpu{gpu}-port{port}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HARNESS_ROBODOJO_RUNTIME="1", CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               TMPDIR=str(tmpdir))
    factory = "benches.robodojo.runtime:create_environment"
    if args.benchmark == "robotwin":
        factory = "benches.robotwin.runtime:create_environment"
        # RoboTwin needs its own cuRobo v0.7.8 first (environment/robotwin_curobo.sh).
        curobo = os.environ.get("HARNESS_ROBOTWIN_CUROBO", str(ROOT / "third_party" / "curobo-0.7.8"))
        env["PYTHONPATH"] = os.pathsep.join([str(Path(curobo) / "src"), str(ROOT), str(ROOT / "src")]
                                            + [p for p in [os.environ.get("PYTHONPATH")] if p])
    log_file = open(output / "workers" / f"worker-{task}.log", "ab")
    process = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "scripts" / "harness"), "serve",
         "--factory", factory,
         "--options", str(options), "--port", str(port),
         *(["--contract", args.contract, "--embodiment", args.embodiment]
           if getattr(args, "contract", "native") != "native" else [])],
        cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.time() + args.boot_timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"worker for {task} exited with {process.returncode}; see workers/worker-{task}.log")
        try:
            RemoteEnvironment(f"http://127.0.0.1:{port}", timeout=10).tools()
            return process
        except Exception:
            time.sleep(5)
    stop_worker(process)
    raise RuntimeError(f"worker for {task} did not become ready in {args.boot_timeout} s")


def stop_worker(process):
    """Stop a worker's process group if it still runs: SIGINT, then SIGKILL after 30 s."""
    if process.poll() is not None:
        return
    # Isaac Kit ignores SIGTERM; SIGINT usually works, SIGKILL always does.
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def done(directory, rerun_errors):
    """An episode counts as done once it has a summary; with rerun_errors, one that
    ended in an error (provider outage, dead worker) is run again."""
    summary = directory / "summary.json"
    if not summary.exists():
        return False
    return not (rerun_errors and json.loads(summary.read_text()).get("status") == "error")


def replacements(task_dir, name="unstable"):
    """<task>/unstable.json: {unstable seed: the layout run in its place}, so a resumed suite runs
    the same replacement again instead of retrying the unstable layout. <task>/displaced.json:
    {requested seed: the layout run for it}, for a requested layout an earlier suite in this
    output had already run in place of an unstable one (that run keeps counting for its owner)."""
    path = task_dir / f"{name}.json"
    return {int(k): int(v) for k, v in json.loads(path.read_text()).items()} if path.exists() else {}


def write_replacements(task_dir, name, mapping):
    """Write <task>/<name>.json atomically: a kill mid-write must not leave broken JSON for the resume."""
    temporary = task_dir / f".{name}.json.tmp"
    temporary.write_text(json.dumps(mapping, indent=2))
    os.replace(temporary, task_dir / f"{name}.json")


def pack_size(args, task):
    """Number of layouts of `task` in the eval pack, or None when the checkout is not readable here."""
    try:
        from benches.robodojo.codec import ENV_CONFIG
        from benches.robodojo.tasks import layout_count, layout_dir

        checkout = os.environ.get("HARNESS_ROBODOJO_ROOT", str(ROOT / "third_party" / "RoboDojo"))
        return layout_count(layout_dir(checkout, ENV_CONFIG, args.layout_pack_seed), task) or None
    except Exception:
        return None


def run_slot(args, slot, gpu, port, tasks, output, run):
    """One slot's loop: take tasks from the queue until it is empty, and for each start a
    worker, run the seeds not done yet, and stop it. On RoboDojo eval layouts an unstable
    layout is replaced by the pack's next unused one; another worker error restarts the
    worker, and once the suite's cost cap is reached the remaining episodes are skipped."""
    while True:
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            return
        task_dir = output / task
        # As RoboDojo's own eval loop: an official layout that does not settle stably is skipped
        # (set aside as seed-<n>.unstable-<time>) and the next unused layout of the pack runs instead.
        spares = args.layout_source == "eval" and args.benchmark == "robodojo"
        replaced = replacements(task_dir) if spares else {}
        displaced = replacements(task_dir, "displaced") if spares else {}
        size = pack_size(args, task) if spares else None
        if spares and size is None:
            log(output, f"[slot {slot}] {task}: layout count unknown (RoboDojo checkout not readable here); "
                        "spares are bounded by --unstable-spares only")
        # A spare is never a layout this suite was asked for, one already used, or one past the pack.
        # It is the next one after all of those (RoboDojo takes the next layout that settles).
        taken = set(args.seed_list) | set(replaced) | set(replaced.values()) | set(displaced.values())

        def next_spare():
            spare = max(taken) + 1
            while spare in taken or (task_dir / f"seed-{spare}").exists():
                spare += 1
            return None if size is not None and spare >= size else spare

        def resolve(seed):
            seed = displaced.get(seed, seed)
            while seed in replaced:
                seed = replaced[seed]
            return seed

        # A requested layout that an earlier suite in this output ran in place of an unstable one
        # still answers for that seed; the requested seed gets a layout of its own, so no layout
        # is counted twice.
        owners = {new: old for old, new in [*replaced.items(), *displaced.items()]}
        for seed in args.seed_list:
            if spares and seed in owners and seed not in displaced:
                spare = next_spare()
                if spare is None:
                    log(output, f"[slot {slot}] {task} seed {seed}: already ran in place of seed {owners[seed]} "
                                "and no free layout is left in the pack")
                    continue
                displaced[seed] = spare
                taken.add(spare)
                write_replacements(task_dir, "displaced", displaced)
                log(output, f"[slot {slot}] {task} seed {seed}: already ran in place of seed {owners[seed]}; "
                            f"running seed {spare} for it")

        pending = list(dict.fromkeys(resolve(seed) for seed in args.seed_list))
        pending = [seed for seed in pending if not done(task_dir / f"seed-{seed}", args.rerun_errors)]
        if not pending:
            log(output, f"[slot {slot}] {task}: all seeds done")
            continue
        log(output, f"[slot {slot}] {task}: starting worker on GPU {gpu} port {port} for seeds {pending}")
        try:
            process = start_worker(args, task, gpu, port, output)
        except Exception as error:
            log(output, f"[slot {slot}] {task}: WORKER FAILED {error}")
            continue
        try:
            config = experiment_for(run, task, pending)
            config.environment = {**config.environment, "endpoint": f"http://127.0.0.1:{port}"}
            if args.max_model_calls:
                config.max_model_calls = args.max_model_calls
            if args.max_cost_usd:
                config.max_cost_usd = args.max_cost_usd
            spares_left = args.unstable_spares - len(replaced) if spares else 0
            retried = set()
            while pending:
                seed = pending.pop(0)
                if args.max_total_cost_usd and SPENT["usd"] >= args.max_total_cost_usd:
                    log(output, f"[slot {slot}] {task} seed {seed}: skipped, suite spent "
                                f"${SPENT['usd']:.2f} of ${args.max_total_cost_usd:.2f}")
                    continue
                directory = output / task / f"seed-{seed}"
                if directory.exists():
                    # An interrupted or failed episode is rerun from scratch; the old run is kept.
                    kind = "error" if (directory / "summary.json").exists() else "incomplete"
                    directory.rename(directory.with_name(f"seed-{seed}.{kind}-{int(time.time())}"))
                scripted = {} if args.script_replies is None else scripted_models(args.script_replies)
                summary = run_program_episode(config, seed, directory, **scripted)
                with LOCK:
                    SPENT["usd"] += summary["cost_usd"]
                log(output, f"[slot {slot}] {task} seed {seed}: {summary['status']} success={summary['success']} "
                            f"reward={summary['reward']} calls={summary['model_calls']} "
                            f"cost=${summary['cost_usd']:.3f} wall={summary['wall_seconds']:.0f}s")
                # RoboDojo drops a layout that does not settle at reset, and an episode it marks
                # unstable mid-way (scripted support arms); both are replaced by the next layout.
                unstable = summary["status"] == "error" and bool(
                    re.search(r"unstable after (reset|step)", str(summary.get("error", ""))))
                spare = next_spare() if unstable and spares_left > 0 else None
                if unstable and spare is None and spares:
                    log(output, f"[slot {slot}] {task} seed {seed}: unstable layout, no spare left; kept as an error")
                if spare is not None:
                    directory.rename(directory.with_name(f"seed-{seed}.unstable-{int(time.time())}"))
                    replaced[seed] = spare
                    taken.add(spare)
                    write_replacements(task_dir, "unstable", replaced)
                    log(output, f"[slot {slot}] {task} seed {seed}: unstable layout, skipped; running seed {spare} instead")
                    pending.append(spare)
                    spares_left -= 1
                    # The worker is still usable: its next reset loads the replacement layout into
                    # the same simulator (RoboDojo itself relaunches every round). In the R4 2-case
                    # run 8 replacements after unstable layouts ran full episodes this way.
                elif summary["status"] == "error" and not unstable and (
                        "Worker" in str(summary.get("error", "")) or not summary.get("steps")):
                    # An unstable layout leaves the worker usable; any other worker error restarts it,
                    # and so does an episode that failed before its first step (a reset that timed
                    # out: the simulator hung, and every later episode on it timed out too). That
                    # layout runs once more on the new worker.
                    log(output, f"[slot {slot}] {task}: worker error, restarting worker")
                    stop_worker(process)
                    process = start_worker(args, task, gpu, port, output)
                    if not summary.get("steps") and seed not in retried:
                        retried.add(seed)
                        pending.insert(0, seed)
        except Exception as error:
            log(output, f"[slot {slot}] {task}: FAILED {type(error).__name__}: {error}")
        finally:
            stop_worker(process)


def experiment_for(run, task, seeds):
    """The driver's configuration of one task of the experiment."""
    return run.task_config(task, seeds)


# The robot of each benchmark (embodiments/<name>), for observation contracts.
BENCH_EMBODIMENT = {"robodojo": "arx_x5"}


def suite_arguments(run, *, gpus=None, workers_per_gpu=None, base_port=None, rerun_errors=False, script=None,
                    max_total_cost_usd=None):
    """The suite's settings: the run's, the deploy's [sim] section, then the command line."""
    sim = dict(getattr(run.deploy, "sim", {}) or {})
    suite = run.suite
    worker = run.worker
    contract = run.agent.runtime.get("contract", "native")
    if contract != "native" and BENCH_EMBODIMENT.get(run.bench) is None:
        raise ValueError(f"The agent's contract {contract!r} needs an embodiment profile, and bench {run.bench} has "
                         "none: set agent.runtime.contract = \"native\" for it")
    gpu_list = gpus if gpus is not None else sim.get("gpus")
    if not gpu_list:
        raise ValueError("No GPUs for simulator workers: give --gpus or a deploy file with [sim] gpus")
    return SimpleNamespace(
        benchmark=run.bench, seed_list=list(run.seeds), layout_source=run.layouts,
        layout_pack_seed=run.layout_pack_seed,
        # Without one, the worker uses its checkout's (RoboDojo harness_dev_layouts, RoboTwin harness_seeds).
        dev_layout_dir=run.tree.get("dev_layout_dir"),
        gpus=",".join(str(g) for g in gpu_list),
        workers_per_gpu=int(workers_per_gpu or sim.get("workers_per_gpu", 1)),
        base_port=int(base_port or sim.get("base_port", 8780)),
        boot_timeout=float(suite.get("boot_timeout", 600)),
        max_model_calls=run.budget.get("max_model_calls"), rerun_errors=rerun_errors,
        max_cost_usd=run.budget.get("max_cost_usd"),
        max_total_cost_usd=max_total_cost_usd or run.budget.get("max_total_cost_usd"),
        unstable_spares=int(suite.get("unstable_spares", 5)),
        record_video=bool(worker.get("record_video", False)), shield=json.dumps(worker["shield"]) if worker.get("shield") else None,
        ground_truth=bool(worker.get("ground_truth", False)),
        robotwin_task_config=worker.get("task_config", "demo_randomized"),
        robotwin_seed_dir=worker.get("seed_cache_dir"),
        contract=contract, embodiment=BENCH_EMBODIMENT.get(run.bench),
        script_replies=script)


def preflight(run, log=print):
    """Check that the services the experiment's agent uses are up before any simulator boots:
    resident ones must answer (harness up), on-demand ones need the deploy's broker
    (started here when it does not run)."""
    from ..services import deploy as deploys
    from ..services.supervisor import listening, start_broker

    deploy = run.deploy
    names = []
    for value in run.agent.placement.values():
        if isinstance(value, dict) and "service" in value and not value.get("dropped"):
            names.append(value["service"])
        elif isinstance(value, dict) and "service" not in value:
            names += [name for name, entry in value.items() if not entry.get("dropped")]
    down = []
    for name in sorted(set(names)):
        placement = deploy.placement(name)
        if placement.mode == "resident" and not listening(deploys.port(deploy, name)):
            down.append(name)
        if placement.mode == "external":
            address = deploys.url(deploy, name).split("://", 1)[-1].split("/", 1)[0]
            host, _, port = address.rpartition(":")
            if port.isdigit() and not listening(int(port), host):
                down.append(name)
    if down:
        raise RuntimeError(f"services {down} do not answer; start them with: harness up {' '.join(down)} "
                           f"--deploy {deploy.name}")
    if any(deploy.placement(name).mode == "on_demand" for name in names):
        start_broker(deploy, log=log)


def run_suite(run, output, **options):
    """Run every task and seed of `run` into `output` (see suite_arguments for options)."""
    preflight(run, log=lambda message: print(message, file=sys.stderr, flush=True))
    args = suite_arguments(run, **options)
    output = Path(output).resolve()
    (output / "workers").mkdir(parents=True, exist_ok=True)
    (output / "truth").mkdir(exist_ok=True)
    tasks = list(run.tasks)
    slots = [(gpu, index) for gpu in args.gpus.split(",") for index in range(args.workers_per_gpu)]
    # Several processes may share one output (one per GPU): merge their tasks and seeds.
    import fcntl

    with open(output / "suite.json.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = json.loads((output / "suite.json").read_text()) if (output / "suite.json").exists() else {}
        (output / "suite.json").write_text(json.dumps({
            "run": str(run.path) if run.path else None, "agent": run.agent.name, "agent_id": run.agent.identity,
            "agent_version": run.agent.lock.get("agent_version"),
            "deploy": getattr(run.deploy, "name", None),
            "tasks": list(dict.fromkeys([*previous.get("tasks", []), *tasks])),
            "seeds": sorted({*previous.get("seeds", []), *args.seed_list}),
            "layout_source": args.layout_source, "layout_pack_seed": args.layout_pack_seed,
            "started": previous.get("started") or time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    work = queue.Queue()
    for task in tasks:
        work.put(task)
    log(output, f"suite: {len(tasks)} tasks x {len(args.seed_list)} seeds on {len(slots)} slots "
                f"(agent {run.agent.name} {run.agent.identity})")
    threads = []
    for number, (gpu, _index) in enumerate(slots):
        thread = threading.Thread(target=run_slot, daemon=True, args=(
            args, number, gpu, args.base_port + number, work, output, run))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    log(output, f"suite: done, model cost ${SPENT['usd']:.2f}")
    return {"output": str(output), "tasks": tasks, "seeds": args.seed_list, "cost_usd": SPENT["usd"]}
