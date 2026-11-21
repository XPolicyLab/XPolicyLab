"""RoboTwin 2.0 evaluation client wrapper for HumanWAM-Goal.

Supports two seed modes over RoboTwin's ``scripts/eval_policy_xpolicylab.py``:
- ``fixed`` (default): evaluates the 100 cached ``(seed, instruction)`` cases per task
  from ``assets/goal_cache`` without re-running the online expert check.
- ``native``: runs ``eval_policy_xpolicylab.py`` with ``expert_check=True`` from seed 100000.
"""

import hashlib
import json
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType


PROTOCOL = "goalwam.official_native.v1"
SEED_MODES = ("fixed", "native")
CASES_PER_TASK = 100
REQUIRED_HOOKS = (
    "main",
    "build_policy_client",
    "eval_remote_policy",
    "build_instruction",
    "class_decorator",
    "build_eval_save_dir",
    "normalize_action_chunk",
)


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_sha256(value):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a lowercase SHA-256 digest")
    return value


class EvaluationAborted(SystemExit):
    """Uncaught exit so upstream ``except Exception`` blocks do not skip a failed case."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(2)


class GoalCacheClient:
    """Attach ``task_config`` and goal-cache metadata to RoboTwin websocket RPCs."""

    _robotwin_protocol = "ws"

    def __init__(self, client, *, task_config, task_name, cache_manifest_sha256, record=None, validate_action=None):
        self.client = client
        self.task_config = task_config
        self.task_name = task_name
        self.cache_manifest_sha256 = require_sha256(cache_manifest_sha256)
        self.record = record or (lambda event: None)
        self.validate_action = validate_action
        self.case = None
        self.initial_verified = False
        self.failed = False

    def abort(self, reason):
        self.failed = True
        self.initial_verified = False
        try:
            self.record(dict(event="aborted", reason=reason, case=self.case))
        except Exception:
            pass
        print(f"[HumanWAM-Goal] evaluation aborted: {reason}", file=sys.stderr, flush=True)
        raise EvaluationAborted(reason)

    def check_ack(self, reply, operation):
        expected = dict(
            protocol=PROTOCOL,
            operation=operation,
            status="prepared" if operation == "prepare_case" else "ready",
            cache_manifest_sha256=self.cache_manifest_sha256,
            case_id=self.case["case_id"],
            instruction_sha256=text_sha256(self.case["instruction"]),
        )
        if not isinstance(reply, dict) or any(reply.get(k) != v for k, v in expected.items()):
            self.abort(f"Unexpected {operation} response from policy server")
        if reply.get("initial_verified") is not (operation == "update_obs"):
            self.abort(f"Unexpected {operation} initial_verified flag")

    def call(self, *, func_name, **kwargs):
        if self.failed:
            self.abort("Policy session already terminated")
        if func_name not in {"prepare_case", "reset", "update_obs", "get_action", "trial_end"}:
            self.abort(f"Unsupported RPC: {func_name}")
        try:
            if func_name == "prepare_case":
                self.case = None
                self.initial_verified = False
                metadata = dict(kwargs["obs"])
                if metadata.get("task_name") != self.task_name or metadata.get("action_type") != "joint":
                    self.abort("Task or action_type does not match client configuration")
                if "task_config" in metadata and metadata["task_config"] != self.task_config:
                    self.abort("task_config does not match client configuration")
                seed, instruction = metadata["seed"], metadata["instruction"]
                if type(seed) is not int or seed < 100000 or not isinstance(instruction, str) or not instruction.strip():
                    self.abort("Invalid episode seed or instruction")
                self.case = dict(
                    task_config=self.task_config,
                    task_name=self.task_name,
                    seed=seed,
                    instruction=instruction,
                    case_id=f"{self.task_config}/{self.task_name}/{seed}",
                )
                metadata.update(
                    task_config=self.task_config,
                    protocol=PROTOCOL,
                    expected_cache_manifest_sha256=self.cache_manifest_sha256,
                )
                kwargs = dict(kwargs, obs=metadata)
            elif self.case is None:
                self.abort(f"{func_name} called before prepare_case")
            if func_name == "reset":
                self.initial_verified = False
            if func_name == "update_obs" and kwargs["obs"].get("instruction") != self.case["instruction"]:
                self.abort("Observation instruction differs from prepared case")
            if func_name == "get_action" and not self.initial_verified:
                self.abort("get_action called before step-0 observation verification")
            reply = self.client.call(func_name=func_name, **kwargs)
            if func_name in ("prepare_case", "update_obs"):
                self.check_ack(reply, func_name)
                if func_name == "update_obs":
                    self.initial_verified = True
            elif func_name == "get_action" and self.validate_action is not None:
                if not self.validate_action(reply):
                    self.abort("Policy returned an empty action chunk")
            elif func_name == "trial_end":
                self.record(dict(event="trial_end", case=self.case, result=kwargs.get("obs")))
                self.case = None
                self.initial_verified = False
            return reply
        except EvaluationAborted:
            raise
        except Exception as error:
            self.abort(f"{func_name} failed: {type(error).__name__}: {error}")

    def close(self):
        try:
            return self.client.close()
        except Exception as error:
            if not self.failed:
                self.abort(f"Policy connection close failed: {type(error).__name__}: {error}")


def evaluator_path(root):
    path = Path(root).resolve() / "scripts/eval_policy_xpolicylab.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing RoboTwin evaluator: {path}")
    return path


def load_evaluator(root):
    source = evaluator_path(root)
    module = ModuleType("_robotwin_eval_xpolicylab")
    module.__file__ = str(source)
    exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    missing = [name for name in REQUIRED_HOOKS if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(f"Incompatible eval_policy_xpolicylab.py; missing callables: {missing}")
    return module


def parse_case_range(text):
    start, sep, stop = str(text).partition(":")
    if not sep or not start.isdigit() or not stop.isdigit() or not 0 <= int(start) < int(stop) <= CASES_PER_TASK:
        raise ValueError(f"HWGOAL_FIXED_CASES must be start:stop with 0 <= start < stop <= {CASES_PER_TASK}")
    return int(start), int(stop)


def fixed_cases(cache_root, cache_manifest_sha256, task_config, task_name, case_range=None):
    raw = (Path(cache_root) / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != require_sha256(cache_manifest_sha256):
        raise ValueError("Goal cache manifest SHA-256 mismatch")
    rows = sorted(
        (r for r in json.loads(raw)["cases"] if r["task_config"] == task_config and r["task"] == task_name),
        key=lambda r: r["seed"],
    )
    seeds = [r["seed"] for r in rows]
    if len(rows) != CASES_PER_TASK or len(set(seeds)) != len(seeds):
        raise ValueError(f"Expected {CASES_PER_TASK} cached seeds for {task_config}/{task_name}")
    cases = []
    for row in rows:
        seed, instruction = row["seed"], row["official_instruction"]
        if type(seed) is not int or seed < 100000 or text_sha256(instruction) != row["official_instruction_sha256"]:
            raise ValueError(f"Invalid goal cache entry: {task_config}/{task_name}/{seed}")
        cases.append(dict(seed=seed, instruction=instruction, case_id=f"{task_config}/{task_name}/{seed}"))
    if case_range is not None:
        start, stop = parse_case_range(case_range)
        cases = cases[start:stop]
    return cases


def build_arguments(config, root, seed_mode="fixed"):
    root_path = Path(root).resolve()
    task_name = str(config.get("task_name") or "").strip()
    task_config = str(config.get("task_config") or "").strip()
    if not task_name or not (root_path / "envs" / f"{task_name}.py").is_file():
        raise ValueError(f"Unknown RoboTwin task: {task_name!r}")
    if task_config not in {"demo_clean", "demo_randomized"}:
        raise ValueError(f"Unsupported task_config: {task_config!r}")
    if seed_mode not in SEED_MODES:
        raise ValueError(f"seed_mode must be one of {SEED_MODES}")
    args = dict(config)
    args.update(
        root_dir=str(root_path),
        bench_name="RoboTwin",
        env_cfg_type="arx_x5",
        protocol="ws",
        seed=0,
        test_num=1 if seed_mode == "fixed" else CASES_PER_TASK,
        expert_check=seed_mode == "native",
        action_type=config.get("action_type", "joint"),
        eval_batch=False,
        frequency=30,
        instruction_type="seen" if task_config == "demo_clean" else "unseen",
    )
    args.setdefault("xpolicylab_root", str(root_path / "XPolicyLab"))
    args.setdefault("device_id", "0")
    return args


@contextmanager
def patched_client_factory(native, cache_manifest_sha256, record=None):
    original = native.build_policy_client

    def build(args):
        try:
            client = original(args)
            return GoalCacheClient(
                client,
                task_config=args["task_config"],
                task_name=args["task_name"],
                cache_manifest_sha256=cache_manifest_sha256,
                record=record,
                validate_action=native.normalize_action_chunk,
            )
        except Exception as error:
            raise EvaluationAborted(f"Policy connection failed: {type(error).__name__}: {error}") from error

    native.build_policy_client = build
    try:
        yield
    finally:
        native.build_policy_client = original


@contextmanager
def pinned_case(native, case, save_dir):
    names = ("eval_remote_policy", "build_instruction", "class_decorator", "build_eval_save_dir")
    originals = {name: getattr(native, name) for name in names}
    outcome = {}

    def evaluate(task_name, task_env, args, model_client, unused_start_seed, **kwargs):
        result = originals["eval_remote_policy"](task_name, task_env, args, model_client, case["seed"], **kwargs)
        outcome["successes"] = int(result[1])
        return result

    def make_env(task_name):
        task_env = originals["class_decorator"](task_name)
        setup = task_env.setup_demo

        def setup_case(*args, **kwargs):
            if kwargs.get("seed") != case["seed"]:
                raise EvaluationAborted(
                    f"Case {case['case_id']} failed setup and attempted fallback seed {kwargs.get('seed')}"
                )
            return setup(*args, **kwargs)

        task_env.setup_demo = setup_case
        return task_env

    native.eval_remote_policy = evaluate
    native.build_instruction = lambda *unused, **unused_kwargs: case["instruction"]
    native.class_decorator = make_env
    native.build_eval_save_dir = lambda *unused: save_dir
    try:
        yield outcome
    finally:
        for name, value in originals.items():
            setattr(native, name, value)


def run_fixed(native, args, cases, record, summary, save):
    started = datetime.now()
    result_dir = Path(
        native.build_eval_save_dir(
            args["task_name"],
            args["policy_name"],
            args["task_config"],
            args["ckpt_setting"],
            started.strftime("%Y-%m-%d_%H-%M-%S_%f"),
        )
    )
    result_dir.mkdir(parents=True, exist_ok=False)
    summary.update(result_dir=str(result_dir), cases_planned=len(cases), cases=[])
    for case in cases:
        case_args = dict(
            args,
            evaluation_id=f"robotwin-fixed-{args['task_config']}-{args['task_name']}",
            trial_id=case["case_id"],
            action_case_id=case["case_id"],
        )
        with pinned_case(native, case, result_dir / "cases" / f"seed_{case['seed']}") as outcome:
            native.main(case_args)
        if outcome.get("successes") not in (0, 1):
            raise EvaluationAborted(f"Case {case['case_id']} produced no episode result")
        entry = dict(case_id=case["case_id"], seed=case["seed"], success=bool(outcome["successes"]))
        summary["cases"].append(entry)
        record(dict(event="case_result", **entry))
        done = len(summary["cases"])
        wins = sum(item["success"] for item in summary["cases"])
        print(f"Success rate: {wins}/{done} => {round(wins / done * 100, 1)}% (seed {case['seed']})", flush=True)
        save()
    successes = sum(entry["success"] for entry in summary["cases"])
    with (result_dir / "_result.txt").open("w", encoding="utf-8") as stream:
        stream.write(f"Timestamp: {started.strftime('%Y-%m-%d_%H-%M-%S_%f')}\n\n")
        stream.write(f"Instruction Type: {args['instruction_type']}\n\n")
        stream.write(str(successes / len(cases)))
    partial = "" if len(cases) == CASES_PER_TASK else f" (partial: {len(cases)}/{CASES_PER_TASK})"
    summary.update(successes=successes, success_rate=successes / len(cases))
    print(f"Data has been saved to {result_dir / '_result.txt'}")
    print(f"Final success rate: {successes}/{len(cases)} = {round(successes / len(cases) * 100, 1)}%{partial}")


def run_native(root, config, cache_manifest_sha256, output, seed_mode="fixed", goal_cache=None, case_range=None):
    evaluator_path(root)
    args = build_arguments(config, root, seed_mode)
    require_sha256(cache_manifest_sha256)
    cases = None
    if seed_mode == "fixed":
        if goal_cache is None:
            raise ValueError("Fixed seed mode requires goal_cache")
        cases = fixed_cases(goal_cache, cache_manifest_sha256, args["task_config"], args["task_name"], case_range)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output / "summary.json"
    summary = dict(
        protocol=PROTOCOL,
        status="running",
        seed_mode=seed_mode,
        cache_manifest_sha256=cache_manifest_sha256,
        arguments=args,
        started_utc=datetime.now(timezone.utc).isoformat(),
    )

    def save():
        tmp = summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(summary_path)

    def record(event):
        with (output / "client_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    save()
    previous_cwd = Path.cwd()
    try:
        os.chdir(Path(root).resolve())
        native = load_evaluator(root)
        with patched_client_factory(native, cache_manifest_sha256, record):
            if seed_mode == "fixed":
                run_fixed(native, args, cases, record, summary, save)
            else:
                native.main(args)
        summary["status"] = "completed"
    except BaseException as error:
        summary.update(status="failed", error_type=type(error).__name__, error=getattr(error, "reason", str(error)))
        raise
    finally:
        os.chdir(previous_cwd)
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
