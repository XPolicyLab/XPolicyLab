"""Run the pinned native RoboTwin loop with fail-closed HumanWAM-Goal RPCs.

Two seed modes share the native episode loop (setup, instruction, observation,
action and success code unchanged):

``fixed`` (default) replays the 10,000 released cases: for each case of the task,
one native episode at its seed with its exact instruction and no expert check.
Native seed selection is not repeatable across machines (CuRobo's L-BFGS CUDA
kernel reads uninitialized memory; MotionGen retries stop at a 10 s wall-clock
timeout), so the released population is replayed instead of re-selected.

``native`` runs the evaluator as published: expert check over seeds from 100000
until 100 trials are accepted. A seed outside the released cache stops the run.

The default command only checks source/configuration on CPU. No simulator,
transport, model, or image API is imported until an explicit ``--run``.
"""

import argparse
import ast
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
ROBOTWIN_COMMIT = "ea8b21121ebb3cd201ff5b3fe361944ac94eda3f"
NATIVE_SOURCE_SHA256 = "78821388d58daf8aa65db55c5952df3742486dd93e717ce6a96374a3795dfc7a"
TASKS = frozenset(
    """adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size
click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block handover_mic
hanging_mug lift_pot move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad
open_laptop open_microwave pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right
place_bread_basket place_bread_skillet place_burger_fries place_can_basket place_cans_plasticbox
place_container_plate place_dual_shoes place_empty_cup place_fan place_mouse_pad
place_object_basket place_object_scale place_object_stand place_phone_stand place_shoe
press_stapler put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object shake_bottle
shake_bottle_horizontally stack_blocks_three stack_blocks_two stack_bowls_three stack_bowls_two
stamp_seal turn_switch""".split()
)


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_sha256(value):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a lowercase SHA256 digest")
    return value


class OfficialNativeFatal(SystemExit):
    """Client-side process failure; upstream ``except Exception`` cannot skip it."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(2)


class FailClosedClient:
    """Enrich native case metadata and require acknowledgements before actions."""

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
        event = dict(event="fatal_policy_boundary", reason=reason, case=self.case)
        try:
            self.record(event)
        except Exception as error:
            reason += f"; audit write failed: {type(error).__name__}: {error}"
        print(f"HumanWAM-Goal official native run aborted: {reason}", file=sys.stderr, flush=True)
        raise OfficialNativeFatal(reason)

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
            self.abort(f"Invalid {operation} acknowledgement; expected case/cache/instruction binding")
        if reply.get("initial_verified") is not (operation == "update_obs"):
            self.abort(f"Invalid {operation} initial_verified acknowledgement")
        self.record(dict(event="verified_ack", **reply))

    def call(self, *, func_name, **kwargs):
        if self.failed:
            self.abort("The policy session already failed; it cannot resume or select another seed")
        if func_name not in {"prepare_case", "reset", "update_obs", "get_action", "trial_end"}:
            self.abort(f"Unsupported native RPC: {func_name}")
        try:
            if func_name == "prepare_case":
                self.case = None
                self.initial_verified = False
                metadata = dict(kwargs["obs"])
                if metadata.get("task_name") != self.task_name or metadata.get("action_type") != "joint":
                    self.abort("Native task/action differs from the declared submission profile")
                if "task_config" in metadata and metadata["task_config"] != self.task_config:
                    self.abort("Native task_config differs from the declared submission profile")
                seed, instruction = metadata["seed"], metadata["instruction"]
                if (
                    type(seed) is not int
                    or seed < 100000
                    or not isinstance(instruction, str)
                    or not instruction.strip()
                ):
                    self.abort("Invalid selected native seed or exact instruction")
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
                self.abort(f"{func_name} requires a prepared native case")
            if func_name == "reset":
                self.initial_verified = False
            if func_name == "update_obs" and kwargs["obs"].get("instruction") != self.case["instruction"]:
                self.abort("Observed instruction differs from the selected native instruction")
            if func_name == "get_action" and not self.initial_verified:
                self.abort("No action is permitted before initial RGB/state/camera verification")
            reply = self.client.call(func_name=func_name, **kwargs)
            if func_name in ("prepare_case", "update_obs"):
                self.check_ack(reply, func_name)
                if func_name == "update_obs":
                    self.initial_verified = True
            elif func_name == "get_action" and self.validate_action is not None:
                if not self.validate_action(reply):
                    self.abort("Policy returned an empty action chunk")
            elif func_name == "trial_end":
                self.record(dict(event="native_trial_end", case=self.case, result=kwargs.get("obs")))
                self.case = None
                self.initial_verified = False
            return reply
        except Exception as error:
            # Websocket error frames become ordinary client WsError exceptions. Make
            # the decision here, outside the remote exception type hierarchy.
            self.abort(f"{func_name} failed: {type(error).__name__}: {error}")

    def close(self):
        try:
            return self.client.close()
        except Exception as error:
            if not self.failed:
                self.abort(f"Policy connection cleanup failed: {type(error).__name__}: {error}")


def checked_source(root):
    path = Path(root).resolve() / "scripts/eval_policy_xpolicylab.py"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != NATIVE_SOURCE_SHA256:
        raise ValueError("Native evaluator differs from the reviewed RoboTwin commit; review before updating its pin")
    tree = ast.parse(raw, filename=str(path))
    if any(
        isinstance(n, ast.ExceptHandler)
        and (
            n.type is None
            or any(isinstance(x, ast.Name) and x.id in {"BaseException", "SystemExit"} for x in ast.walk(n.type))
        )
        for n in ast.walk(tree)
    ):
        raise ValueError("Native evaluator could swallow a fatal client boundary")
    return path, raw


def parse_case_range(text):
    """``start:stop`` over a task's 100 released cases in seed order (Python slice bounds)."""
    start, sep, stop = str(text).partition(":")
    if not sep or not start.isdigit() or not stop.isdigit() or not 0 <= int(start) < int(stop) <= CASES_PER_TASK:
        raise ValueError(f"A fixed case range is start:stop with 0 <= start < stop <= {CASES_PER_TASK}")
    return int(start), int(stop)


def fixed_cases(cache_root, cache_manifest_sha256, task_config, task_name, case_range=None):
    """The released cases of one task, in seed order, from the pinned goal-cache manifest."""
    raw = (Path(cache_root) / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != require_sha256(cache_manifest_sha256):
        raise ValueError("Goal-cache manifest differs from its pinned digest")
    rows = sorted(
        (r for r in json.loads(raw)["cases"] if r["task_config"] == task_config and r["task"] == task_name),
        key=lambda r: r["seed"],
    )
    seeds = [r["seed"] for r in rows]
    if len(rows) != CASES_PER_TASK or len(set(seeds)) != len(seeds):
        raise ValueError(f"Expected {CASES_PER_TASK} distinct released seeds for {task_config}/{task_name}")
    cases = []
    for row in rows:
        seed, instruction = row["seed"], row["official_instruction"]
        if type(seed) is not int or seed < 100000 or text_sha256(instruction) != row["official_instruction_sha256"]:
            raise ValueError(f"Invalid released case {task_config}/{task_name}/{seed}")
        cases.append(dict(seed=seed, instruction=instruction, case_id=f"{task_config}/{task_name}/{seed}"))
    if case_range is not None:
        start, stop = parse_case_range(case_range)
        cases = cases[start:stop]
    return cases


def build_arguments(config, root, seed_mode="fixed"):
    allowed = {
        "task_name",
        "task_config",
        "policy_name",
        "ckpt_setting",
        "policy_server_url",
        "host",
        "port",
        "xpolicylab_root",
        "device_id",
    }
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Unsupported native configuration fields; seed/budget/expert/control overrides are forbidden")
    for key in ("task_name", "task_config", "policy_name", "ckpt_setting"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Missing native configuration: {key}")
    if config["task_name"] not in TASKS or config["task_config"] not in {"demo_clean", "demo_randomized"}:
        raise ValueError("Expected an official task and clean/randomized configuration")
    if not config.get("policy_server_url") and not config.get("port"):
        raise ValueError("Configure policy_server_url or host/port")
    if config.get("action_type", "joint") not in ("joint", "ee"):
        raise ValueError("action_type must be joint or ee")
    if seed_mode not in SEED_MODES:
        raise ValueError(f"seed_mode must be one of {SEED_MODES}")
    args = dict(config)
    args.update(
        root_dir=str(Path(root).resolve()),
        bench_name="RoboTwin",
        env_cfg_type="arx_x5",
        protocol="ws",
        seed=0,
        # fixed: one native episode per released case, its seed already accepted.
        test_num=1 if seed_mode == "fixed" else CASES_PER_TASK,
        expert_check=seed_mode == "native",
        action_type=config.get("action_type", "joint"),
        eval_batch=False,
        frequency=30,
        instruction_type="seen" if config["task_config"] == "demo_clean" else "unseen",
    )
    args.setdefault("xpolicylab_root", str(Path(root).resolve() / "XPolicyLab"))
    args.setdefault("device_id", "0")
    return args


@contextmanager
def guarded_factory(native, cache_manifest_sha256, record=None):
    original = native.build_policy_client

    def build(args):
        try:
            client = original(args)
            return FailClosedClient(
                client,
                task_config=args["task_config"],
                task_name=args["task_name"],
                cache_manifest_sha256=cache_manifest_sha256,
                record=record,
                validate_action=native.normalize_action_chunk,
            )
        except Exception as error:
            raise OfficialNativeFatal(f"Policy connection failed: {type(error).__name__}: {error}") from error

    native.build_policy_client = build
    try:
        yield
    finally:
        native.build_policy_client = original


@contextmanager
def released_case(native, case, save_dir):
    """Pin one native episode to a released case: its seed and instruction, and no other seed."""
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
            # The native loop moves to the next seed after an unstable setup or an
            # environment error before the first action; a released case is never replaced.
            if kwargs.get("seed") != case["seed"]:
                raise OfficialNativeFatal(
                    f"Released case {case['case_id']} did not complete; the native loop "
                    f"requested seed {kwargs.get('seed')} instead"
                )
            return setup(*args, **kwargs)

        task_env.setup_demo = setup_case
        return task_env

    native.eval_remote_policy = evaluate
    # Without the expert episode the native instruction generator lacks its input;
    # the released case carries the exact instruction evaluated for this seed.
    native.build_instruction = lambda *unused, **unused_kwargs: case["instruction"]
    native.class_decorator = make_env
    native.build_eval_save_dir = lambda *unused: save_dir
    try:
        yield outcome
    finally:
        for name, value in originals.items():
            setattr(native, name, value)


def run_fixed(native, args, cases, record, report, save):
    """One native episode per released case; RoboTwin's _result.txt over all of them."""
    started = datetime.now()
    result_dir = Path(
        native.build_eval_save_dir(
            args["task_name"], args["policy_name"], args["task_config"], args["ckpt_setting"],
            started.strftime("%Y-%m-%d_%H-%M-%S_%f"),
        )
    )
    result_dir.mkdir(parents=True, exist_ok=False)
    report.update(result_dir=str(result_dir), cases_planned=len(cases), cases=[])
    for case in cases:
        case_args = dict(
            args,
            evaluation_id=f"robotwin-fixed-{args['task_config']}-{args['task_name']}",
            trial_id=case["case_id"],
            action_case_id=case["case_id"],
        )
        with released_case(native, case, result_dir / "cases" / f"seed_{case['seed']}") as outcome:
            native.main(case_args)
        if outcome.get("successes") not in (0, 1):
            raise OfficialNativeFatal(f"Released case {case['case_id']} returned no native result")
        entry = dict(case_id=case["case_id"], seed=case["seed"], success=bool(outcome["successes"]))
        report["cases"].append(entry)
        record(dict(event="fixed_case_result", **entry))
        save()
    successes = sum(entry["success"] for entry in report["cases"])
    with (result_dir / "_result.txt").open("w", encoding="utf-8") as stream:
        stream.write(f"Timestamp: {started.strftime('%Y-%m-%d_%H-%M-%S_%f')}\n\n")
        stream.write(f"Instruction Type: {args['instruction_type']}\n\n")
        stream.write(str(successes / len(cases)))
    partial = "" if len(cases) == CASES_PER_TASK else (
        f" (partial: {len(cases)} of {CASES_PER_TASK} released cases, seeds {cases[0]['seed']}..{cases[-1]['seed']})"
    )
    report.update(successes=successes, success_rate=successes / len(cases))
    print(f"Data has been saved to {result_dir / '_result.txt'}")
    print(f"Final success rate: {successes}/{len(cases)} = {round(successes / len(cases) * 100, 1)}%{partial}")


def run_native(root, config, cache_manifest_sha256, output, seed_mode="fixed", goal_cache=None, case_range=None):
    """Explicit execution; consumes the upstream loop unchanged."""
    source, raw = checked_source(root)
    args = build_arguments(config, root, seed_mode)
    require_sha256(cache_manifest_sha256)
    cases = None
    if seed_mode == "fixed":
        if goal_cache is None:
            raise ValueError("Fixed seed mode reads its cases from the released goal cache")
        cases = fixed_cases(goal_cache, cache_manifest_sha256, args["task_config"], args["task_name"], case_range)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "native_wrapper_report.json"
    report = dict(
        protocol=PROTOCOL,
        status="starting",
        seed_mode=seed_mode,
        robotwin_commit=ROBOTWIN_COMMIT,
        native_source_sha256=NATIVE_SOURCE_SHA256,
        cache_manifest_sha256=cache_manifest_sha256,
        arguments=args,
        started_utc=datetime.now(timezone.utc).isoformat(),
        native_entry_invoked=False,
    )

    def save():
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(report_path)

    def record(event):
        with (output / "client_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    save()
    previous_cwd = Path.cwd()
    try:
        os.chdir(Path(root).resolve())
        native = ModuleType("_goalwam_pinned_robotwin_native")
        native.__file__ = str(source)
        # Execute the bytes we hashed, not a second unverified filesystem read.
        exec(compile(raw, str(source), "exec"), native.__dict__)
        with guarded_factory(native, cache_manifest_sha256, record):
            report["native_entry_invoked"] = True
            if seed_mode == "fixed":
                run_fixed(native, args, cases, record, report, save)
            else:
                native.main(args)
        report["status"] = "native_loop_completed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=getattr(error, "reason", str(error)))
        raise
    finally:
        os.chdir(previous_cwd)
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Native single-task connection/profile JSON")
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--seed-mode", choices=SEED_MODES, default="fixed")
    parser.add_argument("--goal-cache", type=Path, help="Released goal cache; required with --seed-mode fixed")
    parser.add_argument("--cases", help="Fixed mode: start:stop slice of the 100 released cases (partial run)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Default: CPU source/configuration checks only")
    mode.add_argument("--run", action="store_true", help="Explicitly execute a future native evaluation")
    parser.add_argument("--output", type=Path, help="New wrapper audit directory; required with --run")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source, _ = checked_source(args.robotwin_root)
    profile = build_arguments(config, args.robotwin_root, args.seed_mode)
    require_sha256(args.cache_manifest_sha256)
    cases = None
    if args.seed_mode == "fixed":
        if args.goal_cache is None:
            parser.error("--seed-mode fixed requires --goal-cache")
        cases = fixed_cases(
            args.goal_cache, args.cache_manifest_sha256, profile["task_config"], profile["task_name"], args.cases
        )
    if args.run:
        if args.output is None:
            parser.error("--run requires a new --output directory")
        run_native(
            args.robotwin_root, config, args.cache_manifest_sha256, args.output,
            args.seed_mode, args.goal_cache, args.cases,
        )
    else:
        print(
            json.dumps(
                dict(
                    check_only=True,
                    benchmark_executed=False,
                    protocol=PROTOCOL,
                    source=str(source),
                    source_sha256=NATIVE_SOURCE_SHA256,
                    cache_manifest_sha256=args.cache_manifest_sha256,
                    seed_mode=args.seed_mode,
                    fixed_cases=None if cases is None else len(cases),
                    arguments=profile,
                    limits="Source/config only; no cache availability, model load, simulator or success validation",
                ),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
