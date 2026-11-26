"""Command line: run an episode, smoke-test a backend without a model, consolidate skills."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def _overrides(pairs):
    out = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def cmd_run(args) -> int:
    from knowin_agent.episode import load_config, run_episode

    config = load_config(Path(args.config), _overrides(args.set))
    result = run_episode(config, Path(args.run_dir))
    print(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    return 0


def cmd_host(args) -> int:
    from knowin_agent.bridge.host import run_hosted_episode
    from knowin_agent.episode import load_config

    config = load_config(Path(args.config), _overrides(args.set))
    result = run_hosted_episode(config, Path(args.run_dir))
    print(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    return 0


def smoke(runtime, library, text: str, log) -> dict:
    """Model-free walk through the runtime: concurrency, perception, observation and a skill."""
    report = {"instruction": runtime.ctx.instruction, "budget_start": runtime.budget(), "steps": []}

    def step(name, fn):
        started = time.time()
        try:
            out = fn()
            data = out.data if hasattr(out, "data") else out
            row = {"step": name, "ok": "error" not in data, "seconds": round(time.time() - started, 2), "result": data}
        except Exception as exc:  # noqa: BLE001 - a smoke reports every failure and continues
            row = {"step": name, "ok": False, "seconds": round(time.time() - started, 2),
                   "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=4)}
        report["steps"].append(row)
        log(json.dumps(row, ensure_ascii=False, default=str)[:1500])
        return row

    step("arm_status", runtime.arm_status)
    left_up = (runtime.grasp_point("left") + [0, 0, 0.04]).tolist()
    right_up = (runtime.grasp_point("right") + [0, 0, 0.04]).tolist()
    first = step("left_async_up", lambda: runtime.move_to("left", left_up, wait=False))
    step("right_up_concurrent", lambda: runtime.move_to("right", right_up))
    if first.get("ok") and "track" in first.get("result", {}):
        step("wait_left", lambda: runtime.wait(tracks=[first["result"]["track"]]))
    if runtime.perceiver is None:
        report["budget_end"] = runtime.budget()
        return report
    found = step("locate_head", lambda: runtime.locate(text, "head"))
    objects = found.get("result", {}).get("objects") if found.get("ok") else None
    if objects:
        ref = objects[0]["ref"]
        center = objects[0]["center_xyz"]
        arm = "left" if center[0] < 0 else "right"
        step("observe_down", lambda: runtime.observe(arm, ref, view="down", distance_m=0.22))
        wrist = step("locate_wrist", lambda: runtime.locate(text, f"{arm}_wrist"))
        wobjs = wrist.get("result", {}).get("objects") if wrist.get("ok") else None
        target = wobjs[0]["ref"] if wobjs else ref
        step("grasp_candidates", lambda: runtime.grasp_candidates(target, arm=arm))
        step("pick_and_hold", lambda: library.run(runtime, "pick_and_hold", {"target": target, "arm": arm}))
        step("relocate_after_pick", lambda: runtime.locate(text, f"{arm}_wrist"))
        step("budget_panel", runtime.budget_panel)
        if getattr(runtime.backend, "supports_reset", False):
            step("reset", runtime.reset)
            step("arm_status_after_reset", runtime.arm_status)
            step("locate_after_reset", lambda: runtime.locate(text, "head"))
            step("empty_close", lambda: runtime.gripper("left", "close"))
    report["budget_end"] = runtime.budget()
    report["evaluation_host_only"] = runtime.backend.evaluation()
    return report


def cmd_smoke(args) -> int:
    from knowin_agent.episode import build_stack, load_config

    config = load_config(Path(args.config), _overrides(args.set))
    out_dir = Path(args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = None
    if int(config.get("video_stride", 2)) > 0:
        from knowin_agent.backends.robodojo.video import FrameRecorder

        frames = FrameRecorder(stride=int(config.get("video_stride", 2)))
    started = time.time()
    backend, runtime, library = build_stack(config, frame_sink=frames)
    boot_seconds = round(time.time() - started, 1)
    print(f"backend ready in {boot_seconds}s", flush=True)
    try:
        report = smoke(runtime, library, args.text, lambda line: print(line, flush=True))
        report["boot_seconds"] = boot_seconds
        (out_dir / "smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    finally:
        if frames is not None and frames.frames:
            frames.write(out_dir / "video", fps=backend.profile.control_hz)
        backend.close()
    failed = [s["step"] for s in report["steps"] if not s["ok"]]
    print(json.dumps({"failed_steps": failed, "budget_end": report.get("budget_end")}, default=str))
    return 1 if failed else 0


def cmd_script(args) -> int:
    """Run program files through a ProgramSession, exactly as an agent's exec would, with no model: a scripted
    check of library functions on the real simulator. Receipts go to script.json, a video to video/."""
    from knowin_agent.episode import build_stack, load_config
    from knowin_agent.program.session import ProgramSession

    config = load_config(Path(args.config), _overrides(args.set))
    out_dir = Path(args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = None
    if int(config.get("video_stride", 2)) > 0:
        from knowin_agent.backends.robodojo.video import FrameRecorder

        frames = FrameRecorder(stride=int(config.get("video_stride", 2)))
    backend, runtime, library = build_stack(config, frame_sink=frames)
    receipts = []
    try:
        session = ProgramSession(runtime, phase="practice", library=library,
                                 score_feedback=lambda: backend.evaluation().get("score"))
        for path in args.programs:
            receipt = session.execute(Path(path).read_text())
            receipts.append(dict(receipt, program=path))
            print(json.dumps({"program": path, "ok": receipt["ok"], "stop_reason": receipt["stop_reason"],
                              "error": receipt["error"], "native_steps": receipt["native_steps"],
                              "score": receipt.get("score_feedback"), "stdout": receipt["stdout"][-3000:]},
                             ensure_ascii=False, default=str), flush=True)
            if not receipt["ok"] and not args.keep_going:
                break
        (out_dir / "script.json").write_text(json.dumps(
            {"receipts": receipts, "evaluation_host_only": backend.evaluation()}, ensure_ascii=False, indent=1,
            default=str))
    except Exception:  # noqa: BLE001 - closing the simulator exits the process before a traceback prints
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        if frames is not None and frames.frames:
            frames.write(out_dir / "video", fps=backend.profile.control_hz)
        backend.close()
    return 0 if receipts and all(r["ok"] for r in receipts) else 1


def cmd_consolidate(args) -> int:
    from knowin_agent.skills.consolidation import mine, validate, write_candidates

    if args.action == "mine":
        candidates = mine([Path(p) for p in args.runs], min_count=args.min_count)
        path = write_candidates(candidates, Path(args.out))
        print(f"{len(candidates)} candidates -> {path}")
    else:
        smoke_spec = json.loads(Path(args.smoke).read_text()) if args.smoke else None
        print(json.dumps(validate(Path(args.skills), smoke=smoke_spec), indent=1))
    return 0


def cmd_import(args) -> int:
    from knowin_agent.migrate import import_frozen

    print(json.dumps(import_frozen(Path(args.source), Path(args.dest), label=args.label), indent=1))
    return 0


def cmd_review_stage(args) -> int:
    """Inputs of a review session for an external launcher (the L20 console runs the agent itself)."""
    from knowin_agent.evolution.review import Reviewer

    mount = tuple(args.raw_mount.split("=", 1)) if args.raw_mount else None
    ws = Reviewer(None, test_budget=args.test_budget, raw_mount=mount, oracle=args.oracle).build_inputs(
        args.task, [Path(p) for p in args.practice], Path(args.previous) if args.previous else None,
        Path(args.workspace), feedback=[Path(p) for p in args.feedback or []], test_budget=args.test_budget)
    print(json.dumps({"workspace": str(ws)}))
    return 0


def cmd_review_freeze(args) -> int:
    """Check a finished review's out/ and freeze it; REPORT.md lands next to the frozen dir."""
    from knowin_agent.evolution.review import Reviewer, RunInput

    reviewer = Reviewer(None)
    out = Path(args.workspace) / "out"
    runs = [RunInput.of(Path(p)) for p in args.practice]
    state = {"task": args.task, "round": args.round, "merged_from": [str(r.run_dir) for r in runs],
             "feedback": list(args.feedback or []), "previous": args.previous,
             "warnings": reviewer.validate_out(out)}
    frozen = reviewer.to_frozen(out, Path(args.frozen), runs, state)
    print(json.dumps({"frozen": str(frozen), "warnings": state["warnings"]}, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="knowin-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one episode with the Task/Local agents")
    run.add_argument("--config", required=True)
    run.add_argument("--run-dir", required=True)
    run.add_argument("--set", action="append", help="override a config value, e.g. backend.seed=3")
    run.set_defaults(fn=cmd_run)
    host = sub.add_parser("host", help="run one episode with Task/Local driven by codex exec over MCP")
    host.add_argument("--config", required=True)
    host.add_argument("--run-dir", required=True)
    host.add_argument("--set", action="append")
    host.set_defaults(fn=cmd_host)
    sm = sub.add_parser("smoke", help="model-free backend smoke: dual-arm motion, perception, observe, a skill")
    sm.add_argument("--config", required=True)
    sm.add_argument("--run-dir", required=True)
    sm.add_argument("--text", default="block")
    sm.add_argument("--set", action="append")
    sm.set_defaults(fn=cmd_smoke)
    sc = sub.add_parser("script", help="run program files through a ProgramSession with no model")
    sc.add_argument("--config", required=True)
    sc.add_argument("--run-dir", required=True)
    sc.add_argument("--set", action="append")
    sc.add_argument("--keep-going", action="store_true", help="run every program even after one fails")
    sc.add_argument("programs", nargs="+")
    sc.set_defaults(fn=cmd_script)
    im = sub.add_parser("import-frozen", help="import an H4/H5 run's frozen task knowledge")
    im.add_argument("source")
    im.add_argument("dest")
    im.add_argument("--label", default=None)
    im.set_defaults(fn=cmd_import)
    co = sub.add_parser("consolidate", help="offline skill consolidation")
    co.add_argument("action", choices=["mine", "validate"])
    co.add_argument("--runs", nargs="*", default=[])
    co.add_argument("--out", default="consolidation")
    co.add_argument("--min-count", type=int, default=2)
    co.add_argument("--skills", default="skills")
    co.add_argument("--smoke", default=None)
    co.set_defaults(fn=cmd_consolidate)
    for name, fn in (("review-stage", cmd_review_stage), ("review-freeze", cmd_review_freeze)):
        rv = sub.add_parser(name, help="review merge step driven by an external launcher (L20 console)")
        rv.add_argument("--task", required=True)
        rv.add_argument("--workspace", required=True)
        rv.add_argument("--practice", nargs="+", required=True, help="practice run dirs, best first")
        rv.add_argument("--feedback", nargs="*", default=[], help="test run dirs of the previous merge")
        rv.add_argument("--previous", default=None, help="frozen dir of the previous merge")
        rv.add_argument("--raw-mount", default=None,
                        help="HOST=CONTAINER: the runs directory mounted read-only in the review container")
        if name == "review-stage":
            rv.add_argument("--test-budget", type=int, default=100)
            rv.add_argument("--oracle", action="store_true",
                            help="also give the host's per-predicate record (switch evolution.condition_report)")
        else:
            rv.add_argument("--frozen", required=True)
            rv.add_argument("--round", type=int, default=None)
        rv.set_defaults(fn=fn)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
