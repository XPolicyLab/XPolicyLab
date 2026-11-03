#!/usr/bin/env python3
"""Freeze a method configuration of OEH_Review for one task.

``configs/<ckpt>.template.json`` names the method with paths relative to this adapter (``${ADAPTER}``, ``${XPL_ROOT}``,
``${OPENWAM_CKPT_DIR}``). This resolves them, sets the task's native step limit (``configs/robodojo-step-limits.json``,
from the pinned RoboDojo source) and, for a review configuration, the task the teacher is briefed on, then runs the
OEH freeze (``oeh_policy.cli.freeze``): checkpoint, normalizer, OpenWAM source, kernel and runtime identities are
hashed into ``state/<ckpt>/<task>.frozen.json``, which the policy server loads and re-checks at start.

usage: freeze_config.py --ckpt oeh_fast --task classify_objects --xpolicylab-root /path/to/XPolicyLab
                        [--openwam-ckpt-dir DIR] [--step-limit N] [--force]
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

POLICY_DIR = Path(__file__).resolve().parents[1]
RUNTIME = POLICY_DIR / "runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from oeh_policy import cli  # noqa: E402

STEP_LIMITS = POLICY_DIR / "configs" / "robodojo-step-limits.json"
DIGEST_CACHE = POLICY_DIR / "state" / "digest-cache.json"
# RoboDojo's control rate, as the campaign's matrix block declares it for the review teacher's timing.
ROBODOJO_CONTROL_HZ = 25


def _substitute(value, mapping: dict[str, str]):
    if isinstance(value, str):
        for key, replacement in mapping.items():
            value = value.replace(key, replacement)
        return value
    if isinstance(value, list):
        return [_substitute(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _substitute(item, mapping) for key, item in value.items()}
    return value


def resolve(template: dict, *, adapter: Path, xpl_root: Path, ckpt_dir: Path, task: str, step_limit: int,
            step_source: str) -> dict:
    cfg = _substitute(copy.deepcopy(template), {"${ADAPTER}": str(adapter), "${XPL_ROOT}": str(xpl_root),
                                                "${OPENWAM_CKPT_DIR}": str(ckpt_dir)})
    cfg["native_step_limit"] = step_limit
    cfg["native_step_limit_source"] = step_source
    if cfg.get("mode") == "review":
        review = cfg["review"]
        review["task"] = task
        review["task_source"] = (f"XPolicyLab task_name {task} (the teacher's task context is keyed by the task "
                                 "without its _random suffix)")
        review.setdefault("control_hz", ROBODOJO_CONTROL_HZ)
        # The review mode bounds its first call through review.first_call_extra_seconds, as the campaign compiles it.
        cfg.pop("first_decision_seconds", None)
        cfg.pop("first_decision_seconds_source", None)
    return cfg


def _load_cache() -> None:
    if DIGEST_CACHE.is_file():
        for key, value in json.loads(DIGEST_CACHE.read_text(encoding="utf-8")).items():
            path, size, mtime = key.rsplit("|", 2)
            cli._DIGESTS[(path, int(size), int(mtime))] = value


def _save_cache() -> None:
    DIGEST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DIGEST_CACHE.write_text(json.dumps({f"{p}|{s}|{m}": v for (p, s, m), v in cli._DIGESTS.items()}, indent=1,
                                       sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="method configuration: configs/<ckpt>.template.json")
    parser.add_argument("--task", required=True, help="RoboDojo task_name")
    parser.add_argument("--xpolicylab-root", required=True, type=Path)
    parser.add_argument("--openwam-ckpt-dir", type=Path,
                        help="OpenWAM-Alpha-Sim-RoboDojo checkpoint directory (default: policy/OpenWAM/checkpoints/...)")
    parser.add_argument("--step-limit", type=int, help="native step limit for a task not in configs/robodojo-step-limits.json")
    parser.add_argument("--force", action="store_true", help="re-freeze an existing configuration")
    args = parser.parse_args()

    template_path = POLICY_DIR / "configs" / f"{args.ckpt}.template.json"
    if not template_path.is_file():
        known = sorted(p.name[:-len(".template.json")] for p in (POLICY_DIR / "configs").glob("*.template.json"))
        print(f"[freeze_config] unknown ckpt_name {args.ckpt!r}; configs/ has {known}", file=sys.stderr)
        return 1
    output = POLICY_DIR / "state" / args.ckpt / f"{args.task}.frozen.json"
    if output.is_file() and not args.force:
        print(f"[freeze_config] reusing {output}")
        return 0
    xpl_root = args.xpolicylab_root.resolve()
    ckpt_dir = (args.openwam_ckpt_dir or xpl_root / "policy/OpenWAM/checkpoints/OpenWAM-Alpha-Sim-RoboDojo").resolve()
    if not (ckpt_dir / "normalization_stats.npy").is_file():
        print(f"[freeze_config] no OpenWAM checkpoint at {ckpt_dir} (download_checkpoint.sh, or --openwam-ckpt-dir)",
              file=sys.stderr)
        return 1
    limits = json.loads(STEP_LIMITS.read_text(encoding="utf-8"))
    if args.step_limit is not None:
        step_limit, step_source = args.step_limit, "freeze_config.py --step-limit (operator override)"
    elif args.task in limits["limits"]:
        step_limit, step_source = limits["limits"][args.task], f"{limits['source']} for {args.task}"
    else:
        print(f"[freeze_config] task {args.task!r} is not in {STEP_LIMITS.name}; pass --step-limit", file=sys.stderr)
        return 1
    template = json.loads(template_path.read_text(encoding="utf-8"))
    cfg = resolve(template, adapter=POLICY_DIR, xpl_root=xpl_root, ckpt_dir=ckpt_dir, task=args.task,
                  step_limit=step_limit, step_source=step_source)
    output.parent.mkdir(parents=True, exist_ok=True)
    resolved = output.with_name(f"{args.task}.resolved.json")
    resolved.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if output.exists():
        output.unlink()
    _load_cache()
    try:
        identity = cli.freeze(resolved, output)
    finally:
        _save_cache()
    print(f"[freeze_config] {output}: configuration {identity['configuration_sha256'][:12]} "
          f"checkpoint {identity['checkpoint_sha256'][:12]} native_step_limit {step_limit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
