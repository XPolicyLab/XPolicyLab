"""Read-only selective candidate preflight; never imports simulator/model code.

Machine launchers own flock and launch once. Busy cards are a valid check-only
outcome, not permission to launch. Native task names here are an evaluation plan,
never policy routing. --tasks is a failed/low-task subset, retaining both Gen halves.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import sys

from wall_oss_rerun import (full_tasks, gpu_problems, parse_gpus, process_problems,
                            require, validate_package)

TIER1 = ("solve_equation", "align_blocks", "push_T", "push_T_random",
         "classify_objects_by_language", "swap_blocks", "arrange_largest_number",
         "arrange_largest_number_random", "play_Xylophone", "pack_objects_into_box",
         "pack_objects_into_box_random", "store_laptop_and_headphones",
         "store_laptop_and_headphones_random", "general_pickup", "press_by_number")
TIER2 = ("sweep_blocks", "sweep_blocks_random")
AFIX = ("make_kong", "fasten_screws", "build_tower", "cover_blocks", "match_and_pick_from_conveyor")


def task_plan(robodojo, *, tasks=None, afix=False, tier2=False):
    allowed = TIER1 + (AFIX if afix else ()) + (TIER2 if tier2 else ())
    selected = tuple(tasks.split(",")) if tasks else allowed
    require(bool(selected) and len(set(selected)) == len(selected) and set(selected) <= set(allowed),
            "Only unique Tier1 (+ AFIX/Tier2 explicitly enabled) tasks allowed")
    native = full_tasks(robodojo)
    require(set(selected) <= set(native), "Unknown native task")
    for task in selected:
        base = task.removesuffix("_random")
        if base + "_random" in native:
            require(base in selected and base + "_random" in selected, "Gen requires BOTH halves: " + base)
    return selected


def lock_available(gpu, lock_root):
    """Existing shared inode only: check-only never creates or truncates a lock."""
    try:
        fd = os.open(lock_root / f"wall-oss-hack-gpu{gpu}.lock", os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        finally:
            os.close(fd)
    except OSError:
        return False


def preflight(args):
    gpus = parse_gpus(args.gpus)
    agent = validate_package(args.package, expected_version="1.21.0")
    require("complementary" in agent["sources"]["agent"]["skills"].get("enable", []), "Complementary required")
    tasks = task_plan(args.robodojo, tasks=args.tasks, afix=args.afix, tier2=args.tier2)
    require(not args.out.exists() and not args.out.is_symlink(), "OUT exists: never resume/overwrite")
    links = [args.package / "results", args.robodojo / "smoke_results" / args.out.name]
    links += [args.robodojo / "eval_result/RoboDojo" / task / args.package.name for task in tasks]
    require(all(not p.exists() and not p.is_symlink() for p in links), "Result namespace exists: NEW policy required")
    require(args.auth.is_file(), "Auth file missing (not read by preflight)")
    print(f"agent={agent['lock']['agent_version']} identity={agent['identity']}")
    print("tasks=" + ",".join(tasks))
    print("OUT=" + str(args.out) + "; live vendor/results untouched")
    for name, port in (("WALL external", 19012), ("SAM3 external", 8810), ("relay", 18087)):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                pass
        except OSError as error:
            raise ValueError(f"{name} :{port} unavailable; never restart") from error
        print(f"{name} :{port} reachable")
    free = []
    for gpu in gpus:
        problems = gpu_problems([gpu]) + process_problems([gpu])
        if not args.locks_held and not lock_available(gpu, args.lock_root):
            problems.append(f"GPU{gpu} flock unavailable; wait, never kill")
        if problems:
            print("; ".join(problems))
        else:
            free.append(gpu)
    if args.require_idle:
        require(free == gpus, "Selected GPU subset not idle; refused run")
    print("FREE_GPUS=" + ",".join(free))
    print("CHECK-ONLY OK; " + ("idle subset requires held flock + repeated preflight before --run" if free else "waiting for GPU"))
    return agent, tasks, free


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("package", "robodojo", "out", "auth", "lock-root"):
        p.add_argument("--" + key, type=Path, required=True)
    # R3 Isaac segfault(139) on GPU5/7 is simulator evidence, not a driver fix.
    # Use the conservative next-round subset; explicit --gpus still remains possible.
    p.add_argument("--gpus", default="1,2,3,4")
    p.add_argument("--tasks")
    p.add_argument("--afix", action="store_true")
    p.add_argument("--tier2", action="store_true")
    p.add_argument("--require-idle", action="store_true")
    p.add_argument("--locks-held", action="store_true", help="launcher already holds shared flock descriptors")
    args = p.parse_args(argv)
    try:
        preflight(args)
    except (ValueError, KeyError, OSError) as error:
        print("REFUSED: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
