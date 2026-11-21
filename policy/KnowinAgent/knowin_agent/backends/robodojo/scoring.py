"""Host-only readout of RoboDojo's native score, as the official ``run_eval`` computes it.

Native ``success`` starts True and only means "not failed yet"; an episode is
completed when ``end_flag`` and ``success`` are both set. A completed episode
scores 100; otherwise the task's process score (``reward_manager.get_score``,
present when the task defines ``get_score``) is the partial credit, else 0.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Optional

# Predicates that write evaluator state even when asked for a reading (transition latches, state refreshes);
# a host-side report never calls them. Predicates with an `update` argument are called with update=False.
STATEFUL_PREDICATES = ("is_joint_position_ratio_change_from_above_to_below", "is_joint_position_change",
                       "update_object_state")
PARSER_STATE = ("pre_state", "joint_ratio_transition_state", "robot_origin_endpose")
LAYOUT_TYPES = ("Rigid", "Articulation", "Garment")


def partial_credit_supported(native: Any) -> bool:
    manager = getattr(native, "reward_manager", None)
    return hasattr(native, "get_score") and callable(getattr(manager, "get_score", None))


def read_score(native: Any, env_idx: int = 0) -> dict:
    ended = bool(native.end_flag[env_idx])
    success = ended and bool(native.success[env_idx])
    if success:
        score = 100.0
    elif partial_credit_supported(native):
        score = float(native.reward_manager.get_score()[env_idx])
    else:
        score = 0.0
    if not math.isfinite(score) or not 0.0 <= score <= 100.0:
        raise ValueError(f"native score {score!r} outside [0, 100]")
    return {
        "score": score,
        "success": success,
        "ended": ended,
        "native_steps": int(native.take_action_cnt[env_idx]),
    }


def is_predicate(check: Any) -> bool:
    return isinstance(check, tuple) and len(check) == 2 and isinstance(check[0], str) and isinstance(check[1], dict)


def predicates_in(check: Any) -> list:
    """The (name, args) leaves of a check, a list of checks or a (checks, score) pair, in order."""
    if is_predicate(check):
        return [check]
    if isinstance(check, (list, tuple)):
        return [leaf for part in check for leaf in predicates_in(part)]
    return []


def _plain(value: Any) -> Any:
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value][:12]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    return repr(value)[:60]


def predicate_label(check: tuple) -> str:
    name, args = check
    shown = ", ".join(f"{k}={_plain(v)}" for k, v in args.items() if k not in ("env_idx", "update"))
    return f"{name}({shown})"


def _evaluate(manager: Any, check: tuple, env_idx: int) -> dict:
    name, args = check
    if name in STATEFUL_PREDICATES:
        return {"ok": None, "skipped": "stateful predicate"}
    args = dict(args)
    if "update" in args:
        args["update"] = False
    try:
        value = float(manager.call_func_parser((name, args), env_idx))
    except Exception as exc:  # noqa: BLE001 - a report entry, never an episode failure
        return {"ok": None, "error": f"{type(exc).__name__}: {exc}"[:160]}
    return {"ok": value >= 1.0 - 1e-6, "value": round(value, 4)}


def condition_report(native: Any, env_idx: int = 0) -> Optional[dict]:
    """Host-only: every predicate of the task's score tiers and success check, evaluated now one by one.

    The evaluator's parser state is saved before and restored after, and `update` predicates are read with
    update=False, so the official score sequence is unchanged. Returns None when the task has no reward
    manager. Never give this to an agent: it is the evaluator's ground truth.
    """
    manager = getattr(native, "reward_manager", None)
    if manager is None or not hasattr(manager, "call_func_parser"):
        return None
    parser = getattr(manager, "func_parser", None)
    saved = {k: copy.deepcopy(getattr(parser, k)) for k in PARSER_STATE if hasattr(parser, k)}
    try:
        meta = (getattr(manager, "score_meta", None) or [{}])[env_idx] or {}
        entries = list((getattr(manager, "score_list", None) or [[]])[env_idx] or [])
        success_checks = list((getattr(manager, "check_list", None) or [[]])[env_idx] or [])
        ids: dict[str, str] = {}
        predicates = []

        def ref(check):
            key = predicate_label(check)
            if key not in ids:
                ids[key] = f"p{len(ids) + 1}"
                predicates.append({"id": ids[key], "pred": key, **_evaluate(manager, check, env_idx)})
            return ids[key]

        mode = meta.get("mode")
        gradient = list(meta.get("gradient") or [])
        passed = int((getattr(manager, "score_completed_count", None) or [0])[env_idx] or 0)
        stages = []
        for i, entry in enumerate(entries):
            stage = {"uses": sorted(set(ref(c) for c in predicates_in(entry)), key=lambda s: int(s[1:]))}
            if mode == "transition":
                stage.update(score=gradient[i] if i < len(gradient) else None, passed=i < passed)
            elif mode == "paired" and isinstance(entry, tuple) and len(entry) == 2:
                stage.update(score=entry[1], passed=False)
            stages.append(stage)
        return {"mode": mode, "tiers": gradient, "stages_passed": passed if mode != "paired" else None,
                "stages": stages, "success_check": sorted(set(ref(c) for c in predicates_in(success_checks)),
                                                         key=lambda s: int(s[1:])),
                "predicates": predicates}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}
    finally:
        for key, value in saved.items():
            setattr(parser, key, value)


def object_poses(native: Any, env_idx: int = 0) -> Optional[dict]:
    """Host-only: the simulator's pose of every scene object (position m, quaternion), keyed by instance."""
    parser = getattr(getattr(native, "reward_manager", None), "func_parser", None)
    layout = getattr(parser, "layout_manager", None)
    if layout is None:
        return None
    poses = {}
    try:
        for kind in LAYOUT_TYPES:
            for record in layout.get_layout_records(env_idx, kind) or []:
                name = record.get("inst_name")
                pos, rot = layout.get_instance_pose(inst_name=name, env_idx=env_idx)
                poses[name] = {"label": record.get("label"), "kind": kind,
                               "pos": [round(float(v), 4) for v in _flat(pos)],
                               "quat": [round(float(v), 4) for v in _flat(rot)]}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:200], **poses}
    return poses


def _flat(value: Any) -> list:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "flatten"):
        value = value.flatten()
    return list(value)
