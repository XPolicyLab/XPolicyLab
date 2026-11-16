"""Read-only, host-private snapshots of RoboDojo's native score accumulator."""

import hashlib
import inspect
import math
from pathlib import Path

import numpy as np

SCHEMA = "robodojo-native-score-v1"
MANAGER_FIELDS = (
    "check_list",
    "final_check_list",
    "query_list",
    "trigger_check_list",
    "trigger_query_list",
    "score_list",
    "score_achieved",
    "score_completed_count",
    "score_meta",
    "score_trigger_meta",
    "final_score_list",
    "final_score_achieved",
    "final_score_completed_count",
    "final_score_meta",
    "_gated_score_lst",
)
PARSER_FIELDS = ("pre_state", "robot_origin_endpose", "joint_ratio_transition_state")


def encode(value):
    """Archive values and symbolic predicate names, never pickle simulator objects."""
    if isinstance(value, np.ndarray):
        return encode(value.tolist())
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if callable(value):
        return {"callable": f"{value.__module__}.{value.__qualname__}"}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        # Unlimited joint bounds may legitimately be infinite in predicate metadata.
        return value if math.isfinite(value) else {"nonfinite_float": str(value)}
    if type(value).__module__.startswith("torch") and hasattr(value, "detach"):
        return encode(value.detach().cpu().numpy())
    raise ValueError(f"Unsupported scoring state: {type(value).__name__}")


def source_hashes(native):
    objects = [
        type(native),
        getattr(native, "get_score", None),
        getattr(native, "run_reward", None),
    ]
    manager = getattr(native, "reward_manager", None)
    if manager is not None:
        objects += [type(manager), type(manager.func_parser)]
    sources = {}
    for obj in objects:
        if obj is None:
            continue
        filename = inspect.getsourcefile(obj)
        if filename and Path(filename).is_file():
            path = Path(filename).resolve()
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def snapshot(env):
    native = env.native
    # Native success=True before end_flag means viable, not completed.
    completed = bool(native.end_flag[0]) and bool(native.success[0])
    if completed != bool(env.success):
        raise ValueError("Native completion and adapter completion disagree")
    partial = callable(getattr(native, "get_score", None))
    manager = getattr(native, "reward_manager", None)
    score = 100.0 if completed else float(manager.get_score()[0]) if partial else 0.0
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError(f"Invalid native Score: {score}")
    state = {
        name: encode(getattr(manager, name))
        for name in MANAGER_FIELDS
        if manager is not None and hasattr(manager, name)
    }
    parser = getattr(manager, "func_parser", None)
    state["predicate_state"] = {
        name: encode(getattr(parser, name))
        for name in PARSER_FIELDS
        if parser is not None and hasattr(parser, name)
    }
    if not hasattr(env, "_scoring_source_hashes"):
        env._scoring_source_hashes = source_hashes(native)
    return {
        "schema": SCHEMA,
        "score": score,
        "normalized_score": score / 100,
        "success": completed,
        "partial_credit_supported": partial,
        "native_steps": int(native.take_action_cnt[0]),
        "native_ended": bool(native.end_flag[0]),
        "task": getattr(env, "task_name", None),
        "source_hashes": env._scoring_source_hashes,
        "scorer_state": state,
        "protocol": "workspace-session-v1",
        "budget_matches_official": bool(getattr(env, "official_action_limit", False)),
        "native_action_limit": getattr(env, "native_step_limit", None)
        if getattr(env, "official_action_limit", False) else None,
        "note": (
            "Official task action limit; Score at the recorded stop under EmbodiedEvo sampling."
            if getattr(env, "official_action_limit", False)
            else "Native Score at the recorded stop; historical unlimited action budget."
        ),
    }
