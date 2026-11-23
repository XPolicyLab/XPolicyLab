"""Structural checks for policy responses that are ready to execute."""

import numpy as np


def _schema_for_env(env):
    dims = getattr(env, "robot_action_dim_info", None)
    if not isinstance(dims, dict):
        return None
    cfg = getattr(env, "deploy_cfg", {}) or {}
    # These embodiments include base and body fields beyond arm/ee dimensions.
    if str(cfg.get("env_cfg_type", "")).lower() in {"m92uw", "mobile_m92uw", "mobile"}:
        return None
    action_type = getattr(env, "action_type", None) or cfg.get("action_type") or "joint"
    from XPolicyLab.utils.process_data import get_robot_state_schema
    return get_robot_state_schema(action_type, dims)


def _finite_numeric(value):
    array = np.asarray(value)
    return array.size > 0 and array.dtype.kind in "iuf" and np.isfinite(array).all()


def validate_action_chunk(actions, env=None):
    """Accept a non-empty sequence of control steps."""
    schema = _schema_for_env(env)
    if isinstance(actions, np.ndarray):
        if actions.ndim != 2 or not all(actions.shape) or not _finite_numeric(actions):
            raise ValueError("Expected a non-empty numeric (steps, action_dim) action chunk")
        if schema and actions.shape[-1] != sum(schema.values()):
            raise ValueError("Packed action chunk width does not match robot dimensions")
        return
    if not isinstance(actions, (list, tuple)) or not actions:
        raise ValueError("Expected a non-empty action chunk")
    for step in actions:
        if isinstance(step, dict) and step:
            if schema:
                _validate_robot_fields(step, env.robot_action_dim_info, schema)
            else:
                if not all(_finite_numeric(value) for value in step.values()):
                    raise ValueError("Action chunk contains an empty, nonnumeric or nonfinite field")
                validate_dict = getattr(env, "validate_action_dict", None)
                if callable(validate_dict):
                    validate_dict(step)
            continue
        if isinstance(step, (list, tuple, np.ndarray)):
            vector = np.asarray(step)
            if vector.ndim == 1 and _finite_numeric(vector):
                if schema and vector.size != sum(schema.values()):
                    raise ValueError("Packed action chunk width does not match robot dimensions")
                continue
        raise ValueError("Action chunk contains an invalid control step")


def validate_action_batch(chunks, num_envs, env=None):
    """Validate all chunk lengths before any environment advances. Return the horizon."""
    if isinstance(chunks, np.ndarray):
        if (chunks.ndim != 3 or chunks.shape[0] != num_envs
                or not all(chunks.shape) or chunks.dtype.kind not in "iuf"):
            raise ValueError("Expected a numeric (active_envs, steps, action_dim) action batch")
        schema = _schema_for_env(env)
        if not np.isfinite(chunks).all() or (schema and chunks.shape[-1] != sum(schema.values())):
            raise ValueError("Invalid packed action batch values or robot dimensions")
        return chunks.shape[1]
    if not isinstance(chunks, (list, tuple)) or not chunks or len(chunks) != num_envs:
        raise ValueError(f"Expected one action chunk per active environment ({num_envs})")
    for chunk in chunks:
        validate_action_chunk(chunk, env)
    horizon = len(chunks[0])
    if any(len(chunk) != horizon for chunk in chunks):
        raise ValueError("Batched action chunks must have the same length")
    return horizon


def validate_robot_state_dict(state_dict: dict, robot_action_dim_info: dict, action_type: str = "joint") -> None:
    """
    Validate complete action fields, dimensions and finite numeric values.

    Args:
        state_dict: Action dictionary for one environment.
        robot_action_dim_info: {
            "arm_dim": [...],
            "ee_dim": [...],
        }
        action_type: 'joint' or 'ee'.

    Raises:
        KeyError: if required keys are missing
        ValueError: if unexpected keys, wrong dimensions or non-finite values are found
        TypeError: if values are not real numeric arrays or sequences
    """
    from XPolicyLab.utils.process_data import get_robot_state_schema
    expected = get_robot_state_schema(action_type, robot_action_dim_info)
    _validate_robot_fields(state_dict, robot_action_dim_info, expected)


def _validate_robot_fields(state_dict, robot_action_dim_info, expected):
    from XPolicyLab.utils.process_data import normalize_robot_state_keys
    state_dict = normalize_robot_state_keys(state_dict, robot_action_dim_info)
    missing_keys = [key for key in expected if key not in state_dict]
    if missing_keys:
        raise KeyError(f"Missing action keys: {missing_keys}")
    unexpected_keys = [k for k in state_dict if k not in expected]

    if unexpected_keys:
        raise ValueError(f"Unexpected state keys: {unexpected_keys}")

    for key, expected_dim in expected.items():
        value = state_dict[key]

        if not isinstance(value, (np.ndarray, list, tuple)):
            raise TypeError(
                f"state_dict['{key}'] must be array-like, got {type(value)}"
            )

        arr = np.asarray(value)
        if arr.dtype.kind not in "iuf":
            raise TypeError(f"state_dict['{key}'] must contain real numbers, got {arr.dtype}")
        if not np.isfinite(arr).all():
            raise ValueError(f"state_dict['{key}'] contains NaN or infinity")

        if arr.ndim != 1:
            raise ValueError(
                f"state_dict['{key}'] must be 1D, got shape {arr.shape}"
            )

        if arr.shape[0] != expected_dim:
            raise ValueError(
                f"state_dict['{key}'] dim mismatch: expected {expected_dim}, "
                f"got shape {arr.shape}"
            )
