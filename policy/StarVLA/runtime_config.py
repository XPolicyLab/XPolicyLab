from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_AUTO_VALUES = {None, "", "auto", "none", "null"}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def _checkpoint_run_dir(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path).expanduser()
    if path.is_dir():
        return path.parent if path.name in {"checkpoints", "final_model"} else path
    if path.parent.name in {"checkpoints", "final_model"}:
        return path.parent.parent
    return path.parent


def _read_include_state(config_path: Path) -> bool | None:
    if not config_path.is_file():
        return None
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    try:
        value = config["datasets"]["vla_data"]["include_state"]
    except (KeyError, TypeError):
        return None
    return parse_bool(value)


def resolve_include_state(value: Any, checkpoint_path: str | Path | None) -> bool:
    """Resolve state input from an override or checkpoint-side configuration."""

    normalized = value.strip().lower() if isinstance(value, str) else value
    if normalized not in _AUTO_VALUES:
        return parse_bool(value)
    if checkpoint_path in (None, "", "null", "None"):
        return False

    run_dir = _checkpoint_run_dir(checkpoint_path)
    for name in ("config.yaml", "config.full.yaml"):
        include_state = _read_include_state(run_dir / name)
        if include_state is not None:
            return include_state
    return False


def resolve_checkpoint_framework(checkpoint_path: str | Path | None) -> str | None:
    if checkpoint_path in (None, "", "null", "None"):
        return None

    run_dir = _checkpoint_run_dir(checkpoint_path)
    for name in ("config.yaml", "config.full.yaml"):
        config_path = run_dir / name
        if not config_path.is_file():
            continue
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        framework = config.get("framework", {})
        if isinstance(framework, dict) and framework.get("name"):
            return str(framework["name"])
    return None


def validate_server_runtime_contract(
    metadata: Any,
    *,
    include_state: bool,
    action_dim: int,
    unnorm_key: str | None,
    expected_framework: str | None = None,
    expected_pi_v3_forward: str | None = None,
) -> dict[str, Any]:
    """Reject StarVLA servers that silently use a different inference contract.

    The public upstream server predates RoboDojo server-side state
    normalization.  It accepts the same request shape, so connecting to it can
    produce plausible-looking but invalid robot actions instead of a clear
    error.  Released RoboDojo checkpoints must use the XPolicyLab vendored
    server, which advertises this contract at websocket handshake time.
    """

    if not isinstance(metadata, dict):
        raise ValueError(f"StarVLA server metadata must be a mapping, got {type(metadata).__name__}.")
    contract = metadata.get("runtime_contract")
    if not isinstance(contract, dict):
        raise ValueError(
            "StarVLA server does not advertise runtime_contract. Use "
            "policy/StarVLA/eval.sh or scripts/eval_hf_robodojo.sh so XPolicyLab starts "
            "its vendored server; do not start deployment/model_server/server_policy.py "
            "from an arbitrary upstream StarVLA checkout."
        )

    expected = {
        "version": 1,
        "image_color_order": "rgb",
        "state_input": "raw_env",
        "state_normalization": "training_transform",
        "action_output": "unnormalized_env",
        "action_dim": int(action_dim),
    }
    for key, expected_value in expected.items():
        actual = contract.get(key)
        if actual != expected_value:
            raise ValueError(
                f"Incompatible StarVLA runtime contract: {key}={actual!r}, "
                f"expected {expected_value!r}."
            )

    if expected_framework is not None:
        actual_framework = metadata.get("framework")
        if actual_framework != expected_framework:
            raise ValueError(
                f"Incompatible StarVLA framework: server={actual_framework!r}, "
                f"checkpoint={expected_framework!r}."
            )

    if expected_pi_v3_forward is not None:
        if expected_framework != "QwenPI_v3":
            raise ValueError(
                "A PI-v3 forward requirement can only be attached to a QwenPI_v3 checkpoint."
            )
        actual_forward = contract.get("pi_v3_forward")
        if actual_forward != expected_pi_v3_forward:
            raise ValueError(
                "Incompatible PI-v3 forward contract: "
                f"server={actual_forward!r}, expected={expected_pi_v3_forward!r}."
            )

    if include_state and contract.get("state_normalization") != "training_transform":
        raise ValueError("include_state=true requires server-side training-stat state normalization.")

    available_keys = metadata.get("available_unnorm_keys", [])
    if unnorm_key is not None and unnorm_key not in available_keys:
        raise ValueError(
            f"StarVLA server does not provide unnorm_key={unnorm_key!r}; "
            f"available keys: {available_keys!r}."
        )
    return contract
