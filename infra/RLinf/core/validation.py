"""Checks for contracts hidden by the external xpolicylab model name."""

from pathlib import Path
import subprocess

from .extensions import active_model, policy_extension, simulator_extension


def checked_rlinf_revision():
    """Verify the checkout against the gitlink recorded by XPolicyLab."""
    root = Path(__file__).resolve().parents[3]
    submodule = "infra/RLinf/RLinf"
    expected = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", f":{submodule}"], text=True
    ).strip()
    checkout = root / submodule
    if not (checkout / "rlinf" / "config.py").is_file():
        raise RuntimeError("Initialize the RLinf submodule with git submodule update --init.")
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise RuntimeError(f"Expected RLinf {expected} from the XPolicyLab gitlink, found {actual}.")
    return actual


def validate_adapter_config(cfg) -> None:
    model = active_model(cfg)
    policy = policy_extension(model)
    for mode in ("train", "eval"):
        env = cfg.env.get(mode)
        if env is None or (mode == "train" and cfg.runner.only_eval):
            continue
        for extension in (simulator_extension(env), policy):
            if hasattr(extension, "validate"):
                extension.validate(cfg, model, env, mode)
    rollout = cfg.get("rollout", {})
    if rollout.get("rollout_backend", rollout.get("backend", "huggingface")) != "huggingface":
        raise ValueError("XPolicyLab rollout requires the huggingface backend.")
