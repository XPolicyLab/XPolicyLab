"""Phoenix_WAM checkpoint directories use XPolicyLab's shared resolver."""
from pathlib import Path
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root

POLICY_DIR = Path(__file__).resolve().parent


def resolve_policy_checkpoint(config):
    root = resolve_checkpoint_root(
        config, POLICY_DIR / "checkpoints", policy_dir=POLICY_DIR,
        explicit_keys=("checkpoint_path", "ckpt_path", "model_path"), must_exist=False,
    )
    if not root.is_dir():
        raise FileNotFoundError(f"Phoenix_WAM requires a weights directory: {root}")
    return root
