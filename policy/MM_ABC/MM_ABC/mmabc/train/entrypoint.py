"""Training entrypoint. Launched once per rank by torchrun."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]


def _check_overrides(cfg, overrides: list[str]) -> None:
    """Reject an override that names a key the config does not have.

    ``from_dotlist`` happily invents keys, so a typo becomes a setting that
    silently does nothing: the run proceeds with the default and the log shows
    no sign of it. That is cheap to catch here and expensive to notice a day
    into a run.
    """
    unknown = []
    for item in overrides:
        key = item.split("=", 1)[0]
        if key.startswith("model_overrides."):
            # Validated against the model config when the trainer merges it.
            continue
        if OmegaConf.select(cfg, key, default=_MISSING) is _MISSING:
            unknown.append(key)
    if unknown:
        raise SystemExit(
            "unknown override key(s): "
            + ", ".join(unknown)
            + "\nAdd the key to the config file first, or fix the spelling."
        )


_MISSING = object()


def load_train_config(path: str | Path):
    """Load a train config, following a chain of ``defaults:`` parents."""
    path = Path(path)
    cfg = OmegaConf.load(str(path if path.is_absolute() else REPO_ROOT / path))
    parent = cfg.pop("defaults", None)
    if parent is None:
        return cfg
    return OmegaConf.merge(load_train_config(str(parent)), cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="path to a configs/train/*.yaml")
    ap.add_argument(
        "overrides", nargs="*", help="dotlist overrides, e.g. micro_batch_size=4 max_steps=100"
    )
    args = ap.parse_args()

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    cfg = load_train_config(args.config)
    if args.overrides:
        _check_overrides(cfg, args.overrides)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    from mmabc.train.trainer import Trainer

    Trainer(cfg, REPO_ROOT).train()

    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
