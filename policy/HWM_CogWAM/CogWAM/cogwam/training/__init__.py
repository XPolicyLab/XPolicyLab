"""Training loop, optimizer schedule and checkpoint handling.

Imports are lazy: ``cogwam.training.trainer`` pulls in wandb, DeepSpeed and the
data layer, which tooling that only needs the config helpers should not pay for.
"""

from cogwam.training.config_tracker import AccessTrackedConfig, wrap_config
from cogwam.training.utils import TrainerUtils, normalize_dotlist_args

__all__ = [
    "AccessTrackedConfig",
    "CogWAMTrainer",
    "TrainerUtils",
    "normalize_dotlist_args",
    "setup_optimizer_and_scheduler",
    "wrap_config",
]


def __getattr__(name: str):
    if name == "CogWAMTrainer":
        from cogwam.training.trainer import CogWAMTrainer

        return CogWAMTrainer
    if name == "setup_optimizer_and_scheduler":
        from cogwam.training.optim import setup_optimizer_and_scheduler

        return setup_optimizer_and_scheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
