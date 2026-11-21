"""Model-construction config for inference.

This is the inference-only subset of Falcon's trainer config: a training
snapshot is loaded as plain data and only the sections needed to rebuild the
model are kept. Optimizer, checkpoint, parallelism, logging and other
training-launcher settings in the snapshot are ignored.
"""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any

from omegaconf import DictConfig

from .lazy import LazyConfig


@dataclass
class TrainerConfig:
    """Model architecture and noise-schedule settings read from a snapshot.

    Both fields are ``DictConfig``, so they accept whatever keys the snapshot
    defines. Unknown top-level sections are dropped by
    :func:`dict_to_dataclass`.
    """

    model: DictConfig = field(default_factory=DictConfig)
    noise_scheduler: DictConfig = field(default_factory=DictConfig)

    @staticmethod
    def from_lazy_config(cfg: LazyConfig) -> "TrainerConfig":
        return dict_to_dataclass(TrainerConfig, cfg)

    def to_dict_config(self) -> DictConfig:
        return DictConfig(asdict(self))


def dict_to_dataclass(cls: Any, data: DictConfig) -> Any:
    """Recursively convert a LazyConfig to a dataclass, keeping known fields.

    Args:
        cls (class): The dataclass to convert to.
        data (DictConfig): The LazyConfig to convert.

    Returns:
        The converted dataclass.
    """
    if not is_dataclass(cls):
        return data

    result = {}
    for f in fields(cls):
        if f.name in data:
            value = data.pop(f.name)
            if is_dataclass(f.type) and isinstance(value, DictConfig):
                result[f.name] = dict_to_dataclass(f.type, value)
            else:
                result[f.name] = value
        elif f.type is DictConfig:
            result[f.name] = DictConfig({})

    return cls(**result)
