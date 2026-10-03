"""Public training configuration for the released RoboTwin recipe."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Topology = Literal["full_joint", "h_bridge"]


@dataclass(frozen=True)
class TactileConfig:
    checkpoint: Path
    slot_ids: tuple[int, ...] = tuple(range(8)) + tuple(range(17, 25))
    latent_dim: int = 48
    frame_count: int = 18
    queries_per_frame: int = 16
    observed_frames: int = 2
    hidden_size: int = 1024

    def __post_init__(self) -> None:
        if (self.latent_dim, self.frame_count, self.queries_per_frame, self.observed_frames, self.hidden_size) != (48, 18, 16, 2, 1024):
            raise ValueError("The public recipe uses the unified [18,16,48] tactile contract")


@dataclass(frozen=True)
class DatasetConfig:
    clean_root: Path
    text_cache: Path
    quality_manifest: Path
    episodes_per_task: int = 50
    samples_per_episode: int = 10
    task_count: int = 50


@dataclass(frozen=True)
class TrainingConfig:
    dataset: DatasetConfig
    tactile: TactileConfig
    initialization_checkpoint: Path
    output_dir: Path
    topology: Topology = "full_joint"
    h_bridge_joint_start_layer: int = 8
    h_bridge_joint_end_layer: int = 22
    batch_size: int = 4
    world_size: int = 32
    max_steps: int = 50_000
    learning_rate: float = 5.0e-5
    warmup_steps: int = 1_000
    save_interval: int = 10_000
