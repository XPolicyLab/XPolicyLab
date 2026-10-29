from collections.abc import Sequence
from math import ceil
from typing import Any, Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    GLOBAL_LAYOUT_VERSION = "global_randperm_v1"

    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        if self.batch_size <= 0 or self.num_processes <= 0:
            raise ValueError("Sampler batch size and process count must be positive.")
        self.layout_version = self.GLOBAL_LAYOUT_VERSION
        self.layout_fingerprint = None
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        if callable(getattr(self.dataset, "set_epoch", None)):
            self.dataset.set_epoch(self.epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def state_metadata(self) -> dict[str, Any]:
        return {
            "layout_version": self.layout_version,
            "layout_fingerprint": self.layout_fingerprint,
        }

    def validate_resume_metadata(self, metadata: dict[str, Any] | None) -> None:
        if metadata is None:
            return
        if not isinstance(metadata, dict):
            raise ValueError(
                "Training state is missing sampler layout metadata required by "
                f"{self.layout_version}."
            )
        expected = self.state_metadata()
        if metadata != expected:
            raise ValueError(
                "Sampler layout differs from the saved training state: "
                f"saved={metadata}, active={expected}."
            )

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if self.resume_batch_offset > 0:
            sample_offset = (
                self.resume_batch_offset * self.batch_size * self.num_processes
            )
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


class TaskBalancedEpochSampler(Sampler[int]):
    """Deterministic task-balanced stream aligned to optimizer-level batches."""

    LAYOUT_VERSION = "task_balanced_optimizer_batch_v1"

    def __init__(
        self,
        dataset: Sized,
        task_to_indices: dict[str, Sequence[int]],
        seed: int,
        batch_size: int,
        num_processes: int,
        gradient_accumulation_steps: int,
    ) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        if (
            min(
                self.batch_size,
                self.num_processes,
                self.gradient_accumulation_steps,
            )
            <= 0
        ):
            raise ValueError("Task-balanced sampler batch geometry must be positive.")
        if not isinstance(task_to_indices, dict) or not task_to_indices:
            raise ValueError("Task-balanced sampling requires non-empty task groups.")
        self.task_to_indices = {
            str(task): tuple(int(index) for index in indices)
            for task, indices in task_to_indices.items()
        }
        if any(not indices for indices in self.task_to_indices.values()):
            raise ValueError("Every task-balanced sampler group must be non-empty.")
        flattened = [
            index for indices in self.task_to_indices.values() for index in indices
        ]
        expected = list(range(len(self.dataset)))
        if sorted(flattened) != expected:
            raise ValueError(
                "Task-balanced sampler groups must cover every dataset index "
                "exactly once."
            )
        self.task_names = tuple(sorted(self.task_to_indices))
        self.global_batch_size = (
            self.batch_size * self.num_processes * self.gradient_accumulation_steps
        )
        self.epoch_num_samples = (
            ceil(len(self.dataset) / self.global_batch_size) * self.global_batch_size
        )
        self.optimizer_steps_per_epoch = (
            self.epoch_num_samples // self.global_batch_size
        )
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if callable(getattr(self.dataset, "set_epoch", None)):
            self.dataset.set_epoch(self.epoch)

    def set_epoch_offset(self, epoch_offset: int) -> None:
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int) -> None:
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self) -> None:
        self.resume_batch_offset = 0

    def state_metadata(self) -> dict[str, Any]:
        return {
            "layout_version": self.LAYOUT_VERSION,
            "seed": self.seed,
            "task_names": list(self.task_names),
            "task_sizes": {
                task: len(self.task_to_indices[task]) for task in self.task_names
            },
            "batch_size": self.batch_size,
            "num_processes": self.num_processes,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "global_batch_size": self.global_batch_size,
        }

    def validate_resume_metadata(self, metadata: dict[str, Any] | None) -> None:
        expected = self.state_metadata()
        if metadata != expected:
            raise ValueError(
                "Task-balanced sampler contract differs from saved state: "
                f"saved={metadata}, active={expected}."
            )

    def _optimizer_batch_indices(self, global_step: int) -> list[int]:
        num_tasks = len(self.task_names)
        slots_per_task, remainder = divmod(self.global_batch_size, num_tasks)
        task_slots: list[str] = []
        for task in self.task_names:
            task_slots.extend([task] * slots_per_task)
        remainder_start = int(global_step) % num_tasks
        for offset in range(remainder):
            task_slots.append(self.task_names[(remainder_start + offset) % num_tasks])

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + 1_000_003 * int(global_step))
        task_order = torch.randperm(len(task_slots), generator=generator).tolist()
        indices: list[int] = []
        for slot in task_order:
            task = task_slots[slot]
            candidates = self.task_to_indices[task]
            candidate_slot = int(
                torch.randint(
                    low=0,
                    high=len(candidates),
                    size=(1,),
                    generator=generator,
                ).item()
            )
            indices.append(candidates[candidate_slot])
        return indices

    def __iter__(self) -> Iterator[int]:
        epoch_identity = self.epoch + self.epoch_offset
        first_global_step = epoch_identity * self.optimizer_steps_per_epoch
        indices: list[int] = []
        for optimizer_batch in range(self.optimizer_steps_per_epoch):
            indices.extend(
                self._optimizer_batch_indices(first_global_step + optimizer_batch)
            )
        if self.resume_batch_offset > 0:
            sample_offset = (
                self.resume_batch_offset * self.batch_size * self.num_processes
            )
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return self.epoch_num_samples
