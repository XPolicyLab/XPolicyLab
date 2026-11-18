from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch
from torch import Tensor


@dataclass(frozen=True)
class Wan22DistributedConfig:
    """Explicit multi-rank configuration for the Wan2.2 inference path.

    ``backend="none"`` is the legacy single-rank path. Merely launching with
    torchrun never enables distributed WAM execution.
    """

    backend: Literal["none", "torch"] = "none"
    image_parallel_size: int = 1
    coordinator_rank: int = 0
    mesh_dim_name: str = "image"

    def __post_init__(self) -> None:
        if self.image_parallel_size <= 0:
            raise ValueError("image_parallel_size must be positive")
        if not 0 <= self.coordinator_rank < self.image_parallel_size:
            raise ValueError("coordinator_rank must be in the image-parallel mesh")
        if not self.mesh_dim_name:
            raise ValueError("mesh_dim_name must not be empty")
        if self.backend == "none" and self.image_parallel_size != 1:
            raise ValueError(
                "multi-rank WAM requires an explicitly configured distributed backend"
            )

    @property
    def enabled(self) -> bool:
        return self.backend != "none"


@dataclass(frozen=True)
class Wan22DeviceMesh:
    """Dependency-light description of the one-dimensional image mesh."""

    ranks: tuple[int, ...]
    dim_name: str = "image"

    def __post_init__(self) -> None:
        if not self.ranks:
            raise ValueError("device mesh must contain at least one rank")
        if len(set(self.ranks)) != len(self.ranks):
            raise ValueError("device mesh ranks must be unique")

    def local_rank(self, global_rank: int) -> int:
        try:
            return self.ranks.index(global_rank)
        except ValueError as error:
            raise RuntimeError(f"rank {global_rank} is not in the WAM device mesh") from error


@runtime_checkable
class Wan22Collective(Protocol):
    """Small collective surface used by WAM; suitable for CPU fakes."""

    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    @property
    def backend(self) -> str: ...

    def broadcast(self, tensor: Tensor, *, src: int) -> Tensor: ...

    def verify_step(self, step: int, tag: str) -> None: ...

    def barrier(self) -> None: ...


@runtime_checkable
class Wan22ImageParallelProtocol(Protocol):
    """Backend hook around each core transformer block.

    A production backend may shard image tokens before ``run_block`` and
    exchange sequence/head partitions inside it. The built-in implementation
    intentionally keeps tensors replicated and exists for CPU/gloo validation.
    """

    def run_block(
        self,
        *,
        block_index: int,
        hidden_states: Tensor,
        operation: Callable[[Tensor], tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor]: ...


class ReplicatedImageParallel:
    """Correctness/testing backend; performs no model or token sharding."""

    def __init__(self, context: Wan22DistributedContext) -> None:
        self._context = context

    def run_block(
        self,
        *,
        block_index: int,
        hidden_states: Tensor,
        operation: Callable[[Tensor], tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor]:
        self._context.coordinate(f"core.block.{block_index}.enter")
        result = operation(hidden_states)
        self._context.coordinate(f"core.block.{block_index}.exit")
        return result


class TorchDistributedCollective:
    """torch.distributed process-group adapter without CUDA assumptions."""

    def __init__(self, process_group: object | None = None) -> None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "Wan22 torch backend requires an initialized torch.distributed process group"
            )
        self.process_group = process_group

    @property
    def rank(self) -> int:
        return torch.distributed.get_rank(self.process_group)

    @property
    def world_size(self) -> int:
        return torch.distributed.get_world_size(self.process_group)

    @property
    def backend(self) -> str:
        return str(torch.distributed.get_backend(self.process_group))

    def broadcast(self, tensor: Tensor, *, src: int) -> Tensor:
        global_src = src
        if self.process_group is not None:
            get_global_rank = getattr(torch.distributed, "get_global_rank", None)
            if callable(get_global_rank):
                global_src = get_global_rank(self.process_group, src)
            else:
                get_process_group_ranks = getattr(
                    torch.distributed, "get_process_group_ranks", None
                )
                if not callable(get_process_group_ranks):
                    raise RuntimeError(
                        "This torch version cannot translate a process-group "
                        "rank to a global broadcast source rank"
                    )
                ranks = get_process_group_ranks(self.process_group)
                try:
                    global_src = ranks[src]
                except IndexError as exc:
                    raise ValueError(
                        f"broadcast source rank {src} is outside the process group"
                    ) from exc
        torch.distributed.broadcast(
            tensor, src=global_src, group=self.process_group
        )
        return tensor

    def verify_step(self, step: int, tag: str) -> None:
        # Stable FNV-1a avoids Python's process-randomized hash().
        fingerprint = 2166136261
        for byte in tag.encode("utf-8"):
            fingerprint = ((fingerprint ^ byte) * 16777619) & 0x7FFFFFFF
        device = torch.device("cpu")
        if self.backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL collective validation requires CUDA")
            device = torch.device("cuda", torch.cuda.current_device())
        value = torch.tensor([step, fingerprint], dtype=torch.int64, device=device)
        minimum = value.clone()
        maximum = value.clone()
        torch.distributed.all_reduce(
            minimum, op=torch.distributed.ReduceOp.MIN, group=self.process_group
        )
        torch.distributed.all_reduce(
            maximum, op=torch.distributed.ReduceOp.MAX, group=self.process_group
        )
        if not torch.equal(minimum, maximum):
            raise RuntimeError(
                f"Wan22 collective order diverged at step {step} ({tag!r})"
            )

    def barrier(self) -> None:
        torch.distributed.barrier(group=self.process_group)


class Wan22DistributedContext:
    """Validated rank, mesh, ownership, and collective ordering state."""

    def __init__(
        self,
        config: Wan22DistributedConfig | None = None,
        *,
        collective: Wan22Collective | None = None,
        image_parallel: Wan22ImageParallelProtocol | None = None,
        replicated_for_testing: bool = False,
    ) -> None:
        self.config = config or Wan22DistributedConfig()
        if self.config.enabled and collective is None:
            raise RuntimeError(
                "configured multi-rank WAM backend requires a collective adapter"
            )
        if not self.config.enabled and collective is not None:
            raise RuntimeError("collective adapter requires an enabled WAM backend")
        self.collective = collective
        self.mesh = Wan22DeviceMesh(
            tuple(range(self.config.image_parallel_size)),
            self.config.mesh_dim_name,
        )
        if collective is not None:
            if collective.world_size != self.config.image_parallel_size:
                raise RuntimeError(
                    "Wan22 image_parallel_size does not match process-group world size: "
                    f"{self.config.image_parallel_size} != {collective.world_size}"
                )
            self.mesh.local_rank(collective.rank)
        self._step = 0
        self.image_parallel = image_parallel
        if replicated_for_testing and self.image_parallel is None:
            self.image_parallel = ReplicatedImageParallel(self)
        if self.world_size > 1 and self.image_parallel is None:
            raise RuntimeError(
                "multi-rank WAM requires an explicit image-parallel protocol"
            )

    @classmethod
    def single_rank(cls) -> Wan22DistributedContext:
        return cls()

    @classmethod
    def from_torch(
        cls,
        config: Wan22DistributedConfig,
        *,
        process_group: object | None = None,
        replicated_for_testing: bool = False,
    ) -> Wan22DistributedContext:
        collective = TorchDistributedCollective(process_group)
        return cls(
            config,
            collective=collective,
            replicated_for_testing=replicated_for_testing,
        )

    @property
    def rank(self) -> int:
        return 0 if self.collective is None else self.collective.rank

    @property
    def world_size(self) -> int:
        return 1 if self.collective is None else self.collective.world_size

    @property
    def is_coordinator(self) -> bool:
        return self.rank == self.config.coordinator_rank

    def coordinate(self, tag: str) -> None:
        if self.collective is not None:
            self.collective.verify_step(self._step, tag)
        self._step += 1

    def reset_episode(self) -> None:
        self.coordinate("episode.reset")
        if self.collective is not None:
            self.collective.barrier()
        self._step = 0

    def broadcast(self, tensor: Tensor, *, tag: str) -> Tensor:
        self.coordinate(tag)
        if self.collective is not None:
            return self.collective.broadcast(
                tensor, src=self.config.coordinator_rank
            )
        return tensor

    def owned_tensor(
        self,
        factory: Callable[[], Tensor],
        *,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        tag: str,
    ) -> Tensor:
        value = (
            factory()
            if self.is_coordinator
            else torch.empty(shape, dtype=dtype, device=device)
        )
        if tuple(value.shape) != shape:
            raise RuntimeError(
                f"coordinator produced unexpected shape for {tag}: "
                f"{tuple(value.shape)} != {shape}"
            )
        return self.broadcast(value, tag=tag)

    def validate_state_owner(
        self, owner_rank: int | None, state_world_size: int | None
    ) -> None:
        if owner_rank is not None and owner_rank != self.rank:
            raise RuntimeError(
                f"Wan22 cache owned by rank {owner_rank}, used on rank {self.rank}"
            )
        if state_world_size is not None and state_world_size != self.world_size:
            raise RuntimeError(
                "Wan22 cache world size changed: "
                f"{state_world_size} != {self.world_size}"
            )


def default_distributed_context(
    config: Wan22DistributedConfig | None,
    *,
    image_parallel: Wan22ImageParallelProtocol | None = None,
) -> Wan22DistributedContext:
    config = config or Wan22DistributedConfig()
    if not config.enabled:
        return Wan22DistributedContext(config)
    collective = TorchDistributedCollective()
    return Wan22DistributedContext(
        config,
        collective=collective,
        image_parallel=image_parallel,
    )
