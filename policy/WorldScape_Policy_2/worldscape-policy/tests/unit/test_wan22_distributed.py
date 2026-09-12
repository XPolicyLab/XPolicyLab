from __future__ import annotations

import pytest
import torch

from worldscape_policy.cli.serve import ensure_single_rank
from worldscape_policy.types import WAMInferenceState
from worldscape_policy.wam.wan22 import (
    TorchDistributedCollective,
    Wan22DeviceMesh,
    Wan22DistributedConfig,
    Wan22DistributedContext,
)


class FakeCollective:
    backend = "fake"

    def __init__(self, *, rank: int = 0, world_size: int = 2) -> None:
        self._rank = rank
        self._world_size = world_size
        self.steps = []
        self.broadcasts = []

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def broadcast(self, tensor, *, src):
        self.broadcasts.append((src, tensor.clone()))
        return tensor

    def verify_step(self, step, tag):
        self.steps.append((step, tag))

    def barrier(self):
        self.steps.append(("barrier",))


class FakeImageParallel:
    def __init__(self):
        self.blocks = []

    def run_block(self, *, block_index, hidden_states, operation):
        self.blocks.append(block_index)
        return operation(hidden_states)


def _context(*, rank=0, world_size=2):
    return Wan22DistributedContext(
        Wan22DistributedConfig(
            backend="torch",
            image_parallel_size=world_size,
        ),
        collective=FakeCollective(rank=rank, world_size=world_size),
        image_parallel=FakeImageParallel(),
    )


def test_multi_rank_requires_explicit_backend_and_matching_group():
    with pytest.raises(ValueError, match="explicitly configured"):
        Wan22DistributedConfig(backend="none", image_parallel_size=2)

    with pytest.raises(RuntimeError, match="does not match"):
        Wan22DistributedContext(
            Wan22DistributedConfig(backend="torch", image_parallel_size=2),
            collective=FakeCollective(world_size=3),
            image_parallel=FakeImageParallel(),
        )

    with pytest.raises(RuntimeError, match="image-parallel protocol"):
        Wan22DistributedContext(
            Wan22DistributedConfig(backend="torch", image_parallel_size=2),
            collective=FakeCollective(),
        )


def test_mesh_and_cache_ownership_are_rank_aware():
    mesh = Wan22DeviceMesh((2, 4), "image")
    assert mesh.local_rank(4) == 1
    with pytest.raises(RuntimeError, match="not in"):
        mesh.local_rank(3)

    context = _context(rank=1)
    context.validate_state_owner(1, 2)
    with pytest.raises(RuntimeError, match="owned by rank 0"):
        context.validate_state_owner(0, 2)
    with pytest.raises(RuntimeError, match="world size changed"):
        context.validate_state_owner(1, 4)


def test_collective_order_and_coordinator_noise_are_explicit():
    context = _context()
    value = context.owned_tensor(
        lambda: torch.arange(4),
        shape=(4,),
        dtype=torch.int64,
        device=torch.device("cpu"),
        tag="noise",
    )
    torch.testing.assert_close(value, torch.arange(4))
    assert context.collective.steps == [(0, "noise")]
    assert context.collective.broadcasts[0][0] == 0

    state = WAMInferenceState(cache_owner_rank=context.rank, cache_world_size=2)
    context.validate_state_owner(state.cache_owner_rank, state.cache_world_size)
    context.reset_episode()
    context.coordinate("next")
    assert context.collective.steps[-3:] == [
        (1, "episode.reset"),
        ("barrier",),
        (0, "next"),
    ]


def test_replicated_cpu_protocol_is_opt_in_only():
    collective = FakeCollective()
    context = Wan22DistributedContext(
        Wan22DistributedConfig(backend="torch", image_parallel_size=2),
        collective=collective,
        replicated_for_testing=True,
    )
    output, cache = context.image_parallel.run_block(
        block_index=3,
        hidden_states=torch.tensor([2.0]),
        operation=lambda value: (value + 1, value + 2),
    )
    torch.testing.assert_close(output, torch.tensor([3.0]))
    torch.testing.assert_close(cache, torch.tensor([4.0]))
    assert collective.steps == [
        (0, "core.block.3.enter"),
        (1, "core.block.3.exit"),
    ]


def test_server_rejects_distributed_serving_even_with_wam_backend(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    with pytest.raises(RuntimeError, match="coordinator/worker"):
        ensure_single_rank(_context())
    with pytest.raises(RuntimeError, match="coordinator/worker"):
        ensure_single_rank()


@pytest.mark.parametrize("use_global_rank_api", [True, False])
def test_subgroup_broadcast_translates_relative_source_to_global(
    monkeypatch, use_global_rank_api
):
    group = object()
    collective = TorchDistributedCollective.__new__(TorchDistributedCollective)
    collective.process_group = group
    calls = []

    if use_global_rank_api:
        monkeypatch.setattr(
            torch.distributed,
            "get_global_rank",
            lambda actual_group, rank: 4
            if actual_group is group and rank == 1
            else pytest.fail("unexpected rank translation"),
        )
    else:
        monkeypatch.setattr(torch.distributed, "get_global_rank", None)
        monkeypatch.setattr(
            torch.distributed,
            "get_process_group_ranks",
            lambda actual_group: [2, 4]
            if actual_group is group
            else pytest.fail("unexpected process group"),
        )
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda tensor, *, src, group: calls.append((tensor, src, group)),
    )

    tensor = torch.tensor([1])
    assert collective.broadcast(tensor, src=1) is tensor
    assert calls == [(tensor, 4, group)]
