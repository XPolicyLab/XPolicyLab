"""Bounded weighted sampling with the reference count/weight-balanced pools."""
import hashlib
import math

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def balanced_partitions(weights, world):
    # Same descending blocks and lightest-rank assignment as the reference
    # build_count_balanced_weight_partitions; stable sorts resolve ties.
    if world < 1 or len(weights) < world or torch.count_nonzero(weights) < world:
        raise ValueError('Balanced sampling requires at least one positive-weight item per rank')
    partitions = torch.full((world, math.ceil(len(weights) / world)), -1, dtype=torch.int64)
    mass = torch.zeros(world, dtype=torch.float64)
    order = torch.argsort(weights, descending=True, stable=True)
    for offset in range(0, len(weights), world):
        indices = order[offset:offset + world]
        ranks = torch.argsort(mass, stable=True)[:len(indices)]
        partitions[ranks, offset // world] = indices
        mass[ranks] += weights[indices]
    return partitions, mass


class DistributedWeightedSampler(Sampler):
    """Infinite stream; memory is bounded by dataset size and one draw chunk.

    Every rank has the same whole-batch epoch length. Balanced mode uses
    disjoint reference pools; unpartitioned mode shards a shared draw stream.
    RNG sequences are deterministic here, not bitwise Forge sampler replay.
    """
    def __init__(self, weights, batch_size, *, rank, world, seed, balanced=True, chunk_size=65536):
        self.weights = torch.as_tensor(weights, dtype=torch.float64, device='cpu')
        if (self.weights.ndim != 1 or len(self.weights) == 0
                or not torch.isfinite(self.weights).all() or (self.weights < 0).any()
                or self.weights.sum() <= 0):
            raise ValueError('Sampling weights must be a nonempty finite nonnegative vector with positive sum')
        if batch_size < 1 or world < 1 or not 0 <= rank < world or chunk_size < 1:
            raise ValueError('Invalid distributed sampler dimensions')
        self.batch_size, self.rank, self.world = batch_size, rank, world
        self.epoch_samples = (len(self.weights) // (batch_size * world)) * batch_size
        if self.epoch_samples == 0:
            raise ValueError('Dataset must contain at least one full global batch; reduce batch size')
        self.balanced = balanced and world > 1
        self.indices = torch.arange(len(self.weights))
        if self.balanced:
            partitions, _ = balanced_partitions(self.weights, world)
            self.indices = partitions[rank][partitions[rank] >= 0]
            self.weights = self.weights[self.indices]
        self.seed = ((seed + (rank + 1) * 0x9E3779B97F4A7C15) % ((1 << 63) - 1)
                     if self.balanced else seed)
        self.chunk_size = chunk_size

    def __len__(self):
        return self.epoch_samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        while True:
            count = self.epoch_samples if self.balanced else self.epoch_samples * self.world
            offset = 0
            while offset < count:
                take = min(self.chunk_size, count - offset)
                draws = torch.multinomial(self.weights, take, replacement=True, generator=generator)
                for position, index in enumerate(draws):
                    if self.balanced or ((offset + position) // self.batch_size) % self.world == self.rank:
                        yield int(self.indices[index])
                offset += take


def check_dataset_agreement(weights, control_group):
    """Refuse divergent rank-local data instead of silently training different mixtures."""
    array = torch.as_tensor(weights, dtype=torch.float64).cpu().numpy()
    identity = (array.shape, hashlib.sha256(array.tobytes()).hexdigest())
    identities = [None] * dist.get_world_size()
    dist.all_gather_object(identities, identity, group=control_group)
    if any(other != identity for other in identities):
        raise ValueError('Dataset sampling weights differ between ranks; check shared data/configuration')
