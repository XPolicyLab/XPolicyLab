from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from XPolicyLab.policy.Liber_0_lite.liber0.training.train import backward_batch


class Objective(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 2)

    def forward(self, inputs):
        loss = self.layer(inputs).square().mean()
        return loss, {'loss_video': loss.detach(), 'loss_action': loss.detach() * 0}


def worker(rank, path):
    dist.init_process_group('gloo', init_method=f'file://{path}', rank=rank, world_size=2)
    torch.manual_seed(14)
    model = Objective()
    reference = Objective()
    reference.load_state_dict(model.state_dict())
    wrapped = DistributedDataParallel(model)
    inputs = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 24
    for micro in range(2):
        local = inputs.reshape(2, 2, 2, 3)[micro, rank]
        backward_batch(wrapped, local, 2, synchronize=(micro == 1))
    expected, _ = reference(inputs)
    expected.backward()
    for actual, correct in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, correct.grad, atol=1e-7, rtol=1e-6)
    dist.destroy_process_group()


def test_two_rank_accumulation_matches_global_batch(tmp_path):
    mp.spawn(worker, args=(str(tmp_path / 'gloo'),), nprocs=2, join=True)
