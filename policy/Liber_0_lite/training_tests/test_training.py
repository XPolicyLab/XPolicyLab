import torch
import pytest
from torch import nn

from XPolicyLab.policy.Liber_0_lite.liber0.backbone import LiberModel
from XPolicyLab.policy.Liber_0_lite.liber0.training import TrainingModel
from XPolicyLab.policy.Liber_0_lite.liber0.training.scheduler import TrainingScheduler


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.1))
        self.device = torch.device('cpu')
        self.sample_dtype = torch.float32
        self.patch_size, self.noise_scale, self.timestep_eps = 16, 8., 1e-3

    _patchify = staticmethod(LiberModel._patchify)

    def _joint_forward_single(self, prompt, reference, height, width, state,
                              video, video_time, action, action_time):
        return video * self.weight, action * self.weight, None


def sample(padded=False):
    return dict(video=torch.zeros(1, 3, 2, 16, 16), action=torch.zeros(1, 32, 14),
                proprio=torch.zeros(1, 32, 14), prompt=['Move the cup.'],
                action_is_pad=torch.full((1, 32), padded))


def test_backward_and_optimizer_step():
    torch.manual_seed(42)
    model = ToyModel()
    training = TrainingModel(model)
    optimizer = torch.optim.AdamW(training.parameters(), lr=1e-4)
    before = model.weight.detach().clone()
    loss, metrics = training(sample())
    assert torch.isfinite(loss)
    assert torch.equal(loss.detach(), metrics['loss_video'] + metrics['loss_action'])
    loss.backward()
    assert torch.isfinite(model.weight.grad) and model.weight.grad != 0
    optimizer.step()
    assert not torch.equal(before, model.weight.detach())


def test_padding_removes_action_loss():
    loss, metrics = TrainingModel(ToyModel())(sample(padded=True))
    assert metrics['loss_action'] == 0
    assert torch.equal(loss.detach(), metrics['loss_video'])


def test_prompt_count():
    inputs = sample()
    inputs['prompt'] = []
    with pytest.raises(ValueError, match='prompt'):
        TrainingModel(ToyModel())(inputs)


@pytest.mark.parametrize('shift', [1., 5.])
def test_training_schedule(shift):
    schedule = TrainingScheduler(1000, shift)
    clean, noise = torch.zeros(2, 32, 14), torch.ones(2, 32, 14)
    time = torch.tensor([0., 1000.])
    output = schedule.add_noise(clean, noise, time)
    assert torch.equal(output[0], clean[0]) and torch.equal(output[1], noise[1])
    assert torch.equal(schedule.training_target(clean, noise, time), noise - clean)
    assert torch.isfinite(schedule.training_weight(time)).all()
