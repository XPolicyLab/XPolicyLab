import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from XPolicyLab.policy.Liber_0_lite.liber0.training.state import build_optimizer, build_scheduler, capture_rng, restore_checkpoint, save_checkpoint

CONFIG = dict(max_steps=10, warmup_steps=2, learning_rate=1e-4, min_learning_rate=1e-6, weight_decay=.01)


def objects():
    model = SimpleNamespace(dit=torch.nn.Linear(3, 2))
    optimizer = build_optimizer(model.dit.parameters(), CONFIG)
    scheduler = build_scheduler(optimizer, CONFIG)
    return model, optimizer, scheduler


def update(model, optimizer, scheduler):
    optimizer.zero_grad()
    value = torch.randn(4, 3) + random.random() + np.random.rand()
    loss = model.dit(value).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def test_resume_exact_and_export(tmp_path):
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    model, optimizer, scheduler = objects()
    update(model, optimizer, scheduler)
    destination = tmp_path / 'step_000001'
    progress = dict(step=1, epoch=0, batch=1)
    save_checkpoint(destination, model, optimizer, scheduler, progress=progress,
                    config=CONFIG, rng_states=[capture_rng()], inference_config='inference: {}\n', stats='{}')
    expected = [update(model, optimizer, scheduler) for _ in range(4)]
    restored, opt, sched = objects()
    assert restore_checkpoint(destination, restored, opt, sched, expected_config=CONFIG, rank=0, world_size=1) == progress
    actual = [update(restored, opt, sched) for _ in range(4)]
    assert expected == actual
    assert all(torch.equal(a, b) for a, b in zip(model.dit.parameters(), restored.dit.parameters()))
    assert optimizer.param_groups[0]['lr'] == opt.param_groups[0]['lr']
    exported = torch.load(destination / 'model.pt', weights_only=True)
    assert exported['step'] == 1 and exported['action_expert_stability_fp32']
    assert set(exported['dit']) == {'weight', 'bias'}
    assert not destination.with_name(destination.name + '.pending').exists()


def test_resume_rejects_changed_config(tmp_path):
    model, optimizer, scheduler = objects()
    save_checkpoint(tmp_path / 'saved', model, optimizer, scheduler, progress=dict(step=0),
                    config=CONFIG, rng_states=[capture_rng()], inference_config='{}', stats='{}')
    with pytest.raises(ValueError, match='configuration'):
        restore_checkpoint(tmp_path / 'saved', model, optimizer, scheduler,
                           expected_config=dict(CONFIG, learning_rate=2e-4), rank=0, world_size=1)


def test_partial_save_not_loadable(tmp_path):
    model, optimizer, scheduler = objects()
    with pytest.raises(FileNotFoundError):
        restore_checkpoint(tmp_path, model, optimizer, scheduler, expected_config=CONFIG, rank=0, world_size=1)
