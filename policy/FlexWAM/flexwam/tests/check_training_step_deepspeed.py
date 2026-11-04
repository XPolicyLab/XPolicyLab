"""Run with torchrun and PYTHONPATH=src; exercises the production step controller.

Example: torchrun --standalone --nproc-per-node=2 tests/check_training_step_deepspeed.py
         --zero-stage 2 --gas 2 --output /tmp/training-step.json
"""
import argparse
import copy
import json
from pathlib import Path

import torch
from accelerate import Accelerator, DeepSpeedPlugin
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad
from torch.utils.data import DataLoader, TensorDataset

from flexwam.training.optimization import TrainingStep, configure_deepspeed_clipping
from flexwam.training.trainer import FlexWAMTrainer
from test_training_optimization import TinyCFModel

parser = argparse.ArgumentParser()
parser.add_argument('--zero-stage', type=int, choices=[1, 2], required=True)
parser.add_argument('--gas', type=int, default=1)
parser.add_argument('--output', required=True)
args = parser.parse_args()
torch.set_num_threads(1)
torch.manual_seed(71)
root = Path(__file__).resolve().parents[1]
config = json.loads((root / f'scripts/train/ds_configs/ds_zero{args.zero_stage}_config.json').read_text())
config.update(train_micro_batch_size_per_gpu=2, gradient_accumulation_steps=args.gas)
acc = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=args.gas,
                  step_scheduler_with_optimizer=False, deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=config))
configure_deepspeed_clipping(acc, .37)
model = TinyCFModel()
optimizer = torch.optim.AdamW([
    {'params': list(model.body.parameters()) + list(model.router.parameters())},
    {'params': model.critic.parameters()},
], lr=.001, weight_decay=.1, betas=(.9, .95))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: .99 ** step)
loader = DataLoader(TensorDataset(torch.randn(256, 2)), batch_size=2)
model, optimizer, loader, scheduler = acc.prepare(model, optimizer, loader, scheduler)
raw = acc.unwrap_model(model)
controller = TrainingStep(acc, model, optimizer, max_grad_norm=.37, critic_group_index=1)
trainer = FlexWAMTrainer.__new__(FlexWAMTrainer)
trainer.model = model
trainer.accelerator = acc
trainer.training_step = controller
engine = controller.engine
assert engine.gradient_clipping() == .37
assert engine.optimizer.clip_grad == .37
base_optimizer = engine.basic_optimizer
critic_partition = base_optimizer.param_groups[1]['params'][0]

def values():
    return {role: torch.cat([safe_get_full_fp32_param(p).detach().float().flatten().cpu()
                              for p in module.parameters()])
            for role, module in [('body', raw.body), ('router', raw.router), ('critic', raw.critic)]}

def unchanged_state(old, new):
    assert old.keys() == new.keys()
    for key in old:
        if torch.is_tensor(old[key]):
            assert torch.equal(old[key], new[key]), key
        else:
            assert old[key] == new[key], key

records = []
data = iter(loader)
# CF in windows 2 and 5, only on the first microbatch: this catches clearing
# already-accumulated critic gradients on a later non-CF microbatch. Other
# windows represent warmup, skipped probes, and post-CF stage 5.
for window in range(1, 8):
    if window == 4:
        # Continue through both skipped-CF and CF windows after state loading.
        saved = copy.deepcopy(base_optimizer.state_dict())
        base_optimizer.load_state_dict(saved)
        assert base_optimizer.param_groups[1]['params'][0] is critic_partition
    start = values()
    state_before = copy.deepcopy(base_optimizer.state[critic_partition])
    engine_before = engine.global_steps
    if window == 6:
        raw.critic.requires_grad_(False)
    reference = TinyCFModel().to(device=acc.device, dtype=next(raw.parameters()).dtype)
    reference.load_state_dict(raw.state_dict())
    reference.critic.requires_grad_(window < 6)
    expected_grads = {name: torch.zeros_like(p, dtype=torch.float32)
                      for name, p in reference.named_parameters() if p.requires_grad}
    gradient_error = 0.
    for micro in range(args.gas):
        with acc.accumulate(model):
            x, = next(data)
            cf = window in [2, 5] and micro == 0
            ref_main = reference.main_loss(x, cf)
            ref_cf, _ = reference.run_deferred_pixel_critic_cf()
            ((ref_main if ref_cf is None else ref_main + ref_cf) / args.gas).backward()
            for name, p in reference.named_parameters():
                if p.grad is not None:
                    expected_grads[name].add_(p.grad.float())
            reference.zero_grad(set_to_none=True)
            loss = raw.main_loss(x, cf)
            trainer._backward_main_loss(loss)
            trainer._backward_deferred_pixel_cf(loss, {})
            # Neither backward is allowed to update parameters.
            assert engine.global_steps == engine_before
            for role, tensor in values().items():
                assert torch.equal(tensor, start[role]), ('updated during backward', window, micro, role)
            if acc.sync_gradients:
                for name, p in raw.named_parameters():
                    if name not in expected_grads:
                        continue
                    expected = acc.reduce(expected_grads[name], reduction='mean')
                    actual = safe_get_full_grad(p).float()
                    error = float((actual - expected).norm() / expected.norm().clamp_min(1e-6))
                    gradient_error = max(gradient_error, error)
                    assert error < .03, ('wrong accumulated gradient', window, name, error)
            norm = controller.step()
            if acc.sync_gradients:
                assert norm is not None
                scheduler.step()
            else:
                assert norm is None
    after = values()
    assert engine.global_steps == engine_before + 1
    assert scheduler.scheduler.last_epoch == window
    change = {role: float((after[role] - start[role]).abs().max()) for role in start}
    if window not in [2, 5]:
        assert change['critic'] == 0., (window, change)
        unchanged_state(state_before, base_optimizer.state[critic_partition])
    else:
        assert change['critic'] > 0., ('CF gradient was lost', window)
    assert change['body'] > 0. and change['router'] > 0.
    records.append({'window': window, 'cf': window in [2, 5], 'updates': engine.global_steps,
                    'max_parameter_change': change,
                    'critic_adam_step': float(base_optimizer.state[critic_partition].get('step', 0)),
                    'max_gradient_relative_error': gradient_error,
                    'grad_norm': float(norm)})

engine_counts = acc.gather(torch.tensor([engine.global_steps], device=acc.device)).cpu().tolist()
assert engine_counts == [7] * acc.num_processes
if acc.is_main_process:
    result = {'zero_stage': args.zero_stage, 'world_size': acc.num_processes,
              'gradient_accumulation_steps': args.gas, 'engine_updates_all_ranks': engine_counts,
              'gradient_clipping': engine.gradient_clipping(), 'windows': records,
              'scope': 'Production trainer backward helpers and TrainingStep; real CUDA AdamW/ZeRO, tiny network.'}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print('PASS', json.dumps(result), flush=True)
acc.wait_for_everyone()
acc.end_training()
