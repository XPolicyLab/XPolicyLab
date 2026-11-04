"""Real AdamW regressions for deferred CF and accumulation boundaries."""
import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from accelerate import Accelerator

from flexwam.training.optimization import TrainingStep, configure_deepspeed_clipping
from flexwam.training.trainer import FlexWAMTrainer


class TinyCFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(2, 2)
        self.router = nn.Linear(2, 2)
        self.critic = nn.Linear(2, 2)
        self._deferred_pixel_cf_ctx = None
        self.router_cf_lambda = 1.0

    def main_loss(self, x, cf):
        x = x.to(self.body.weight.dtype)
        feature = self.body(x)
        self._deferred_pixel_cf_ctx = feature.detach() if cf else None
        return self.router(feature).float().square().mean() + self.critic(feature.detach()).float().sum() * 0

    def run_deferred_pixel_critic_cf(self):
        feature = self._deferred_pixel_cf_ctx
        self._deferred_pixel_cf_ctx = None
        if feature is None:
            return None, {}
        logits = self.critic(feature).float()
        teacher = logits.detach().softmax(-1)
        log_student = self.router(feature).float().log_softmax(-1)
        kl = (teacher * (teacher.log() - log_student)).sum(-1).mean()
        loss = (logits - 1).square().mean() + .5 * kl
        return loss, {"loss_router_cf": float(loss.detach())}


class TrainingOptimizationTests(unittest.TestCase):
    def setup_training(self, gas=2):
        torch.manual_seed(23)
        accelerator = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=gas,
                                  step_scheduler_with_optimizer=False)
        model = TinyCFModel()
        optimizer = torch.optim.AdamW([
            {"params": list(model.body.parameters()) + list(model.router.parameters())}, {"params": model.critic.parameters()},
        ], lr=.01, weight_decay=.1)
        model, optimizer = accelerator.prepare(model, optimizer)
        controller = TrainingStep(accelerator, model, optimizer, max_grad_norm=1., critic_group_index=1)
        trainer = FlexWAMTrainer.__new__(FlexWAMTrainer)
        trainer.accelerator = accelerator
        trainer.model = model
        trainer.training_step = controller
        return accelerator, model, optimizer, controller, trainer

    def test_no_cf_preserves_critic_weights_and_adam_state(self):
        acc, model, opt, controller, trainer = self.setup_training()
        raw = acc.unwrap_model(model)
        x = torch.tensor([[1., 2.], [3., -1.]])
        for cf in [True, False]:
            with acc.accumulate(model):
                loss = raw.main_loss(x, cf)
                trainer._backward_main_loss(loss)
                trainer._backward_deferred_pixel_cf(loss, {})
                controller.step()
        saved = {p: (p.detach().clone(), copy.deepcopy(opt.state[p])) for p in raw.critic.parameters()}
        for _ in range(4):
            with acc.accumulate(model):
                loss = raw.main_loss(x, False)
                trainer._backward_main_loss(loss)
                trainer._backward_deferred_pixel_cf(loss, {})
                controller.step()
        for p, (value, state) in saved.items():
            self.assertTrue(torch.equal(p, value))
            for key, expected in state.items():
                self.assertTrue(torch.equal(opt.state[p][key], expected), key)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_cf_on_first_microbatch_is_not_lost(self):
        acc, model, opt, controller, trainer = self.setup_training()
        raw = acc.unwrap_model(model)
        before = [p.detach().clone() for p in raw.critic.parameters()]
        x = torch.ones(2, 2)
        with acc.accumulate(model):
            loss = raw.main_loss(x, True)
            trainer._backward_main_loss(loss)
            trainer._backward_deferred_pixel_cf(loss, {})
            self.assertIsNone(controller.step())
            self.assertTrue(controller.critic_has_grad)
            for p, old in zip(raw.critic.parameters(), before):
                self.assertTrue(torch.equal(p, old))
        with acc.accumulate(model):
            loss = raw.main_loss(x, False)
            trainer._backward_main_loss(loss)
            trainer._backward_deferred_pixel_cf(loss, {})
            self.assertTrue(controller.critic_has_grad)
            controller.step()
        self.assertFalse(controller.critic_has_grad)
        self.assertTrue(any(not torch.equal(p, old) for p, old in zip(raw.critic.parameters(), before)))
        self.assertTrue(all(int(opt.state[p]['step']) == 1 for p in model.parameters()))

    def test_deferred_matches_joint_loss_update(self):
        acc, model, opt, controller, trainer = self.setup_training(gas=1)
        raw = acc.unwrap_model(model)
        reference = copy.deepcopy(raw)
        ref_opt = torch.optim.AdamW([
            {'params': list(reference.body.parameters()) + list(reference.router.parameters())}, {'params': reference.critic.parameters()},
        ], lr=.01, weight_decay=.1)
        x = torch.tensor([[1., 2.], [3., -1.]])
        for _ in range(3):
            with acc.accumulate(model):
                loss = raw.main_loss(x, True)
                trainer._backward_main_loss(loss)
                trainer._backward_deferred_pixel_cf(loss, {})
                controller.step()
            main = reference.main_loss(x, True)
            cf, _ = reference.run_deferred_pixel_critic_cf()
            (main + cf).backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
            ref_opt.step()
            ref_opt.zero_grad(set_to_none=True)
        for p, expected in zip(raw.parameters(), reference.parameters()):
            torch.testing.assert_close(p, expected, rtol=0, atol=0)

    def test_clipping_config_is_set_before_engine_initialization(self):
        plugin = SimpleNamespace(deepspeed_config={})
        acc = SimpleNamespace(state=SimpleNamespace(deepspeed_plugin=plugin))
        configure_deepspeed_clipping(acc, .37)
        self.assertEqual(plugin.deepspeed_config['gradient_clipping'], .37)
        for invalid in [-1., float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                configure_deepspeed_clipping(acc, invalid)


if __name__ == '__main__':
    unittest.main()
