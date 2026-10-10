"""Verify frozen-WM output pruning with the real coupled Transformer on CPU."""
import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers import GemmaConfig, GemmaForCausalLM

from focus_vlwa.model.focus_vlwa import FocusVLWA
from focus_vlwa.model.joint_experts import JointExperts


class TinyPolicy(FocusVLWA):
    def __init__(self, checkpointing=False):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(action_horizon=3)
        self.use_discrete_state = True
        self.use_world_model = True
        self.compute_world_model_supervision = True
        self.world_model_horizon = 4
        self.world_model_event_tokens = 1
        self.world_model_dim = 4
        self.subtask_aux = False
        self.world_model_loss_weight = 1.0
        self.world_model_event_loss_weight = 1.0
        self.world_model_state_loss_weight = 0.3
        self.world_model_action_loss_weight = 0.0
        self.gradient_checkpointing_enabled = checkpointing
        self.prefix_projection = nn.Linear(4, 32)
        self.action_input_projection = nn.Linear(4, 32)
        self.action_output_projection = nn.Linear(32, 4)
        self.time_mlp_input = nn.Linear(32, 32)
        self.time_mlp_output = nn.Linear(32, 32)
        self.world_model_input_projection = nn.Linear(4, 32)
        self.world_model_output_projection = nn.Linear(32, 4)
        cfg = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                   num_attention_heads=8, num_key_value_heads=1, head_dim=4,
                   vocab_size=16, attention_dropout=0.0)
        prefix = GemmaForCausalLM(GemmaConfig(**cfg, use_adarms=False)).model
        coupled = JointExperts.__new__(JointExperts)
        nn.Module.__init__(coupled)
        coupled.vision_language_model = nn.Module()
        coupled.vision_language_model.language_model = prefix
        coupled.vision_language_model.model = SimpleNamespace(language_model=prefix)
        coupled.vision_language_model.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=2))
        for name in ('action_expert', 'world_model_expert'):
            expert = GemmaForCausalLM(GemmaConfig(**cfg, use_adarms=True, adarms_cond_dim=32))
            expert.model.embed_tokens = None
            expert.model.gradient_checkpointing = checkpointing
            setattr(coupled, name, expert)
        self.joint_experts = coupled

    def _preprocess_observation(self, obs, *, train=True):
        return [], [], None, None, obs.state, obs

    def embed_prefix(self, images, masks, tokens, token_masks, observation):
        prefix = self.prefix_projection(observation.prefix)
        mask = torch.ones(prefix.shape[:2], dtype=torch.bool)
        return prefix, mask, torch.zeros_like(mask)


def inputs():
    obs = SimpleNamespace(prefix=torch.randn(2, 3, 4), state=torch.randn(2, 4),
                          world_state=torch.randn(2, 4, 4), world_state_mask=torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 1.]]),
                          event_action=torch.randn(2, 4), event_action_mask=torch.tensor([1., 0.]),
                          action_mask=torch.ones(2, 3, 4))
    return obs, torch.randn(2, 3, 4)


class WMSupervisionTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(47)

    def test_action_outputs_gradients_rng_and_adamw_update_match(self):
        for checkpointing in (False, True):
            with self.subTest(checkpointing=checkpointing), patch.dict(
                os.environ, {'FOCUS_VLWA_NO_GRAD_CKPT': '0' if checkpointing else '1'}
            ):
                baseline = TinyPolicy(checkpointing)
                baseline.freeze_world_model_expert()
                optimized = copy.deepcopy(baseline)
                optimized.freeze_world_model_expert(skip_supervision=True)
                self.assertEqual(baseline.state_dict().keys(), optimized.state_dict().keys())
                obs, actions = inputs()
                optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-4) for m in (baseline, optimized)]
                results = []
                for model in (baseline, optimized):
                    torch.manual_seed(101)
                    with patch.object(model.world_model_output_projection, 'forward', wraps=model.world_model_output_projection.forward) as output_head:
                        result = model(obs, actions)
                        result['action'].mean().backward()
                        count = output_head.call_count
                    results.append((result, torch.get_rng_state(), count))
                self.assertGreater(results[0][2], 0)
                self.assertEqual(results[1][2], 0)
                self.assertEqual(set(results[1][0]), {'action'})
                torch.testing.assert_close(results[0][0]['action'], results[1][0]['action'], rtol=0, atol=0)
                self.assertTrue(torch.equal(results[0][1], results[1][1]))
                self.assertGreater(baseline.prefix_projection.weight.grad.abs().sum().item(), 0)
                self.assertGreater(baseline.time_mlp_input.weight.grad.abs().sum().item(), 0)
                for (name, original), (_, changed) in zip(baseline.named_parameters(), optimized.named_parameters()):
                    self.assertEqual(original.grad is None, changed.grad is None, name)
                    if original.grad is not None:
                        torch.testing.assert_close(original.grad, changed.grad, rtol=0, atol=0, msg=name)
                frozen = {n: p.clone() for n, p in optimized.named_parameters() if not p.requires_grad}
                for optimizer in optimizers:
                    optimizer.step()
                for (name, original), (_, changed) in zip(baseline.named_parameters(), optimized.named_parameters()):
                    torch.testing.assert_close(original, changed, rtol=0, atol=0, msg=name)
                    if name in frozen:
                        torch.testing.assert_close(changed, frozen[name], rtol=0, atol=0)
                # WM supervision must remain an input to the action branch after pruning.
                changed_obs = copy.copy(obs)
                changed_obs.world_state = obs.world_state + 5
                torch.manual_seed(101)
                original_action = optimized(obs, actions)['action']
                torch.manual_seed(101)
                changed_action = optimized(changed_obs, actions)['action']
                self.assertFalse(torch.equal(original_action, changed_action))

    def test_joint_supervision_and_gradients_are_retained(self):
        model = TinyPolicy()
        obs, actions = inputs()
        result = model(obs, actions)
        self.assertEqual(set(result), {'action', 'world_model', 'world_model_event', 'world_model_state'})
        (result['action'].mean() + result['world_model'].mean()).backward()
        self.assertGreater(model.world_model_output_projection.weight.grad.abs().sum().item(), 0)
