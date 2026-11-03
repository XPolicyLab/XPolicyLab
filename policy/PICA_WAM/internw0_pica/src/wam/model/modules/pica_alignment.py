"""PICA-WAM cross-appearance self-alignment with an FP32 EMA of the action expert.

The video conditioner uses live weights under no_grad.
The EMA is kept outside module registration so DDP/DeepSpeed never optimize it.
"""
import copy

import torch
from torch import nn

from .mot.mixture_of_transformers import MoT


class CrossAppearanceAlignment(nn.Module):
    def __init__(self, action_dim, num_layers, *, loss_weight=0.05, student_layer=8,
                 teacher_layer=20, sigma_gap_max=0.2, head_hidden=512, ema_decay=0.9999):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.student_layer = int(student_layer)
        self.teacher_layer = int(teacher_layer)
        self.sigma_gap_max = float(sigma_gap_max)
        self.ema_decay = float(ema_decay)
        if not 0 <= self.student_layer < self.teacher_layer < num_layers:
            raise ValueError("PICA alignment requires 0 <= student_layer < teacher_layer < num_layers.")
        if not 0 < self.ema_decay < 1 or not 0 <= self.sigma_gap_max <= 1:
            raise ValueError("PICA alignment requires EMA decay in (0,1) and sigma gap in [0,1].")
        if self.loss_weight <= 0 or int(head_hidden) <= 0:
            raise ValueError("PICA alignment loss weight and projection width must be positive.")
        self.head = nn.Sequential(nn.Linear(action_dim, int(head_hidden)), nn.GELU(),
                                  nn.Linear(int(head_hidden), action_dim, bias=False))
        self.ema_steps = 0
        self._teacher = []
        self._pending_shadow = None

    def teacher(self, live_mot):
        if not self._teacher:
            live = live_mot.mixtures["action"]
            mirror = copy.deepcopy(live).requires_grad_(False).eval()
            mirror.use_gradient_checkpointing = False
            shadow = {name: p.detach().float().clone() for name, p in live.named_parameters()}
            if self._pending_shadow is not None:
                if shadow.keys() != self._pending_shadow.keys():
                    raise ValueError("PICA alignment EMA checkpoint parameter names do not match the action expert.")
                for name, value in self._pending_shadow.items():
                    if value.shape != shadow[name].shape:
                        raise ValueError(f"PICA alignment EMA checkpoint shape mismatch: {name}")
                    shadow[name].copy_(value)
                self._pending_shadow = None
                for name, p in mirror.named_parameters():
                    p.data.copy_(shadow[name])
            teacher = MoT(
                mixtures={name: mirror if name == "action" else live_mot.mixtures[name]
                          for name in live_mot.expert_order},
                mot_checkpoint_mixed_attn=False,
                attention_backend=live_mot.attention_backend,
                flex_block_size=live_mot.flex_block_size,
            )
            # Calling eval() on the shared video expert would change the student's mode.
            teacher.training = False
            self._teacher.append((teacher, mirror, shadow))
        return self._teacher[0][0]

    @torch.no_grad()
    def update_ema(self, live_mot):
        self.teacher(live_mot)
        _, mirror, shadow = self._teacher[0]
        live = dict(live_mot.mixtures["action"].named_parameters())
        for name, p in mirror.named_parameters():
            if p.shape != live[name].shape:
                raise ValueError("PICA alignment EMA requires replicated parameters (ZeRO-1 or ZeRO-2).")
            shadow[name].mul_(self.ema_decay).add_(live[name].detach().float(), alpha=1-self.ema_decay)
            p.copy_(shadow[name])
        buffers = dict(live_mot.mixtures["action"].named_buffers())
        for name, b in mirror.named_buffers():
            b.copy_(buffers[name])
        self.ema_steps += 1

    def checkpoint_state(self):
        shadow = self._teacher[0][2] if self._teacher else self._pending_shadow
        return {"head": self.head.state_dict(), "shadow": shadow, "ema_steps": self.ema_steps}

    def load_checkpoint_state(self, state):
        self.head.load_state_dict(state["head"], strict=True)
        self._teacher.clear()
        self._pending_shadow = state["shadow"]
        self.ema_steps = int(state["ema_steps"])
