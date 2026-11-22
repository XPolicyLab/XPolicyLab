import logging
import math

import torch
from torch.optim import Optimizer

_logger = logging.getLogger(__name__)


class DebugAdamW(Optimizer):
    """
    Adamw optimizer, for debug purpose
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        param_names=None,
    ):
        defaults = dict(
            lr=lr,
            adamw_betas=betas,
            adamw_eps=eps,
            weight_decay=weight_decay,
        )

        # save param name mapping (id -> name)
        self.param_names = param_names or {}

        super().__init__(params, defaults)
        self._step_count = 0
        self._print_dtype = True

    @classmethod
    def from_model(
        cls, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, **kwargs
    ):
        params = list(model.parameters())
        param_names = {id(p): name for name, p in model.named_parameters()}
        return cls(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            param_names=param_names,
            **kwargs,
        )

    def get_param_name(self, param):
        """Get parameter name"""
        return self.param_names.get(id(param), f"unknown_{id(param)}")

    def set_print_dtype(self, enable: bool):
        self._print_dtype = enable

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        fp32_count = 0
        bf16_count = 0

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if self._print_dtype and self._step_count == 1:
                    if p.dtype == torch.float32:
                        fp32_count += 1
                    elif p.dtype == torch.bfloat16:
                        bf16_count += 1

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["momentum_sq_buffer"] = torch.zeros_like(g)

                state["step"] += 1
                step = state["step"]
                buf1 = state["momentum_buffer"]
                buf2 = state["momentum_sq_buffer"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)
                g = buf1 / (eps + buf2.sqrt())
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


# Modified from Moonlight https://github.com/MoonshotAI/Moonlight
class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_weight_decay: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        param_groups,
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        super().__init__(param_groups, defaults)

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_muon = group.get("use_muon", False)

            lr = group["lr"]
            weight_decay = group["weight_decay"]

            if use_muon:
                ############################
                #           Muon           #
                ############################
                momentum = group["momentum"]

                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)

                    if not torch.isfinite(g).all():
                        _logger.warning(
                            "[Check 1] Raw gradient has NaN/Inf, shape=%s",
                            g.shape,
                        )

                    state = self.state[p]
                    # Lazy init momentum buffer
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)

                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)

                    # Check 2: momentum buffer.
                    if not torch.isfinite(buf).all():
                        _logger.warning("[Check 2] Momentum buffer has NaN/Inf")

                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                    u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                    if not torch.isfinite(u).all():
                        _logger.error(
                            "[Check 4] Newton-Schulz output has NaN/Inf! "
                            f"Input g norm was: {g.float().norm().item():.6e}",
                        )

                    adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                    p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(u, alpha=-adjusted_lr)

            else:
                ############################
                #       AdamW backup       #
                ############################
                beta1, beta2 = group["adamw_betas"]
                eps = group["adamw_eps"]

                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue

                    state = self.state[p]
                    # Lazy init AdamW state
                    if "step" not in state:
                        state["step"] = 0
                        state["moment1"] = torch.zeros_like(g)
                        state["moment2"] = torch.zeros_like(g)

                    state["step"] += 1
                    step = state["step"]
                    buf1 = state["moment1"]
                    buf2 = state["moment2"]
                    buf1.lerp_(g, 1 - beta1)
                    buf2.lerp_(g.square(), 1 - beta2)

                    g = buf1 / (eps + buf2.sqrt())

                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                    scale = bias_correction1 / bias_correction2**0.5
                    p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(g, alpha=-lr / scale)

        return loss


def _can_use_muon(param):
    cond = []
    cond.append(param.ndim >= 2)
    return all(cond)


def default_filter(name):
    cond = []
    cond.append("embed_tokens" not in name)
    cond.append("lm_head" not in name)
    cond.append("visual.patch_embed" not in name)
    cond.append("conv1d" not in name)
    return all(cond)


def build_muon_optimizer(
    model,
    lr=1e-3,
    weight_decay=0.1,
    eps=1e-8,
    beta_1=0.9,
    beta_2=0.95,
    filter_fn=default_filter,
    param_groups=None,
    **kwargs,
):
    all_muon_groups = []
    all_adamw_groups = []
    id_to_name = {id(p): name for name, p in model.named_parameters()}
    if param_groups is None:
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.requires_grad and _can_use_muon(p) and filter_fn(name)
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if p.requires_grad and not (_can_use_muon(p) and filter_fn(name))
        ]
        all_muon_groups.append({"params": muon_params, "use_muon": True})
        all_adamw_groups.append({"params": adamw_params, "use_muon": False})
    else:
        for group in param_groups:
            params = group["params"]
            group_kwargs = {k: v for k, v in group.items() if k != "params"}

            muon_p, adamw_p = [], []
            for p in params:
                name = id_to_name.get(id(p), "")
                if _can_use_muon(p) and filter_fn(name):
                    muon_p.append(p)
                else:
                    adamw_p.append(p)

            if muon_p:
                all_muon_groups.append(
                    {"params": muon_p, "use_muon": True, **group_kwargs}
                )
            if adamw_p:
                all_adamw_groups.append(
                    {"params": adamw_p, "use_muon": False, **group_kwargs}
                )

    return Muon(
        param_groups=all_muon_groups + all_adamw_groups,
        lr=lr,
        weight_decay=weight_decay,
        adamw_betas=(beta_1, beta_2),
        adamw_eps=eps,
    )
