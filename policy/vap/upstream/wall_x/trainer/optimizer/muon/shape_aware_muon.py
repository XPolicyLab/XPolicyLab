import logging

import torch
import torch.distributed as dist

from .muon import Muon, zeropower_via_newtonschulz5

_logger = logging.getLogger(__name__)


class MuonZero0(Muon):
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
        named_params=None,
        orig_shapes=None,
        sharded_tensor_dict=None,
    ):
        self.orig_shapes = orig_shapes
        self.name_to_param = {id(p): name for name, p in named_params}
        self.sharded_tensor_dict = sharded_tensor_dict

        super().__init__(
            param_groups,
            lr,
            weight_decay,
            momentum,
            nesterov,
            ns_steps,
            adamw_betas,
            adamw_eps,
        )

    def _get_orig_shape(self, param):
        """get original shape of the param"""
        name = self.name_to_param.get(id(param))
        name = clean_name(name)
        shape = self.orig_shapes.get(name)
        if shape is None:
            raise KeyError(f"_get_orig_shape: {name} not found in orig_shapes.")
        return shape

    def all_gather_varlen(self, src, dst, all_size):
        """all gather varlen tensor"""
        dst_list = []
        offset = 0
        for size in all_size:
            dst_list.append(dst.narrow(0, offset, size))
            offset += size
        handle = dist.all_gather(dst_list, src.view(-1), async_op=True)
        return handle

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        rank = torch.distributed.get_rank()

        for group in self.param_groups:
            use_muon = group.get("use_muon", False)

            lr = group["lr"]
            weight_decay = group["weight_decay"]

            if use_muon:
                ############################
                #           Muon           #
                ############################
                momentum = group["momentum"]

                complete_grads = {}
                complete_momentums = {}
                handles_grad = {}
                handles_momentum = {}
                sharded_params = {}
                for p in group["params"]:
                    name = self.name_to_param.get(id(p))
                    if name in self.sharded_tensor_dict:
                        all_size = self.sharded_tensor_dict[name]["all_size"]
                        complete_grad = self.sharded_tensor_dict[name]["complete_grad"]
                        handle_grad = self.all_gather_varlen(
                            p.grad if p.grad is not None else torch.zeros_like(p),
                            complete_grad,
                            all_size,
                        )
                        state = self.state[p]
                        if "momentum_buffer" not in state and p.grad is not None:
                            state["momentum_buffer"] = torch.zeros_like(p.grad)
                        complete_momentum = self.sharded_tensor_dict[name][
                            "complete_momentum"
                        ]
                        handle_momentum = self.all_gather_varlen(
                            (
                                state["momentum_buffer"]
                                if p.grad is not None
                                else torch.zeros_like(p)
                            ),
                            complete_momentum,
                            all_size,
                        )
                        complete_grads[name] = complete_grad
                        complete_momentums[name] = complete_momentum
                        handles_grad[name] = handle_grad
                        handles_momentum[name] = handle_momentum
                        sharded_params[name] = p

                for p in group["params"]:
                    name = self.name_to_param.get(id(p))
                    complete_grad = None
                    index_begin = 0
                    if name in self.sharded_tensor_dict:
                        continue
                    if p.grad is None:
                        continue
                    rank = torch.distributed.get_rank()

                    # Use full param to compute svd
                    orig_shape = self._get_orig_shape(p)
                    g = p.grad if complete_grad is None else complete_grad
                    g = g.view(orig_shape)
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p.grad)

                    # Reshape buffer to match gradient shape for computation
                    buf = state["momentum_buffer"].view(g.shape)
                    buf.mul_(momentum).add_(g)
                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                    u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                    u = u.view(p.grad.shape)
                    adjusted_lr = self.adjust_lr_for_muon(lr, orig_shape)
                    p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(u, alpha=-adjusted_lr)

                # Process sharded params
                for name, p in sharded_params.items():
                    handle_grad = handles_grad[name]
                    handle_momentum = handles_momentum[name]
                    handle_grad.wait()
                    handle_momentum.wait()
                    if p.grad is None:
                        continue
                    index_begin = self.sharded_tensor_dict[name]["index"][rank]

                    orig_shape = self._get_orig_shape(p)
                    g = complete_grads[name]
                    g = g.view(orig_shape)
                    buf = complete_momentums[name].view(orig_shape)

                    state = self.state[p]

                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                        buf = buf.view(g.shape)

                    buf.mul_(momentum).add_(g)
                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                    u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    g_partial_shape = p.grad.shape[0]
                    assert (
                        p.ndim == 1
                    ), f"rank {rank} name: {name} p.grad shape: {p.grad.shape}"
                    index_end = index_begin + g_partial_shape
                    # Copy the local shard of momentum back to state
                    state["momentum_buffer"].copy_(buf.view(-1)[index_begin:index_end])
                    u = u.view(-1)
                    u = u[index_begin:index_end]
                    u = u.view(p.grad.shape)
                    adjusted_lr = self.adjust_lr_for_muon(lr, orig_shape)
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


def clean_name(name):
    import re

    clean_name = re.sub(r"(_fsdp_wrapped_module\.)+", "", name)
    return clean_name


def _can_use_muon(name, origin_shapes):
    cleaned = clean_name(name)
    if cleaned not in origin_shapes:
        raise KeyError(
            f"{name!r} (cleaned to {cleaned!r}) is not in origin_shapes. "
            "Pre-wrap shape capture is stale — likely causes: a param was "
            "added after capture, FSDP renamed flat-params, or clean_name() "
            "no longer strips the current wrapper prefix. "
            f"Known keys (sample): {list(origin_shapes)[:5]}..."
        )
    return len(origin_shapes[cleaned]) >= 2


def default_filter(name):
    cond = []
    cond.append("embed_tokens" not in name)
    cond.append("lm_head" not in name)
    cond.append("visual.patch_embed" not in name)
    return all(cond)


def build_shape_aware_muon_optimizer(
    model,
    origin_shapes,
    lr=1e-3,
    weight_decay=0.1,
    eps=1e-8,
    beta_1=0.9,
    beta_2=0.95,
    filter_fn=default_filter,
    param_groups=None,
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Gather numel of all parameters
    local_numels = {}
    for name, p in model.named_parameters():
        local_numels[name] = p.numel()
    all_numels = [None] * world_size
    dist.all_gather_object(all_numels, local_numels)
    all_sharded_tensor_dict = {}

    for name, p in model.named_parameters():
        cleaned = clean_name(name)
        if cleaned not in origin_shapes:
            raise KeyError(
                f"{name!r} (cleaned to {cleaned!r}) is not in origin_shapes. "
                "Pre-wrap shape capture is stale — likely causes: a param was "
                "added after capture, FSDP renamed flat-params, or "
                "clean_name() no longer strips the current wrapper prefix. "
                f"Known keys (sample): {list(origin_shapes)[:5]}..."
            )
        origin_shape = origin_shapes[cleaned]
        flat_num = 1
        for dim in origin_shape:
            flat_num *= dim
        all_numel = torch.tensor([m[name] for m in all_numels])
        if all_numel.nonzero().numel() > 1 and (
            all_numel.unique().numel() > 1
            or all_numel.unique().numel() == 1
            and all_numel.unique()[0] != flat_num
        ):
            index = torch.nn.functional.pad(all_numel, (1, 0))[:-1].cumsum(0)
            all_sharded_tensor_dict[name] = {
                "index": index,
                "all_size": all_numel,
                "complete_grad": torch.empty(
                    flat_num,
                    device=p.device,
                    dtype=p.grad.dtype if p.grad is not None else p.dtype,
                ),
                "complete_momentum": torch.empty(
                    flat_num,
                    device=p.device,
                    dtype=p.grad.dtype if p.grad is not None else p.dtype,
                ),
            }

    # Build id -> name mapping
    id_to_name = {id(p): name for name, p in model.named_parameters()}
    muon_param_groups = []
    adamw_param_groups = []

    if param_groups is None:
        # Default: single group, split by muon/adamw
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.requires_grad
            and _can_use_muon(name, origin_shapes)
            and filter_fn(name)
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if p.requires_grad
            and not (_can_use_muon(name, origin_shapes) and filter_fn(name))
        ]
        muon_param_groups = [{"params": muon_params, "use_muon": True}]
        adamw_param_groups = [{"params": adamw_params, "use_muon": False}]
    else:
        # Custom groups: split each group into muon/adamw subgroups
        for group in param_groups:
            params = group["params"]
            group_kwargs = {k: v for k, v in group.items() if k != "params"}

            muon_p, adamw_p = [], []
            for p in params:
                name = id_to_name.get(id(p), "")
                if _can_use_muon(name, origin_shapes) and filter_fn(name):
                    muon_p.append(p)
                else:
                    adamw_p.append(p)

            if muon_p:
                muon_param_groups.append(
                    {"params": muon_p, "use_muon": True, **group_kwargs}
                )
            if adamw_p:
                adamw_param_groups.append(
                    {"params": adamw_p, "use_muon": False, **group_kwargs}
                )

    # One-shot dump at optimizer construction; skip the string-building work
    # entirely when INFO is filtered out.
    if _logger.isEnabledFor(logging.INFO):
        for r in range(world_size):
            if rank == r:
                lines = ["", "=" * 60, f"RANK {rank}", "=" * 60, "MUON PARAMS:"]
                for group in muon_param_groups:
                    for p in group["params"]:
                        lines.append(
                            f"  {id_to_name[id(p)]}: {list(p.shape)}  "
                            f"p.dtype {p.dtype} "
                            f"p.grad.dtype {p.grad.dtype if p.grad is not None else None} "
                            f"lr {group.get('lr', lr)} "
                            f"weight_decay {group.get('weight_decay', weight_decay)}"
                        )
                lines.append("ADAMW PARAMS:")
                for group in adamw_param_groups:
                    for p in group["params"]:
                        lines.append(
                            f"  {id_to_name[id(p)]}: {list(p.shape)}  "
                            f"p.dtype {p.dtype} "
                            f"p.grad.dtype {p.grad.dtype if p.grad is not None else None} "
                            f"lr {group.get('lr', lr)} "
                            f"weight_decay {group.get('weight_decay', weight_decay)}"
                        )
                _logger.info("\n".join(lines))
            torch.distributed.barrier()

    named_params = list(model.named_parameters())

    return MuonZero0(
        param_groups=muon_param_groups + adamw_param_groups,
        lr=lr,
        weight_decay=weight_decay,
        named_params=named_params,
        orig_shapes=origin_shapes,
        sharded_tensor_dict=all_sharded_tensor_dict,
        adamw_betas=(beta_1, beta_2),
        adamw_eps=eps,
    )


def get_param_groups(model, base_lr):
    vision_params = []
    llm_params = []
    action_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "vision" in name:
            vision_params.append(param)
        elif "action_preprocessor" in name:
            action_params.append(param)
        else:
            llm_params.append(param)

    return [
        {"params": vision_params, "lr": base_lr * 5.0, "name": "vision"},
        {"params": action_params, "lr": base_lr * 0.5, "name": "action"},
        {"params": llm_params, "lr": base_lr * 1.0, "name": "llm"},
    ]
