"""Trainer-side helpers: CLI parsing, parameter groups, freezing, checkpoint discovery.

The checkpoint filename convention encoded by ``_CHECKPOINT_PATTERN`` is
load-bearing: released weights, the resume scan and ``tools/convert_checkpoint.py``
all agree on ``steps_<N>_pytorch_model.pt`` / ``steps_<N>_model.safetensors``.
"""

from __future__ import annotations

import os
import re

import torch
import torch.distributed as dist
from accelerate.logging import get_logger

logger = get_logger(__name__)

# Model-only checkpoints written by ``checkpoint.save_checkpoint``. Anything that
# scans a run directory for resumable weights must use exactly this pattern.
_CHECKPOINT_PATTERN = re.compile(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$")


def normalize_dotlist_args(args):
    """
    Convert CLI overrides into OmegaConf dotlist entries.

    Accepted forms:
      ['--x.y', 'val'] / ['--flag'] / ['--x.y=val'] → ['x.y=val', 'flag=true']
      ['x.y=val'] → ['x.y=val']  (bare OmegaConf-style overrides)
    """
    normalized = []
    skip = False
    for i in range(len(args)):
        if skip:
            skip = False
            continue

        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                normalized.append(key)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                normalized.append(f"{key}={args[i + 1]}")
                skip = True
            else:
                normalized.append(f"{key}=true")
        elif "=" in arg and not arg.startswith("-"):
            # Bare OmegaConf-style override, e.g. trainer.is_resume=true.
            normalized.append(arg)
        else:
            pass  # skip orphaned values
    return normalized


def build_param_lr_groups(model, cfg):
    """
    build multiple param groups based on cfg.trainer.learning_rate.
    support specifying different learning rates for different modules, the rest use base.

    Args:
        model: nn.Module model object
        cfg: config object, requires cfg.trainer.learning_rate dictionary

    Returns:
        List[Dict]: param_groups that can be used to build optimizer with torch.optim
    """

    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 1e-4)  # default base learning rate

    freeze_modules = cfg.trainer.get("freeze_modules", "")
    if not isinstance(freeze_modules, str):
        freeze_modules = ""
    freeze_patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]

    used_params = set()
    frozen_params = set()
    param_groups = []

    for freeze_path in freeze_patterns:
        module = model
        try:
            for attr in freeze_path.split("."):
                module = getattr(module, attr)
            frozen_params.update(id(p) for p in module.parameters())
        except AttributeError:
            print(f"⚠️ freeze module path does not exist: {freeze_path}")
            continue

    named_parameters = list(model.named_parameters())

    def append_group(group_name: str, lr, selected: list[tuple[str, torch.nn.Parameter]]) -> None:
        selected = [
            (name, parameter)
            for name, parameter in selected
            if parameter.requires_grad and id(parameter) not in frozen_params and id(parameter) not in used_params
        ]
        if not selected:
            return
        params = [parameter for _, parameter in selected]
        param_groups.append({"params": params, "lr": lr, "name": group_name})
        used_params.update(id(parameter) for parameter in params)

    for module_name, lr in lr_cfg.items():
        if module_name == "base":
            continue
        # try to find the module under the model by module_name (support nested paths)
        module = model
        try:
            for attr in module_name.split("."):
                module = getattr(module, attr)
            # filter out frozen parameters.  Internally frozen teachers must not
            # occupy Adam/ZeRO optimizer state even when they are frozen by the
            # model rather than trainer.freeze_modules.
            module_param_ids = {id(parameter) for parameter in module.parameters()}
            append_group(
                module_name,
                lr,
                [(name, parameter) for name, parameter in named_parameters if id(parameter) in module_param_ids],
            )
        except AttributeError:
            logger.warning("⚠️ module path `%s` not found in model", module_name)

    # assign base learning rate to the remaining unused parameters (exclude frozen ones)
    other_named = [
        (name, parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad and id(parameter) not in used_params and id(parameter) not in frozen_params
    ]
    append_group("base", base_lr, other_named)

    return param_groups


def _is_safetensors_path(path) -> bool:
    """Check if a path refers to a safetensors file."""
    return str(path).endswith(".safetensors")


def _strict_load_full_model(
    model: torch.nn.Module,
    checkpoint: dict[str, torch.Tensor],
) -> dict[str, object]:
    """Strictly restore a complete model-state for a new training stage."""

    target_state_dict = model.state_dict()
    target_keys = set(target_state_dict)
    checkpoint_keys = set(checkpoint)
    missing_keys = sorted(target_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - target_keys)
    shape_mismatches = [
        {
            "key": key,
            "expected": list(target_state_dict[key].shape),
            "actual": list(checkpoint[key].shape),
        }
        for key in sorted(target_keys & checkpoint_keys)
        if target_state_dict[key].shape != checkpoint[key].shape
    ]
    if missing_keys or unexpected_keys or shape_mismatches:
        raise RuntimeError(
            "Strict full-model load failed: "
            f"missing={missing_keys[:8]}; unexpected={unexpected_keys[:8]}; "
            f"shape={shape_mismatches[:8]}"
        )

    used_assign = any(tensor.is_meta for tensor in target_state_dict.values())
    model.load_state_dict(checkpoint, strict=True, assign=used_assign)
    return {
        "module_path": "<full_model>",
        "target_key_count": len(target_state_dict),
        "loaded_key_count": len(checkpoint),
        "excluded_checkpoint_key_count": 0,
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
        "used_assign": used_assign,
    }


class TrainerUtils:
    @staticmethod
    def freeze_backbones(model, freeze_modules=""):
        """
        directly freeze the specified submodules based on the relative module path list (patterns), no longer
        recursively find all submodule names:
          - patterns: read from config.trainer.freeze_modules, separated by commas to get the "relative path" list
            for example "qwen_vl_interface, action_model.net",
            it means to freeze model.qwen_vl_interface and model.action_model.net.

        Args:
            model: nn.Module model object
            freeze_modules: relative module path list (patterns)

        Returns:
            model: nn.Module model object
        """
        frozen = []
        if freeze_modules and isinstance(freeze_modules, str):
            # split and remove whitespace
            patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]

            for path in patterns:
                # split the "relative path" by dots, for example "action_model.net" → ["action_model", "net"]
                attrs = path.split(".")
                module = model
                try:
                    for attr in attrs:
                        module = getattr(module, attr)
                    # if the module is successfully get, freeze it and its all submodule parameters
                    for param in module.parameters():
                        param.requires_grad = False
                    frozen.append(path)
                except AttributeError:
                    # if the attribute does not exist, skip and print warning
                    print(f"⚠️ module path does not exist, cannot freeze: {path}")
                    continue

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"🔒 Frozen modules with re pattern: {frozen}")
        return model

    @staticmethod
    def print_trainable_parameters(model):
        """
        print the total number of parameters and trainable parameters of the model
        :param model: PyTorch model instance
        """
        if dist.is_initialized() and dist.get_rank() != 0:
            return None
        print("📊 model parameter statistics:")
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"# Parameters (in millions): {num_params / 10**6:.3f} Total, "
            f"{num_trainable_params / 10**6:.3f} Trainable"
        )
        return num_params, num_trainable_params

    @staticmethod
    def load_pretrained_backbones(
        model,
        checkpoint_path=None,
        reload_modules=None,
        strict=False,
    ):
        """
        load checkpoint:
        - if reload_modules is set, load by path part
        - otherwise → load the entire model parameters (overwrite model)

        ``strict`` routes the full-model branch through ``_strict_load_full_model``,
        which refuses any missing/unexpected/shape-mismatched key instead of
        letting a partially initialized stage start training.
        """
        if not checkpoint_path:
            return model
        is_main = not dist.is_initialized() or dist.get_rank() == 0
        if is_main:
            print(f"📦 loading checkpoint: {checkpoint_path}")
        try:
            if _is_safetensors_path(checkpoint_path):
                from safetensors.torch import load_file

                checkpoint = load_file(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"❌ loading checkpoint failed: {e}") from e

        if strict and reload_modules:
            raise ValueError(
                "strict pretrained loading is only defined for a full-model "
                "checkpoint; trainer.reload_modules must be empty"
            )

        if reload_modules:  # partial load
            module_paths = [p.strip() for p in reload_modules.split(",") if p.strip()]
            for path in module_paths:
                module = model
                try:
                    for module_name in path.split("."):  # find the module to modify level by level
                        module = getattr(module, module_name)
                    prefix = path + "."
                    sub_state_dict = {k[len(prefix) :]: v for k, v in checkpoint.items() if k.startswith(prefix)}
                    if sub_state_dict:
                        module.load_state_dict(sub_state_dict, strict=True)
                        if is_main:
                            print(f"✅ parameters loaded to module '{path}'")
                    else:
                        print(f"⚠️ parameters not found in checkpoint '{path}'")
                except AttributeError:
                    print(f"❌ cannot find module path: {path}")
        else:  # full load
            try:
                if strict:
                    audit = _strict_load_full_model(model, checkpoint)
                    if is_main:
                        print(
                            "✅ loaded <full_model> model parameters (strict, "
                            f"keys={audit['loaded_key_count']}, assign={audit['used_assign']})"
                        )
                else:
                    incompatible = model.load_state_dict(checkpoint, strict=False)
                    if is_main:
                        missing = list(incompatible.missing_keys)
                        unexpected = list(incompatible.unexpected_keys)
                        print(
                            "✅ loaded <full_model> model parameters "
                            f"(missing={len(missing)}, unexpected={len(unexpected)})"
                        )
                        if missing:
                            print(f"⚠️ first missing keys: {missing[:8]}")
                        if unexpected:
                            print(f"⚠️ first unexpected keys: {unexpected[:8]}")
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"❌ loading full model failed: {e}") from e
        return model

    @staticmethod
    def setup_distributed_training(accelerator, *components):
        """
        use Accelerator to prepare distributed training components
        :param accelerator: Accelerate instance
        :param components: any number of components (such as model, optimizer, dataloader, etc.)
        :return: prepared distributed components (in the same order as input)
        """
        return accelerator.prepare(*components)

    @staticmethod
    def _reset_dataloader(dataloader, epoch_counter):
        """safe reset dataloader iterator"""
        # 1. update epoch counter
        epoch_counter += 1

        # 2. set new epoch (distributed core)
        if hasattr(dataloader, "sampler") and callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(epoch_counter)

        # 3. create new iterator
        return iter(dataloader), epoch_counter

    def _get_latest_checkpoint(self, checkpoint_dir):
        """Find the latest model-only checkpoint in the directory based on step number."""
        if not os.path.exists(checkpoint_dir):
            self.accelerator.print(f"No checkpoint directory found at {checkpoint_dir}")
            return None, 0

        # Find all checkpoints matching the naming convention, supports .pt and .safetensors
        checkpoints = [
            f
            for f in os.listdir(checkpoint_dir)
            if _CHECKPOINT_PATTERN.match(f) and os.path.isfile(os.path.join(checkpoint_dir, f))
        ]

        if not checkpoints:
            self.accelerator.print(f"No checkpoints found in {checkpoint_dir}")
            return None, 0

        # Extract step numbers and sort
        try:
            checkpoints_with_steps = [(ckpt, int(_CHECKPOINT_PATTERN.match(ckpt).group(1))) for ckpt in checkpoints]
        except AttributeError as e:
            self.accelerator.print(f"Error parsing checkpoint filenames: {e}")
            return None, 0

        # Sort by step number and get the latest checkpoint
        checkpoints_with_steps.sort(key=lambda x: x[1])
        latest_checkpoint, completed_steps = checkpoints_with_steps[-1]

        latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        self.accelerator.print(f"Latest checkpoint found: {latest_checkpoint_path}")
        return latest_checkpoint_path, completed_steps


# ============================================================================
# Gradient-norm instrumentation
# ============================================================================


def _module_trainable_params(model, path):
    m = model
    for attr in path.split("."):
        if not hasattr(m, attr):
            return []
        m = getattr(m, attr)
    if not isinstance(m, torch.nn.Module):
        return []
    return [p for p in m.parameters() if p.requires_grad]


def build_grad_norm_groups(model):
    """Split the trainable parameters into the shared backbone and the heads.

    CogWAM optimizes one objective per step, so the interesting decomposition is
    "how much of the gradient lands on the VLM the two streams share" versus
    "how much lands on the planner queries and the World--Action MoT". Missing
    modules do not create empty groups.
    """
    seen = set()

    def collect(paths):
        out = []
        for path in paths:
            for p in _module_trainable_params(model, path):
                if id(p) in seen:
                    continue
                seen.add(id(p))
                out.append(p)
        return out

    groups = {}
    shared = collect(["qwen_vl_interface"])
    if shared:
        groups["shared"] = shared
    head = collect(["action_plan_queries", "world_plan_queries", "action_model"])
    if head:
        groups["head"] = head
    return groups


def _full_grad(p, prefer_ds: bool):
    """Return a full gradient, gathering ZeRO shards when requested.

    ``safe_get_full_grad`` is collective and therefore must be called
    consistently on every rank. Non-DeepSpeed gradients are already global.
    """
    if prefer_ds:
        try:
            from deepspeed.utils import safe_get_full_grad

            g = safe_get_full_grad(p)
            if g is not None:
                return g
        except Exception:
            pass
    return getattr(p, "grad", None)


@torch.no_grad()
def group_grad_sqnorm(params, prefer_ds: bool):
    """Return the group's global squared L2 gradient norm on-device.

    The DeepSpeed path is collective. Aggregation remains on the GPU and only
    the final scalar needs host synchronization. Returns ``None`` if no
    parameter has a gradient.
    """
    total = None
    for p in params:
        g = _full_grad(p, prefer_ds)
        if g is None:
            continue
        s = g.detach().float().pow(2).sum()
        total = s if total is None else total + s
    return total


@torch.no_grad()
def group_grad_local_sqnorm(params):
    """Return the local-shard squared L2 gradient norm without a collective.

    Under ZeRO-2, ``safe_get_local_grad`` reads each rank's disjoint shard. One
    caller-side ``all_reduce(SUM)`` then recovers the global squared norm.
    Non-DeepSpeed execution falls back to ``p.grad`` and must not reduce it
    again. A tensor is always returned so every rank can follow one control
    path even when no gradients exist.
    """
    try:
        from deepspeed.utils import safe_get_local_grad
    except Exception:
        safe_get_local_grad = None
    total = None
    device = None
    for p in params:
        if device is None and hasattr(p, "device"):
            device = p.device
        g = None
        if safe_get_local_grad is not None:
            try:
                g = safe_get_local_grad(p)
            except Exception:
                g = None
        if g is None:
            g = getattr(p, "grad", None)
        if g is None or g.numel() == 0:
            continue
        s = g.detach().float().pow(2).sum()
        total = s if total is None else total + s
    if total is None:
        total = torch.zeros((), dtype=torch.float32, device=(device or "cpu"))
    return total


__all__ = [
    "TrainerUtils",
    "build_grad_norm_groups",
    "build_param_lr_groups",
    "group_grad_local_sqnorm",
    "group_grad_sqnorm",
    "normalize_dotlist_args",
]
