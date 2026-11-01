"""DuoFlow 的原生联合策略 LoRA；保留完整视觉语言融合路径。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ARM_ACTION_DIM = 14
MODEL_ACTION_DIM = 32
ACTION_STEPS = 50


def _value(batch: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in batch and batch[key] is not None:
            return batch[key]
    return default


class ExpertLoRALinear(nn.Module):
    """冻结官方 Linear，只在 expert stream 上训练 FP32 rank-r 残差。"""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear) or rank <= 0 or alpha <= 0:
            raise ValueError("ExpertLoRALinear requires Linear, rank>0 and alpha>0")
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_a = nn.Parameter(
            torch.empty(self.rank, base.in_features, device=base.weight.device, dtype=torch.float32)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, self.rank, device=base.weight.device, dtype=torch.float32)
        )
        nn.init.normal_(self.lora_a, mean=0.0, std=1.0 / self.rank**0.5)

    def forward(self, value: Tensor) -> Tensor:
        base_dtype = self.base.weight.dtype
        result = self.base(value.to(dtype=base_dtype))
        residual = F.linear(value.to(dtype=self.lora_a.dtype), self.lora_a)
        residual = F.linear(residual.to(dtype=self.lora_b.dtype), self.lora_b)
        return result + (residual * self.scaling).to(dtype=result.dtype)

    def _apply(self, fn: Any) -> nn.Module:
        super()._apply(fn)
        with torch.no_grad():
            self.lora_a.data = self.lora_a.data.float()
            self.lora_b.data = self.lora_b.data.float()
        return self

    @property
    def weight(self) -> Tensor:
        """转发官方 Linear.weight，兼容 expert 的 dtype 读取。"""
        return self.base.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base.bias






class DuoFlowPolicy(nn.Module):
    """在上游骨干动作专家中挂载 FP32 LoRA。"""

    def __init__(self, model: nn.Module, mode: str = "FT") -> None:
        super().__init__()
        if mode != "FT":
            raise ValueError("DuoFlow uses the ordinary joint LoRA policy only")
        self.model = model
        self.mode = mode
        self._lora_paths: list[str] = []

    @staticmethod
    def _module_at(root: nn.Module, path: str) -> nn.Module:
        current: Any = root
        for part in path.split("."):
            current = current[int(part)] if part.isdigit() else getattr(current, part)
        return current

    @staticmethod
    def _set_module(root: nn.Module, path: str, value: nn.Module) -> None:
        parts = path.split(".")
        parent: Any = root
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        leaf = parts[-1]
        if leaf.isdigit():
            parent[int(leaf)] = value
        else:
            setattr(parent, leaf, value)

    def _install_expert_lora(self, rank: int, alpha: float) -> list[str]:
        layers = getattr(getattr(self.model, "paligemma_with_expert", None), "layers", None)
        if layers is None:
            raise AttributeError("official model lacks paligemma_with_expert.layers")
        paths: list[str] = []
        for index in range(len(layers)):
            paths.extend(
                f"paligemma_with_expert.layers.{index}.self_attn.{name}.1"
                for name in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
            paths.extend(
                f"paligemma_with_expert.layers.{index}.mlps.1.{name}"
                for name in ("gate_proj", "up_proj", "down_proj")
            )
        for path in paths:
            current = self._module_at(self.model, path)
            if isinstance(current, ExpertLoRALinear):
                continue
            if not isinstance(current, nn.Linear):
                raise TypeError(f"expected expert Linear at {path}, got {type(current).__name__}")
            self._set_module(self.model, path, ExpertLoRALinear(current, rank, alpha))
        self._lora_paths = paths
        return list(paths)

    @staticmethod
    def _is_lora(name: str) -> bool:
        return name.endswith(".lora_a") or name.endswith(".lora_b")

    @staticmethod
    def _is_adapter(name: str) -> bool:
        return False

    def configure_trainable(self, rank: int = 8, alpha: float = 16.0) -> dict[str, Any]:
        """冻结骨干参数，只训练动作专家的 LoRA。"""
        if rank <= 0 or alpha <= 0:
            raise ValueError("rank and alpha must be positive")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._install_expert_lora(int(rank), float(alpha))
        for name, parameter in self.model.named_parameters():
            if self._is_lora(name):
                parameter.requires_grad_(True)
        trainable = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        lora_names = [name for name in trainable if self._is_lora(name)]
        adapter_names = [name for name in trainable if self._is_adapter(name)]
        return {
            "mode": self.mode,
            "lora_rank": int(rank),
            "lora_alpha": float(alpha),
            "lora_paths": list(self._lora_paths),
            "lora_parameter_names": lora_names,
            "adapter_parameter_names": adapter_names,
            "lora_parameter_count": sum(trainable[name].numel() for name in lora_names),
            "adapter_parameter_count": sum(trainable[name].numel() for name in adapter_names),
            "trainable_parameter_count": sum(parameter.numel() for parameter in trainable.values()),
        }

    def training_prediction(
        self,
        batch: Mapping[str, Any],
        x_t: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        """调用官方完整 joint forward，并返回 physical `[B,50,14]` velocity。"""
        images = _value(batch, "images")
        img_masks = _value(batch, "img_masks", "image_masks")
        lang_tokens = _value(batch, "lang_tokens")
        lang_masks = _value(batch, "lang_masks")
        emb_ids = _value(batch, "emb_ids", "embodiment_id")
        proprioception = _value(batch, "proprioception", "observation.proprioception")
        required = {
            "images": images,
            "img_masks": img_masks,
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
            "emb_ids": emb_ids,
            "proprioception": proprioception,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise KeyError(f"missing official batch keys: {missing}")
        if x_t.ndim != 3 or tuple(x_t.shape[1:]) != (ACTION_STEPS, MODEL_ACTION_DIM):
            raise ValueError(f"x_t must be [B,50,32], got {tuple(x_t.shape)}")
        output = self.model(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            x_t,
            timestep,
            emb_ids,
            lang_att_masks=_value(batch, "lang_att_masks"),
            fast_action_indicator=_value(batch, "fast_action_indicator"),
            subtask_indicator=_value(batch, "subtask_indicator"),
            image_grid_thw=_value(batch, "image_grid_thw"),
            lang_loss_masks=_value(batch, "lang_loss_masks"),
            state_memory=_value(batch, "state_memory", "observation.state_memory"),
            state_memory_masks=_value(batch, "state_memory_masks", "observation.state_memory_mask"),
            proprioception=proprioception,
            agent_pos_mask=_value(batch, "agent_pos_mask", "observation.agent_pos_mask"),
            proprioception_present=_value(
                batch, "proprioception_present", "observation.proprioception_present"
            ),
            state=_value(batch, "state", "observation.state"),
            action_emb_ids=_value(batch, "action_emb_ids"),
            flow_matching_batch_mul=int(_value(batch, "flow_matching_batch_mul", default=1)),
        )
        velocity = output["v_t"]
        if velocity.ndim != 3 or tuple(velocity.shape[1:]) != (ACTION_STEPS, MODEL_ACTION_DIM):
            raise RuntimeError(f"official v_t must be [B,50,32], got {tuple(velocity.shape)}")
        return velocity[..., :ARM_ACTION_DIM]

    def trainable_state_dict(self) -> dict[str, dict[str, Tensor]]:
        """返回分组状态；``lora`` 可单独作为 FT/DuoFlow 共同初始化。"""
        lora: dict[str, Tensor] = {}
        adapter: dict[str, Tensor] = {}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            value = parameter.detach().cpu().clone()
            if self._is_lora(name):
                lora[name] = value
            elif self._is_adapter(name):
                adapter[name] = value
        return {"lora": lora, "adapter": adapter}

    def load_trainable_state_dict(
        self,
        state: Mapping[str, Any],
        strict: bool = True,
    ) -> list[str]:
        """加载完整状态，或只加载共享 ``lora`` 状态到 FT/DuoFlow。"""
        if not isinstance(state, Mapping):
            raise TypeError("trainable state must be a mapping")
        if "lora" in state or "adapter" in state:
            groups: dict[str, Mapping[str, Tensor]] = {
                "lora": state.get("lora", {}),
                "adapter": state.get("adapter", {}),
            }
        else:
            groups = {
                "lora": {name: value for name, value in state.items() if self._is_lora(str(name))},
                "adapter": {name: value for name, value in state.items() if self._is_adapter(str(name))},
            }
        parameters = dict(self.model.named_parameters())
        expected_lora = {name for name in parameters if self._is_lora(name)}
        expected_adapter = {name for name in parameters if self._is_adapter(name)}
        loaded: list[str] = []
        for group_name, values in groups.items():
            if not isinstance(values, Mapping):
                raise TypeError(f"state group {group_name!r} must be a mapping")
            for raw_name, value in values.items():
                name = str(raw_name)
                if name.startswith("model.") and name[6:] in parameters:
                    name = name[6:]
                parameter = parameters.get(name)
                if parameter is None:
                    if strict:
                        raise KeyError(f"unknown trainable parameter: {raw_name}")
                    continue
                if not parameter.requires_grad:
                    if strict:
                        raise ValueError(f"refusing frozen parameter: {name}")
                    continue
                if tuple(parameter.shape) != tuple(value.shape):
                    raise ValueError(f"shape mismatch for {name}: {tuple(value.shape)} vs {tuple(parameter.shape)}")
                with torch.no_grad():
                    parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
                loaded.append(name)
        provided_lora = set(groups["lora"])
        if strict and provided_lora and provided_lora != expected_lora:
            missing = sorted(expected_lora - provided_lora)
            if missing:
                raise KeyError(f"missing LoRA parameters: {missing[:5]}")
        if strict and groups["adapter"] and set(groups["adapter"]) != expected_adapter:
            missing = sorted(expected_adapter - set(groups["adapter"]))
            if missing:
                raise KeyError(f"missing adapter parameters: {missing[:5]}")
        if strict and not loaded and (groups["lora"] or groups["adapter"]):
            raise RuntimeError("no trainable parameters loaded")
        return sorted(set(loaded))


__all__ = [
    "ExpertLoRALinear",
    "DuoFlowPolicy",
]
