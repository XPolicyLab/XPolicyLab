from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from psi_policy.model.observation.encoders.base import BaseModalityEncoder


FEATURE_DIM_MAP = {
    "vit_small_r26_s32_224": 384,
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
    "vit_tiny_patch16_224": 192,
    "deit_tiny_patch16_224": 192,
    "vit_tiny_patch16_384": 192,
    "test_vit.r160_in1k": 192,
    "mobilevit_xxs": 320,
    "mobilevit_xs": 384,
}


class TimmRGBEncoder(BaseModalityEncoder):
    """共享一个 timm backbone，对多个 RGB 视角分别编码为 token。"""

    def __init__(
        self,
        *,
        key: str | None = None,
        keys: Sequence[str] | None = None,
        obs_shapes: Mapping[str, Sequence[int]] | None,
        obs_horizon: int = 1,
        model_name: str = "vit_small_r26_s32_224",
        image_size: int | Sequence[int] = 224,
        pretrained: bool = True,
        tokens_per_frame: int = 1,
        **_: object,
    ) -> None:
        resolved_keys = _resolve_keys(key=key, keys=keys)
        self._keys = resolved_keys
        super().__init__(obs_shapes=obs_shapes, obs_horizon=obs_horizon)

        self.model_name = str(model_name)
        self.image_size = _normalize_spatial_size(image_size)
        self.tokens_per_frame = int(tokens_per_frame)

        self.vit_head = self._build_vision_backbone(pretrained=pretrained)
        self._feature_dim = int(self._infer_feature_dim())

    @property
    def input_keys(self) -> Sequence[str]:
        return list(self._keys)

    @property
    def output_names(self) -> Sequence[str]:
        return list(self._keys)

    @property
    def embedding_dim(self) -> int:
        return int(self._feature_dim)

    def _build_vision_backbone(self, *, pretrained: bool) -> nn.Module:
        try:
            import timm  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("TimmRGBEncoder 依赖 timm，请安装后重试") from exc

        return timm.create_model(
            self.model_name,
            pretrained=bool(pretrained),
            global_pool="",
            num_classes=0,
        )

    def _infer_feature_dim(self) -> int:
        if self.model_name in FEATURE_DIM_MAP:
            return int(FEATURE_DIM_MAP[self.model_name])

        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size[0], self.image_size[1])
            tokens = self._forward_backbone_tokens(dummy)
            return int(tokens.shape[-1])

    def _forward_backbone_tokens(self, x: torch.Tensor) -> torch.Tensor:
        output = self.vit_head(x)
        if output.ndim == 2:
            return output[:, None, :]
        if output.ndim == 3:
            return output
        if output.ndim == 4:
            return output.flatten(2).transpose(1, 2)
        raise ValueError(f"unsupported backbone output shape: {tuple(output.shape)}")

    def _select_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.tokens_per_frame == 1:
            return tokens[:, :1]
        if self.tokens_per_frame > 1:
            return tokens[:, : self.tokens_per_frame]
        return tokens

    def build_example_obs(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        example: Dict[str, torch.Tensor] = {}
        for key in self._keys:
            example[key] = torch.zeros(
                (batch_size, self.obs_horizon, 3, self.image_size[0], self.image_size[1]),
                device=device,
                dtype=dtype,
            )
        return example

    def encode_rgb(self, rgb_data: torch.Tensor) -> torch.Tensor:
        bsz, steps = rgb_data.shape[:2]
        rgb = rgb_data.reshape(bsz * steps, 3, self.image_size[0], self.image_size[1])
        tokens = self._forward_backbone_tokens(rgb)
        tokens = self._select_tokens(tokens)
        return tokens.reshape(bsz, steps * tokens.shape[1], tokens.shape[-1])

    def validate_obs_contract(self, obs_dict: Mapping[str, torch.Tensor]) -> None:
        for key in self._keys:
            value = obs_dict.get(key)
            if value is None:
                raise KeyError(f"TimmRGBEncoder 缺少输入 key: {key}")
            _validate_canonical_rgb(value=value, key=key, image_size=self.image_size)

    def forward(self, obs_dict: Mapping[str, torch.Tensor], training: bool = True) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        for key in self._keys:
            outputs[key] = self.encode_rgb(obs_dict[key])

        return outputs



def _resolve_keys(*, key: str | None, keys: Sequence[str] | None) -> list[str]:
    candidates: list[str] = []
    if key is not None:
        candidates.append(str(key))
    if keys is not None:
        candidates.extend(str(v) for v in keys)

    cleaned: list[str] = []
    seen = set()
    for raw in candidates:
        cur = str(raw).strip()
        if not cur or cur in seen:
            continue
        seen.add(cur)
        cleaned.append(cur)

    if not cleaned:
        raise ValueError("TimmRGBEncoder 至少需要一个 key")
    return cleaned


def _normalize_spatial_size(raw: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
        if len(values) != 2:
            raise ValueError("image_size 列表长度必须为 2")
        height = int(values[0])
        width = int(values[1])
    else:
        size = int(raw)
        height = size
        width = size

    if height <= 0 or width <= 0:
        raise ValueError("image_size 必须为正整数")
    return (height, width)


def _validate_canonical_rgb(
    *,
    value: torch.Tensor,
    key: str,
    image_size: tuple[int, int],
) -> None:
    if value.ndim != 5:
        raise ValueError(f"obs[{key}] 必须为 5 维 [B,T,C,H,W]，实际 {tuple(value.shape)}")
    if value.shape[2] != 3:
        raise ValueError(f"obs[{key}] 必须为 channels-first RGB，实际 shape={tuple(value.shape)}")
    if not torch.is_floating_point(value):
        raise TypeError(f"obs[{key}] 必须为 float tensor，实际 dtype={value.dtype}")
    if tuple(value.shape[-2:]) != image_size:
        raise ValueError(
            f"obs[{key}] 空间尺寸必须为 {image_size}，实际 {tuple(value.shape[-2:])}"
        )
    if value.numel() == 0:
        return
    min_value = float(value.min().item())
    max_value = float(value.max().item())
    if min_value < -1.0 - 1.0e-4 or max_value > 1.0 + 1.0e-4:
        raise ValueError(
            f"obs[{key}] 期望 normalizer 后的 RGB 值域为 [-1,1]，"
            f"实际 min={min_value:.6f}, max={max_value:.6f}"
        )


__all__ = ["TimmRGBEncoder", "FEATURE_DIM_MAP"]
