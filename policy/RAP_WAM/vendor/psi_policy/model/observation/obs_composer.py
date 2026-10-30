from __future__ import annotations

from collections.abc import Mapping as AbcMapping
from collections.abc import Sequence as AbcSequence
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import hydra
import torch
import torch.nn as nn

from psi_policy.common.obs_spec import resolve_obs_spec
from psi_policy.model.observation.concat.base import BaseConcatStrategy
from psi_policy.model.observation.concat.token_level import TokenLevelConcat
from psi_policy.model.observation.encoders.base import BaseModalityEncoder
from psi_policy.model.observation.fusions.base import BaseFusion


class ObsComposer(nn.Module):
    """v2 观测编排器：encoder 路由 -> 融合插件 -> token 拼接。"""

    def __init__(
            self,
            *,
            obs_spec: Any | None = None,
            obs_shapes: Mapping[str, Sequence[int]],
            obs_horizon: int = 1,
            obs_keys: Sequence[str] | None = None,
            encoders: Any | None = None,
            modalities: Any | None = None,
            fusions: Any | None = None,
            concat: Any | None = None,
            output_dim: int | None = None,
            **_: object,
    ) -> None:
        super().__init__()
        if output_dim is not None:
            raise ValueError("obs_composer.output_dim 已废弃，请迁移到 concat.output_dim")
        if modalities is not None:
            raise ValueError("obs_composer.modalities 已废弃，请改用 obs_composer.encoders")

        self._resolved_obs = _resolve_obs_spec(obs_spec=obs_spec, obs_keys=obs_keys)
        self._obs_keys = list(self._resolved_obs.get("obs_keys", []))
        if not self._obs_keys:
            raise ValueError("obs_spec 不能为空：至少需要一个 obs key")

        self._obs_shapes = _normalize_obs_shapes(obs_shapes)
        self.obs_horizon = int(obs_horizon)
        if self.obs_horizon <= 0:
            raise ValueError("obs_horizon 必须为正整数")

        self._obs_keys_resolved = _clean_keys(self._obs_keys)
        missing = [k for k in self._obs_keys_resolved if k not in self._obs_shapes]
        if missing:
            raise ValueError("obs_keys 在 manifest 中缺失: " + ", ".join(sorted(missing)))

        self.encoder_modules = nn.ModuleList(self._build_encoders(encoders=encoders))
        if len(self.encoder_modules) == 0:
            raise ValueError("至少需要一个 encoder")
        self._token_input_dims = self._collect_token_input_dims()

        self.fusion_modules = nn.ModuleList(self._build_fusions(fusions=fusions))
        self.concat_strategy = self._build_concat_strategy(concat=concat, input_dims=self._token_input_dims)

    @property
    def obs_keys(self) -> Sequence[str]:
        return list(self._obs_keys)

    @property
    def obs_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return dict(self._obs_shapes)

    @property
    def obs_keys_resolved(self) -> list[str]:
        return list(self._obs_keys_resolved)

    @property
    def output_dim(self) -> int:
        return int(self.concat_strategy.output_dim)

    @property
    def device(self) -> torch.device:
        param = next(self.parameters(), None)
        return param.device if param is not None else torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        param = next(self.parameters(), None)
        return param.dtype if param is not None else torch.float32

    @torch.no_grad()
    def output_shape(self) -> tuple[tuple[int, int, int], list[int]]:
        example = self.build_example_obs(batch_size=1, device=self.device, dtype=self.dtype)
        encoded = self.forward(example, training=False)

        if encoded.ndim != 3:
            raise ValueError(f"obs_composer 输出必须为 3 维 [B,L,D]，实际 {tuple(encoded.shape)}")
        if encoded.shape[0] != 1:
            raise ValueError(f"output_shape 推导期望 batch=1，实际 {encoded.shape[0]}")
        if int(encoded.shape[2]) != int(self.output_dim):
            raise ValueError(
                f"output_dim 与 forward 输出不一致: output_dim={self.output_dim}, got={encoded.shape[2]}"
            )

        obs_part_length = self.get_obs_part_length(example_obs=example,training=False)
        part_lengths = [int(v) for v in obs_part_length]
        if not part_lengths:
            raise ValueError("obs_part_length 不能为空")
        if any(v <= 0 for v in part_lengths):
            raise ValueError(f"obs_part_length 必须全部为正整数，实际 {part_lengths}")
        if sum(part_lengths) != int(encoded.shape[1]):
            raise ValueError(
                "obs_part_length 与 token_len 不一致: "
                f"sum(obs_part_length)={sum(part_lengths)}, token_len={encoded.shape[1]}"
            )

        shape = (int(encoded.shape[0]), int(encoded.shape[1]), int(encoded.shape[2]))
        return shape, part_lengths

    def build_example_obs(
            self,
            *,
            batch_size: int = 1,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> Dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")

        device = device or torch.device("cpu")
        dtype = dtype or torch.float32

        example: Dict[str, torch.Tensor] = {}
        for module in self.encoder_modules:
            builder = getattr(module, "build_example_obs", None)
            if not callable(builder):
                continue
            module_example = builder(batch_size=batch_size, device=device, dtype=dtype)
            if not isinstance(module_example, AbcMapping):
                raise TypeError(f"{type(module).__name__}.build_example_obs 必须返回 Mapping，实际 {type(module_example)}")
            example.update({str(k): v for k, v in module_example.items()})

        for key in self._obs_keys_resolved:
            if key in example:
                continue
            tail = _shape_for_tensor(self._obs_shapes[key])
            example[key] = torch.zeros((batch_size, self.obs_horizon) + tail, dtype=dtype, device=device)
        return example

    def _build_encoders(self, *, encoders: Any | None) -> list[BaseModalityEncoder]:
        if encoders is None:
            raise ValueError("ObsComposer 需要通过 encoders 显式声明 encoder")

        modules: list[BaseModalityEncoder] = []
        for idx, cfg in enumerate(encoders):
            module = _instantiate_component(
                cfg=cfg,
                base_type=BaseModalityEncoder,
                name=f"encoder[{idx}]",
                init_kwargs=(
                    {"obs_shapes": self.obs_shapes, "obs_horizon": self.obs_horizon, "obs_spec": self._resolved_obs},
                    {"obs_shapes": self.obs_shapes, "obs_horizon": self.obs_horizon},
                    {},
                ),
            )
            modules.append(module)
        _validate_unique_output_names(modules)
        return modules

    def _build_fusions(self, *, fusions: Any | None) -> list[BaseFusion]:
        if fusions is None:
            return []

        modules: list[BaseFusion] = []
        for idx, cfg in enumerate(_iter_component_cfgs(fusions)):
            fusion = _instantiate_component(
                cfg=cfg,
                base_type=BaseFusion,
                name=f"fusion[{idx}]",
                init_kwargs=(
                    {"obs_shapes": self.obs_shapes, "obs_horizon": self.obs_horizon, "obs_spec": self._resolved_obs},
                    {},
                ),
            )
            modules.append(fusion)
        return modules

    def _build_concat_strategy(
            self,
            *,
            concat: Any | None,
            input_dims: Mapping[str, int],
    ) -> BaseConcatStrategy:
        if concat is None:
            raise ValueError(
                "ObsComposer 需要显式配置 concat，且 concat.order 必须显式声明"
            )

        strategy = _instantiate_component(
            cfg=concat,
            base_type=BaseConcatStrategy,
            name="concat",
            init_kwargs=(
                {
                    "obs_shapes": self.obs_shapes,
                    "obs_horizon": self.obs_horizon,
                    "obs_spec": self._resolved_obs,
                    "input_dims": input_dims,
                },
                {"input_dims": input_dims},
                {"obs_shapes": self.obs_shapes, "obs_horizon": self.obs_horizon, "obs_spec": self._resolved_obs},
                {},
            ),
        )

        binder = getattr(strategy, "bind_input_dims", None)
        if callable(binder):
            binder(input_dims)
        return strategy

    def _collect_token_input_dims(self) -> Dict[str, int]:
        merged: Dict[str, int] = {}
        for module in self.encoder_modules:
            dim = int(module.embedding_dim)
            if dim <= 0:
                raise ValueError(
                    f"{type(module).__name__}.embedding_dim 必须为正整数，实际 {dim}"
                )
            for name in module.output_names:
                merged[str(name)] = dim
        return merged

    def _encode_all_modalities(
            self,
            obs_dict: Mapping[str, torch.Tensor],
            *,
            training: bool,
    ) -> Dict[str, torch.Tensor]:
        token_map: Dict[str, torch.Tensor] = {}
        for module in self.encoder_modules:
            outputs = module(obs_dict, training=training)
            _validate_encoder_outputs(module=module, outputs=outputs)
            token_map.update(outputs)
        _validate_batch_consistency(token_map)
        return token_map

    def _apply_fusions(
            self,
            token_map: Dict[str, torch.Tensor],
            obs_dict: Mapping[str, torch.Tensor],
            *,
            training: bool,
    ) -> Dict[str, torch.Tensor]:
        fused = token_map
        for fusion in self.fusion_modules:
            validator = getattr(fusion, "validate_token_map", None)
            if callable(validator):
                validator(fused)
            fused = fusion(token_map=fused, obs_dict=obs_dict, training=training)
            _validate_token_map_contract(
                token_map=fused,
                expected_dims=self._token_input_dims,
                source=f"type={type(fusion).__name__}",
            )
        return fused

    def _encode_and_concat(
            self,
            obs_dict: Mapping[str, torch.Tensor],
            *,
            training: bool,
    ) -> tuple[torch.Tensor, list[int]]:
        token_map = self._encode_all_modalities(obs_dict, training=training)
        fused = self._apply_fusions(token_map, obs_dict, training=training)
        encoded, part_lengths = self.concat_strategy(token_map=fused)
        return encoded, [int(v) for v in part_lengths]

    def forward(self, obs_dict: Dict[str, torch.Tensor], training: bool = True) -> torch.Tensor:
        encoded, _ = self._encode_and_concat(obs_dict, training=training)
        return encoded

    def validate_obs_contract(self, obs_dict: Mapping[str, torch.Tensor]) -> None:
        validated = False
        for module in self.encoder_modules:
            validator = getattr(module, "validate_obs_contract", None)
            if callable(validator):
                validator(obs_dict)
                validated = True

        if not validated:
            raise ValueError(
                "ObsComposer 未找到任何模态输入契约校验器。"
                "请为模态 encoder 实现 validate_obs_contract()。"
            )

    @torch.no_grad()
    def get_obs_part_length(
            self,
            *,
            example_obs: Dict[str, torch.Tensor],
            training: bool = False,
    ) -> list[int]:
        _, part_lengths = self._encode_and_concat(example_obs, training=training)
        return part_lengths


def _iter_component_cfgs(raw: Any) -> Iterable[Any]:
    if isinstance(raw, AbcSequence) and not isinstance(raw, (str, bytes, bytearray)):
        return list(raw)

    if isinstance(raw, AbcMapping):
        if "_target_" in raw:
            return [raw]
        return list(raw.values())

    return [raw]


def _instantiate_component(
        *,
        cfg: Any,
        base_type: type,
        name: str,
        init_kwargs: Sequence[Mapping[str, Any]],
):
    if isinstance(cfg, base_type):
        return cfg

    if isinstance(cfg, nn.Module):
        raise TypeError(f"{name} 类型不匹配，期望 {base_type.__name__}，实际 {type(cfg).__name__}")

    errors: list[Exception] = []
    for kwargs in init_kwargs:
        try:
            instance = hydra.utils.instantiate(cfg, **dict(kwargs))
            if not isinstance(instance, base_type):
                raise TypeError(
                    f"{name} 实例类型不匹配，期望 {base_type.__name__}，实际 {type(instance).__name__}"
                )
            return instance
        except Exception as exc:
            if _is_unexpected_kw_error(exc):
                errors.append(exc)
                continue
            raise

    try:
        instance = hydra.utils.instantiate(cfg)
    except Exception as exc:
        details = "\n".join(str(e) for e in errors)
        raise RuntimeError(f"实例化 {name} 失败：{exc}\n{details}") from exc

    if not isinstance(instance, base_type):
        raise TypeError(f"{name} 实例类型不匹配，期望 {base_type.__name__}，实际 {type(instance).__name__}")
    return instance


def _is_unexpected_kw_error(exc: Exception) -> bool:
    message = str(exc)
    return "unexpected keyword argument" in message


def _validate_unique_output_names(modules: Sequence[BaseModalityEncoder]) -> None:
    seen = set()
    duplicate: list[str] = []
    for module in modules:
        for name in module.output_names:
            key = str(name)
            if key in seen:
                duplicate.append(key)
                continue
            seen.add(key)

    if duplicate:
        dup_text = ", ".join(sorted(set(duplicate)))
        raise ValueError(f"模态输出名重复: {dup_text}")


def _validate_encoder_outputs(
        *,
        module: BaseModalityEncoder,
        outputs: Mapping[str, torch.Tensor],
) -> None:
    if not isinstance(outputs, AbcMapping):
        raise TypeError(f"{type(module).__name__}.forward 必须返回 Mapping，实际 {type(outputs)}")

    expected_names = [str(v) for v in module.output_names]
    actual_names = [str(v) for v in outputs.keys()]
    missing = [name for name in expected_names if name not in outputs]
    unexpected = [name for name in actual_names if name not in set(expected_names)]
    if missing or unexpected:
        raise ValueError(
            f"{type(module).__name__} 输出 key 不符合声明: "
            f"missing={missing}, unexpected={unexpected}"
        )

    expected_dim = int(module.embedding_dim)
    if expected_dim <= 0:
        raise ValueError(f"{type(module).__name__}.embedding_dim 必须为正整数，实际 {expected_dim}")

    for name in expected_names:
        _validate_token_tensor(name=name, token=outputs[name], expected_dim=expected_dim)


def _validate_token_map_contract(
        *,
        token_map: Mapping[str, torch.Tensor],
        expected_dims: Mapping[str, int],
        source: str,
) -> None:
    actual_keys = {str(v) for v in token_map.keys()}
    expected_keys = {str(v) for v in expected_dims.keys()}
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"{source} 输出 token 集合不符合约束: missing={missing}, unexpected={unexpected}"
        )

    for name, dim in expected_dims.items():
        _validate_token_tensor(name=str(name), token=token_map[str(name)], expected_dim=int(dim))
    _validate_batch_consistency(token_map)


def _validate_token_tensor(*, name: str, token: torch.Tensor, expected_dim: int) -> None:
    if not isinstance(token, torch.Tensor):
        raise TypeError(f"token_map[{name}] 必须为 Tensor，实际 {type(token)}")
    if token.ndim != 3:
        raise ValueError(f"token_map[{name}] 必须为 [B,L,D]，实际 {tuple(token.shape)}")
    if int(token.shape[-1]) != int(expected_dim):
        raise ValueError(
            f"token_map[{name}] 输出维度不一致: expected={expected_dim}, actual={token.shape[-1]}"
        )


def _validate_batch_consistency(token_map: Mapping[str, torch.Tensor]) -> None:
    bsz: int | None = None
    for name, token in token_map.items():
        current = int(token.shape[0])
        if bsz is None:
            bsz = current
            continue
        if current != bsz:
            raise ValueError(f"batch 维不一致：{name} 的 batch={current}, expected={bsz}")


def _resolve_obs_spec(*, obs_spec: Any | None, obs_keys: Sequence[str] | None) -> dict[str, Any]:
    if obs_spec is not None:
        return resolve_obs_spec(obs_spec)

    if obs_keys is None:
        raise ValueError("obs_spec 不能为空")

    keys = _clean_keys(obs_keys)
    return {
        "obs_keys": keys,
        "groups": {},
    }


def _clean_keys(raw_values: Any) -> list[str]:
    if raw_values is None:
        return []

    if isinstance(raw_values, (str, bytes)):
        candidates = [raw_values]
    else:
        try:
            candidates = list(iter(raw_values))
        except TypeError:
            candidates = [raw_values]

    cleaned: list[str] = []
    seen = set()
    for raw in candidates:
        key = str(raw).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(key)
    return cleaned


def _normalize_obs_shapes(obs_shapes: Mapping[str, Sequence[int]]) -> Dict[str, Tuple[int, ...]]:
    normalized: Dict[str, Tuple[int, ...]] = {}
    for raw_key, raw_shape in obs_shapes.items():
        key = str(raw_key).strip()
        if not key:
            continue

        if raw_shape is None:
            shape = ()
        elif isinstance(raw_shape, AbcSequence) and not isinstance(raw_shape, (str, bytes, bytearray)):
            shape = tuple(int(v) for v in raw_shape)
        else:
            shape = (int(raw_shape),)

        if any(v < 0 for v in shape):
            raise ValueError(f"obs_shapes[{key}] 不能包含负维度: {shape}")

        normalized[key] = shape
    return normalized


def _shape_for_tensor(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(shape) == 0:
        return (1,)
    return tuple(int(v) for v in shape)


__all__ = ["ObsComposer"]
