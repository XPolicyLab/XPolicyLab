"""Train-aligned RoboDojo ME-U0 state/action normalization codec.

RoboDojo/XPolicyLab exposes ARX-X5 joint vectors in native order::

    [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]

The ME_U0 map-style training config selects those values into named fields,
normalizes them with ME-U0's ``LinearNormalizer`` and concatenates them in
``shape_meta`` order::

    [left_arm(6), right_arm(6), left_gripper(1), right_gripper(1), pad(12)]

This module builds the exact same joint-action transforms and normalizer from
the resolved experiment config and source-derived stats JSON. It deliberately
does not infer statistics from evaluation observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from leap.data.me_u0.utils.normalizer import LinearNormalizer, load_dataset_stats_from_json


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _metadata_dim(items: Iterable[Mapping[str, Any]]) -> int:
    return sum(int(item.get("shape", item.get("raw_shape"))) for item in items)


@dataclass(frozen=True)
class RoboDojoCodecContract:
    dataset_name: str
    source_stats_path: str
    norm_mode: str
    action_type: str
    action_state_transforms: tuple[str, ...]
    normalized_action_clip_value: Optional[float]
    raw_action_dim: int
    raw_state_dim: int
    canonical_action_dim: int
    canonical_state_dim: int
    model_action_dim: int
    model_state_dim: int


class MEU0RoboDojoCodec:
    """Bidirectional native-XPolicy ↔ normalized-ME_U0 codec."""

    def __init__(
        self,
        *,
        dataset_name: str,
        shape_meta: Mapping[str, Any],
        stats_path: str,
        norm_mode: str,
        use_stepwise_action_norm: bool,
        action_target_dim: int,
        state_target_dim: int,
        action_type: str = "joint",
        action_state_transforms: Optional[Iterable[Any]] = None,
        norm_exception_mode: Optional[Mapping[str, Any]] = None,
        clip_normalized_actions: bool = True,
        normalized_action_clip_value: float = 1.0,
        clip_grippers: bool = True,
    ) -> None:
        self.dataset_name = str(dataset_name)
        self.shape_meta = _plain(shape_meta)
        self.stats_path = str(stats_path)
        self.norm_mode = str(norm_mode)
        self.action_type = str(action_type)
        if self.action_type != "joint":
            raise ValueError(
                "ME_U0 RoboDojo codec only supports joint actions, "
                f"got {self.action_type!r}"
            )
        self.action_state_transforms = list(action_state_transforms or [])
        self.action_target_dim = int(action_target_dim)
        self.state_target_dim = int(state_target_dim)
        self.clip_normalized_actions = bool(clip_normalized_actions)
        self.normalized_action_clip_value = float(normalized_action_clip_value)
        if self.normalized_action_clip_value <= 0:
            raise ValueError("normalized_action_clip_value must be positive")
        self.clip_grippers = bool(clip_grippers)

        self._validate_meta("action")
        self._validate_meta("state")
        self.raw_action_dim = self._raw_dim("action")
        self.raw_state_dim = self._raw_dim("state")
        self.canonical_action_dim = _metadata_dim(self.shape_meta["action"])
        self.canonical_state_dim = _metadata_dim(self.shape_meta["state"])
        if self.canonical_action_dim > self.action_target_dim:
            raise ValueError("canonical action dimensions exceed action_target_dim")
        if self.canonical_state_dim > self.state_target_dim:
            raise ValueError("canonical state dimensions exceed state_target_dim")

        stats = load_dataset_stats_from_json(self.stats_path)
        self.normalizer = LinearNormalizer(
            shape_meta=self.shape_meta,
            use_stepwise_action_norm=bool(use_stepwise_action_norm),
            default_mode=self.norm_mode,
            exception_mode=_plain(norm_exception_mode),
            stats=stats,
        )
        self.contract = RoboDojoCodecContract(
            dataset_name=self.dataset_name,
            source_stats_path=self.stats_path,
            norm_mode=self.norm_mode,
            action_type=self.action_type,
            action_state_transforms=tuple(
                type(transform).__name__ for transform in self.action_state_transforms
            ),
            normalized_action_clip_value=(
                self.normalized_action_clip_value
                if self.clip_normalized_actions
                else None
            ),
            raw_action_dim=self.raw_action_dim,
            raw_state_dim=self.raw_state_dim,
            canonical_action_dim=self.canonical_action_dim,
            canonical_state_dim=self.canonical_state_dim,
            model_action_dim=self.action_target_dim,
            model_state_dim=self.state_target_dim,
        )

    def _validate_meta(self, kind: str) -> None:
        items = self.shape_meta.get(kind)
        if not items:
            raise ValueError(f"shape_meta.{kind} is empty")
        occupied: set[int] = set()
        for item in items:
            key = str(item["key"])
            indices = [int(index) for index in item.get("select_indices", [])]
            raw_width = int(item.get("raw_shape", len(indices)))
            if len(indices) != raw_width:
                raise ValueError(
                    f"shape_meta.{kind}.{key}: select_indices={indices} "
                    f"does not match raw_shape={raw_width}"
                )
            overlap = occupied.intersection(indices)
            if overlap:
                raise ValueError(f"shape_meta.{kind}.{key}: duplicate native indices {sorted(overlap)}")
            occupied.update(indices)

    def _raw_dim(self, kind: str) -> int:
        indices = [
            int(index)
            for item in self.shape_meta[kind]
            for index in item.get("select_indices", [])
        ]
        if not indices:
            raise ValueError(f"shape_meta.{kind} has no select_indices")
        expected = set(range(max(indices) + 1))
        if set(indices) != expected:
            raise ValueError(
                f"shape_meta.{kind} must cover native dimensions 0..{max(indices)}, got {sorted(indices)}"
            )
        return max(indices) + 1

    @staticmethod
    def _as_2d(value: Any, expected_dim: int, label: str) -> torch.Tensor:
        tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[-1] != expected_dim:
            raise ValueError(
                f"{label} must have shape ({expected_dim},) or (N,{expected_dim}), got {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{label} contains NaN/Inf")
        return tensor

    def _native_to_fields(self, native: torch.Tensor, kind: str) -> dict[str, torch.Tensor]:
        fields: dict[str, torch.Tensor] = {}
        for item in self.shape_meta[kind]:
            indices = torch.as_tensor(item["select_indices"], dtype=torch.long)
            fields[str(item["key"])] = native.index_select(-1, indices)
        return fields

    def _fields_to_canonical(self, fields: Mapping[str, torch.Tensor], kind: str) -> torch.Tensor:
        return torch.cat([fields[str(item["key"])] for item in self.shape_meta[kind]], dim=-1)

    def _canonical_to_fields(self, canonical: torch.Tensor, kind: str) -> dict[str, torch.Tensor]:
        fields: dict[str, torch.Tensor] = {}
        offset = 0
        for item in self.shape_meta[kind]:
            width = int(item["shape"])
            fields[str(item["key"])] = canonical[:, offset : offset + width]
            offset += width
        return fields

    def _fields_to_native(self, fields: Mapping[str, torch.Tensor], kind: str) -> torch.Tensor:
        raw_dim = self.raw_action_dim if kind == "action" else self.raw_state_dim
        first = next(iter(fields.values()))
        native = torch.empty((first.shape[0], raw_dim), dtype=first.dtype, device=first.device)
        for item in self.shape_meta[kind]:
            native[:, item["select_indices"]] = fields[str(item["key"])]
        return native

    def _transform_batch(
        self,
        *,
        action: Optional[dict[str, torch.Tensor]],
        state: dict[str, torch.Tensor],
        inverse: bool,
    ) -> tuple[Optional[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        batch: dict[str, Any] = {"state": state}
        if action is not None:
            batch["action"] = action
        transforms = (
            reversed(self.action_state_transforms)
            if inverse
            else self.action_state_transforms
        )
        for transform in transforms:
            batch = transform.backward(batch) if inverse else transform.forward(batch)
        return batch.get("action"), batch["state"]

    def _state_fields(self, native_state: Any) -> dict[str, torch.Tensor]:
        native = self._as_2d(native_state, self.raw_state_dim, "RoboDojo native state")
        return self._native_to_fields(native, "state")

    def encode_state(self, native_state: Any) -> torch.Tensor:
        """Return normalized canonical state padded to the model's 26-D width."""
        fields = self._state_fields(native_state)
        _, fields = self._transform_batch(action=None, state=fields, inverse=False)
        for key, field_norm in self.normalizer.normalizers["state"].items():
            fields[key] = field_norm.forward(fields[key])
        canonical = self._fields_to_canonical(fields, "state")
        return torch.nn.functional.pad(canonical, (0, self.state_target_dim - canonical.shape[-1]))

    def encode_action(self, native_action: Any, native_state: Any = None) -> torch.Tensor:
        """Normalize a native action; primarily used by the round-trip doctor test."""
        native = self._as_2d(native_action, self.raw_action_dim, "RoboDojo native action")
        fields = self._native_to_fields(native, "action")
        if self.action_state_transforms:
            if native_state is None:
                raise ValueError("native_state is required by RoboDojo delta action transforms")
            fields, _ = self._transform_batch(
                action=fields,
                state=self._state_fields(native_state),
                inverse=False,
            )
            assert fields is not None
        for key, field_norm in self.normalizer.normalizers["action"].items():
            fields[key] = field_norm.forward(fields[key])
        canonical = self._fields_to_canonical(fields, "action")
        return torch.nn.functional.pad(canonical, (0, self.action_target_dim - canonical.shape[-1]))

    def decode_action(self, normalized_action: Any, native_state: Any = None) -> np.ndarray:
        """Inverse source-stats normalization and restore XPolicyLab's native order."""
        tensor = torch.as_tensor(normalized_action, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[-1] < self.canonical_action_dim:
            raise ValueError(
                "normalized action must be (T,D) with "
                f"D>={self.canonical_action_dim}, got {tuple(tensor.shape)}"
            )
        canonical = tensor[:, : self.canonical_action_dim]
        if self.clip_normalized_actions:
            canonical = canonical.clamp(
                -self.normalized_action_clip_value,
                self.normalized_action_clip_value,
            )
        fields = self._canonical_to_fields(canonical, "action")
        for key, field_norm in self.normalizer.normalizers["action"].items():
            fields[key] = field_norm.backward(fields[key])
        if self.action_state_transforms:
            if native_state is None:
                raise ValueError("native_state is required by RoboDojo delta action transforms")
            fields, _ = self._transform_batch(
                action=fields,
                state=self._state_fields(native_state),
                inverse=True,
            )
            assert fields is not None
        native = self._fields_to_native(fields, "action")
        if self.clip_grippers:
            for item in self.shape_meta["action"]:
                if "gripper" in str(item["key"]).lower():
                    native[:, item["select_indices"]] = native[:, item["select_indices"]].clamp(0.0, 1.0)
        return native.cpu().numpy().astype(np.float32)

    def assert_round_trip(
        self,
        native_action: Any,
        native_state: Any = None,
        *,
        atol: float = 2e-5,
    ) -> float:
        """Golden check that config ordering and source stats are mutually consistent."""
        old_clip = self.clip_normalized_actions
        self.clip_normalized_actions = False
        try:
            encoded = self.encode_action(native_action, native_state=native_state)
            decoded = self.decode_action(encoded, native_state=native_state)
        finally:
            self.clip_normalized_actions = old_clip
        expected = np.asarray(native_action, dtype=np.float32).reshape(decoded.shape)
        error = float(np.max(np.abs(decoded - expected)))
        if error > atol:
            raise AssertionError(f"RoboDojo action codec round-trip error {error} > {atol}")
        return error


def build_robodojo_codecs_from_config(
    cfg: Any,
    *,
    clip_normalized_actions: Optional[bool] = None,
    stats_path_override: Optional[str] = None,
) -> dict[str, MEU0RoboDojoCodec]:
    """Build codecs for RoboDojo map-style datasets in a resolved experiment config.

    ``stats_path_override`` is the serving-time equivalent of the public
    evaluator's ``--stats`` option. A single path cannot be applied
    unambiguously when a config contains multiple RoboDojo datasets, so fail
    explicitly in that case instead of silently assigning the wrong artifact.
    """
    codecs: dict[str, MEU0RoboDojoCodec] = {}
    action_type = str(cfg.get("robodojo_action_type", "joint"))
    if clip_normalized_actions is None:
        clip_normalized_actions = bool(
            cfg.get("robodojo_clip_normalized_actions", True)
        )
    from leap.core.config import instantiate

    train = cfg["data"]["train"]
    robodojo_datasets = []
    for dataset in train.get("datasets", []) or []:
        name = str(dataset.get("dataset_name", ""))
        target = str(dataset.get("_target_", ""))
        if "robodojo" not in name.lower() and "robodojo" not in target.lower():
            continue
        robodojo_datasets.append(dataset)

    if stats_path_override is not None and len(robodojo_datasets) != 1:
        raise ValueError(
            "stats_path_override requires exactly one RoboDojo dataset, "
            f"found {len(robodojo_datasets)}"
        )

    for dataset in robodojo_datasets:
        name = str(dataset.get("dataset_name", ""))
        processor = dataset["processor"]
        merger = processor["action_state_merger"]
        transform_cfg = processor.get("action_state_transforms")
        transforms = [] if transform_cfg is None else list(instantiate(transform_cfg))
        codecs[name] = MEU0RoboDojoCodec(
            dataset_name=name,
            shape_meta=dataset["shape_meta"],
            stats_path=str(
                stats_path_override
                if stats_path_override is not None
                else dataset["pretrained_norm_stats"]
            ),
            norm_mode=str(processor["norm_default_mode"]),
            use_stepwise_action_norm=bool(processor.get("use_stepwise_action_norm", False)),
            action_target_dim=int(merger["action_target_dim"]),
            state_target_dim=int(merger["state_target_dim"]),
            action_type=action_type,
            action_state_transforms=transforms,
            norm_exception_mode=processor.get("norm_exception_mode"),
            clip_normalized_actions=clip_normalized_actions,
            normalized_action_clip_value=float(
                cfg.get("robodojo_normalized_action_clip_value", 1.0)
            ),
        )
    if not codecs:
        raise ValueError("No RoboDojo map-style dataset found at cfg.data.train.datasets")
    return codecs
