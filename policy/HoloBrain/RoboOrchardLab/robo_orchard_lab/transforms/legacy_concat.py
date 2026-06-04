# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


from __future__ import annotations
from typing import Any, Mapping, Sequence

from pydantic import Field
from robo_orchard_core.utils.config import ClassType
from typing_extensions import deprecated

from robo_orchard_lab.transforms.base import (
    DictTransform,
    DictTransformPipeline,
    DictTransformPipelineConfig,
    _flatten_runtime_transforms,
)

_LEGACY_CONCAT_VALIDATED_FIELDS = {
    "input_columns",
    "missing_input_columns_as_none",
    "output_column_mapping",
    "check_return_columns",
    "keep_input_columns",
}
_LEGACY_CONCAT_DEFAULT_FIELD_VALUES = {
    "input_columns": None,
    "missing_input_columns_as_none": False,
    "output_column_mapping": {},
    "check_return_columns": False,
    "keep_input_columns": True,
}


class _ImmutableLegacyOutputColumnMapping(dict[str, str]):
    """Read-only outer mapping placeholder for legacy concat configs."""

    def _raise_mutation(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "ConcatDictTransformConfig.output_column_mapping is read-only. "
            "Configure output_column_mapping on child transforms instead."
        )

    __delitem__ = _raise_mutation
    __setitem__ = _raise_mutation
    clear = _raise_mutation
    pop = _raise_mutation
    popitem = _raise_mutation
    setdefault = _raise_mutation
    update = _raise_mutation


def _freeze_legacy_output_column_mapping(
    value: Mapping[str, str] | None,
) -> _ImmutableLegacyOutputColumnMapping:
    if value is None:
        return _ImmutableLegacyOutputColumnMapping()
    return _ImmutableLegacyOutputColumnMapping(dict(value))


def _get_legacy_concat_field_values(
    config: ConcatDictTransformConfig | None = None,
    /,
    **overrides: Any,
) -> dict[str, Any]:
    values = dict(_LEGACY_CONCAT_DEFAULT_FIELD_VALUES)
    if config is not None:
        config_dict = getattr(config, "__dict__", {})
        for field_name in _LEGACY_CONCAT_VALIDATED_FIELDS:
            if field_name in config_dict:
                values[field_name] = getattr(config, field_name)
    values.update(overrides)
    return values


def _validate_legacy_concat_field_values(values: Mapping[str, Any]) -> None:
    if values["input_columns"] is not None:
        raise ValueError(
            "ConcatDictTransformConfig does not support input_columns. "
            "Configure legacy concat behavior on child transforms instead."
        )
    if values["missing_input_columns_as_none"] is not False:
        raise ValueError(
            "ConcatDictTransformConfig does not support "
            "missing_input_columns_as_none. Configure legacy concat "
            "behavior on child transforms instead."
        )
    if values["output_column_mapping"]:
        raise ValueError(
            "ConcatDictTransformConfig does not support "
            "output_column_mapping. Configure legacy concat behavior on "
            "child transforms instead."
        )
    if values["check_return_columns"] is not False:
        raise ValueError(
            "ConcatDictTransformConfig does not support "
            "check_return_columns. Configure legacy concat behavior on "
            "child transforms instead."
        )
    if values["keep_input_columns"] is not True:
        raise ValueError(
            "ConcatDictTransformConfig does not support "
            "keep_input_columns. Configure legacy concat behavior on "
            "child transforms instead."
        )


def _raise_unsupported_leaf_api(name: str, replacement: str) -> RuntimeError:
    return RuntimeError(
        f"ConcatDictTransform does not implement {name}. "
        f"Use the {replacement} instead."
    )


@deprecated(
    "ConcatDictTransform is deprecated. Use DictTransformPipeline instead."
)
class ConcatDictTransform(DictTransformPipeline):
    """Legacy compatibility wrapper around DictTransformPipeline.

    ``ConcatDictTransform`` stays ``DictTransform``-compatible for legacy
    ``isinstance`` checks, but it does not support leaf-only APIs such as
    ``input_columns``, ``output_columns``, or ``transform(...)``.
    ``+`` no longer returns ``ConcatDictTransform``; construct it explicitly
    only when a deprecated compatibility shim is still required.

    Use ``__call__(row)`` or ``apply(row)`` for row-aware execution.

    Example:
        >>> transform = ConcatDictTransformConfig(transforms=[...])()
        >>> final_row = transform({"value": 1})
    """

    cfg: ConcatDictTransformConfig

    @classmethod
    def from_transforms(
        cls,
        transforms: Sequence[DictTransform[Any] | DictTransformPipeline],
    ) -> ConcatDictTransform:
        instance = cls.__new__(cls)
        instance._transforms = _flatten_runtime_transforms(transforms)
        instance.cfg = ConcatDictTransformConfig(
            transforms=[transform.cfg for transform in instance._transforms]
        )
        return instance

    @property
    def input_columns(self) -> list[str]:
        raise _raise_unsupported_leaf_api(
            "input_columns",
            "mapped_input_columns property",
        )

    @property
    def output_columns(self) -> list[str]:
        raise _raise_unsupported_leaf_api(
            "output_columns",
            "mapped_output_columns property",
        )

    def transform(self, **kwargs) -> dict:
        raise _raise_unsupported_leaf_api(
            "transform method",
            "__call__ method",
        )


@deprecated(
    "ConcatDictTransformConfig is deprecated. "
    "Use DictTransformPipelineConfig instead."
)
class ConcatDictTransformConfig(DictTransformPipelineConfig):
    """Legacy compatibility wrapper around DictTransformPipelineConfig.

    Use ``__call__()`` to build the runtime concat transform. ``+`` now
    canonicalizes back to ``DictTransformPipelineConfig`` rather than
    returning another ``ConcatDictTransformConfig``.
    """

    class_type: ClassType[ConcatDictTransform] = ConcatDictTransform
    input_columns: None = None
    missing_input_columns_as_none: bool = False
    output_column_mapping: dict[str, str] = Field(
        default_factory=_ImmutableLegacyOutputColumnMapping
    )
    check_return_columns: bool = False
    keep_input_columns: bool = True

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        _validate_legacy_concat_field_values(
            _get_legacy_concat_field_values(self)
        )
        object.__setattr__(
            self,
            "output_column_mapping",
            _freeze_legacy_output_column_mapping(self.output_column_mapping),
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _LEGACY_CONCAT_VALIDATED_FIELDS:
            _validate_legacy_concat_field_values(
                _get_legacy_concat_field_values(self, **{name: value})
            )
        if name == "output_column_mapping":
            value = _freeze_legacy_output_column_mapping(value)
        super().__setattr__(name, value)


ConcatDictTransform.__module__ = "robo_orchard_lab.transforms.base"
ConcatDictTransformConfig.__module__ = "robo_orchard_lab.transforms.base"
DictTransform.register(ConcatDictTransform)
