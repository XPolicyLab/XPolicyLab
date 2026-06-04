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
import functools
import inspect
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, is_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
)

from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
)
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassType,
    Config,
    ConfigInstanceOf,
    load_from,
)
from typing_extensions import TypeVar

from robo_orchard_lab.utils.state import (
    State,
    StateSaveLoadMixin,
)

if TYPE_CHECKING:
    from robo_orchard_lab.transforms.legacy_concat import (
        ConcatDictTransform,
        ConcatDictTransformConfig,
    )

__all__ = [
    "Config",
    "ClassType",
    "ConfigInstanceOf",
    "DictRowTransform",
    "DictRowTransformConfig",
    "DictTransform",
    "DictTransformType",
    "DictTransformConfig",
    "DictTransformPipeline",
    "DictTransformPipelineConfig",
    "ConcatDictTransform",
    "ConcatDictTransformConfig",
]


SemanticOutputT = TypeVar("SemanticOutputT")
DictRowTransformType = TypeVar(
    "DictRowTransformType", bound="DictRowTransform[Any]", covariant=True
)


@dataclass(frozen=True)
class _TransformReflectionMetadata:
    signature_input_columns: tuple[str, ...]
    input_defaults: dict[str, Any]
    inferred_output_columns: tuple[str, ...] | None


def semantic_output_to_dict(output: Any) -> dict[str, Any]:
    """Normalize a supported semantic output into a dict view."""
    if isinstance(output, dict):
        return output
    if isinstance(output, BaseModel):
        return {k: getattr(output, k) for k in output.model_fields}
    if is_dataclass(output) and not isinstance(output, type):
        return {
            field.name: getattr(output, field.name)
            for field in output.__dataclass_fields__.values()
        }
    raise TypeError(
        "Expected a semantic output to be a dict, BaseModel, or dataclass "
        f"instance, but got {type(output)}."
    )


def _to_dict(obj: Mapping[str, Any] | BaseModel | Any) -> dict[str, Any] | Any:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, Mapping):
        return dict(obj)
    if isinstance(obj, BaseModel) or (
        is_dataclass(obj) and not isinstance(obj, type)
    ):
        return semantic_output_to_dict(obj)
    return obj


def _flatten_runtime_transforms(
    transforms: Sequence[DictRowTransform[Any]],
) -> list[DictTransform[Any]]:
    flat_transforms: list[DictTransform[Any]] = []
    for transform in transforms:
        flat_transforms.extend(transform._leaf_transforms())
    return flat_transforms


def _flatten_transform_configs(
    configs: Sequence[DictRowTransformConfig[Any]],
) -> list[DictTransformConfig[Any]]:
    flat_configs: list[DictTransformConfig[Any]] = []
    for config in configs:
        if isinstance(config, DictTransformPipelineConfig):
            flat_configs.extend(config.transforms)
        else:
            flat_configs.append(config)
    return flat_configs


def _instantiate_transform_from_config(
    transform_cfg: DictTransformConfig[Any],
) -> DictTransform[Any]:
    """Instantiate one child transform while preserving config contracts.

    The standard `InitFromConfig = True` path keeps the original child config
    object attached to the runtime transform. Configs that rely on custom
    `__call__`, `create_instance_by_cfg`, or kwargs-based construction still
    go through `transform_cfg()` so those hooks remain honored.
    """

    class_type = transform_cfg.class_type
    if (
        getattr(class_type, "InitFromConfig", False)
        and type(transform_cfg).__call__ is ClassConfig.__call__
        and type(transform_cfg).create_instance_by_cfg
        is ClassConfig.create_instance_by_cfg
    ):
        return class_type(transform_cfg)
    transform = transform_cfg()
    if getattr(transform, "cfg", None) is not transform_cfg:
        # Keep the runtime child bound to the exact config object stored on the
        # pipeline so later cfg mutations still flow through to the child.
        transform.cfg = transform_cfg
    return transform


@functools.lru_cache(maxsize=2048)
def _get_cached_ordered_mapped_columns(
    transform_columns: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    input_columns: list[str] = []
    input_seen: set[str] = set()
    output_columns: list[str] = []
    output_seen: set[str] = set()

    for cur_inputs_tuple, cur_outputs_tuple in transform_columns:
        cur_inputs = list(cur_inputs_tuple)
        consumed_outputs = output_seen.intersection(cur_inputs)
        if consumed_outputs:
            output_columns = [
                name for name in output_columns if name not in consumed_outputs
            ]
            output_seen.difference_update(consumed_outputs)
        for name in cur_inputs:
            if name in consumed_outputs or name in input_seen:
                continue
            input_columns.append(name)
            input_seen.add(name)
        for name in cur_outputs_tuple:
            if name in output_seen:
                continue
            output_columns.append(name)
            output_seen.add(name)

    return tuple(input_columns), tuple(output_columns)


def _get_pipeline_transform_columns_key(
    transforms: Sequence[DictTransform[Any]],
) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    return tuple(
        (
            tuple(transform.mapped_input_columns),
            tuple(transform.mapped_output_columns),
        )
        for transform in transforms
    )


def _infer_output_columns_from_annotation(
    return_annotation: Any,
) -> tuple[str, ...] | None:
    if is_dataclass(return_annotation):
        return tuple(return_annotation.__dataclass_fields__.keys())
    try:
        if issubclass(return_annotation, BaseModel):
            return tuple(return_annotation.model_fields.keys())
    except TypeError:
        pass
    return None


def _build_transform_reflection_metadata(
    transform: Any,
) -> _TransformReflectionMetadata:
    sig = inspect.signature(transform, eval_str=True)
    input_columns = []
    input_defaults = {}
    for param in sig.parameters.values():
        if (
            param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            and param.name != "self"
        ):
            input_columns.append(param.name)
            if param.default != inspect.Parameter.empty:
                input_defaults[param.name] = param.default
    return _TransformReflectionMetadata(
        signature_input_columns=tuple(input_columns),
        input_defaults=input_defaults,
        inferred_output_columns=_infer_output_columns_from_annotation(
            sig.return_annotation
        ),
    )


@functools.cache
def _get_cached_transform_reflection_metadata(
    transform_callable: Any,
) -> _TransformReflectionMetadata:
    return _build_transform_reflection_metadata(transform_callable)


def _get_transform_reflection_metadata(
    transform: Any,
) -> _TransformReflectionMetadata:
    transform_callable = getattr(transform, "__func__", transform)
    try:
        hash(transform_callable)
    except TypeError:
        return _build_transform_reflection_metadata(transform)
    return _get_cached_transform_reflection_metadata(transform_callable)


def _normalize_input_columns_config_key(
    input_columns: dict[str, str] | Sequence[str] | None,
) -> tuple[str, tuple[Any, ...] | None]:
    if isinstance(input_columns, Mapping):
        return (
            "mapping",
            tuple(input_columns.items()),
        )
    if isinstance(input_columns, (list, tuple)):
        return (
            "sequence",
            tuple(input_columns),
        )
    if input_columns is None:
        return ("none", None)
    raise TypeError(
        f"Expected input_columns to be a dict, list, tuple or None, but got "
        f"{type(input_columns)}."
    )


@functools.lru_cache(maxsize=2048)
def _get_cached_input_column_views(
    signature_input_columns: tuple[str, ...],
    normalized_input_columns: tuple[str, tuple[Any, ...] | None],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    kind, payload = normalized_input_columns
    input_columns = list(signature_input_columns)
    if kind == "mapping":
        input_columns.extend(value for _, value in payload or ())
    elif kind == "sequence":
        input_columns.extend(payload or ())
    input_columns_tuple = tuple(dict.fromkeys(input_columns))
    if kind == "mapping":
        inverted_mapping = {value: key for key, value in payload or ()}
        mapped_input_columns = tuple(
            inverted_mapping.get(col, col) for col in input_columns_tuple
        )
    elif kind == "sequence":
        mapped_input_columns = tuple(payload or ())
    else:
        mapped_input_columns = input_columns_tuple
    return input_columns_tuple, mapped_input_columns


class DictRowTransform(
    StateSaveLoadMixin, Generic[SemanticOutputT], metaclass=ABCMeta
):
    """Shared row-aware transform contract for leaf and pipeline objects.

    This interface intentionally covers only the row application boundary that
    both single-stage transforms and transform pipelines implement. Leaf-only
    semantics such as ``transform(...)``, ``input_columns``, and
    ``output_columns`` still belong to :class:`DictTransform`.
    """

    cfg: DictRowTransformConfig[Any]

    @abstractmethod
    def apply(
        self, row: Mapping[str, Any]
    ) -> tuple[SemanticOutputT, dict[str, Any]]:
        """Apply the row-aware transform and return raw plus merged outputs."""
        raise NotImplementedError

    def __call__(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Call the row-aware transform and return the merged row only."""
        _, final_row = self.apply(row)
        return final_row

    @property
    @abstractmethod
    def mapped_input_columns(self) -> list[str]:
        """The external input columns consumed by the row-aware transform."""
        raise NotImplementedError

    @property
    @abstractmethod
    def mapped_output_columns(self) -> list[str]:
        """The external output columns produced by the row-aware transform."""
        raise NotImplementedError

    @abstractmethod
    def __add__(
        self,
        other: DictRowTransform[Any],
    ) -> DictTransformPipeline:
        """Concatenate another row-aware transform into a pipeline."""
        raise NotImplementedError

    @classmethod
    def from_config(
        cls: type[DictRowTransformType], config_path: str
    ) -> DictRowTransformType:
        """Load a row-aware transform from a configuration file."""
        cfg = load_from(config_path, ensure_type=DictRowTransformConfig)
        if not issubclass(cfg.class_type, cls):
            raise TypeError(
                f"Config class_type {cfg.class_type} is not a subclass of "
                f"{cls.__name__}."
            )
        return cfg()

    @abstractmethod
    def _leaf_transforms(self) -> tuple[DictTransform[Any], ...]:
        """Return the flattened leaf transforms that back this object."""
        raise NotImplementedError


class DictTransform(DictRowTransform[SemanticOutputT]):
    """A class that defines the interface for transforming a dict.

    The dict is usually a row in a dataset, and the transform
    will take the input columns and produce a semantic transform output.
    Row merging, output-column mapping, and compatibility dict assembly
    happen later through ``apply(...)`` and ``__call__(...)``.

    User should implement the `transform` method to define the specific
    transformation logic.

    If you use a dataclass or BaseModel as the return type of
    the transform method, `output_columns` will be automatically
    inferred from the fields of the dataclass or BaseModel. Otherwise,
    you should implement the `output_columns` property to return the
    expected output columns for the transform.


    For Transform whose input and output columns are not known
    at the time of configuration, the input and output columns
    properties should not be used, and the `check_return_columns`
    configuration should be set to False. This will prevent runtime
    errors when the transform is called with a row dict.

    """

    InitFromConfig: bool = True

    cfg: DictTransformConfig

    @abstractmethod
    def transform(self, **kwargs) -> SemanticOutputT:
        """Transform input columns into a semantic stage output.

        All input columns will be passed as keyword arguments. The return
        value may be a ``dict``, dataclass instance, or ``BaseModel``.
        ``apply(...)`` and ``__call__(...)`` handle row-level merging.
        """
        raise NotImplementedError

    def apply(
        self, row: Mapping[str, Any]
    ) -> tuple[SemanticOutputT, dict[str, Any]]:
        """Apply the transform once and return both raw and merged outputs.

        This method will extract the input columns from the row dict,
        call the transform method, and return a new row dict with the
        transformed values added to the original row dict.

        Use the first return value when callers need the structured semantic
        output of ``transform(...)``. Use ``__call__(...)`` or the second
        return value when callers need the final row dict after output-column
        mapping and row merge.

        Returns:
            tuple[SemanticOutputT, dict[str, Any]]: The raw
                ``transform(...)`` result and the final merged row dict that
                ``__call__`` would return.

        """
        row = _to_dict(row)
        if not isinstance(row, dict):
            raise TypeError(f"Expected row to be a dict, but got {type(row)}.")
        return self._apply_dict(row)

    def _apply_dict(
        self, row: dict[str, Any]
    ) -> tuple[SemanticOutputT, dict[str, Any]]:
        """Apply the transform assuming ``row`` is already a plain dict."""

        reflection_metadata = _get_transform_reflection_metadata(
            self.transform
        )
        normalized_input_columns = _normalize_input_columns_config_key(
            self.cfg.input_columns
        )
        input_column_mapping_items = (
            normalized_input_columns[1]
            if normalized_input_columns[0] == "mapping"
            else None
        )
        mapped_input = (
            row if input_column_mapping_items is None else row.copy()
        )
        if input_column_mapping_items is not None:
            for src_name, dst_name in input_column_mapping_items:
                if src_name not in mapped_input:
                    raise ValueError(
                        f"Input column {src_name} not found in row dict."
                    )
                mapped_input[dst_name] = mapped_input.pop(src_name)

        ts_input = {}
        input_columns, _ = _get_cached_input_column_views(
            reflection_metadata.signature_input_columns,
            normalized_input_columns,
        )
        for col in input_columns:
            if col not in mapped_input:
                if col in reflection_metadata.input_defaults:
                    # Use the default value from the function signature
                    ts_input[col] = reflection_metadata.input_defaults[col]
                elif not self.cfg.missing_input_columns_as_none:
                    raise KeyError(
                        f"Input column `{col}` not found in row dict."
                    )
                else:
                    ts_input[col] = None
            else:
                ts_input[col] = mapped_input[col]

        transform_result = self.transform(**ts_input)
        columns_after = semantic_output_to_dict(transform_result)
        output_column_mapping = self.cfg.output_column_mapping
        if columns_after is transform_result and output_column_mapping:
            # Keep raw dict outputs unchanged when key remapping mutates the
            # compatibility row patch below.
            columns_after = columns_after.copy()

        # check that the output columns match the expected output columns
        if self.cfg.check_return_columns:
            for col in columns_after.keys():
                if col not in self.output_columns:
                    raise ValueError(
                        f"Output column {col} not in expected output columns: "
                        f"{self.output_columns}."
                    )

        for src_name, dst_name in output_column_mapping.items():
            if dst_name in columns_after:
                raise ValueError(
                    f"Output column {dst_name} already exists in transformed "
                    "columns."
                )
            if src_name in columns_after:
                columns_after[dst_name] = columns_after.pop(src_name)
        if self.cfg.keep_input_columns:
            ret = row.copy()
        else:
            ret = {}
        ret.update(columns_after)
        return transform_result, ret

    def __call__(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Call the transform on a row dict and return the merged row only."""
        return super().__call__(row)

    @property
    def input_columns(self) -> list[str]:
        """The input columns that this transform requires.

        This should be a list of column names that are required
        for the transformation. The transform method will be called
        with these columns as keyword arguments.

        The input columns are determined by the signature of the
        `transform` method, and if the `input_columns` configuration
        is set, it will be used as well to determine the input columns.

        """
        reflection_metadata = _get_transform_reflection_metadata(
            self.transform
        )
        normalized_input_columns = _normalize_input_columns_config_key(
            self.cfg.input_columns
        )
        input_columns, _ = _get_cached_input_column_views(
            reflection_metadata.signature_input_columns,
            normalized_input_columns,
        )
        return list(input_columns)

    @property
    def mapped_input_columns(self) -> list[str]:
        """The input columns that this transform requires for column mapping.

        If no mapping is required, this will be the same as input_columns.

        """

        reflection_metadata = _get_transform_reflection_metadata(
            self.transform
        )
        normalized_input_columns = _normalize_input_columns_config_key(
            self.cfg.input_columns
        )
        _, mapped_input_columns = _get_cached_input_column_views(
            reflection_metadata.signature_input_columns,
            normalized_input_columns,
        )
        return list(mapped_input_columns)

    @property
    def output_columns(self) -> list[str]:
        """The output columns that this transform produces.

        This should be a list of column names that the transform will
        produce as output. The transform method will return a dict
        with these keys.

        Note that this property contains all possible output columns,
        not just the ones that are actually produced by the transform.
        The transform method may return a subset of these columns,
        but the output_columns property should list all columns that
        the transform can produce.

        """
        inferred_output_columns = _get_transform_reflection_metadata(
            self.transform
        ).inferred_output_columns
        if inferred_output_columns is None:
            return_annotation = inspect.signature(
                self.transform, eval_str=True
            ).return_annotation
            raise NotImplementedError(
                "Cannot determine output columns for "
                f"{self.__class__.__name__}. Return type "
                f"{return_annotation} "
                "is not a dataclass or BaseModel. You should implement the "
                "output_columns property to return the expected output "
                "columns for this transform."
            )
        return list(inferred_output_columns)

    @property
    def mapped_output_columns(self) -> list[str]:
        """The output columns that this transform produces after mapping."""
        old_output_columns = self.output_columns
        return [
            self.cfg.output_column_mapping.get(col, col)
            for col in old_output_columns
        ]

    def __repr__(self) -> str:
        ret = f"{self.__class__.__name__}("
        ret += f"cfg={self.cfg.to_dict(exclude_defaults=True)}"
        ret += ")"
        return ret

    def _get_state(self) -> State:
        """Get the state of the object for saving."""
        # pull out cfg from state for better clarity
        ret = super()._get_state()
        ret.config = ret.state.pop("cfg", None)
        return ret

    def _set_state(self, state: State) -> None:
        """Set the state of the object from the unpickled state."""
        # push cfg back to state for consistency
        state.state["cfg"] = state.config
        state.config = None
        super()._set_state(state)

    def __add__(
        self,
        other: DictRowTransform[Any],
    ) -> DictTransformPipeline:
        """Concatenate another DictTransform to this one.

        This returns a new canonical pipeline and does not mutate the
        left-hand side. It intentionally does not preserve the deprecated
        ``ConcatDictTransform`` wrapper as a ``+`` result.
        """
        if not isinstance(other, DictRowTransform):
            raise TypeError(
                "Can only concatenate DictTransform or "
                "DictTransformPipeline objects."
            )
        return _get_runtime_add_result_pipeline_type(
            self, other
        ).from_transforms([self, other])

    @classmethod
    def from_config(
        cls: type[DictTransformType], config_path: str
    ) -> DictTransformType:
        """Load a transform from a configuration file.

        During the current concat-to-pipeline migration, the legacy
        ``DictTransform`` loader still accepts row-aware pipeline configs so
        callers can migrate config files before they migrate load sites.
        """
        cfg = load_from(config_path, ensure_type=DictRowTransformConfig)
        if cls is DictTransform and issubclass(
            cfg.class_type, DictTransformPipeline
        ):
            return cfg()  # type: ignore[return-value]
        if not issubclass(cfg.class_type, cls):
            raise TypeError(
                f"Config class_type {cfg.class_type} is not a subclass of "
                f"{cls.__name__}."
            )
        return cfg()

    def _leaf_transforms(self) -> tuple[DictTransform[Any], ...]:
        return (self,)


DictTransformType = TypeVar(
    "DictTransformType", bound=DictTransform[Any], covariant=True
)


class DictRowTransformConfig(ClassConfig[DictRowTransformType]):
    """Shared row-aware transform config contract.

    Both single-stage and pipeline config objects implement this interface.
    """

    class_type: ClassType[DictRowTransformType]

    def __add__(
        self,
        other: DictRowTransformConfig[Any],
    ) -> DictTransformPipelineConfig:
        """Concatenate two row-aware configs into a pipeline config."""
        raise NotImplementedError


class DictTransformConfig(DictRowTransformConfig[DictTransformType]):
    class_type: ClassType[DictTransformType]

    input_columns: dict[str, str] | Sequence[str] | None = Field(
        validation_alias=AliasChoices("input_column_mapping", "input_columns"),
        default=None,
    )
    """The input columns that need to be mapped to fit
    the transform's input_columns, or a list of input columns
    as extra input columns.

    If this is a dict, it should map the source column names
    to the destination column names, and the destination column names
    will be used as the input columns for the transform.

    """

    missing_input_columns_as_none: bool = Field(default=False)
    """If True, missing input columns will be set to None.

    If False, an error will be raised if any input column is missing."""

    output_column_mapping: dict[str, str] = Field(default_factory=dict)
    """The output columns that the transform will produce.
    This should be a mapping from the output column names to the
    names that the transform will use to return the transformed values.
    If the transform does not produce any output, this can be an empty dict.
    """

    check_return_columns: bool = False
    """Whether to check that the output columns of the transform
    match the expected output columns. If this is set to True,
    the transform will raise an error if the output columns do not
    match the expected output columns.

    For Transform that does not properly implement the `output_columns`
    property, or if the output columns are not known at the time of
    configuration, this should be set to False to avoid runtime errors.
    """

    keep_input_columns: bool = True
    """Whether to keep the input columns in the output dict.

    If this is set to True, the input columns will be included in the
    output dict. If this is set to False, the input columns will be
    removed from the output dict.
    """

    # overload + operator to return the canonical pipeline config
    def __add__(
        self,
        other: DictRowTransformConfig[Any],
    ) -> DictTransformPipelineConfig:
        """Concatenate two DictTransformConfig objects.

        This returns a new canonical pipeline config and does not mutate the
        left-hand side. It intentionally does not preserve the deprecated
        ``ConcatDictTransformConfig`` wrapper as a ``+`` result.
        """
        if not isinstance(other, DictRowTransformConfig):
            raise TypeError(
                "Can only concatenate DictTransformConfig objects."
            )
        return _get_config_add_result_pipeline_type(self, other).from_configs(
            [self, other]
        )


def _get_runtime_add_result_pipeline_type(
    left: DictRowTransform[Any],
    right: DictRowTransform[Any],
) -> type[DictTransformPipeline]:
    """Pick the pipeline type produced by ``+``.

    Deprecated legacy concat wrappers are normalized back to the canonical
    ``DictTransformPipeline`` surface so `+` does not keep propagating the old
    compatibility type.
    """

    from robo_orchard_lab.transforms.legacy_concat import ConcatDictTransform

    if isinstance(left, ConcatDictTransform) or isinstance(
        right, ConcatDictTransform
    ):
        return DictTransformPipeline
    if (
        isinstance(left, DictTransformPipeline)
        and type(left) is not DictTransformPipeline
    ):
        return type(left)
    if (
        isinstance(right, DictTransformPipeline)
        and type(right) is not DictTransformPipeline
    ):
        return type(right)
    return DictTransformPipeline


def _get_config_add_result_pipeline_type(
    left: DictRowTransformConfig[Any],
    right: DictRowTransformConfig[Any],
) -> type[DictTransformPipelineConfig]:
    """Pick the pipeline-config type produced by ``+``.

    Deprecated legacy concat wrappers are normalized back to the canonical
    ``DictTransformPipelineConfig`` surface so `+` does not keep propagating
    the old compatibility type.
    """

    from robo_orchard_lab.transforms.legacy_concat import (
        ConcatDictTransformConfig,
    )

    if isinstance(left, ConcatDictTransformConfig) or isinstance(
        right, ConcatDictTransformConfig
    ):
        return DictTransformPipelineConfig
    if (
        isinstance(left, DictTransformPipelineConfig)
        and type(left) is not DictTransformPipelineConfig
    ):
        return type(left)
    if (
        isinstance(right, DictTransformPipelineConfig)
        and type(right) is not DictTransformPipelineConfig
    ):
        return type(right)
    return DictTransformPipelineConfig


class DictTransformPipeline(DictRowTransform[Any]):
    """A pipeline that applies multiple row-aware dict transforms in order.

    ``DictTransformPipeline`` implements the shared row-aware contract through
    ``apply(...)``, ``__call__(...)``, and mapped column metadata. It does not
    expose leaf-only semantics such as ``transform(...)``, ``input_columns``,
    or ``output_columns`` because those concepts stay owned by each child
    transform.
    """

    InitFromConfig: bool = True

    cfg: DictTransformPipelineConfig
    _transforms: list[DictTransform[Any]]

    def __init__(self, cfg: DictTransformPipelineConfig) -> None:
        self.cfg = cfg
        self._transforms = [
            _instantiate_transform_from_config(transform_cfg)
            for transform_cfg in cfg.transforms
        ]

    @classmethod
    def from_transforms(
        cls,
        transforms: Sequence[DictRowTransform[Any]],
    ) -> DictTransformPipeline:
        """Build a pipeline container around existing runtime transforms.

        This preserves runtime transform identity, so repeating the same
        runtime stage in the input sequence will reuse that same instance.
        """
        instance = cls.__new__(cls)
        instance._transforms = _flatten_runtime_transforms(transforms)
        instance.cfg = DictTransformPipelineConfig(
            transforms=[transform.cfg for transform in instance._transforms]
        )
        return instance

    def __getitem__(self, index: int) -> DictTransform[Any]:
        return self._transforms[index]

    def __add__(
        self,
        other: DictRowTransform[Any],
    ) -> DictTransformPipeline:
        """Return a new pipeline extended with ``other``."""
        if not isinstance(other, DictRowTransform):
            raise TypeError(
                "Can only concatenate DictTransform or "
                "DictTransformPipeline objects."
            )
        return _get_runtime_add_result_pipeline_type(
            self, other
        ).from_transforms([self, other])

    def __iadd__(
        self,
        other: DictRowTransform[Any],
    ) -> DictTransformPipeline:
        """Append another transform or pipeline to this pipeline in place."""
        if not isinstance(other, DictRowTransform):
            raise TypeError(
                "Can only concatenate DictTransform or "
                "DictTransformPipeline objects."
            )
        new_transforms = _flatten_runtime_transforms([other])
        self._transforms.extend(new_transforms)
        self.cfg.transforms = tuple(self.cfg.transforms) + tuple(
            transform.cfg for transform in new_transforms
        )
        return self

    @property
    def mapped_input_columns(self) -> list[str]:
        mapped_input_columns, _ = _get_cached_ordered_mapped_columns(
            _get_pipeline_transform_columns_key(self._transforms)
        )
        return list(mapped_input_columns)

    @property
    def mapped_output_columns(self) -> list[str]:
        _, mapped_output_columns = _get_cached_ordered_mapped_columns(
            _get_pipeline_transform_columns_key(self._transforms)
        )
        return list(mapped_output_columns)

    def apply(self, row: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Apply each child once and return the final stage output."""
        last_transform_result: Any = None
        current_row = _to_dict(row)
        if not isinstance(current_row, dict):
            raise TypeError(
                f"Expected row to be a dict, but got {type(current_row)}."
            )
        for transform in self._transforms:
            last_transform_result, current_row = transform._apply_dict(
                current_row
            )
        return last_transform_result, current_row

    def __call__(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return super().__call__(row)

    def _leaf_transforms(self) -> tuple[DictTransform[Any], ...]:
        return tuple(self._transforms)

    def _get_state(self) -> State:
        return State(
            state=dict(transforms=list(self._transforms)),
            config=self.cfg,
            hierarchical_save=None,
            class_type=type(self),
        )

    def _set_state(self, state: State) -> None:
        self.cfg = state.config  # type: ignore
        self._transforms = state.state["transforms"]
        self.cfg.transforms = tuple(
            transform.cfg for transform in self._transforms
        )


class DictTransformPipelineConfig(
    DictRowTransformConfig[DictTransformPipeline]
):
    """Config for a row-aware pipeline built from child transform configs."""

    class_type: ClassType[DictTransformPipeline] = DictTransformPipeline

    transforms: Sequence[ConfigInstanceOf[DictTransformConfig]] = Field(
        min_length=1
    )

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(
            self,
            "transforms",
            tuple(_flatten_transform_configs(self.transforms)),
        )

    @classmethod
    def from_configs(
        cls,
        configs: Sequence[DictRowTransformConfig[Any]],
    ) -> DictTransformPipelineConfig:
        """Build a flattened pipeline config from one or more configs.

        This preserves child config identity, but runtime instantiation still
        creates one child transform per listed config entry.
        """
        return cls(transforms=tuple(_flatten_transform_configs(configs)))

    def __getitem__(self, item):
        return self.transforms[item]

    def __add__(
        self,
        other: DictRowTransformConfig[Any],
    ) -> DictTransformPipelineConfig:
        if not isinstance(other, DictRowTransformConfig):
            raise TypeError(
                "Can only concatenate DictTransformConfig objects."
            )
        return _get_config_add_result_pipeline_type(self, other).from_configs(
            [self, other]
        )

    def __iadd__(
        self,
        other: DictRowTransformConfig[Any],
    ) -> DictTransformPipelineConfig:
        if not isinstance(other, DictRowTransformConfig):
            raise TypeError(
                "Can only concatenate DictTransformConfig objects."
            )
        self.transforms = tuple(self.transforms) + tuple(
            _flatten_transform_configs([other])
        )
        return self


def __getattr__(name: str) -> Any:
    if name in {"ConcatDictTransform", "ConcatDictTransformConfig"}:
        from robo_orchard_lab.transforms.legacy_concat import (
            ConcatDictTransform,
            ConcatDictTransformConfig,
        )

        globals()["ConcatDictTransform"] = ConcatDictTransform
        globals()["ConcatDictTransformConfig"] = ConcatDictTransformConfig
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
