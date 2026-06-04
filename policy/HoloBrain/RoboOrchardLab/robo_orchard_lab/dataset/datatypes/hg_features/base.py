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
import pickle
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import datasets as hg_datasets
import numpy as np
import pyarrow as pa
import pyzstd
import torch
from datasets.features.features import register_feature
from pydantic import BaseModel
from robo_orchard_core.utils.config import (
    callable_to_string,
    string_to_callable,
)
from typing_extensions import TypeVar

__all__ = [
    "RODataFeature",
    "RODictDataFeature",
    "TypedDictFeatureDecode",
    "FeatureDecodeMixin",
    "PickleFeature",
    "hg_dataset_feature",
    "check_fields_consistency",
    "guess_hg_features",
]


class FeatureDecodeMixin(metaclass=ABCMeta):
    """Mixin class for features that support decoding."""

    @abstractmethod
    def decode_example(self, value: Any, **kwargs) -> Any:
        """Decode the example value from its stored format."""
        raise NotImplementedError(
            "Subclasses must implement decode_example method."
        )


class TypedDictFeatureDecode(FeatureDecodeMixin):
    """Helper class for decoding typed dictionary features.

    This class uses `_dict` as the feature decoding schema to decode the
    example value. User should extend this class and set the `_dict`
    attribute to represent the schema.

    Note that `_dict` will not be included in the serialized output, and
    also not be used in the initialization of the class. You should
    initialize `_dict` in `__post_init__` or in the class body.

    """

    _dict: dict = field(init=False, repr=False)
    _decode_type: type = field(init=False, repr=False)
    decode: bool = True

    def decode_example(self, value: Any, **kwargs) -> Any:
        if not self.decode:
            raise RuntimeError(
                "This feature does not support decoding. "
                "Set decode=True to enable decoding."
            )
        ret: dict = hg_datasets.features.features.decode_nested_example(
            schema=self._dict, obj=value
        )
        return self._decode_type(**ret)


@dataclass
class RODataFeature(metaclass=ABCMeta):
    """Base class for RoboOrchard dataset features.

    User should implement the `pa_type` property and `encode_example` method
    to define the specific feature type and how to encode example values.

    This class does not include `decode_example` method, as it is not
    required for all features. If you need to decode the example values,
    you can inherit from `FeatureDecodeMixin`.

    """

    _type: str = field(init=False, repr=False)
    """The class name of the feature type. Needed for serialization
    and deserialization. Should be set in subclasses."""

    def __call__(self) -> pa.DataType:
        """Return the pyarrow data type for this feature."""
        return self.pa_type

    @property
    @abstractmethod
    def pa_type(self) -> pa.DataType:
        """Return the pyarrow data type for this feature."""
        raise NotImplementedError(
            "Subclasses must implement pa_type property."
        )

    @abstractmethod
    def encode_example(self, value: Any) -> Any:
        """Encode the example value into a format suitable for storage."""
        raise NotImplementedError(
            "Subclasses must implement encode_example method."
        )


class RODictDataFeature(RODataFeature):
    """A feature that is composed of a dictionary of features.

    This class does not inherit from `dict` but use `dict` to define features.
    It is useful for defining complex features that are
    composed of multiple fields. The user should define the `_dict` attribute
    as a dictionary mapping field names to features. The keys of the dictionary
    are the field names, and the values are the features.

    This class does not include `decode_example` method, as it is not
    required for all features. If you need to decode the example values,
    you can inherit from `DictFeatureDecodeMixin`.

    """

    _dict: dict = field(init=False, repr=False)

    @property
    def pa_type(self) -> pa.DataType:
        """Return the pyarrow data type for this feature."""
        return hg_datasets.features.features.get_nested_type(self._dict)

    def encode_example(self, value: Any) -> Any:
        return hg_datasets.features.features.encode_nested_example(
            schema=self._dict, obj=value
        )

    def adapt_for_pa_type(self, pa_struct: pa.StructType) -> bool:
        """Adapt the feature to match the given pyarrow struct type.

        This method is useful for ensuring that the feature schema matches
        the expected schema when loading data from a dataset. It checks that
        the fields in the `_dict` match the fields in the `pa_struct`, and
        raises a TypeError if there are any mismatches.

        Args:
            pa_struct (pa.StructType): The pyarrow struct type to adapt to.

        Returns:
            bool: True if the feature is successfully adapted, False otherwise.

        Raises:
            NotImplementedError: If the method is not implemented in the
            subclass.

        """
        raise NotImplementedError(
            "Subclasses must implement adapt_for_pa_type method."
        )

    def reset(self):
        """Reset the feature to its initial state.

        This is useful for features that have internal state that needs
        to be reset.

        Raise:
            NotImplementedError: If the method is not implemented in the
            subclass.

        """
        raise NotImplementedError(
            "Subclasses must implement reset method if they have "
            "internal state."
        )

    def items(self):
        """Return the items of the dictionary."""
        return self._dict.items()

    def keys(self):
        """Return the keys of the dictionary."""
        return self._dict.keys()

    def values(self):
        """Return the values of the dictionary."""
        return self._dict.values()


@runtime_checkable
class ToDataFeatureMixin(Protocol):
    """Protocol for features that can be converted to a pyarrow DataType."""

    @classmethod
    def dataset_feature(cls) -> RODataFeature: ...

    def get(self, key: str, default: Any = None) -> Any:
        """Get the value of the feature by key."""
        ...


RODataFeatureType = TypeVar("RODataFeatureType", bound=RODataFeature)


def hg_dataset_feature(
    cls: type[RODataFeatureType],
) -> type[RODataFeatureType]:
    """Decorator to register a feature class with its type."""
    if not issubclass(cls, RODataFeature):
        raise TypeError("Feature class must inherit from RODataFeature.")
    cls._type = cls.__qualname__
    register_feature(cls, cls._type)
    return cls


def check_fields_consistency(
    cls: type[BaseModel],
    pa_struct: pa.StructType,
):
    pydantic_fields = set(cls.model_fields.keys())
    pa_fields = set([field.name for field in pa_struct.fields])
    if pydantic_fields != pa_fields:
        raise TypeError(
            f"Pydantic fields {pydantic_fields} do not match "
            f"pyarrow fields {pa_fields} for {cls.__name__}."
            " This means that the feature is not fully implemented."
        )


@hg_dataset_feature
@dataclass
class PickleFeature(RODataFeature, FeatureDecodeMixin):
    """A feature that uses pickle to serialize and deserialize data.

    Args:
        class_type (type | str): The class type of the object to be
            serialized/deserialized. It should be a class for initialization,
            or a string from `callable_to_string` for deserialization.
        binary_type (Literal["binary", "large_binary"], optional): The
            type of binary storage to use. Defaults to "binary".

    """

    class_type: type | str

    decode: bool = True

    binary_type: Literal["binary", "large_binary"] = "binary"

    compression: Literal["zstd"] | None = None

    def __post_init__(self):
        # make sure that class_type is string for serialization
        if not isinstance(self.class_type, str):
            self.class_type = callable_to_string(self.class_type)

    def _get_class_type(self) -> type:
        """Get the class type from the class_type attribute."""
        if hasattr(self, "_class_type"):
            return self._class_type  # type: ignore
        else:
            self._class_type = (
                string_to_callable(self.class_type)
                if isinstance(self.class_type, str)
                else self.class_type
            )
            return self._class_type  # type: ignore

    def __setattr__(self, name, value):
        if name == "class_type":
            self.__dict__.pop("_class_type", None)
            self.__dict__[name] = value
        else:
            super().__setattr__(name, value)

    @property
    def pa_type(self):
        """Return the pyarrow data type for this feature."""
        if self.binary_type == "binary":
            return pa.binary()
        else:
            return pa.large_binary()

    def encode_example(self, value: Any) -> bytes:
        class_type = self._get_class_type()
        if not isinstance(value, class_type):
            # special handling for torch.Tensor
            # as hugggingface datasets convert them to numpy arrays
            # or lists automatically
            if class_type is torch.Tensor and isinstance(
                value, (np.ndarray, list, tuple)
            ):
                return self._encode_obj(value)

            raise TypeError(
                f"Value must be of type {class_type}, but got {type(value)}."
            )
        return self._encode_obj(value)

    def decode_example(
        self,
        value: bytes,
        **kwargs,
    ) -> Any:
        if not self.decode:
            raise RuntimeError(
                "Decoding is disabled for this feature. Please use "
                "PickleFeature(decode=True) instead."
            )
        ret = self._decode_obj(value)
        class_type = self._get_class_type()
        if class_type is torch.Tensor and not isinstance(ret, torch.Tensor):
            ret = torch.tensor(ret)

        return ret

    def _encode_obj(self, obj: Any) -> bytes:
        ret = pickle.dumps(obj)
        if self.compression == "zstd":
            ret = pyzstd.compress(ret)
        return ret

    def _decode_obj(self, data: bytes) -> Any:
        if self.compression == "zstd":
            data = pyzstd.decompress(data)
        return pickle.loads(data)


def guess_hg_features(
    data: dict,
    dataset_feature_kwargs: dict | None = None,
) -> hg_datasets.features.Features:
    """Guess the Hugging Face dataset features from an dict.

    If the object contains a list or tuple, it will try to guess the feature
    type from the first non-null value. If all values are None or empty,
    it raises a ValueError.

    Try to avoid any None values in the input object, as it may lead to
    incorrect feature type inference.

    Args:
        data (dict): The input data to guess features from.
        dataset_feature_kwargs (dict, optional): Additional keyword arguments
            to pass to the `dataset_feature` method of the feature classes.
            Defaults to None.

    """

    if not isinstance(data, dict):
        raise TypeError(
            "Input data must be a dictionary mapping field names to values."
        )

    if dataset_feature_kwargs is None:
        dataset_feature_kwargs = {}

    from robo_orchard_lab.dataset.datatypes.hg_features.tensor import (
        TypedTensorFeature,
    )

    def guess_feature(obj: Any):
        if isinstance(obj, (hg_datasets.Features, RODataFeature)):
            return obj
        elif isinstance(obj, dict):
            return {k: guess_feature(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            idx, value = hg_datasets.features.features.first_non_null_value(
                obj
            )
            if idx < 0:
                raise ValueError(
                    "Cannot guess features from a list or tuple with all "
                    "None values or empty."
                )
            return hg_datasets.Sequence(feature=guess_feature(value))
        elif isinstance(obj, ToDataFeatureMixin):
            return obj.dataset_feature(**dataset_feature_kwargs)
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            dtype = (
                obj.dtype if isinstance(obj, np.ndarray) else obj.numpy().dtype
            )
            as_torch_tensor = isinstance(obj, torch.Tensor)
            return TypedTensorFeature(
                dtype=str(dtype),
                as_torch_tensor=as_torch_tensor,
            )
        else:
            return hg_datasets.features.features.generate_from_arrow_type(
                pa.array([obj]).type
            )

    feature_dict = guess_feature(data)
    assert isinstance(feature_dict, dict), (
        "The guessed features must be a dictionary mapping field names to "
        "Hugging Face dataset features."
    )
    return hg_datasets.Features(**(feature_dict))
