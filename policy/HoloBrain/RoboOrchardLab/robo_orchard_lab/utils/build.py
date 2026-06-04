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

import copy
import importlib
from typing import Annotated, Any

import torch
from pydantic.functional_validators import PlainValidator
from robo_orchard_core.utils.config import ClassConfig, callable_to_string

__all__ = ["build", "DelayInitDictType"]


def build(obj: ClassConfig | dict | Any, *args) -> Any:
    """Instantiates an object based on the provided configuration or object.

    This function serves as a flexible factory. It can build an object
    from a `ClassConfig` instance, a dictionary specifying the object's
    type and parameters, or simply return the object if it's of any other type.

    Special handling is implemented for `torch.nn.GroupNorm`, where if `*args`
    are provided, the first argument is assumed to be `num_channels`.

    Args:
        obj: The configuration or object to build from.

            - If `ClassConfig`: An instance of `ClassConfig` which contains
              the class type and initialization parameters.

            - If `dict`: A dictionary containing a 'class_type' or 'type' key
              specifying the class to instantiate (either as a callable or
              a string like 'module.name:ClassName' or 'module.name'),
              and other keys as keyword arguments for the constructor.

            - If any other type: The object itself is returned.

        *args: Additional positional arguments to pass to the constructor
            of the object being built.

    Returns:
        The instantiated object, or `obj` itself if no instantiation logic
        applies to its type.

    Raises:
        KeyError: If `obj` is a dictionary and is missing both 'class_type'
            and 'type' keys.
        ImportError: If `obj` is a dictionary and the class string refers
            to a module or class that cannot be imported.
        AttributeError: If `obj` is a dictionary and the class string refers
            to a module but the class name cannot be found within it.
    """

    if isinstance(obj, ClassConfig):
        if obj.class_type == torch.nn.GroupNorm:
            return obj(num_channels=args[0])
        else:
            return obj()
    elif isinstance(obj, dict):
        cfg = copy.deepcopy(obj)

        if "class_type" in cfg:
            cls = cfg.pop("class_type")
        elif "type" in cfg:
            cls = cfg.pop("type")
        else:
            raise KeyError("Missing type key `class_type` or `type`")

        if isinstance(cls, str):
            if ":" in cls:
                module_name, cls_name = cls.split(":")
                module = importlib.import_module(module_name)
                cls = getattr(module, cls_name)
            else:
                cls = importlib.import_module(cls)
        if cls == torch.nn.GroupNorm and len(args) == 1:
            return cls(num_channels=args[0], **cfg)  # type: ignore
        return cls(*args, **cfg)  # type: ignore
    else:
        return obj


def _validate_delay_init_dict(x: dict) -> dict:
    """Recursively validates and transforms a dictionary for delayed initialization.

    This function standardizes the structure of a configuration dictionary,
    primarily by ensuring that any class type information is stored under
    a 'type' key as a string. It processes nested dictionaries and lists/tuples
    of dictionaries recursively.

    This is intended as a helper function, often for use with Pydantic's
    `PlainValidator`, to prepare configurations that will be used to
    instantiate objects later.

    Args:
        x (dict): The input dictionary to validate and transform.

    Returns:
        A new dictionary with a standardized structure. Specifically,
        class type information is moved to a 'type' key as a string,
        and nested structures are similarly processed. If no 'class_type'
        or 'type' key is found, the original dictionary (deep-copied)
        is returned.
    """  # noqa: E501
    x = copy.deepcopy(x)

    # If no class type keys are present, it might be a plain data dict or already processed.  # noqa: E501
    if "class_type" not in x and "type" not in x:
        return x

    if "class_type" in x:
        class_type = x.pop("class_type")
    elif "type" in x:
        class_type = x.pop("type")

    if not isinstance(class_type, str):
        class_type = callable_to_string(class_type)

    for key, value in x.items():
        if isinstance(value, dict):
            x[key] = _validate_delay_init_dict(value)
        elif isinstance(value, (list, tuple)):
            converted = []
            for value_i in value:
                if isinstance(value_i, dict):
                    converted.append(_validate_delay_init_dict(value_i))
                else:
                    converted.append(value_i)
            x[key] = type(value)(converted)

    return dict(type=class_type, **x)


DelayInitDictType = Annotated[
    dict,
    PlainValidator(_validate_delay_init_dict),
]
"""A type alias for dictionaries intended for delayed initialization.

This annotated type signifies that a dictionary conforms to a structure
suitable for later instantiation of an object. It uses the
`_validate_delay_init_dict` function as a Pydantic `PlainValidator`,
meaning that when data is assigned to a Pydantic model field with this
type, `_validate_delay_init_dict` will be called to validate and
transform the input dictionary.

The transformed dictionary will typically have a 'type' key containing
the string representation of the class to be instantiated, and other
keys representing parameters for its constructor.
"""
