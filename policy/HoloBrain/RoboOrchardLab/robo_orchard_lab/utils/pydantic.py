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

import base64
import pickle

__all__ = [
    "pydantic_serialize_with_pickle",
    "pydantic_deserialize_with_pickle",
]


def pydantic_serialize_with_pickle(value):
    buff = pickle.dumps(value)
    return base64.b64encode(buff).decode("utf-8")


def pydantic_deserialize_with_pickle(value):
    if isinstance(value, str):
        buff = base64.b64decode(value)
        return pickle.loads(buff)
    else:
        return value
