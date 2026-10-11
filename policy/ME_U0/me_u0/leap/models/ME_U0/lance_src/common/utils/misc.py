# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# coding: utf-8

from dataclasses import dataclass

@dataclass
class AutoEncoderParams:
    downsample_spatial: int
    downsample_temporal: int
    z_channels: int
    # for flux
    scale_factor: float = 0.3611
    shift_factor: float = 0.1159

def tuple_mul(a: tuple, b: tuple) -> tuple:
    """
    返回两个同长度 tuple 的按位乘积。

    参数：
        a (tuple of numbers)：第一个元组
        b (tuple of numbers)：第二个元组，长度需与 a 一致

    返回：
        tuple：按位相乘后的结果
    """
    if len(a) != len(b):
        raise ValueError("两个元组长度必须相等")
    return tuple(x * y for x, y in zip(a, b))
