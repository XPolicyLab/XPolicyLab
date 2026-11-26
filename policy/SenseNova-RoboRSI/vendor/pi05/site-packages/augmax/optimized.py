# Copyright 2024 Konrad Heidler
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
from .base import BaseChain, Transformation, InputType
from .geometric import GeometricChain, GeometricTransformation
from .colorspace import ColorspaceChain, ColorspaceTransformation


class Chain(BaseChain):
    def __init__(self, *transforms: Transformation, input_types=None):
        geometric = []
        colorspace = []
        other = []

        sub_chains = []
        for transform in transforms:
            if isinstance(transform, GeometricTransformation):
                if other:
                    sub_chains.append(BaseChain(*other))
                    other = []
                if colorspace:
                    sub_chains.append(ColorspaceChain(*colorspace))
                    colorspace = []
                geometric.append(transform)
            elif isinstance(transform, ColorspaceTransformation):
                if other:
                    sub_chains.append(BaseChain(*other))
                    other = []
                if geometric:
                    sub_chains.append(GeometricChain(*geometric))
                    geometric = []
                colorspace.append(transform)
            else:
                if geometric:
                    sub_chains.append(GeometricChain(*geometric))
                    geometric = []
                if colorspace:
                    sub_chains.append(ColorspaceChain(*colorspace))
                    colorspace = []
                other.append(transform)

        if other:
            sub_chains.append(BaseChain(*other))
            other = []
        if geometric:
            sub_chains.append(GeometricChain(*geometric))
            geometric = []
        if colorspace:
            sub_chains.append(ColorspaceChain(*colorspace))
            colorspace = []

        super().__init__(*sub_chains, input_types=input_types)


class OptimizedChain(BaseChain):
    def __init__(self, *transforms: Transformation, input_types=None):
        geometric = []
        colorspace = []
        other = []
        for transform in transforms:
            if isinstance(transform, GeometricTransformation):
                geometric.append(transform)
            elif isinstance(transform, ColorspaceTransformation):
                colorspace.append(transform)
            else:
                other.append(transform)

        sub_chains = []
        if geometric:
            sub_chains.append(GeometricChain(*geometric))
        if colorspace:
            sub_chains.append(ColorspaceChain(*colorspace))
        if other:
            sub_chains.append(Chain(*other))

        super().__init__(*sub_chains, input_types=input_types)
