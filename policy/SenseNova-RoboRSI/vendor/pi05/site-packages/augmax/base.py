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
import jax
import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Dict, Hashable
from enum import Enum

 
class InputType(Enum):
    IMAGE = 'image'
    MASK = 'mask'
    DENSE = 'dense'
    CONTOUR = 'contour'
    KEYPOINTS = 'keypoints'
    METADATA = 'metadata'

PyTree = Union[jnp.ndarray,
               Tuple['PyTree', ...],
               List['PyTree'],
               Dict[Hashable, 'PyTree'],
               InputType,
               None]
RNGKey = Union[jax.typing.ArrayLike, None]


def same_type(left_type, right_type):
    if isinstance(left_type, InputType):
        left_type = left_type.value
    if isinstance(right_type, InputType):
        right_type = right_type.value
    return left_type.lower() == right_type.lower()


class Transformation(ABC):
    def __init__(self, input_types: PyTree=None):
        self.input_types = input_types

    def __call__(self, rng: jnp.ndarray, inputs: PyTree, input_types: PyTree=None) -> PyTree:
        if input_types is None:
          input_types = self.input_types
        if input_types is None:
          input_types = jax.tree_util.tree_map(lambda _: InputType.IMAGE, inputs)
        try:
          jax.tree_util.tree_map(lambda _x, _y: None, inputs, self.input_types)
        except ValueError:
            raise ValueError(f"PyTrees `inputs` and `input_types` are incompatible for Augmentation")

        augmented = self.apply(rng, inputs, input_types)
        return augmented

    def invert(self, rng: jnp.ndarray, inputs: PyTree, input_types: PyTree=None) -> PyTree:
        if input_types is None:
          input_types = self.input_types
        if input_types is None:
          input_types = jax.tree_util.tree_map(lambda _: InputType.IMAGE, inputs)
        try:
          jax.tree_util.tree_map(lambda _x, _y: None, inputs, self.input_types)
        except ValueError:
            raise ValueError(f"PyTrees `inputs` and `input_types` are incompatible for Augmentation")
        augmented = self.apply(rng, inputs, input_types, invert=True)
        return augmented

    @abstractmethod
    def apply(self, rng: jnp.ndarray, inputs: PyTree, input_types: PyTree=None, invert=False) -> PyTree:
        return inputs


class BaseChain(Transformation):
    def __init__(self, *transforms: Transformation, input_types=None):
        super().__init__(input_types)
        self.transforms = transforms

    def apply(self, rng: RNGKey, inputs: PyTree, input_types: PyTree, invert=False) -> PyTree:
        N = len(self.transforms)
        subkeys = [None]*N if rng is None else jax.random.split(rng, N)

        transforms = self.transforms
        if invert:
            transforms = reversed(transforms)
            subkeys = reversed(subkeys)

        values = inputs
        for transform, subkey in zip(transforms, subkeys):
            values = transform.apply(subkey, values, input_types, invert=invert)
        return values

    def __repr__(self):
        members_repr = ",\n".join(str(t) for t in self.transforms)
        members_repr = '\n'.join(['\t'+line for line in members_repr.split('\n')])
        return f'{self.__class__.__name__}(\n{members_repr}\n)'
