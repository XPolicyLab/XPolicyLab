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

from .geometric import (
  GeometricTransformation,
  GeometricChain,
  HorizontalFlip,
  VerticalFlip,
  Rotate,
  CenterCrop,
  Warp,
  Crop,
  Translate,
  Resize,
  RandomCrop,
  Rotate90,
  RandomSizedCrop,
)

from .colorspace import (
  ColorspaceTransformation,
  ColorspaceChain,
  ByteToFloat,
  Normalize,
  ChannelShuffle,
  RandomGrayscale,
  RandomGamma,
  RandomBrightness,
  RandomContrast,
  ColorJitter,
  Solarization,
  RandomChannelGamma,
  ChannelDrop,
)

from .imagelevel import GridShuffle, Blur, GaussianBlur

from .optimized import Chain, OptimizedChain

import importlib.metadata

try:
  __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
  __version__ = "0.0.0"  # Fallback for development mode
