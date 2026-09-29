# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from cogwam.data.lerobot.transform.base import (
    ComposedModalityTransform,
    InvertibleModalityTransform,
    ModalityTransform,
)
from cogwam.data.lerobot.transform.concat import ConcatTransform
from cogwam.data.lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)

# The upstream video-augmentation transforms (VideoCrop, VideoColorJitter, ...)
# are not ported: this recipe instantiates none of them, and they were the
# only reason the data layer needed albumentations and OpenCV.
__all__ = [
    "ComposedModalityTransform",
    "ConcatTransform",
    "InvertibleModalityTransform",
    "ModalityTransform",
    "StateActionToTensor",
    "StateActionTransform",
]
