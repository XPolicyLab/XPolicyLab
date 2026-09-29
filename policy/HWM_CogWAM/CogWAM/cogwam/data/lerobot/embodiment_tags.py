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

from enum import Enum


class EmbodimentTag(Enum):
    """Embodiment identity carried by dataset metadata and mixture statistics.

    Upstream enumerated a dozen research embodiments plus a projector-index
    mapping used by the shared GR00T action head.  The RoboDojo recipe trains a
    single dual-arm ARX-X5 embodiment through the new-embodiment slot and its
    action expert has no per-embodiment projector bank, so only this member
    survives.  The string value is load-bearing: it keys ``dataset_statistics``
    in the run directory and therefore the deployment-side denormalizer.
    """

    NEW_EMBODIMENT = "new_embodiment"


__all__ = ["EmbodimentTag"]
