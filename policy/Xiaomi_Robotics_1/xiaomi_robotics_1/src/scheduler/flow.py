# Copyright (C) 2026 Xiaomi Corporation.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

from typing import Any, Callable

import torch


class RectifiedFlow:
    @torch.no_grad()
    def generate(
        self,
        x0: torch.Tensor,
        num_steps: int = 5,
        forward_func: Callable[..., torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if forward_func is None:
            raise ValueError("forward_func is required for rectified-flow inference.")
        dt = 1.0 / num_steps
        z = x0.clone()
        for step in range(num_steps):
            t = torch.ones((z.shape[0], 1, 1), device=z.device, dtype=z.dtype) * step / num_steps
            z = z + forward_func(z, t, **kwargs) * dt
        return z
