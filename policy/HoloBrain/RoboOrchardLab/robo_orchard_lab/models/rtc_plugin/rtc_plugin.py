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

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class RTCInferencePlugin(nn.Module):
    """Plugin to use real-time chunking (RTC) during asynchronous inference.

    In an async inference setup, when a new action chunk is predicted, there
    might still be a portion of the previously predicted actions that have not
    yet been executed by the agent. To prevent abrupt changes or "jumps"
    between the unexecuted old actions and the new ones,
    this plugin smoothly blends them.

    It works by applying a weighted average to the initial part of the new
    prediction and the remaining part of the old one, over a "delay horizon"
    that corresponds to the system's latency. The weight for the old actions
    starts at 1 and linearly decreases to 0. This ensures that actions closer
    to the present time step follow the old trajectory for continuity, while
    gradually transitioning to fully adopt the new prediction.

    This process results in a smoother, more coherent action trajectory, which
    can significantly improve task success rates by reducing action jitter.
    """

    def __init__(self, max_horizon=32):
        super().__init__()
        self.max_horizon = max_horizon

    def forward(self, pred, remaining_actions, delay_horizon):
        """Using remaining actions to smooth the prediction this time.

        Args:
            pred (torch.Tensor): The newly predicted actions of shape
                (B, N_r, N_j, C), where B is batch size, N_r is the length of
                last remaining actions, N_j is number of joints.
            remaining_actions (torch.Tensor): The unexecuted actions last time
                of shape (B, N_p, N_j, C), where B is batch size, N_p is the
                length of current predicted actions, N_j is number of joints.
            delay_horizon (int): The number of time steps to blend, which
                corresponds to the system's latency.
        """
        max_horizon = min(
            pred.shape[1], remaining_actions.shape[1], self.max_horizon
        )
        delay_horizon = min(delay_horizon, max_horizon)
        weights = torch.ones(delay_horizon)

        if delay_horizon < max_horizon:
            h = max_horizon - delay_horizon
            ci = torch.torch.linspace(1, 0, steps=h + 2)[1:-1]
            decay = ci * (torch.exp(ci) - 1) / (torch.ones(h) * torch.e - 1)
            weights = torch.cat([weights, decay])
        else:
            logger.warning(
                "Received 'remaining_actions' is too short; "
                "this may result in motion jitter. "
                f"remaining steps: {remaining_actions.shape[1]}."
            )

        weights = weights.to(pred)
        weights = weights[:max_horizon, None, None]
        rtc_dim = remaining_actions.shape[-1]
        pred[:, :max_horizon, :, :rtc_dim] = (
            weights * remaining_actions[:, :max_horizon]
            + (1 - weights) * pred[:, :max_horizon, :, :rtc_dim]
        )
        return pred
