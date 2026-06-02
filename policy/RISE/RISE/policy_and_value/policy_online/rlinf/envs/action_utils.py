# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch



def prepare_actions_for_roborl(
    raw_chunk_actions,
    model_name,
) -> np.ndarray:

    chunk_actions = raw_chunk_actions

    return chunk_actions


def prepare_actions(
    raw_chunk_actions,
    simulator_type,
    model_name,
    num_action_chunks,
    action_dim,
    action_scale: float = 1.0,
    policy: str = "widowx_bridge",
) -> torch.Tensor | np.ndarray:
    
    if simulator_type == "roborl":
        chunk_actions = prepare_actions_for_roborl(
            raw_chunk_actions=raw_chunk_actions,
            model_name=model_name,
        )        
    else:
        raise NotImplementedError

    return chunk_actions
