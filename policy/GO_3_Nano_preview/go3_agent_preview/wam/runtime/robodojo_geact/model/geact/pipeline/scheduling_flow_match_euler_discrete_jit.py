# Copyright 2025 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
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

import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


class FlowMatchEulerDiscreteSchedulerJiT(FlowMatchEulerDiscreteScheduler):
    @classmethod
    def from_scheduler(cls, scheduler):
        jit_scheduler = cls(
            num_train_timesteps=scheduler.config.num_train_timesteps,
            shift=getattr(scheduler.config, 'shift', 1.0),
            use_dynamic_shifting=getattr(scheduler.config, 'use_dynamic_shifting', False),
            base_shift=getattr(scheduler.config, 'base_shift', 0.5),
            max_shift=getattr(scheduler.config, 'max_shift', 1.15),
            base_image_seq_len=getattr(scheduler.config, 'base_image_seq_len', 256),
            max_image_seq_len=getattr(scheduler.config, 'max_image_seq_len', 4096),
            invert_sigmas=getattr(scheduler.config, 'invert_sigmas', False),
            shift_terminal=getattr(scheduler.config, 'shift_terminal', None),
            use_karras_sigmas=getattr(scheduler.config, 'use_karras_sigmas', False),
            use_exponential_sigmas=getattr(scheduler.config, 'use_exponential_sigmas', False),
            use_beta_sigmas=getattr(scheduler.config, 'use_beta_sigmas', False),
            time_shift_type=getattr(scheduler.config, 'time_shift_type', 'exponential'),
            stochastic_sampling=getattr(scheduler.config, 'stochastic_sampling', False),
        )
        jit_scheduler.sigmas = scheduler.sigmas.to("cpu")
        jit_scheduler.sigma_min = scheduler.sigma_min
        jit_scheduler.sigma_max = scheduler.sigma_max
        jit_scheduler.timesteps = scheduler.timesteps
        jit_scheduler._step_index = None
        jit_scheduler._begin_index = None
        return jit_scheduler

    def step(
        self,
        model_output,
        timestep,
        sample,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        generator=None,
        per_token_timesteps=None,
        return_dict=True,
    ):
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `FlowMatchEulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sample = sample.to(torch.float32)

        if per_token_timesteps is not None:
            per_token_sigmas = per_token_timesteps / self.config.num_train_timesteps

            sigmas = self.sigmas[:, None, None]
            lower_mask = sigmas < per_token_sigmas[None] - 1e-6
            lower_sigmas = lower_mask * sigmas
            lower_sigmas, _ = lower_sigmas.max(dim=0)

            current_sigma = per_token_sigmas[..., None]
            next_sigma = lower_sigmas[..., None]
            dt = current_sigma - next_sigma
        else:
            sigma_idx = self.step_index
            sigma = self.sigmas[sigma_idx]
            sigma_next = self.sigmas[sigma_idx + 1]

            current_sigma = sigma
            next_sigma = sigma_next
            dt = sigma_next - sigma

        if self.config.stochastic_sampling:
            noise = torch.randn_like(sample)
            prev_sample = (1.0 - next_sigma) * model_output + next_sigma * noise
        else:
            if current_sigma > 0.01:
                v_pred = (sample - model_output) / current_sigma.clamp_min(5e-2)
                prev_sample = sample + dt * v_pred
            else:
                prev_sample = model_output

        self._step_index += 1
        if per_token_timesteps is None:
            prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample,)

        from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
