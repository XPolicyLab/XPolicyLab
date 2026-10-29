from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch
from diffusers import AutoencoderDC
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor

from policy_core.model.WAM_policy.ltx_models.transformer_ltx_multiview_action_expert import (
    ActionExpert,
)
from policy_core.model.WAM_policy.pipeline.va_binding_pipeline import VABindingPipe


logger = logging.getLogger(__name__)


def normalize_rcs_timestep_ratios(values: Sequence[float]) -> tuple[float, ...]:
    ratios = tuple(float(value) for value in values)
    if not ratios:
        raise ValueError("rcs_timestep_ratios must not be empty")
    if any(not 0.0 <= value < 1.0 for value in ratios):
        raise ValueError(
            f"rcs_timestep_ratios must be in [0, 1), got {ratios}"
        )
    return ratios


def select_best_candidates(
    candidates: torch.Tensor,
    scores: torch.Tensor,
    *,
    batch_size: int,
    num_proposals: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidates.shape[0] != batch_size * num_proposals:
        raise ValueError(
            "candidate batch does not match batch_size*num_proposals: "
            f"{candidates.shape[0]} != {batch_size}*{num_proposals}"
        )
    if scores.shape != (batch_size, num_proposals):
        raise ValueError(
            f"scores must have shape {(batch_size, num_proposals)}, got {tuple(scores.shape)}"
        )
    best_indices = scores.argmax(dim=1)
    flat_indices = (
        torch.arange(batch_size, device=scores.device) * num_proposals
        + best_indices
    )
    return candidates.index_select(0, flat_indices), best_indices


class RCSTTCVABindingPipe(VABindingPipe):
    def __init__(
        self,
        vae: AutoencoderDC,
        transformer: ActionExpert,
        scheduler_action=FlowMatchEulerDiscreteScheduler(),
        dtype=torch.bfloat16,
        args=None,
        ON_LINE_TRAINING=False,
        num_proposals: int = 4,
        rcs_timestep_ratios: Sequence[float] = (0.5,),
        rcs_noise_seed: int | None = None,
    ) -> None:
        super().__init__(
            vae=vae,
            transformer=transformer,
            scheduler_action=scheduler_action,
            dtype=dtype,
            args=args,
            ON_LINE_TRAINING=ON_LINE_TRAINING,
        )
        self.num_proposals = int(num_proposals)
        if self.num_proposals <= 0:
            raise ValueError("num_proposals must be positive")
        self.rcs_timestep_ratios = normalize_rcs_timestep_ratios(
            rcs_timestep_ratios
        )
        self.rcs_noise_seed = (
            None if rcs_noise_seed is None else int(rcs_noise_seed)
        )
        self.last_rcs_scores: torch.Tensor | None = None
        self.last_best_indices: torch.Tensor | None = None

    @classmethod
    def from_pipeline(
        cls,
        pipeline: VABindingPipe,
        *,
        num_proposals: int,
        rcs_timestep_ratios: Sequence[float],
        rcs_noise_seed: int | None = None,
    ) -> "RCSTTCVABindingPipe":
        if int(num_proposals) <= 0:
            raise ValueError("num_proposals must be positive")
        if isinstance(pipeline, cls):
            pipeline.num_proposals = int(num_proposals)
            pipeline.rcs_timestep_ratios = normalize_rcs_timestep_ratios(
                rcs_timestep_ratios
            )
            pipeline.rcs_noise_seed = rcs_noise_seed
            return pipeline
        return cls(
            vae=pipeline.vae,
            transformer=pipeline.transformer,
            scheduler_action=pipeline.scheduler_action,
            args=pipeline.args,
            num_proposals=num_proposals,
            rcs_timestep_ratios=rcs_timestep_ratios,
            rcs_noise_seed=rcs_noise_seed,
        )

    @staticmethod
    def _repeat(value: Any, count: int, batch_size: int) -> Any:
        if value is None or count == 1:
            return value
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                "RCS TTC only supports tensor conditioning values, got "
                f"{type(value).__name__}"
            )
        if value.shape[0] != batch_size:
            raise ValueError(
                f"conditioning batch mismatch: {value.shape[0]} != {batch_size}"
            )
        return value.repeat_interleave(count, dim=0)

    @torch.no_grad()
    def _rcs_scores(
        self,
        candidates: torch.Tensor,
        *,
        batch_size: int,
        vision_feature: torch.Tensor,
        state_cond: torch.Tensor,
        action_mask: torch.Tensor | None,
        view_ids: torch.Tensor | None,
        body_emb_id: torch.Tensor | None,
    ) -> torch.Tensor:
        device = candidates.device
        dtype = self.transformer.dtype
        proposal_batch, action_chunk, action_dim = candidates.shape
        num_steps = len(self.scheduler_action.timesteps)
        if num_steps <= 0:
            raise RuntimeError("action scheduler has no inference timesteps")

        indices = [
            min(int(num_steps * ratio), num_steps - 1)
            for ratio in self.rcs_timestep_ratios
        ]
        score_generator = None
        if self.rcs_noise_seed is not None:
            score_generator = torch.Generator(device=device).manual_seed(
                self.rcs_noise_seed
            )
        shared_noise = randn_tensor(
            (batch_size, action_chunk, action_dim),
            generator=score_generator,
            device=device,
            dtype=dtype,
        ).repeat_interleave(self.num_proposals, dim=0)

        mask = action_mask
        if mask is not None:
            mask = mask.to(device=device, dtype=dtype)
            if mask.ndim == 2:
                mask = mask.unsqueeze(1)
            shared_noise = shared_noise * mask


        target_velocity = shared_noise.float() - candidates.float()
        score_sum = torch.zeros(proposal_batch, device=device, dtype=torch.float32)
        was_training = self.transformer.training
        self.transformer.eval()
        try:
            for index in indices:
                sigma = self.scheduler_action.sigmas[index].to(
                    device=device, dtype=dtype
                )
                timestep = self.scheduler_action.timesteps[index].to(device=device)
                noisy_actions = (
                    (1.0 - sigma) * candidates.to(dtype)
                    + sigma * shared_noise
                )
                action_timestep = timestep.reshape(1, 1).expand(
                    proposal_batch, action_chunk
                )
                prediction = self.transformer(
                    action_states=noisy_actions,
                    action_timestep=action_timestep,
                    vision_velocity=vision_feature,
                    state_cond=state_cond,
                    view_ids=view_ids,
                    body_emb_id=body_emb_id,
                )["action"].float()
                squared_error = (prediction - target_velocity).square()
                if mask is not None:
                    squared_error = squared_error * mask.float()
                    denominator = (
                        mask.float().expand(-1, action_chunk, -1).sum(dim=(1, 2))
                    ).clamp_min(1.0)
                    mse = squared_error.sum(dim=(1, 2)) / denominator
                else:
                    mse = squared_error.mean(dim=(1, 2))
                score_sum -= mse
        finally:
            self.transformer.train(was_training)

        return (score_sum / len(indices)).view(batch_size, self.num_proposals)

    @torch.no_grad()
    def infer(self, **kwargs: Any):
        if self.num_proposals == 1:
            return super().infer(**kwargs)
        if kwargs.get("noise") is not None:
            raise ValueError(
                "explicit replay noise is incompatible with num_proposals > 1"
            )

        vision_feature = kwargs.get("vision_feature")
        if not isinstance(vision_feature, torch.Tensor):
            raise TypeError("vision_feature must be a tensor for RCS TTC")
        batch_size = vision_feature.shape[0]
        expanded = dict(kwargs)
        for key in (
            "vision_feature",
            "state_cond",
            "action_mask",
            "view_ids",
            "body_emb_id",
        ):
            expanded[key] = self._repeat(
                kwargs.get(key), self.num_proposals, batch_size
            )

        candidates = super().infer(**expanded)[0]["action"]
        scores = self._rcs_scores(
            candidates,
            batch_size=batch_size,
            vision_feature=expanded["vision_feature"],
            state_cond=expanded["state_cond"],
            action_mask=expanded["action_mask"],
            view_ids=expanded["view_ids"],
            body_emb_id=expanded["body_emb_id"],
        )
        selected, best_indices = select_best_candidates(
            candidates,
            scores,
            batch_size=batch_size,
            num_proposals=self.num_proposals,
        )
        self.last_rcs_scores = scores.detach().cpu()
        self.last_best_indices = best_indices.detach().cpu()
        logger.info(
            "RCS TTC selected proposals=%s scores=%s",
            self.last_best_indices.tolist(),
            self.last_rcs_scores.tolist(),
        )
        return ({
            "action": selected,
            "rcs_scores": scores,
            "rcs_best_indices": best_indices,
        },)
