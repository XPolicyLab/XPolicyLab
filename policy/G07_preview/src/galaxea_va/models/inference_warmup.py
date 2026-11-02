"""Shared startup warmup for AR and single-frame optional IDM inference."""

from __future__ import annotations

import time
from typing import Optional

import torch

from galaxea_va.utils.logging_config import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def warmup_ar_inference(
    self,
    *,
    video_size: tuple[int, int],
    prompt: str,
    num_video_frames: int,
    num_action_denoise_steps: int,
    num_video_denoise_steps: int,
    action_infer_mode: str,
    sigma_shift: Optional[float] = None,
    rand_device: str = "cpu",
    proprio: Optional[torch.Tensor] = None,
    reference_image: Optional[torch.Tensor] = None,
) -> list[float]:
    """Compile and exercise reachable cache lengths before accepting requests.

    A full synthetic episode covers growth and initial cropping. Two calls
    in a second episode cover cold-start and incremental replay after reset.
    The normal periodic reset is exercised if reached before two full-window
    updates, using the window/reset limits supplied by the model's warmup entry.
    Each eager solver loop uses at most two steps to cover initial noise and
    an updated latent; compiled cores do not specialize on the step count.
    Subsequent inference uses the caller's full solver schedule. The caller
    must keep the same mode, shapes and dtype, or run another warmup.
    Pass proprio when normal requests include it: Wan22 appends its condition
    token only when a proprio tensor is supplied.

    Returns per-call wall times in seconds, including CUDA completion. Both
    successful and failed warmups leave empty Transformer/VAE history while
    preserving compiled functions.
    """
    update_frames = num_video_frames - 1
    latents_per_chunk = update_frames // self.vae.temporal_downsample_factor
    if update_frames <= 0 or update_frames % self.vae.temporal_downsample_factor:
        raise ValueError("Warmup requires complete VAE observation chunks.")
    capacity = max(1, self.infer_sliding_window * latents_per_chunk)
    chunks_to_fill = (capacity - 1 + latents_per_chunk - 1) // latents_per_chunk
    calls_per_episode = min(self.infer_reset_every + 1, 1 + chunks_to_fill + 2)
    height, width = video_size
    # Match the policy's [T,C,H,W] -> [1,C,T,H,W] observation layout.
    observed = torch.zeros(
        update_frames, 3, height, width, device=self.device, dtype=self.torch_dtype,
    ).permute(1, 0, 2, 3).unsqueeze(0)
    infer = self.infer_action
    elapsed = []
    try:
        for episode, episode_calls in enumerate((calls_per_episode, 2)):
            for step in range(episode_calls):
                logger.info(
                    "AR compile warmup | mode=%s | episode=%d/2 | call=%d/%d",
                    action_infer_mode, episode + 1, step + 1, episode_calls,
                )
                started = time.perf_counter()
                infer(
                    prompt=prompt,
                    observed_video=observed[:, :, :1] if step == 0 else observed,
                    num_video_frames=num_video_frames,
                    num_action_denoise_steps=min(num_action_denoise_steps, 2),
                    num_video_denoise_steps=min(num_video_denoise_steps, 2),
                    sigma_shift=sigma_shift,
                    rand_device=rand_device,
                    compile_action_infer=True,
                    action_infer_mode=action_infer_mode,
                    reset_cache=step == 0,
                    proprio=proprio,
                    **({"reference_image": reference_image} if reference_image is not None else {}),
                )
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                elapsed.append(time.perf_counter() - started)
    finally:
        self.reset_ar_cache()
    logger.info("AR compile warmup complete | elapsed=%.2fs", sum(elapsed))
    return elapsed
