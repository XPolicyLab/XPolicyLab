"""Shared attention layout for one frame/action diffusion window."""

import torch


def build_frame_memory_attention_mask(
    *,
    video_seq_len: int,
    action_seq_len: int,
    video_tokens_per_frame: int,
    num_condition_frames: int,
    device: torch.device,
    num_future_delta_tokens: int = 0,
    num_anchor_condition_frames: int = 1,
) -> torch.Tensor:
    """Build the mask for real video + video-side delta queries + action."""
    video_seq_len = int(video_seq_len)
    action_seq_len = int(action_seq_len)
    num_future_delta_tokens = int(num_future_delta_tokens)
    num_anchor_condition_frames = int(num_anchor_condition_frames)
    tokens_per_frame = int(video_tokens_per_frame)
    num_condition_frames = int(num_condition_frames)
    real_video_seq_len = video_seq_len - num_future_delta_tokens

    total_len = video_seq_len + action_seq_len
    mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=device)
    for frame_idx in range(num_condition_frames):
        query = slice(
            frame_idx * tokens_per_frame, (frame_idx + 1) * tokens_per_frame
        )
        mask[query, : (frame_idx + 1) * tokens_per_frame] = True
    future_start = num_condition_frames * tokens_per_frame
    # The noisy future video is a privileged teacher and never reads delta
    # queries, keeping the teacher representation independent of the student.
    mask[future_start:real_video_seq_len, :real_video_seq_len] = True
    if num_future_delta_tokens:
        delta_start = real_video_seq_len
        local_video_start = (
            num_anchor_condition_frames * tokens_per_frame
        )
        # Delta queries read local clean video and each other, but never the
        # ground-truth/noisy future video or Action tokens.
        mask[
            delta_start:video_seq_len,
            local_video_start:future_start,
        ] = True
        mask[
            delta_start:video_seq_len,
            delta_start:video_seq_len,
        ] = True
    action_start = video_seq_len
    mask[action_start:, :future_start] = True
    if num_future_delta_tokens:
        mask[action_start:, real_video_seq_len:video_seq_len] = True
    mask[action_start:, action_start:] = True
    return mask
