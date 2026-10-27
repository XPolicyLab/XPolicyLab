"""Temporal-window extent shared by map-style providers."""

def full_temporal_future_offset(
    *,
    num_frames: int,
    sample_stride: int,
    action_start_offset: int = 0,
) -> int:
    """Return the largest native-frame offset queried from one anchor."""

    num_frames = int(num_frames)
    sample_stride = int(sample_stride)
    action_start_offset = int(action_start_offset)
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    if sample_stride < 1:
        raise ValueError("sample_stride must be positive")
    if action_start_offset < 0:
        raise ValueError("action_start_offset must be non-negative")
    video_future_offset = (num_frames - 1) * sample_stride
    action_future_offset = (
        num_frames - 2 + action_start_offset
    ) * sample_stride
    return max(video_future_offset, action_future_offset)
