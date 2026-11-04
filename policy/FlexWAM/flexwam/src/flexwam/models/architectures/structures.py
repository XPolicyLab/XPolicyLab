"""Token layout helpers. Dense layout semantics are unchanged."""
from typing import Any

def _video_visual_view_slices(
    video_seq_len: int,
    video_meta: dict[str, Any],
) -> list[tuple[int, int]]:
    """Return visual-token ranges for each view, excluding shared prefixes."""
    num_views = int(video_meta.get("num_views", 1))
    prefix_len = int(video_meta.get("prefix_len", 0))
    language_len = int(video_meta.get("context_len", 0))
    state_len = int(video_meta.get("state_len", 0))
    action_len = int(video_meta.get("action_len", max(0, prefix_len - language_len - state_len)))
    if min(language_len, state_len, action_len) < 0 or language_len + state_len + action_len != prefix_len:
        raise ValueError(
            "Video prefix metadata mismatch: "
            f"context_len={language_len}, state_len={state_len}, "
            f"action_len={action_len}, prefix_len={prefix_len}."
        )
    prefix_layout = str(video_meta.get("prefix_layout", "per_view"))
    if prefix_layout == "global":
        visual_tokens_per_view = int(
            video_meta.get(
                "visual_tokens_per_view",
                video_meta.get("video_tokens_per_view", 0),
            )
        )
        expected = prefix_len + num_views * visual_tokens_per_view
        if prefix_len < 0 or visual_tokens_per_view <= 0 or video_seq_len != expected:
            raise ValueError(
                "Invalid global-prefix video metadata: "
                f"seq_len={video_seq_len}, prefix_len={prefix_len}, "
                f"visual_tokens_per_view={visual_tokens_per_view}, num_views={num_views}."
            )
        return [
            (
                prefix_len + view_idx * visual_tokens_per_view,
                prefix_len + (view_idx + 1) * visual_tokens_per_view,
            )
            for view_idx in range(num_views)
        ]

    if video_seq_len % num_views != 0:
        raise ValueError(
            f"Video seq len must be divisible by num_views, got {video_seq_len} and {num_views}."
        )
    tokens_per_view = video_seq_len // num_views
    video_tokens_per_view = int(
        video_meta.get("video_tokens_per_view", tokens_per_view - prefix_len)
    )
    if prefix_len < 0 or video_tokens_per_view <= 0 or prefix_len + video_tokens_per_view != tokens_per_view:
        raise ValueError(
            "Invalid per-view video metadata: "
            f"tokens_per_view={tokens_per_view}, prefix_len={prefix_len}, "
            f"video_tokens_per_view={video_tokens_per_view}."
        )
    return [
        (
            view_idx * tokens_per_view + prefix_len,
            view_idx * tokens_per_view + tokens_per_view,
        )
        for view_idx in range(num_views)
    ]
