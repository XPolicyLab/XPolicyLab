"""Shared observation-grid selection for compressed visual history."""


def history_offsets(frame_stride: int) -> tuple[int, ...]:
    """Return eight uniformly spaced strictly past offsets for this release."""
    if type(frame_stride) is not int or frame_stride <= 0:
        raise ValueError("history frame_stride must be a positive integer")
    return tuple(-index * frame_stride for index in range(8, 0, -1))


def select_history_indices(
    current_index: int,
    offsets: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return requested and clamped episode indices, including the anchor."""
    requested = (0, *(current_index + offset for offset in offsets))
    return requested, tuple(max(0, index) for index in requested)
