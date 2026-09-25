from __future__ import annotations


def resolve_recent_history_index(
    history_length: int,
    offset: int,
    *,
    episode_step: int | None = None,
) -> tuple[int, bool]:
    """Resolve a retained-history index and whether it is a real observation.

    Histories keep the episode anchor followed by a recent tail.  Indexing is
    therefore based on the retained list, while validity is based on real
    episode progress when it is available.  This deliberately does not use a
    replan counter: replanning cadence and observation age are independent.
    """
    history_length = int(history_length)
    offset = int(offset)
    if history_length <= 0:
        raise ValueError(
            f"history_length must be positive, got {history_length}."
        )
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}.")

    current_step = (
        history_length - 1 if episode_step is None else int(episode_step)
    )
    if current_step < 0:
        raise ValueError(f"episode_step must be non-negative, got {current_step}.")

    is_real = current_step >= offset
    retained_index = history_length - 1 - offset
    if not is_real:
        return 0, False
    if retained_index < 0:
        raise ValueError(
            "Recent observation is old enough to be real but is no longer in "
            f"the retained history: length={history_length}, offset={offset}, "
            f"episode_step={current_step}."
        )
    return retained_index, True
