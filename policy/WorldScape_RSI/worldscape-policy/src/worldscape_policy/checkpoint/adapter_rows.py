"""Select category-specific WAM adapter rows during pretrained initialization."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor


_POLICY_PREFIXES = (
    "wam.core.state_encoder.",
    "wam.core.action_encoder.",
    "wam.core.action_decoder.",
)
_CORE_PREFIXES = (
    "state_encoder.",
    "action_encoder.",
    "action_decoder.",
)


def is_adapter_parameter(key: str) -> bool:
    """Return whether ``key`` belongs to a category-specific WAM adapter."""

    return key.startswith(_POLICY_PREFIXES + _CORE_PREFIXES)


def select_pretrained_adapter_row(
    source: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
    *,
    source_row: int | None,
) -> dict[str, Tensor]:
    """Adapt a multi-row pretrained state to a single-row target.

    Exact-shape adapter tensors are left unchanged, so a one-row native
    checkpoint can initialize another one-row model even when a platform
    source-row hint remains in the recipe. Non-adapter tensors are never
    modified and remain subject to the caller's strict validation.
    """

    result = dict(source)
    if source_row is None:
        return result
    if isinstance(source_row, bool) or not isinstance(source_row, int):
        raise TypeError("pretrained action-adapter source row must be an integer")
    if source_row < 0:
        raise ValueError("pretrained action-adapter source row must be non-negative")

    source_category_counts: set[int] = set()
    selected = 0
    for key in sorted(set(source) & set(target)):
        if not is_adapter_parameter(key):
            continue
        source_tensor = source[key]
        target_tensor = target[key]
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            continue
        if source_tensor.ndim < 1 or target_tensor.ndim != source_tensor.ndim:
            raise ValueError(f"Adapter tensor rank mismatch for {key!r}")
        if target_tensor.shape[0] != 1:
            # Multi-row targets must use exact loading; do not silently reshape.
            continue
        if tuple(source_tensor.shape[1:]) != tuple(target_tensor.shape[1:]):
            raise ValueError(
                f"Adapter tensor trailing shape mismatch for {key!r}: "
                f"{tuple(source_tensor.shape)} != {tuple(target_tensor.shape)}"
            )
        source_categories = int(source_tensor.shape[0])
        if source_categories <= 1:
            continue
        if source_row >= source_categories:
            raise ValueError(
                f"Adapter source row {source_row} is outside {key!r} "
                f"with {source_categories} categories"
            )
        source_category_counts.add(source_categories)
        result[key] = source_tensor[source_row : source_row + 1]
        selected += 1

    if len(source_category_counts) > 1:
        raise ValueError(
            "Pretrained adapter tensors disagree on category count: "
            + ", ".join(str(value) for value in sorted(source_category_counts))
        )
    if source_category_counts and selected == 0:
        raise ValueError("No category-specific adapter tensors were selected")
    return result


def export_single_adapter_row(
    source: Mapping[str, Tensor],
    *,
    source_row: int,
) -> dict[str, Tensor]:
    """Slice every category-specific adapter tensor for a one-row export."""

    if isinstance(source_row, bool) or not isinstance(source_row, int):
        raise TypeError("adapter source row must be an integer")
    if source_row < 0:
        raise ValueError("adapter source row must be non-negative")
    result = dict(source)
    category_counts: set[int] = set()
    adapter_tensors = 0
    selected = 0
    for key, tensor in source.items():
        if not is_adapter_parameter(key):
            continue
        adapter_tensors += 1
        if tensor.ndim < 1:
            raise ValueError(f"Adapter tensor {key!r} must have at least one dimension")
        categories = int(tensor.shape[0])
        if categories <= 1:
            continue
        if source_row >= categories:
            raise ValueError(
                f"Adapter source row {source_row} is outside {key!r} "
                f"with {categories} categories"
            )
        category_counts.add(categories)
        result[key] = tensor[source_row : source_row + 1]
        selected += 1
    if not adapter_tensors:
        raise ValueError("Checkpoint contains no adapter tensors")
    if not selected:
        return result
    if len(category_counts) != 1:
        raise ValueError(
            "Adapter tensors disagree on category count: "
            + ", ".join(str(value) for value in sorted(category_counts))
        )
    return result


__all__ = [
    "export_single_adapter_row",
    "is_adapter_parameter",
    "select_pretrained_adapter_row",
]
