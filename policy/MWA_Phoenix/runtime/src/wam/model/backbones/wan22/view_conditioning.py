"""Camera-slot IDs for packed multi-camera video latents."""

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import torch


def _normalize_metadata(value: Any, batch_size: int, *, default: str) -> list[str]:
    if value is None:
        return [default] * int(batch_size)
    if isinstance(value, str):
        return [value] * int(batch_size)
    if isinstance(value, Sequence):
        values = [default if item is None else str(item) for item in value]
        if len(values) == int(batch_size):
            return values
        if len(values) == 1:
            return values * int(batch_size)
    return [str(value)] * int(batch_size)


def _view_count(view_names: str, *, default: int) -> int:
    names = [name for name in str(view_names).split("|") if name]
    return max(1, len(names) if names else int(default))


def _split_sizes(total: int, parts: int) -> list[int]:
    parts = max(1, int(parts))
    base = int(total) // parts
    sizes = [base] * parts
    sizes[-1] += int(total) - sum(sizes)
    return sizes


def _fill_region(
    view_ids: torch.Tensor,
    *,
    top: int,
    bottom: int,
    left: int,
    right: int,
    view_id: int,
) -> None:
    if bottom <= top or right <= left:
        return
    view_ids[top:bottom, left:right] = int(view_id)


@lru_cache(maxsize=128)
def _sample_view_ids(
    *,
    layout: str,
    view_count: int,
    height: int,
    width: int,
    padding_view_id: int,
) -> torch.Tensor:
    layout = str(layout).strip().lower()
    if int(view_count) > int(padding_view_id):
        raise ValueError(
            f"View count {view_count} exceeds configured max_views={padding_view_id}."
        )
    device = torch.device("cpu")
    view_ids = torch.full(
        (int(height), int(width)),
        int(padding_view_id),
        dtype=torch.long,
        device=device,
    )
    if layout in {"horizontal", "latent_horizontal"}:
        count = int(view_count)
        cursor = 0
        for view_id, slot_w in enumerate(_split_sizes(width, count)):
            _fill_region(
                view_ids,
                top=0,
                bottom=height,
                left=cursor,
                right=cursor + slot_w,
                view_id=view_id,
            )
            cursor += slot_w
    elif layout == "vertical":
        count = int(view_count)
        # A single-view vertical sample is duplicated into two slots upstream.
        region_count = 2 if count == 1 else count
        cursor = 0
        for region_id, slot_h in enumerate(_split_sizes(height, region_count)):
            _fill_region(
                view_ids,
                top=cursor,
                bottom=cursor + slot_h,
                left=0,
                right=width,
                view_id=min(region_id, count - 1),
            )
            cursor += slot_h
    elif layout == "grid2x2":
        count = min(4, int(view_count))
        mid_h, mid_w = height // 2, width // 2
        regions = (
            (0, mid_h, 0, mid_w),
            (0, mid_h, mid_w, width),
            (mid_h, height, 0, mid_w),
            (mid_h, height, mid_w, width),
        )
        for view_id, (top, bottom, left, right) in enumerate(regions):
            if view_id >= count:
                continue
            _fill_region(
                view_ids,
                top=top,
                bottom=bottom,
                left=left,
                right=right,
                view_id=view_id,
            )
    elif layout in {"robotwin", "robotwin_tshape", "tshape"}:
        count = min(3, int(view_count))
        top_h = (height * 2) // 3
        _fill_region(
            view_ids,
            top=0,
            bottom=top_h,
            left=0,
            right=width,
            view_id=0,
        )
        if count <= 2:
            _fill_region(
                view_ids,
                top=top_h,
                bottom=height,
                left=0,
                right=width,
                view_id=max(0, count - 1),
            )
        else:
            mid_w = width // 2
            _fill_region(
                view_ids,
                top=top_h,
                bottom=height,
                left=0,
                right=mid_w,
                view_id=1,
            )
            _fill_region(
                view_ids,
                top=top_h,
                bottom=height,
                left=mid_w,
                right=width,
                view_id=2,
            )
    else:
        _fill_region(
            view_ids,
            top=0,
            bottom=height,
            left=0,
            right=width,
            view_id=0,
        )

    return view_ids


def build_view_ids(
    *,
    video_layout: Any,
    video_view_names: Any,
    batch_size: int,
    height: int,
    width: int,
    padding_view_id: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the camera slot for each packed spatial patch."""

    layouts = _normalize_metadata(video_layout, batch_size, default="single")
    names = _normalize_metadata(video_view_names, batch_size, default="")
    default_counts = {
        "horizontal": 2,
        "latent_horizontal": 2,
        "vertical": 2,
        "grid2x2": 4,
        "robotwin": 3,
        "robotwin_tshape": 3,
        "tshape": 3,
    }
    ids = [
        _sample_view_ids(
            layout=str(layouts[index]).strip().lower(),
            view_count=_view_count(
                names[index],
                default=default_counts.get(str(layouts[index]).strip().lower(), 1),
            ),
            height=int(height),
            width=int(width),
            padding_view_id=int(padding_view_id),
        )
        for index in range(int(batch_size))
    ]
    return torch.stack(ids, dim=0).to(device=device, non_blocking=True)
