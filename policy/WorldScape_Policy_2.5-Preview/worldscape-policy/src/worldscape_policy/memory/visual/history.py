"""Causal RGB payload for the opt-in visual history prefix."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def prepare_history_frames(views: Tensor, *, layout="mosaic_T") -> Tensor:
    """Native uint8 [T,V,H,W,C] -> normalized frames, shared by train/eval.

    The 5B canvas retains FP16 at 384x320.
    """
    from worldscape_policy.visual_mosaic import prepare_diffusion_mosaic_T
    if views.ndim != 5 or views.shape[1] != 3 or views.shape[-1] != 3 or views.dtype != torch.uint8:
        raise ValueError('history expects uint8 [T,3,H,W,3]')
    if layout != "mosaic_T":
        raise ValueError(f"Unknown history layout: {layout}")
    return prepare_diffusion_mosaic_T(
        views.permute(0, 1, 4, 2, 3).unsqueeze(0).float().div(255),
        input_range='zero_one',
    )[0].half()


@dataclass
class CausalVisualHistory:
    frames: Tensor  # [B,N,3,H,W], unique real images, normalized [-1,1]
    step_ids: Tensor  # [B,N], original episode frame IDs
    mask: Tensor  # [B,N], real images vs batch padding
    anchor_ids: Tensor  # [B]
    episode_ids: tuple[str, ...]
    clip_step_ids: Tensor  # [B,9], includes repeated frame-zero padding
    include_current: bool = True

    @property
    def batch_size(self):
        return self.mask.shape[0]

    def validate(self):
        if type(self.include_current) is not bool:
            raise ValueError('history include_current must be bool')
        if self.frames.ndim != 5 or self.frames.shape[2] != 3 or not self.frames.is_floating_point():
            raise ValueError('history frames must be floating [B,N,3,H,W]')
        b, n = self.frames.shape[:2]
        if self.step_ids.shape != (b, n) or self.mask.shape != (b, n) or self.mask.dtype != torch.bool:
            raise ValueError('history IDs/mask do not match frames')
        if self.anchor_ids.shape != (b,) or len(self.episode_ids) != b:
            raise ValueError('history anchors/episode IDs do not match batch')
        if self.step_ids.dtype != torch.long or self.anchor_ids.dtype != torch.long:
            raise ValueError('history IDs must be int64')
        if bool((((self.step_ids > self.anchor_ids[:, None]) | (self.step_ids < 0)) & self.mask).any()):
            raise ValueError('negative or future frame in causal history')
        slots = self.clip_step_ids
        if slots is None or slots.dtype != torch.long or slots.shape != (b, 9):
            raise ValueError('clip_step_ids must be int64 [B,9]')
        anchors = self.anchor_ids.to(slots.device)
        if bool((slots < 0).any()) or bool((slots > anchors[:, None]).any()):
            raise ValueError('negative or future history clip frame ID')
        if bool((slots[:, 0] != 0).any()):
            raise ValueError('history clip requires episode first frame')
        if self.include_current:
            if bool((slots[:, -1] != anchors).any()):
                raise ValueError('history clip requires exact current anchor')
        else:
            # At episode step zero no past exists: all slots use the first image.
            if bool(((slots >= anchors[:, None]) & (anchors[:, None] > 0)).any()):
                raise ValueError('past-only history clip must exclude the current anchor')
            if bool(((self.step_ids >= self.anchor_ids[:, None])
                     & (self.anchor_ids[:, None] > 0) & self.mask).any()):
                raise ValueError('past-only RGB history must exclude the current anchor')
        if bool((slots[:, 1:] < slots[:, :-1]).any()):
            raise ValueError('history clip slots must be ordered, with repeats allowed')
        for row, valid, clip in zip(self.step_ids, self.mask, slots, strict=True):
            ids = row[valid]
            if len(ids) > 1 and bool((ids[1:] <= ids[:-1]).any()):
                raise ValueError('valid history IDs must be unique and ordered')
            if not bool(torch.isin(clip.to(row.device), ids).all()):
                raise ValueError('history clip slot missing from RGB history')

    def select(self, indices):
        positions = indices.nonzero().flatten().tolist() if indices.dtype == torch.bool else indices.tolist()
        def select(value):
            return value[indices.to(value.device)]
        return CausalVisualHistory(
            select(self.frames), select(self.step_ids), select(self.mask),
            select(self.anchor_ids), tuple(self.episode_ids[i] for i in positions),
            select(self.clip_step_ids),
            include_current=self.include_current,
        )


def collate_histories(histories):
    if all(h is None for h in histories):
        return None
    if any(h is None for h in histories):
        raise ValueError('mixed history-enabled and history-disabled samples')
    for h in histories:
        h.validate()
        if h.batch_size != 1:
            raise ValueError('collator expects single-example histories')
    if any(h.include_current != histories[0].include_current for h in histories):
        raise ValueError('mixed history include_current settings')
    n = max(h.mask.shape[1] for h in histories)
    first = histories[0]
    frames = first.frames.new_zeros(len(histories), n, *first.frames.shape[2:])
    ids = first.step_ids.new_zeros(len(histories), n)
    mask = first.mask.new_zeros(len(histories), n)
    for i, h in enumerate(histories):
        size = h.mask.shape[1]
        frames[i, :size], ids[i, :size], mask[i, :size] = h.frames[0], h.step_ids[0], h.mask[0]
    return CausalVisualHistory(
        frames, ids, mask, torch.cat([h.anchor_ids for h in histories]),
        tuple(h.episode_ids[0] for h in histories),
        torch.cat([h.clip_step_ids for h in histories]),
        include_current=first.include_current,
    )
