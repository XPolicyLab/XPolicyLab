from __future__ import annotations

import torch
from torch import nn

from worldscape_policy.visual_mosaic import (
    compose_three_view_mosaic,
    compose_three_view_robotwin,
    prepare_diffusion_mosaic,
    prepare_diffusion_robotwin_concat,
)
from worldscape_policy.wam.wan22.visual_codec import WanVisualCodec


class _UnusedVAE(nn.Module):
    def encode(self, video, **kwargs):
        del kwargs
        return video


def test_three_view_mosaic_matches_dreamzero_quadrant_layout() -> None:
    views = torch.stack(
        (
            torch.full((1, 1, 1, 2, 3), 10, dtype=torch.uint8),
            torch.full((1, 1, 1, 2, 3), 20, dtype=torch.uint8),
            torch.full((1, 1, 1, 2, 3), 30, dtype=torch.uint8),
        ),
        dim=2,
    )

    mosaic = compose_three_view_mosaic(views)

    assert mosaic.shape == (1, 1, 1, 4, 6)
    assert torch.equal(mosaic[..., :2, :3], torch.full((1, 1, 1, 2, 3), 10))
    assert torch.equal(mosaic[..., 2:, :3], torch.full((1, 1, 1, 2, 3), 20))
    assert torch.equal(mosaic[..., :2, 3:], torch.full((1, 1, 1, 2, 3), 30))
    assert torch.count_nonzero(mosaic[..., 2:, 3:]) == 0


def test_diffusion_mosaic_normalizes_and_resizes_to_single_view_shape() -> None:
    views = torch.zeros((2, 3, 3, 3, 4, 8), dtype=torch.uint8)
    views[:, :, 0] = 255

    result = prepare_diffusion_mosaic(views, input_range="uint8")

    assert result.shape == (2, 3, 3, 4, 8)
    assert result.dtype == torch.float32
    assert result.amin() >= -1
    assert result.amax() <= 1


def test_robotwin_concat_resizes_raw_views_directly_to_final_layout() -> None:
    views = torch.stack(
        (
            torch.full((1, 2, 3, 480, 640), 255, dtype=torch.uint8),
            torch.zeros((1, 2, 3, 480, 640), dtype=torch.uint8),
            torch.full((1, 2, 3, 480, 640), 127, dtype=torch.uint8),
        ),
        dim=2,
    )

    composed = compose_three_view_robotwin(views.float())
    result = prepare_diffusion_robotwin_concat(views, input_range="uint8")

    assert composed.shape == (1, 2, 3, 384, 320)
    assert result.shape == (1, 2, 3, 384, 320)
    torch.testing.assert_close(
        result[..., :256, :],
        torch.ones_like(result[..., :256, :]),
    )
    torch.testing.assert_close(
        result[..., 256:, :160],
        -torch.ones_like(result[..., 256:, :160]),
    )
    torch.testing.assert_close(
        result[..., 256:, 160:],
        torch.full_like(result[..., 256:, 160:], 127 / 127.5 - 1),
    )


def test_visual_codec_dispatches_robotwin_concat() -> None:
    codec = WanVisualCodec(
        _UnusedVAE(),
        visual_input_range="uint8",
        diffusion_view_layout="robotwin_concat",
    )

    result = codec.prepare_diffusion_video(
        torch.zeros((1, 1, 3, 3, 240, 320), dtype=torch.uint8)
    )

    assert result.shape == (1, 1, 3, 384, 320)
