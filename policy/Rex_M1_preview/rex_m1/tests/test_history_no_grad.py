# Copyright (C) 2026 the Rex-M1-preview authors. Licensed under the Apache License, Version 2.0.
"""With history_detach, history frames pass the vision tower separately under no_grad. The outputs
(main and deepstack, before and after pooling) and the parameter gradients must match a single joint
forward (small fp32 model on CPU; history and current frames interleave to check the reassembly order)."""
import types
from functools import partial

import pytest
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

from mibot.models.VLM.qwen3vl import Qwen3VLModel, Qwen3VLVisionModel

# Per image (t, h, w) in patches before merging; pooling needs >= 4x4 after merging.
GRIDS = [(1, 8, 12), (1, 8, 8), (1, 12, 8), (1, 8, 8), (1, 10, 8), (1, 8, 10)]


def _vision():
    torch.manual_seed(0)
    cfg = Qwen3VLVisionConfig(
        depth=3, hidden_size=32, intermediate_size=64, num_heads=2, in_channels=3, patch_size=4,
        temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=24, num_position_embeddings=64,
        deepstack_visual_indexes=[0, 2],
    )
    cfg._attn_implementation = "eager"
    vis = Qwen3VLVisionModel._from_config(cfg, dtype=torch.float32).train()
    vis.gradient_checkpointing_enable()
    return vis


def _model(vis):
    m = types.SimpleNamespace(visual=vis)
    m._visual_history_no_grad = partial(Qwen3VLModel._visual_history_no_grad, m)
    return m


def _run(vis, mask, split):
    m = _model(vis)
    if not split:  # reference: one forward over every image
        m._visual_history_no_grad = lambda pv, grid, *_: vis(pv, grid_thw=grid)
    grid = torch.tensor(GRIDS[: len(mask)])
    torch.manual_seed(1)
    pv = torch.randn(int(grid.prod(-1).sum()), 3 * 2 * 4 * 4)
    vis.zero_grad(set_to_none=True)
    emb, deep = Qwen3VLModel.get_image_features(
        m, pv, grid, image_pool_mask=torch.tensor(mask), image_pool_detach=True
    )
    out = torch.cat(list(emb) + deep, dim=0)
    torch.manual_seed(2)
    loss = (out * torch.randn_like(out)).sum()
    if loss.requires_grad:
        loss.backward()
    grads = {n: p.grad.clone() for n, p in vis.named_parameters() if p.grad is not None}
    return out.detach(), grads


@pytest.mark.parametrize("mask", [
    [True, False, True, True, False, True],   # history and current interleaved
    [False, False, False],                    # no history frames: no split
    [True, True, True, False],                # a single current frame
])
def test_split_matches_joint(mask):
    vis = _vision()
    out0, g0 = _run(vis, mask, split=False)
    out1, g1 = _run(vis, mask, split=True)
    assert out1.shape == out0.shape
    torch.testing.assert_close(out1, out0, rtol=1e-5, atol=1e-6)
    assert g1.keys() == g0.keys() and len(g0) > 0
    for n in g0:
        torch.testing.assert_close(g1[n], g0[n], rtol=1e-4, atol=1e-6, msg=n)


def test_all_history_forward_only():
    vis = _vision()
    mask = [True, True]
    out0, _ = _run(vis, mask, split=False)
    out1, g1 = _run(vis, mask, split=True)
    torch.testing.assert_close(out1, out0, rtol=1e-5, atol=1e-6)
    assert not g1  # all-history batch: the vision tower gets no gradient


def test_inference_path_untouched():
    vis = _vision()
    m = _model(vis)
    m._visual_history_no_grad = lambda *a: pytest.fail("the split path must not run under no_grad")
    grid = torch.tensor(GRIDS[:3])
    pv = torch.randn(int(grid.prod(-1).sum()), 96)
    with torch.no_grad():
        Qwen3VLModel.get_image_features(m, pv, grid, image_pool_mask=torch.tensor([True, False, True]),
                                        image_pool_detach=True)
