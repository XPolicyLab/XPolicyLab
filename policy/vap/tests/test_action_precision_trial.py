"""Inference-only precision tests; no real model required."""
import importlib.util
import os
from pathlib import Path
import sys

import pytest
import torch

from test_vapv2_inference import tiny
from wall_x.model.vap import sampling
from wall_x.model.vapv2.action_precision import configure_action_precision
from wall_x.model.vapv2.inference import FullForwardSamplingModel


def make_model(mode):
    model, text, kw = tiny()
    model.core.to(torch.bfloat16)
    model.torch_dtype = torch.bfloat16
    configure_action_precision(model.core, mode)
    return model, text, kw


@torch.no_grad()
def test_fp32_head_values_keys_and_autocast():
    model, _, _ = make_model("legacy")
    before = {k: v.clone() for k, v in model.core.action_head.state_dict().items()}
    configure_action_precision(model.core, "fp32_head_accum")
    head = model.core.action_head
    assert set(head.state_dict()) == set(before)
    for name, tensor in head.state_dict().items():
        assert tensor.dtype == torch.float32
        assert torch.equal(tensor, before[name].float())
    x = torch.randn(1, 4, 6).bfloat16()
    t = torch.randn(1, 4, 6).bfloat16()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = head(x, t)
    assert out.dtype == torch.float32
    assert torch.equal(out, head(x.float(), t.float()))
    configure_action_precision(model.core, "legacy")
    for name, tensor in model.core.action_head.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_reject_bad_mode_and_live_graph():
    model, _, _ = make_model("legacy")
    with pytest.raises(ValueError):
        configure_action_precision(model.core, "typo")
    model.core._v2_action_accelerator = object()
    with pytest.raises(RuntimeError, match="disable"):
        configure_action_precision(model.core, "fp32_head_accum")


def test_fp32_head_restores_checkpoint_values_not_bf16_roundtrip():
    model, _, _ = make_model("legacy")
    original = {k: torch.full_like(v, 1.001, dtype=torch.float32)
                for k, v in model.core.action_head.state_dict().items()}
    model.core._v2_action_head_checkpoint_fp32 = original
    configure_action_precision(model.core, "fp32_head_accum")
    for k, v in model.core.action_head.state_dict().items():
        assert torch.equal(v, original[k])
        assert not torch.equal(v, original[k].bfloat16().float())
    configure_action_precision(model.core, "legacy")
    for k, v in model.core.action_head.state_dict().items():
        assert torch.equal(v, original[k].bfloat16())


@pytest.mark.parametrize("chunk", [0, 1])
@pytest.mark.parametrize("mode", ["legacy", "fp32_accum", "fp32_head_accum"])
@torch.no_grad()
def test_sampler_dtype_prefix_memory_and_kv(chunk, mode):
    model, text, kw = make_model(mode)
    facade = FullForwardSamplingModel(model)
    method = facade.sample_first_action_chunk_given_video
    if chunk:
        kw.update(clean_prefix_video_latents=torch.randn(1, 1, 8, 2, 1, 1),
                  clean_prefix_action=torch.randn(1, 2, 4))
        method = facade.sample_next_action_chunk_given_prefix
    old_memory = kw['first_frame_memory'].latents.clone()
    out = method(text_context=text, use_kv_cache=True, **kw)
    expected_dtype = torch.bfloat16 if mode == "legacy" else torch.float32
    assert out.actions.dtype == expected_dtype
    assert torch.equal(kw['first_frame_memory'].latents, old_memory)
    if chunk:
        expected_prefix = kw['clean_prefix_action'].bfloat16().to(expected_dtype)
        assert torch.equal(out.actions[:, :2], expected_prefix)
    full = method(text_context=text, use_kv_cache=False, **kw)
    torch.testing.assert_close(out.generated_chunk.float(), full.generated_chunk.float(),
                               atol=0.02, rtol=0.02)
    # Invalid history remains causally invisible, including in FP32 modes.
    kw['first_frame_memory'].latents[:, :, :, 1:3] += 1000
    masked = method(text_context=text, use_kv_cache=True, **kw)
    assert torch.equal(out.generated_chunk, masked.generated_chunk)


@pytest.mark.skipif(not os.environ.get("FROZEN_SAMPLING_SOURCE"), reason="Historical pre-ablation source fixture is not included in the release")
@pytest.mark.parametrize("chunk", [0, 1])
@torch.no_grad()
def test_disabled_bitwise_matches_frozen_source(chunk):
    path = Path(os.environ['FROZEN_SAMPLING_SOURCE'])
    name = 'wall_x.model.vap._frozen_sampling_for_precision_test'
    spec = importlib.util.spec_from_file_location(name, path)
    original = importlib.util.module_from_spec(spec)
    sys.modules[name] = original
    spec.loader.exec_module(original)
    model, text, kw = make_model("legacy")
    facade = FullForwardSamplingModel(model)
    fn_name = 'sample_first_action_chunk_given_video'
    if chunk:
        kw.update(clean_prefix_video_latents=torch.randn(1, 1, 8, 2, 1, 1),
                  clean_prefix_action=torch.randn(1, 2, 4))
        fn_name = 'sample_next_action_chunk_given_prefix'
    for use_cache in (False, True):
        a = facade._sample(getattr(original, fn_name), text_context=text,
                           use_kv_cache=use_cache, **kw)
        b = getattr(facade, fn_name)(text_context=text, use_kv_cache=use_cache, **kw)
        assert torch.equal(a.actions, b.actions)


def test_small_updates_are_not_discarded_in_fp32_state():
    state = torch.ones(1, 2, 1, dtype=torch.bfloat16)
    precise = state.float()
    velocity = torch.full_like(state, 0.016)
    for _ in range(16):
        for name, value in (("legacy", state), ("fp32", precise)):
            out = sampling._replace_action_chunk(value, velocity,
                chunk_index=0, steps_per_chunk=2, delta_sigma=torch.tensor([0.0625]))
            if name == "legacy":
                state = out
            else:
                precise = out
    assert torch.equal(state, torch.ones_like(state))
    assert (precise > 1.015).all()
