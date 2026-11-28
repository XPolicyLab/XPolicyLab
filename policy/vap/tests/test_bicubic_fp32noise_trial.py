"""Private ablation: full-frame PIL Bicubic and native FP32 action noise."""
import importlib.util
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision.transforms import functional as F, InterpolationMode

from test_action_precision_trial import make_model
from wall_x.model.vap import sampling
from wall_x.model.vapv2.action_precision import configure_action_precision
from wall_x.model.vapv2.inference import FullForwardSamplingModel


def bridge():
    path = Path(__file__).resolve().parents[1]/'bridge.py'
    spec = importlib.util.spec_from_file_location('trial_bridge', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('shape', [(480, 640), (256, 320), (120, 160)])
def test_full_frame_pil_bicubic_exact_and_deterministic(monkeypatch, shape):
    monkeypatch.setenv('VAP_IMAGE_RESIZE_MODE', 'pil_bicubic')
    m = bridge()
    rgb = np.random.default_rng(42).integers(0, 256, (*shape, 3), dtype=np.uint8)
    before = rgb.copy()
    # Training's RandomResizedCrop functional, with the entire input selected.
    expected = np.asarray(F.resized_crop(Image.fromarray(rgb), 0, 0, *shape,
                                        [256, 320], InterpolationMode.BICUBIC))
    for seed in (0, 42):
        torch.manual_seed(seed)
        observed = m.resize_rgb(rgb, height=256, width=320, name='rgb')
        assert observed.dtype == np.uint8 and observed.flags.c_contiguous
        assert np.array_equal(observed, expected)
    assert np.array_equal(before, rgb)


def test_disabled_resize_matches_original_and_bad_mode_fails(monkeypatch):
    monkeypatch.delenv('VAP_IMAGE_RESIZE_MODE', raising=False)
    m = bridge()
    rgb = np.random.default_rng(8).integers(0, 256, (480,640,3), dtype=np.uint8)
    expected = F.resize(torch.from_numpy(rgb).permute(2,0,1), [256,320],
                        interpolation=InterpolationMode.BILINEAR).permute(1,2,0).numpy()
    assert np.array_equal(m.resize_rgb(rgb,height=256,width=320,name='rgb'), expected)
    monkeypatch.setenv('VAP_IMAGE_RESIZE_MODE','typo')
    with pytest.raises(ValueError):
        m.resize_rgb(rgb,height=256,width=320,name='rgb')


def test_rgb_camera_and_history_order(monkeypatch):
    monkeypatch.setenv('VAP_IMAGE_RESIZE_MODE','pil_bicubic')
    m = bridge()
    colors = ((255,0,0),(0,255,0),(0,0,255))
    frames = {cam: np.full((480,640,3), color, np.uint8)
              for cam,color in zip(m.CAMERA_NAMES, colors)}
    obs = {'vision':{k:{'color':v} for k,v in frames.items()}}
    views = m.extract_resized_views(obs,height=256,width=320)
    hist = m.extract_executed_video(
        {'executed_video':{k:np.stack([v,v]) for k,v in frames.items()}},
        height=256,width=320,video_frames=2)
    for i,color in enumerate(colors):
        assert np.all(views[i] == color)
        assert np.array_equal(hist[i,0], views[i])
        assert np.array_equal(hist[i,1], views[i])


@pytest.mark.parametrize('seed',[0,42,12345])
def test_direct_noise_exact_dtype_and_repeatability(seed):
    model, _, kw = make_model('fp32_head_accum')
    configure_action_precision(model.core,'fp32_head_accum',initial_noise_dtype='fp32')
    kw.update(video_noise=None,action_noise=None,text_context=torch.zeros(1,4,6))
    state = sampling._prepare_sampling_state(model, generator=torch.Generator().manual_seed(seed), **kw)
    gen = torch.Generator().manual_seed(seed)
    expected_video = torch.randn(1,1,8,4,1,1,dtype=torch.bfloat16,generator=gen)
    expected_action = torch.randn(1,4,4,dtype=torch.float32,generator=gen)
    assert state.action.dtype == torch.float32
    assert torch.equal(state.action,expected_action)
    assert not torch.equal(state.action,state.action.bfloat16().float())
    assert torch.equal(state.video[:,:,:,1:],expected_video)
    again = sampling._prepare_sampling_state(model, generator=torch.Generator().manual_seed(seed), **kw)
    assert torch.equal(state.action,again.action)
    kw['action_noise'] = expected_action
    supplied = sampling._prepare_sampling_state(model, generator=torch.Generator().manual_seed(seed), **kw)
    assert torch.equal(supplied.action,expected_action)


@pytest.mark.parametrize('chunk',[0,1])
@torch.no_grad()
def test_native_fp32_kv_full_and_prefix_memory_unchanged(chunk):
    model,text,kw = make_model('fp32_head_accum')
    configure_action_precision(model.core,'fp32_head_accum',initial_noise_dtype='fp32')
    facade = FullForwardSamplingModel(model)
    kw.update(action_noise=None)
    method=facade.sample_first_action_chunk_given_video
    if chunk:
        kw.update(clean_prefix_video_latents=torch.randn(1,1,8,2,1,1),
                  clean_prefix_action=torch.randn(1,2,4))
        method=facade.sample_next_action_chunk_given_prefix
    memory=kw['first_frame_memory'].latents.clone()
    a=method(text_context=text,generator=torch.Generator().manual_seed(0),use_kv_cache=True,**kw)
    b=method(text_context=text,generator=torch.Generator().manual_seed(0),use_kv_cache=False,**kw)
    assert a.actions.dtype == torch.float32
    torch.testing.assert_close(a.generated_chunk,b.generated_chunk,atol=0.02,rtol=0.02)
    assert torch.equal(memory,kw['first_frame_memory'].latents)
    if chunk:
        assert torch.equal(a.actions[:,:2],kw['clean_prefix_action'].bfloat16().float())


def test_invalid_noise_mode_rejected():
    model,_,_=make_model('legacy')
    with pytest.raises(ValueError):
        configure_action_precision(model.core,'fp32_head_accum',initial_noise_dtype='typo')
    with pytest.raises(ValueError):
        configure_action_precision(model.core,'legacy',initial_noise_dtype='fp32')
