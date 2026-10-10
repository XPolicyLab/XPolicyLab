"""CPU tests for the bundled inference components."""
import importlib
import hashlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
from torchvision.transforms import Resize, functional as TF

runtime = importlib.import_module('XPolicyLab.policy.Liber_0_lite.liber0')
preprocessing = importlib.import_module(runtime.__name__ + '.preprocessing')
scheduler = importlib.import_module(runtime.__name__ + '.scheduler')
backbone = importlib.import_module(runtime.__name__ + '.backbone')
attention = importlib.import_module(runtime.__name__ + '.attention')
batching = importlib.import_module(runtime.__name__ + '.batching')


def test_image_preprocessing_order():
    rng = np.random.default_rng(417)
    images = [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(4)]
    expected = []
    for image in images:
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = Resize([240, 320])(tensor)
        tensor = TF.resized_crop(tensor, 6, 8, 228, 304, [240, 320],
                                  interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        expected.append(TF.resize(tensor, [144, 192], interpolation=TF.InterpolationMode.BILINEAR, antialias=True))
    head, left, right, cue = expected
    mosaic = (torch.cat([torch.cat([head, cue], -1), torch.cat([left, right], -1)], -2) - 0.5) / 0.5
    assert torch.equal(preprocessing.pack_images(images, [240, 320], 0.95), mosaic)


def test_normalization_roundtrip():
    entry = {'default': {'global_mean': [1, 2], 'global_std': [2, 3]}}
    norm = preprocessing.build_norm({'state': entry, 'action': entry})
    x = torch.tensor([[2., 4.]])
    normalized = torch.clamp(x * norm['state_scale'] + norm['state_offset'], -5, 5)
    torch.testing.assert_close(preprocessing.denormalize(normalized, norm['action_scale'], norm['action_offset']), x)


@pytest.mark.parametrize('shift', [3.0, 5.0])
def test_schedule(shift):
    time, delta = scheduler.FlowScheduler(1000, shift).build_inference_schedule(10, torch.device('cpu'), torch.bfloat16)
    u = torch.linspace(1., 0., 11, dtype=torch.float32)
    sigma = shift * u / (1. + (shift - 1.) * u)
    assert torch.equal(time, (sigma[:-1] * 1000).bfloat16())
    assert torch.equal(delta, (sigma[1:] - sigma[:-1]).bfloat16())


def test_patch_roundtrip():
    image = torch.arange(3 * 32 * 48).reshape(1, 3, 32, 48)
    patches = backbone.LiberModel._patchify(image, 16)
    assert torch.equal(backbone.LiberModel._unpatchify(patches, 32, 48, 16), image)


def test_attention_isolates_action_keys_from_visual_queries():
    mask = attention.build_batched_joint_mask(torch.ones(1, 3, dtype=torch.bool), 4, 4, 2, torch.bfloat16)
    assert (mask[..., :12, 12:] == torch.finfo(torch.bfloat16).min).all()
    assert (mask[..., 12:, :] == 0).all()


def test_batch_noise_matches_single_order_and_is_independent():
    model = SimpleNamespace(device=torch.device('cpu'), sample_dtype=torch.bfloat16,
                            noise_scale=8., dit=SimpleNamespace(action_dim=14))
    images = torch.zeros(3, 3, 32, 48)
    seeds = [4, 9, 12]
    video, action = batching.initial_noise(model, images, 32, seeds)
    for index, seed in enumerate(seeds):
        generator = torch.Generator('cpu').manual_seed(seed)
        expected_video = torch.randn((1, 3, 32, 48), generator=generator).bfloat16() * 8.
        expected_action = torch.randn((1, 32, 14), generator=generator).bfloat16()
        assert torch.equal(video[index:index + 1], expected_video)
        assert torch.equal(action[index:index + 1], expected_action)
    reordered_video, reordered_action = batching.initial_noise(model, images, 32, list(reversed(seeds)))
    assert torch.equal(video.flip(0), reordered_video)
    assert torch.equal(action.flip(0), reordered_action)


def test_padding_mask_preserves_unpadded_attention():
    valid = torch.tensor([[False, False, True, True, True], [True, True, True, True, True]])
    mask = attention.build_batched_joint_mask(valid, 4, 4, 2, torch.bfloat16)
    expected = attention.build_batched_joint_mask(torch.ones(1, 3, dtype=torch.bool), 4, 4, 2, torch.bfloat16)
    assert torch.equal(mask[0:1, :, 2:, 2:], expected)
    assert (mask[0, :, 2:, :2] == torch.finfo(torch.bfloat16).min).all()


@pytest.mark.parametrize('corrupt', [False, True])
def test_download_manifest_and_pinned_assets(monkeypatch, tmp_path, corrupt):
    import huggingface_hub
    download = importlib.import_module('XPolicyLab.policy.Liber_0_lite.download_checkpoint')
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        if len(calls) > 1:
            return
        root = Path(kwargs['local_dir'])
        root.mkdir()
        hashes = {}
        for name in ('model.pt', 'config.yaml', 'dataset_stats.json', 'README.md'):
            data = name.encode()
            (root / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        (root / 'manifest.json').write_text(json.dumps({'sha256': hashes}))
        if corrupt:
            (root / 'model.pt').write_bytes(b'corrupt')

    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot)
    monkeypatch.setattr(sys, 'argv', ['download_checkpoint.py', '--destination', str(tmp_path / 'checkpoint'),
                                     '--assets-dir', str(tmp_path / 'assets')])
    if corrupt:
        with pytest.raises(ValueError, match='checksum mismatch'):
            download.main()
        assert len(calls) == 1
    else:
        download.main()
        assert len(calls) == 2
        assert all(len(call['revision']) == 40 for call in calls)
        assert calls[1]['repo_id'] == 'HiDream-ai/HiDream-O1-Image'
    assert calls[0]['repo_id'] == 'LiberAI/Liber0-Lite-Robodojo'
    assert calls[0]['revision'] == '270e1146e9f278ed1122c1ea614e5600dcf33ec3'
    assert 'runtime.tar.gz' not in calls[0]['allow_patterns']
    assert 'policy.tar.gz' not in calls[0]['allow_patterns']
