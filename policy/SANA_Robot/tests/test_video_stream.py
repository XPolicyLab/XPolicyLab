"""The view stream of SANA's causal LTX-2 VAE, on a small random encoder: the stream gives the latents of one encode of the whole video."""

import pytest
import torch

pytest.importorskip("sana_robot")

from XPolicyLab.policy.SANA_Robot import runtime  # noqa: E402
from XPolicyLab.policy.SANA_Robot.ltx2_causal_vae import AutoencoderKLCausalLTX2Video  # noqa: E402
from XPolicyLab.policy.SANA_Robot.video_stream import CausalViewStream  # noqa: E402

SIZES = ((64, 96), (32, 64), (32, 64))
# The layout of the LTX-2.3 VAE (its diffusers config), with small widths and one layer per block.
LTX_2_3_LAYOUT = dict(
    in_channels=3,
    out_channels=3,
    latent_channels=4,
    block_out_channels=(8, 16, 16, 16),
    decoder_block_out_channels=(8, 16, 16, 16),
    layers_per_block=(1, 1, 1, 1, 1),
    decoder_layers_per_block=(1, 1, 1, 1, 1),
    spatio_temporal_scaling=(True, True, True, True),
    decoder_spatio_temporal_scaling=(True, True, True, True),
    decoder_inject_noise=(False, False, False, False, False),
    down_block_types=("LTX2VideoDownBlock3D",) * 4,
    downsample_type=("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
    decoder_upsample_type=("spatiotemporal", "spatiotemporal", "temporal", "spatial"),
    upsample_residual=(False, False, False, False),
    upsample_factor=(2, 2, 1, 2),
    timestep_conditioning=False,
    patch_size=4,
    patch_size_t=1,
    resnet_norm_eps=1e-6,
    scaling_factor=1.0,
    encoder_causal=True,
    decoder_causal=False,
    encoder_spatial_padding_mode="zeros",
    decoder_spatial_padding_mode="zeros",
    spatial_compression_ratio=32,
    temporal_compression_ratio=8,
)


@pytest.fixture(scope="module")
def stream_parts():
    torch.manual_seed(3)
    encoder = AutoencoderKLCausalLTX2Video(**LTX_2_3_LAYOUT).eval()
    encoder.decoder = None
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.normal_(0.0, 0.05)
    encoder = encoder.float().requires_grad_(False)
    mean, std = torch.randn(1, 4, 1, 1, 1), torch.rand(1, 4, 1, 1, 1) + 0.5
    return encoder, lambda latents: (latents - mean) / std


def video(frames, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [torch.rand(3, frames, *size, generator=generator) * 2 - 1 for size in SIZES]


def one_shot(parts, clips):
    """The latents of every view's whole video in one encode, packed as the policy reads them."""
    encoder, normalize = parts
    latents = [normalize(encoder.encode(clip[None]).latent_dist.mode()) for clip in clips]
    return CausalViewStream._pack(latents)


@pytest.mark.parametrize("frozen", [0, 6, 8])
def test_the_stream_gives_the_latents_of_one_encode_of_the_whole_video(stream_parts, frozen):
    """Also when the video opens on copies of its first frame, as the windows of the phases before the episode do."""
    clips = video(25)
    for clip in clips:
        clip[:, 1 : 1 + frozen] = clip[:, :1]
    whole = one_shot(stream_parts, clips)
    stream = CausalViewStream(*stream_parts)
    pieces = [stream.start([clip[:, :1] for clip in clips])]
    pieces += [stream.encode_chunk([clip[:, first : first + 8] for clip in clips]) for first in (1, 9, 17)]
    assert [tuple(piece.shape) for piece in pieces] == [(1, 4, 1, 1, 10)] * 4      # 2 x 3 + 1 x 2 + 1 x 2 latents
    assert all(piece.dtype == torch.float32 for piece in pieces)
    assert torch.equal(torch.cat(pieces, dim=2), whole)


def test_an_observation_is_its_frame_alone_and_leaves_the_stream(stream_parts):
    clips = video(17)
    stream = CausalViewStream(*stream_parts)
    stream.start([clip[:, :1] for clip in clips])
    first = stream.encode_chunk([clip[:, 1:9] for clip in clips])
    observation = stream.encode_observation([clip[:, 8:9] for clip in clips])
    assert torch.equal(observation, one_shot(stream_parts, [clip[:, 8:9] for clip in clips]))
    second = stream.encode_chunk([clip[:, 9:17] for clip in clips])
    assert torch.equal(torch.cat((first, second), dim=2), one_shot(stream_parts, clips)[:, :, 1:])


def test_a_saved_state_resumes_the_stream_and_stays_as_it_is(stream_parts):
    """One state serves several continuations, as the phases of a replanning deployment need."""
    clips, other = video(17), video(17, seed=1)
    stream = CausalViewStream(*stream_parts)
    stream.start([clip[:, :1] for clip in clips])
    opened = stream.save()
    ahead = stream.encode_chunk([clip[:, 1:9] for clip in other])
    stream.load(opened)
    first = stream.encode_chunk([clip[:, 1:9] for clip in clips])
    stream.load(opened)
    again = stream.encode_chunk([clip[:, 1:9] for clip in clips])
    assert torch.equal(first, again) and not torch.equal(first, ahead)
    assert torch.equal(first, one_shot(stream_parts, [clip[:, :9] for clip in clips])[:, :, 1:])
    middle = stream.save()
    second = stream.encode_chunk([clip[:, 9:17] for clip in clips])
    stream.start([clip[:, :1] for clip in other])
    stream.load(middle)
    assert torch.equal(stream.encode_chunk([clip[:, 9:17] for clip in clips]), second)
    assert torch.equal(second, one_shot(stream_parts, clips)[:, :, 2:])


def test_the_stream_refuses_frames_it_cannot_place(stream_parts):
    clips = video(9)
    stream = CausalViewStream(*stream_parts)
    with pytest.raises(ValueError, match="first frame alone"):
        stream.start([clip[:, :2] for clip in clips])
    stream.start([clip[:, :1] for clip in clips])
    with pytest.raises(ValueError, match="multiples of 8"):
        stream.encode_chunk([clip[:, 1:6] for clip in clips])
    with pytest.raises(ValueError, match="3 views"):
        stream.encode_chunk([clip[:, 1:9] for clip in clips[:2]])


def test_use_causal_vae_replaces_the_deployments_stream(stream_parts, monkeypatch):
    import XPolicyLab.policy.SANA_Robot.video_stream as video_stream

    loaded = []
    monkeypatch.setattr(video_stream, "load_causal_encoder", lambda path, device: loaded.append((path, device)) or stream_parts)
    deployment = type("Deployment", (), {"config": type("Config", (), {"vae": "/vae"}), "device": "cuda", "stream": None})()
    runtime.use_causal_vae(deployment)
    assert isinstance(deployment.stream, CausalViewStream) and loaded == [("/vae", "cuda")]
