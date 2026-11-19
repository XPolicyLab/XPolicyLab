"""The video latents of an episode from SANA's causal LTX-2 VAE.

The checkpoints were trained on latents of ``AutoencoderKLCausalLTX2Video`` (``ltx2_causal_vae.py``, a copy of
``diffusion/model/ltx2/causal_vae.py`` of the SANA repository). On frames that repeat the first frame of an episode
another implementation of the same encoder gives bf16 latents that differ by far more than rounding, so the adapter
encodes with that class, one view at a time as the training latents were built. ``CausalViewStream`` has the
interface of ``sana_robot.deploy.memory.view_stream.ViewStream``, which it replaces in the deployment.
"""

from __future__ import annotations

import torch
from diffusers.models.autoencoders import AutoencoderKLLTX2Video
from sana_robot.data.ar.two_stream import strip_view_latents

from .ltx2_causal_vae import AutoencoderKLCausalLTX2Video, EncoderCacheManager

LATENT_BUFFERS = ("latents_mean", "latents_std")


def load_causal_encoder(model_path, device, dtype=torch.bfloat16):
    """Load SANA's causal LTX-2 encoder with the weights of the diffusers VAE.

    Args:
        model_path: The directory of the VAE weights in the diffusers layout, with the ``vae`` subfolder.
        device: The device to encode on.
        dtype: The precision of the weights.

    Returns:
        The encoder, and ``normalize``, which maps its latent means to the normalized latents the policy reads.
    """
    vae = AutoencoderKLLTX2Video.from_pretrained(str(model_path), subfolder="vae", torch_dtype=dtype)
    arguments = {name: value for name, value in dict(vae.config).items() if not name.startswith("_")}
    if "upsample_type" in arguments:
        arguments["decoder_upsample_type"] = arguments.pop("upsample_type")
    arguments = {name: tuple(value) if isinstance(value, list) else value for name, value in arguments.items()}
    encoder = AutoencoderKLCausalLTX2Video(**arguments).eval()
    encoder.decoder = None
    weights = {
        name: value
        for name, value in vae.state_dict().items()
        if name.startswith("encoder.") or name in LATENT_BUFFERS
    }
    loaded = encoder.load_state_dict(weights, strict=False)
    missing = [name for name in loaded.missing_keys if name.startswith("encoder.") or name in LATENT_BUFFERS]
    if loaded.unexpected_keys or missing:
        raise RuntimeError(f"the causal encoder does not take the VAE's weights: {loaded.unexpected_keys} {missing}")
    encoder = encoder.to(device=device, dtype=dtype).requires_grad_(False)
    mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
    std = vae.latents_std.view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
    scale = float(vae.config.scaling_factor)

    def normalize(latents: torch.Tensor) -> torch.Tensor:
        return (latents - mean) * scale / std

    return encoder, normalize


def copy_cache(entry, tensors: bool):
    """Return an encoder cache (a list of tensors, None and small dicts) with its containers copied, and its tensors too when ``tensors`` is set."""
    if torch.is_tensor(entry):
        return entry.clone() if tensors else entry
    if isinstance(entry, dict):
        return {name: copy_cache(value, tensors) for name, value in entry.items()}
    if isinstance(entry, (list, tuple)):
        return type(entry)(copy_cache(value, tensors) for value in entry)
    return entry


class CausalViewStream:
    """The latents of an episode's camera views from SANA's causal VAE, encoded while the frames arrive.

    Every view has a stream of its own: the first frame alone gives the sink, and the frames observed while a chunk
    ran, encoded after everything before them, give the chunk's target latents, as one encode of the whole video
    would. The methods are those of ``sana_robot``'s ``ViewStream``.
    """

    def __init__(self, encoder: AutoencoderKLCausalLTX2Video, normalize) -> None:
        """Build a stream.

        Args:
            encoder: The encoder of ``load_causal_encoder``.
            normalize: Its latent normalization.
        """
        self.encoder = encoder
        self.normalize = normalize
        self.caches = []

    def start(self, frames) -> torch.Tensor:
        """Open the stream of a new episode with the first frame of every view, [3, 1, H_v, W_v] in [-1, 1]; returns the sink latent [1, C, 1, 1, sum(h_v * w_v)]."""
        self.caches = [EncoderCacheManager() for _ in frames]
        return self._pack([self._encode(clip, cache, True) for clip, cache in zip(frames, self.caches)])

    def encode_chunk(self, frames) -> torch.Tensor:
        """Encode the frames observed while a chunk ran, [3, 8 * f, H_v, W_v] per view, after everything the stream has seen; returns the chunk's target latents [1, C, f, 1, sum(h_v * w_v)]."""
        if len(frames) != len(self.caches):
            raise ValueError(f"the stream holds {len(self.caches)} views, got frames of {len(frames)}")
        return self._pack([self._encode(clip, cache, False) for clip, cache in zip(frames, self.caches)])

    def encode_observation(self, frames) -> torch.Tensor:
        """Encode one frame of every view alone, [3, 1, H_v, W_v], as the observation of a chunk; the stream stays as it is. Returns the observation latent [1, C, 1, 1, sum(h_v * w_v)]."""
        return self._pack([self._encode(clip, EncoderCacheManager(), True) for clip in frames])

    def save(self) -> list:
        """Return a copy of the stream's state for ``load``."""
        return [copy_cache(cache.feat_map, tensors=True) for cache in self.caches]

    def load(self, saved: list) -> None:
        """Resume from a saved state, which stays as it is: the encoder replaces the entries of the cache it encodes on."""
        self.caches = [EncoderCacheManager() for _ in saved]
        for cache, state in zip(self.caches, saved):
            cache._feat_map = copy_cache(state, tensors=False)

    @torch.no_grad()
    def _encode(self, clip: torch.Tensor, cache: EncoderCacheManager, first: bool) -> torch.Tensor:
        """Encode the next frames of one view on its cache and return their normalized latents [1, C, f, h, w] as float32."""
        encoder = self.encoder
        frames = clip.shape[1]
        if (first and frames != 1) or (not first and frames % encoder.temporal_compression_ratio):
            raise ValueError(
                f"a stream takes its first frame alone, then multiples of {encoder.temporal_compression_ratio} "
                f"frames; got {frames}"
            )
        own = encoder._encoder_cache
        encoder._encoder_cache = cache
        try:
            pixels = clip[None].to(device=encoder.device, dtype=encoder.dtype)
            posterior = encoder.streaming_causal_encode(pixels, causal=True, reset_cache=first).latent_dist
        finally:
            encoder._encoder_cache = own
        cache.check_pending_consumed()
        return self.normalize(posterior.mode()).float()

    @staticmethod
    def _pack(latents) -> torch.Tensor:
        """Return the views' latents [1, C, f, h_v, w_v] side by side as the policy reads them, [1, C, f, 1, sum(h_v * w_v)]."""
        return strip_view_latents(latents).unsqueeze(3)
