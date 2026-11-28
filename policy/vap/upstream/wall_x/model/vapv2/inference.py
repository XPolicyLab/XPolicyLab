"""Request-local full-forward or exact prefix-cached action sampling for joint-v2.

No mutable text-mask side channel or training-path changes.
Inputs retain VAP's I-A0-V0-A1-V1 layout and clean/noisy timestep semantics.
"""

from __future__ import annotations

import torch

from wall_x.model.vap import sampling
from .text import TextCondition


class _BoundCore:
    def __init__(self, core, text: TextCondition):
        self._core, self._text = core, text

    def __getattr__(self, name):
        return getattr(self._core, name)

    def __call__(self, *args, **kwargs):
        if kwargs.get("capture_action_prefix_rank") is not None:
            raise ValueError("Use forward_action_stage_with_prefix_cache for capture")
        kwargs["text_valid_mask"] = self._text.valid_mask
        return self._core(*args, **kwargs)

    def forward_action_stage_with_prefix_cache(self, *args, **kwargs):
        kwargs["text_valid_mask"] = self._text.valid_mask
        return self._core.forward_action_stage_with_prefix_cache(*args, **kwargs)


class _BoundModel:
    """A read-only view: sampler eval/train bookkeeping never mutates the model."""

    training = False

    def __init__(self, model, text):
        self._model = model
        self.core = _BoundCore(model.core, text)

    def __getattr__(self, name):
        return getattr(self._model, name)

    def eval(self):
        return self

    def train(self, mode=True):
        if mode:
            raise RuntimeError("Inference view cannot enter training mode")
        return self


class FullForwardSamplingModel:
    """Delegate VAE/config access; bind explicit text validity per sampling call.

    This is not an nn.Module and never owns or renames checkpoint parameters.
    The wrapped model must already be in eval mode.
    """

    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        return getattr(self._model, name)

    @torch.no_grad()
    def _sample(self, function, *, text_context, use_kv_cache=False, **kwargs):
        if use_kv_cache and self._model.core.state_film is not None:
            raise NotImplementedError(
                "VAPv2 prefix cache does not support State-FiLM; use full-forward"
            )
        if self._model.training:
            raise RuntimeError("Call model.eval() before VAPv2 inference")
        if not isinstance(text_context, TextCondition):
            raise TypeError(
                "VAPv2 sampling requires TextCondition with explicit Qwen validity"
            )
        parameter = next(self._model.core.parameters())
        text = text_context.to(device=parameter.device, dtype=self._model.torch_dtype)
        expected = (
            self.config.text_feature_seq_len,
            self.config.text_feature_hidden_dim,
        )
        if tuple(text.hidden_states.shape[1:]) != expected:
            raise ValueError(
                f"Qwen context shape must be [B,{expected[0]},{expected[1]}]"
            )
        lengths = text.lengths
        right_padded = (
            torch.arange(expected[0], device=lengths.device)[None] < lengths[:, None]
        )
        if not torch.equal(text.valid_mask, right_padded):
            raise ValueError("Qwen validity must describe right padding including EOS")
        if not torch.isfinite(text.hidden_states[text.valid_mask]).all():
            raise ValueError("Valid Qwen features must be finite")
        return function(
            _BoundModel(self._model, text),
            text_context=text.hidden_states,
            use_kv_cache=use_kv_cache,
            **kwargs,
        )

    def sample_first_action_chunk_given_video(self, **kwargs):
        return self._sample(sampling.sample_first_action_chunk_given_video, **kwargs)

    def sample_next_action_chunk_given_prefix(self, **kwargs):
        return self._sample(sampling.sample_next_action_chunk_given_prefix, **kwargs)
