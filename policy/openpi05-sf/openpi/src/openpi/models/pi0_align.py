from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import projectors
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at


class Pi0Align(pi0.Pi0):
    def __init__(self, config: Any, rngs: nnx.Rngs):
        super().__init__(config.model, rngs)
        paligemma_config = _gemma.get_config(config.model.paligemma_variant)
        self.vla_layers_align = (
            config.vla_layers_align
            if config.vla_layers_align >= 0
            else paligemma_config.depth + config.vla_layers_align
        )
        self.align_projector = projectors.AlignProjector(paligemma_config.width, config, rngs)
        self.align_loss_coeff = config.align_loss_coeff

    def compute_align_losses(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        align_targets: at.Float[at.Array, "b s d"],
        align_mask: at.Bool[at.Array, "b s"],
        *,
        train: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train, sf_align=True)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        image_token_count = sum((image.shape[1] // 14) * (image.shape[2] // 14) for image in observation.images.values())
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        hidden_capture = _gemma.HiddenCaptureSpec(
            expert_index=0,  # paligemma, not action_expert
            layer_index=self.vla_layers_align,
            token_count=image_token_count,
        )
        (prefix_out, suffix_out), _, vision_hidden = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            hidden_capture=hidden_capture,
        )

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        action_loss = jnp.mean(jnp.square(v_t - u_t))
        align_loss = self.align_projector(vision_hidden, align_targets, align_mask)
        loss = action_loss + self.align_loss_coeff * align_loss

        return loss, action_loss, align_loss
