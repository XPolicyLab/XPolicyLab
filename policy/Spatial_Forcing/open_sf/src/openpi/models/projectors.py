"""Additional projector modules for JAX OpenPI models."""

from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class AlignProjector(nnx.Module):
    """Project VLA image-token embeddings and compute cosine alignment loss."""

    def __init__(self, llm_dim: int, config: Any, rngs: nnx.Rngs):
        self.llm_dim = llm_dim
        self.vggt_dim = config.vggt_dim
        self.vla_adapt_layers_align = bool(config.vla_adapt_layers_align)
        self.vggt_adapt_layers_align = bool(config.vggt_adapt_layers_align)
        self.iSF_zscore = bool(config.iSF_zscore)
        self.use_vlm_norm = bool(config.use_vlm_norm)

        if self.vla_adapt_layers_align or self.vggt_adapt_layers_align:
            raise NotImplementedError("JAX-align currently supports fixed VLA/VGGT layer alignment only.")

        self.fc1 = nnx.Linear(self.llm_dim, 2 * self.vggt_dim, rngs=rngs)
        self.fc2 = nnx.Linear(2 * self.vggt_dim, 2 * self.vggt_dim, rngs=rngs)
        self.vlm_norm = nnx.LayerNorm(self.llm_dim, rngs=rngs) if self.use_vlm_norm else None

    def spatial_zscore(self, feat: jax.Array, eps: float = 1e-6) -> jax.Array:
        mean = jnp.mean(feat, axis=1, keepdims=True)
        std = jnp.std(feat, axis=1, keepdims=True)
        return (feat - mean) / (std + eps)

    def align_dimension(self, llm_embedding: jax.Array) -> jax.Array:
        if self.vlm_norm is not None:
            llm_embedding = self.vlm_norm(llm_embedding)
        projected = self.fc1(llm_embedding)
        projected = jax.nn.gelu(projected)
        return self.fc2(projected)

    def __call__(
        self,
        llm_embs: jax.Array,
        target_emb: jax.Array,
        align_mask: jax.Array,
    ) -> jax.Array:
        if llm_embs.ndim != 3:
            raise ValueError(f"Expected llm_embs rank 3 [batch, token, dim], got shape {llm_embs.shape}.")

        llm_emb = self.align_dimension(llm_embs)
        target_emb = target_emb.astype(llm_emb.dtype)
        if self.iSF_zscore:
            target_emb = self.spatial_zscore(target_emb)

        llm_emb = llm_emb.astype(jnp.float32)
        target_emb = target_emb.astype(jnp.float32)
        llm_emb = llm_emb / jnp.maximum(jnp.linalg.norm(llm_emb, axis=-1, keepdims=True), 1e-6)
        target_emb = target_emb / jnp.maximum(jnp.linalg.norm(target_emb, axis=-1, keepdims=True), 1e-6)

        per_token_loss = 1.0 - jnp.sum(llm_emb * target_emb, axis=-1)
        align_mask = align_mask.astype(jnp.float32)
        denom = jnp.maximum(jnp.sum(align_mask, axis=-1), 1.0)
        return jnp.mean(jnp.sum(per_token_loss * align_mask, axis=-1) / denom)
