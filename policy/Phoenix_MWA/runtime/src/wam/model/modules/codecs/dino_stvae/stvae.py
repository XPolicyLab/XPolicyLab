"""Spatial Transformer + causal temporal VAE for unpooled DINO features.

Spatiotemporal variational autoencoder.
Original SHA256: 7ac8c86e332467aa03d639e84831685f0c72f9aa882acb66551066e47095cfc4.
Architecture and state-dict names are unchanged for STVAE-90k compatibility.

Spatial blocks retain OpenWAM SVAE's parameter layout. Temporal operations are
Wan-inspired, independently implemented (no spatial downsampling or Wan weights).
Input [B,C,1+4k,H,W] -> latent [B,L,1+k,H,W]. Cameras belong in B.
"""

from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

MODEL_TYPE = "spatial_transformer_temporal_vae"
FORMAT_VERSION = 1


class ChannelRMSNorm(nn.Module):
    """Normalize channels independently at every (batch,time,y,x) position."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1, 1))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(x.dtype)


class CausalTimeConv(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(in_channels, out_channels, (kernel_size, 1, 1))
        self.left_padding = kernel_size - 1

    def forward(self, x):
        return super().forward(F.pad(x, (0, 0, 0, 0, self.left_padding, 0)))


class TemporalResidual(nn.Module):
    """Bottleneck causal residual; spatial context comes from the Transformer."""

    def __init__(self, channels, hidden_dim):
        super().__init__()
        self.norm = ChannelRMSNorm(channels)
        self.in_proj = nn.Conv3d(channels, hidden_dim, 1)
        self.time_conv = CausalTimeConv(hidden_dim, hidden_dim)
        self.out_proj = nn.Conv3d(hidden_dim, channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        h = self.in_proj(F.silu(self.norm(x)))
        h = self.time_conv(F.silu(h))
        return x + self.out_proj(F.silu(h))


class TemporalDownsample(nn.Module):
    """Preserve first frame; output positions correspond to 0,2,4,... .

    Each later output uses a learned kernel over (2j-2,2j-1,2j).
    Two stages therefore yield endpoint positions 0,4,8,... .
    """

    def __init__(self, channels, hidden_dim):
        super().__init__()
        self.residual = TemporalResidual(channels, hidden_dim)
        self.down = nn.Conv3d(channels, channels, (3, 1, 1), stride=(2, 1, 1))

    def forward(self, x):
        if x.shape[2] % 2 != 1:
            raise ValueError("TemporalDownsample requires odd T")
        x = self.residual(x)
        if x.shape[2] == 1:
            return x
        return torch.cat((x[:, :, :1], self.down(x)), dim=2)


class TemporalUpsample(nn.Module):
    """One initial position plus two positions per subsequent latent.

    This is causal on the latent/chunk axis, not a framewise forecasting decoder.
    """

    def __init__(self, channels, hidden_dim):
        super().__init__()
        self.expand = CausalTimeConv(channels, 2 * channels)
        self.residual = TemporalResidual(channels, hidden_dim)

    def forward(self, x):
        b, c, t, h, w = x.shape
        if t > 1:
            future = self.expand(x)[:, :, 1:]
            future = future.reshape(b, 2, c, t - 1, h, w)
            future = future.permute(0, 2, 3, 1, 4, 5).reshape(b, c, 2 * (t - 1), h, w)
            x = torch.cat((x[:, :, :1], future), dim=2)
        return self.residual(x)


def spatial_block(dim, heads, intermediate_size, dropout):
    return nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=heads,
        dim_feedforward=intermediate_size,
        dropout=dropout,
        activation="gelu",
        layer_norm_eps=1e-12,
        batch_first=True,
        norm_first=True,
    )


class STVAE(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        latent_dim=48,
        num_heads=16,
        num_layers=3,
        intermediate_size=2048,
        dropout=0.0,
        temporal_hidden_dim=256,
        gradient_checkpointing=False,
    ):
        super().__init__()
        if (
            min(
                input_dim,
                latent_dim,
                num_heads,
                num_layers,
                intermediate_size,
                temporal_hidden_dim,
            )
            < 1
        ):
            raise ValueError("model dimensions must be positive")
        if input_dim % num_heads or not 0 <= dropout < 1:
            raise ValueError("invalid heads/dropout")
        self.input_dim, self.latent_dim = input_dim, latent_dim
        self.num_heads, self.num_layers = num_heads, num_layers
        self.intermediate_size, self.dropout = intermediate_size, dropout
        self.temporal_hidden_dim = temporal_hidden_dim
        self.gradient_checkpointing = gradient_checkpointing
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.enc_blocks = nn.ModuleList(
            [
                spatial_block(input_dim, num_heads, intermediate_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.enc_norm = nn.LayerNorm(input_dim, eps=1e-12)
        self.enc_proj = nn.Linear(input_dim, 2 * latent_dim)
        self.dec_proj = nn.Linear(latent_dim, input_dim)
        self.dec_blocks = nn.ModuleList(
            [
                spatial_block(input_dim, num_heads, intermediate_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.dec_norm = nn.LayerNorm(input_dim, eps=1e-12)
        self.temporal_encoder = nn.ModuleList(
            [TemporalDownsample(input_dim, temporal_hidden_dim) for _ in range(2)]
        )
        self.temporal_decoder = nn.ModuleList(
            [TemporalUpsample(input_dim, temporal_hidden_dim) for _ in range(2)]
        )

    def config_dict(self):
        return {
            k: getattr(self, k)
            for k in (
                "input_dim",
                "latent_dim",
                "num_heads",
                "num_layers",
                "intermediate_size",
                "dropout",
                "temporal_hidden_dim",
                "gradient_checkpointing",
            )
        }

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.input_mean = self.input_mean.float()
        self.input_std = self.input_std.float()
        return self

    def set_input_stats(self, mean, std, eps=1e-6):
        mean = mean.detach().reshape(-1).to(self.input_mean)
        std = std.detach().reshape(-1).to(self.input_std)
        if mean.numel() != self.input_dim or std.numel() != self.input_dim:
            raise ValueError("statistics channel count mismatch")
        if (
            not torch.isfinite(mean).all()
            or not torch.isfinite(std).all()
            or (std < 0).any()
        ):
            raise ValueError("invalid feature statistics")
        self.input_mean.copy_(mean)
        self.input_std.copy_(std.clamp_min(eps))

    def _validate(self, x, channels, uncompressed=False):
        if x.ndim != 5 or x.shape[1] != channels or min(x.shape) < 1:
            raise ValueError(
                f"expected nonempty [B,{channels},T,H,W], got {tuple(x.shape)}"
            )
        if uncompressed and (x.shape[2] - 1) % 4:
            raise ValueError("unpooled feature sequence must have T=1+4k")

    def standardize(self, x):
        return (
            x.float() - self.input_mean[None, :, None, None, None]
        ) / self.input_std[None, :, None, None, None]

    @staticmethod
    def _to_seq(x):
        b, c, t, h, w = x.shape
        return x.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c), (b, t, h, w)

    @staticmethod
    def _from_seq(x, shape):
        b, t, h, w = shape
        return x.reshape(b, t, h, w, -1).permute(0, 4, 1, 2, 3).contiguous()

    def _run(self, module, x):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def _encode_params(self, x):
        self._validate(x, self.input_dim, uncompressed=True)
        seq, shape = self._to_seq(self.standardize(x))
        for block in self.enc_blocks:
            seq = self._run(block, seq)
        h = self._from_seq(seq, shape)
        for block in self.temporal_encoder:
            h = self._run(block, h)
        seq, shape = self._to_seq(h)
        params = self.enc_proj(self.enc_norm(seq))
        mu, logvar = self._from_seq(params, shape).chunk(2, dim=1)
        return mu, logvar.float().clamp(-30.0, 20.0)

    def encode(self, x):
        """Training returns (sample,mu,logvar); eval returns deterministic mean."""
        mu, logvar = self._encode_params(x)
        if not self.training:
            return mu
        z = (mu.float() + (0.5 * logvar).exp() * torch.randn_like(mu.float())).to(
            mu.dtype
        )
        return z, mu, logvar

    @contextmanager
    def _deterministic(self):
        states = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            yield
        finally:
            for module, training in states:
                module.training = training

    def encode_mean(self, x):
        """Deterministic even if an enclosing policy recursively calls train()."""
        with self._deterministic():
            return self._encode_params(x)[0]

    def _decode_std(self, z):
        self._validate(z, self.latent_dim)
        seq, shape = self._to_seq(z)
        h = self._from_seq(self.dec_proj(seq), shape)
        for block in self.temporal_decoder:
            h = self._run(block, h)
        seq, shape = self._to_seq(h)
        for block in self.dec_blocks:
            seq = self._run(block, seq)
        return self._from_seq(self.dec_norm(seq), shape)

    def decode(self, z):
        """Reconstruct unpooled features in original DINO scale."""
        x = self._decode_std(z).float()
        return (
            x * self.input_std[None, :, None, None, None]
            + self.input_mean[None, :, None, None, None]
        )

    def forward(self, x):
        mu, logvar = self._encode_params(x)
        z = mu
        if self.training:
            z = (mu.float() + (0.5 * logvar).exp() * torch.randn_like(mu.float())).to(
                mu.dtype
            )
        return dict(
            recon=self._decode_std(z), target=self.standardize(x), mu=mu, logvar=logvar
        )


def stvae_loss(out, beta=1e-4, cos_weight=1.0):
    recon, target = out["recon"].float(), out["target"].float()
    if recon.shape != target.shape:
        raise ValueError("reconstruction/target shape mismatch")
    mse = F.mse_loss(recon, target)
    cos = 1.0 - F.cosine_similarity(recon, target, dim=1).mean()
    mu, logvar = out["mu"].float(), out["logvar"].float()
    kl = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).mean()
    loss = mse + cos_weight * cos + beta * kl
    return loss, {
        k: v.detach() for k, v in dict(loss=loss, mse=mse, cos=cos, kl=kl).items()
    }


def load_stvae(path, device="cpu"):
    """Load model-only stvae.pt with weights_only=True; reject plain S-VAE."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        payload.get("model_type") != MODEL_TYPE
        or payload.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("not a supported ST-VAE checkpoint")
    model = STVAE(**payload["model_config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False)

