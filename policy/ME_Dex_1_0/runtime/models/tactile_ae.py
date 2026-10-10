"""Frozen unified tactile encoder used by the ME-Dex runtime."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .tactile_encoder.layout import surface_slots
from .tactile_encoder.model import UnifiedTactileAE


ROBOTWIN_SLOT_IDS = (*range(8), *range(17, 25))


class PositiveForceNormalizer(nn.Module):
    """Checkpoint-defined physical-force normalization for RoboTwin."""

    def __init__(self, parameters):
        super().__init__()
        self.deadband_low_n = float(parameters["deadband_low_n"])
        self.deadband_high_n = float(parameters["deadband_high_n"])
        knee = torch.tensor(parameters["knee_n"], dtype=torch.float32)
        upper = torch.tensor(parameters["upper_n"], dtype=torch.float32)
        self.register_buffer("knee_n", knee)
        self.register_buffer("denominator", torch.asinh(upper / knee))

    @staticmethod
    def _channel_view(values, channels):
        shape = [1] * values.ndim
        shape[-3] = 3
        return channels.reshape(shape)

    def normalize(self, physical_force, support_mask):
        support = support_mask.to(dtype=physical_force.dtype)
        physical = physical_force * support
        magnitude = physical.float().norm(dim=-3, keepdim=True)
        unit = (magnitude - self.deadband_low_n) / (
            self.deadband_high_n - self.deadband_low_n
        )
        unit = unit.clamp(0.0, 1.0)
        cleaned = physical * (unit.square() * (3.0 - 2.0 * unit))
        normalized = torch.asinh(
            cleaned / self._channel_view(cleaned, self.knee_n)
        ) / self._channel_view(cleaned, self.denominator)
        normalized = normalized.clamp(-1.0, 1.0) * support
        normal = normalized.select(-3, 0).clamp_min(0.0)
        normalized = normalized.clone()
        normalized.select(-3, 0).copy_(normal)
        return normalized, cleaned, support


class TactileAE:
    """Load the unified encoder and expose the RoboTwin sequence contract."""

    def __init__(self, *, checkpoint_path, device, dtype):
        checkpoint = Path(checkpoint_path).expanduser().resolve()

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = UnifiedTactileAE()
        self.model.load_state_dict(state["ema"]["shadow"], strict=True)
        self.model.eval().requires_grad_(False).to(device=device, dtype=dtype)
        spec = state["config"]["normalizations"]["robotwin_si"]
        if spec["type"] != "positive_normal_si":
            raise ValueError("RoboTwin requires positive-normal physical force")
        self.normalizer = PositiveForceNormalizer(spec["parameters"])
        self.normalizer.eval().requires_grad_(False).to(device=device)
        self.device = device
        self.dtype = dtype
        self.slots = surface_slots("robotwin").to(device)
        self.slot_ids = torch.tensor(ROBOTWIN_SLOT_IDS, device=device)
        if self.slots.flatten().tolist() != list(ROBOTWIN_SLOT_IDS):
            raise ValueError("Unexpected RoboTwin tactile slot layout")

    @staticmethod
    def _sequence_support(support, batch, time, device):
        support = support.to(device=device, dtype=torch.bool)
        if support.ndim == 5:
            support = support[:, None].expand(batch, time, 4, 1, 10, 14)
        if tuple(support.shape) != (batch, time, 4, 1, 10, 14):
            raise ValueError(f"Unexpected tactile support shape: {tuple(support.shape)}")
        if not torch.equal(support, support[:, :1].expand_as(support)):
            raise ValueError("Tactile structural support must remain fixed in a window")
        return support

    @staticmethod
    def _dataset_support(support, time):
        expected = (4, time, 1, 10, 14)
        if support.ndim != 6 or tuple(support.shape[1:]) != expected:
            raise ValueError(f"Expected tactile support [B,4,{time},1,10,14]")
        return support.permute(0, 2, 1, 3, 4, 5)

    @staticmethod
    def _valid_tokens(support):
        support = support.squeeze(3)
        quadrants = (
            support[..., :5, :7],
            support[..., :5, 7:],
            support[..., 5:, :7],
            support[..., 5:, 7:],
        )
        return torch.stack(
            [quadrant.any(dim=(-2, -1)) for quadrant in quadrants], dim=-1
        ).flatten(2, 3)

    @torch.no_grad()
    def _encode(self, force, support):
        if force.ndim != 6 or tuple(force.shape[2:]) != (4, 3, 10, 14):
            raise ValueError("Expected tactile force [B,T,4,3,10,14]")
        force = force.to(device=self.device, dtype=torch.float32)
        batch, time = force.shape[:2]
        support = self._sequence_support(support, batch, time, self.device)
        normalized, _, _ = self.normalizer.normalize(force, support)
        static_support = support[:, 0]
        packed_force = normalized.permute(0, 2, 1, 3, 4, 5).flatten(0, 1)
        packed_support = static_support.flatten(0, 1)
        owner = torch.arange(batch, device=self.device).repeat_interleave(4)
        slots = self.slots.repeat(batch, 1)
        output = self.model(
            packed_force.to(self.dtype),
            packed_support,
            owner,
            slots,
            batch,
        )
        latent = output["frame_tokens"].index_select(2, self.slot_ids)
        valid = output["frame_token_valid"].index_select(2, self.slot_ids)
        expected = (batch, time, 16, 48)
        if tuple(latent.shape) != expected:
            raise RuntimeError(f"Unexpected unified tactile shape: {tuple(latent.shape)}")
        return latent, valid

    @torch.no_grad()
    def encode_raw(
        self,
        observed_source,
        future_source,
        *,
        observed_support_source,
        future_support_source,
        **_,
    ):
        force = torch.cat((observed_source, future_source), dim=2).permute(
            0, 2, 1, 3, 4, 5
        )
        support = torch.cat(
            (
                self._dataset_support(observed_support_source, 2),
                self._dataset_support(future_support_source, 16),
            ),
            dim=1,
        )
        return self._encode(force, support)

    @torch.no_grad()
    def encode_condition(self, observed_source, *, observed_support_source, **_):
        force = observed_source.permute(0, 2, 1, 3, 4, 5)
        support = self._dataset_support(observed_support_source, 2)
        return self._encode(force, support)
