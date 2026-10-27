"""Optional debug decoding on CPU, independent of the inference VAE."""
from copy import deepcopy

import torch

from policy_core.utils.WAM_policy.data_utils import decode_latents


class CpuVideoDecoder:
    def __init__(self, source_vae, stats_tracker=None):
        # Construct directly on CPU: deepcopy(source_vae) would first duplicate
        # GPU parameters. Restore CPU RNG so debug output cannot affect inference.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            self.vae = type(source_vae).from_config(source_vae.config)
        self.vae.to(device="cpu", dtype=torch.float32)
        self.vae.load_state_dict({k: v.detach().cpu() for k, v in source_vae.state_dict().items()})
        self.vae.eval().requires_grad_(False)
        self.vae.vae_encoding_mode = getattr(source_vae, "vae_encoding_mode", "3dvae")
        for name in ("latents_mean", "latents_std"):
            if self.vae.vae_encoding_mode.lower().strip() == "3dvae":
                value = getattr(source_vae, name, None)
                if value is not None:
                    setattr(self.vae, name, torch.as_tensor(value).detach().to("cpu", torch.float32).clone())
        # Slicing/tiling are model memory settings, not compilation backends.
        for flag, method in (("use_slicing", "enable_slicing"), ("use_tiling", "enable_tiling")):
            if getattr(source_vae, flag, False):
                getattr(self.vae, method)()
        self.stats_tracker = None
        if stats_tracker is not None:
            self.stats_tracker = deepcopy(stats_tracker).to(device="cpu", dtype=torch.float32).eval()

    @torch.no_grad()
    def decode(self, latents):
        with torch.random.fork_rng(devices=[]):
            return decode_latents(
                self.vae, latents.detach().to(device="cpu", dtype=torch.float32),
                dtype=torch.float32, stats_tracker=self.stats_tracker,
            )
