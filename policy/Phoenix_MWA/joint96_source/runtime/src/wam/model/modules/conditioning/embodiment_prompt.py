"""Per-embodiment learned context tokens for action cross-attention."""
import torch
from torch import nn

from wam.embodiment_registry import EmbodimentRegistry


class EmbodimentPrompt(nn.Module):
    def __init__(self, num_embodiments, num_tokens, context_dim, registry_path=None):
        super().__init__()
        if any(type(v) is not int or v <= 0 for v in (num_embodiments, num_tokens, context_dim)):
            raise ValueError("Prompt dimensions must be positive integers")
        self.num_embodiments = num_embodiments
        self.num_tokens = num_tokens
        self.context_dim = context_dim
        self.registry = EmbodimentRegistry(registry_path) if registry_path else None
        if self.registry and self.registry.num_embodiments != num_embodiments:
            raise ValueError("Model/registry embodiment count mismatch")
        # Do not shift initialization of subsequent existing model modules.
        with torch.random.fork_rng(devices=[]):
            self.embedding = nn.Embedding(num_embodiments, num_tokens * context_dim, device="cpu")
            nn.init.normal_(self.embedding.weight, std=0.02)
        if self.registry and self.registry.reserved_ids:
            with torch.no_grad():
                self.embedding.weight[list(self.registry.reserved_ids)] = 0
            # Derived from the hashed registry; not an extra checkpoint tensor.
            self.register_buffer("_assigned_slots", torch.tensor(
                [i in self.registry.assigned_ids for i in range(num_embodiments)],
                dtype=torch.bool, device="cpu"), persistent=False)

    def forward(self, context, context_mask, embodiment_ids):
        if context.ndim != 3 or context.shape[-1] != self.context_dim:
            raise ValueError("Expected context [B,L,context_dim]")
        if context_mask.dtype != torch.bool or context_mask.shape != context.shape[:2]:
            raise ValueError("context_mask must be bool [B,L], True=visible")
        if not isinstance(embodiment_ids, torch.Tensor) or embodiment_ids.dtype != torch.long:
            raise TypeError("Enabled soft prompts require int64 embodiment_ids")
        if embodiment_ids.shape != (context.shape[0],):
            raise ValueError("Expected embodiment_ids [B]")
        valid = (embodiment_ids >= 0) & (embodiment_ids < self.num_embodiments)
        if hasattr(self, "_assigned_slots"):
            slots = self._assigned_slots.to(embodiment_ids.device)
            valid = valid & slots[embodiment_ids.clamp(0, self.num_embodiments - 1)]
        valid = valid.all()
        if embodiment_ids.device.type == "cpu":
            if not bool(valid):
                raise ValueError("embodiment_ids out of range or reserved")
        else:
            torch._assert_async(valid, "embodiment_ids out of range or reserved")
        ids = embodiment_ids.to(device=self.embedding.weight.device)
        tokens = self.embedding(ids).reshape(context.shape[0], self.num_tokens, self.context_dim).to(context)
        visible = torch.ones((context.shape[0], self.num_tokens), dtype=torch.bool, device=context.device)
        return torch.cat((tokens, context), dim=1), torch.cat((visible, context_mask.to(context.device)), dim=1)

    def metadata(self):
        if self.registry is None:
            raise ValueError("Checkpointing prompts requires a registered embodiment identity")
        return {"schema": "wam.action_soft_prompt.v1", "num_embodiments": self.num_embodiments,
                "num_tokens": self.num_tokens, "context_dim": self.context_dim, **self.registry.metadata()}


def action_prompt_kwargs(expert, sample):
    if getattr(expert, "embodiment_prompt", None) is None:
        return {}
    if "embodiment_id" not in sample:
        raise ValueError("Enabled soft prompts require sample['embodiment_id']")
    return {"embodiment_ids": sample["embodiment_id"]}
