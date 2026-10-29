"""Decide from the instruction whether an episode needs the long-memory AR context.

The policy's own Cosmos3 UND tower answers a fixed yes/no question in the chat template; only the "Yes"/"No"
rows of the released ``lm_head`` are loaded. margin = logit(Yes) - logit(No); margin > 0 selects memory.

"""
import json
import os

import torch
from safetensors import safe_open

from galaxea_va.models.cosmos3.loader import COSMOS3_NANO_ROOT

PROMPT = (
    'Robot instruction: "{instruction}"\n'
    'Does this instruction use the word "swap" or "remember", or ask the robot to repeat an action '
    "a required number of times? Answer Yes or No."
)


class MemoryRouter:
    def __init__(self, ar_model):
        self.backbone = ar_model.backbone
        self.tokenizer = ar_model.tokenizer
        weight = self.backbone.embed_tokens.weight
        self.device, self.dtype = weight.device, weight.dtype
        answer_ids = [self.tokenizer.encode(word, add_special_tokens=False) for word in ("Yes", "No")]
        if any(len(ids) != 1 for ids in answer_ids):
            raise ValueError(f"Yes/No must each be one token, got {answer_ids}")
        transformer_dir = os.path.join(COSMOS3_NANO_ROOT, "transformer")
        with open(os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors.index.json")) as f:
            shard = json.load(f)["weight_map"]["lm_head.weight"]
        with safe_open(os.path.join(transformer_dir, shard), "pt", device=str(self.device)) as f:
            rows = f.get_slice("lm_head.weight")
            # [2, hidden]: the Yes and No rows of the released language-model head.
            self.answer_head = torch.cat([rows[i:i + 1] for (i,) in answer_ids]).to(self.dtype)

    @torch.no_grad()
    def margin(self, instruction: str) -> float:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(instruction=instruction)}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = torch.tensor(self.tokenizer(text, add_special_tokens=False)["input_ids"], device=self.device)
        length = ids.shape[0]
        hidden = self.backbone.embed_tokens(ids)[None]  # [1, L, hidden]
        # Text only: the three mRoPE axes share one position, which reduces to plain 1D RoPE.
        cos, sin = self.backbone.rotary_emb(torch.arange(length, device=self.device).expand(3, length),
                                            self.device, self.dtype)
        cos, sin = cos[0], sin[0]  # [L, head_dim]
        causal = torch.ones(length, length, dtype=torch.bool, device=self.device).tril()[None]
        for layer in self.backbone.layers:
            attn = layer.self_attn
            q, k, v = attn._project(layer.input_layernorm(hidden), attn.to_q, attn.to_k, attn.to_v,
                                    attn.norm_q, attn.norm_k, cos, sin)
            hidden = hidden + attn.to_out(attn._attend(q, k, v, causal))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        yes, no = (self.backbone.norm(hidden[0, -1]) @ self.answer_head.T).float()
        return float(yes - no)
