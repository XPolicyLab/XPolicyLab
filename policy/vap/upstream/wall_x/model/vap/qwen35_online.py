"""Optional frozen online text provider. No Transformers imports in T5 mode.

Exact B300 cache recipe: plain text + EOS, causal final hidden states, BF16
[B,L,Dq], right-zero-padding AFTER encoding. No generation, images, cache or
cross-example state. The trainable projector belongs to VAP core, not here.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import torch

from .text_conditioning import qwen35_encode_ids, qwen35_recipe_for_truncation

RECIPE = "plain_text_eos_last_hidden_v1"


def model_identity(path):
    """Hash local model/tokenizer contents once on the configuration-writing rank."""
    root = Path(path)
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") not in {"qwen3_5", "qwen3_5_text"}:
        raise ValueError("Expected a dense Qwen3.5 checkpoint")
    files = sorted(p for p in root.iterdir() if p.is_file() and
                   p.suffix in {".json", ".safetensors", ".model", ".txt", ".jinja"})
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("No Qwen safetensors weights")
    index = root / "model.safetensors.index.json"
    if index.exists():
        for shard in set(json.loads(index.read_text())["weight_map"].values()):
            if not (root / shard).is_file():
                raise FileNotFoundError(root / shard)
    hashes = {}
    for f in files:
        h = hashlib.sha256()
        with f.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                h.update(block)
        hashes[f.name] = h.hexdigest()
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return identity, int(config.get("text_config", config)["hidden_size"])


def configure_qwen_online(config, path, *, seq_len=512, micro_batch=1):
    from .text_conditioning import text_init_missing_keys, validate_text_config
    from types import SimpleNamespace
    identity, hidden_dim = model_identity(path)
    model = config.setdefault("model", {})
    model.update(text_encoder_backend="qwen35_online", qwen35_model_path=str(path),
                 qwen35_model_id=identity, qwen35_text_recipe=RECIPE,
                 qwen35_encode_micro_batch=int(micro_batch),
                 text_feature_hidden_dim=hidden_dim, text_feature_seq_len=int(seq_len),
                 text_projector_output_dim=4096, text_feature_cache_path=None,
                 text_feature_cache_id=None, text_feature_cache_online_fallback=False,
                 use_qwen25=False, use_qwen35=False, qwen25_cascade_t5=False,
                 qwen35_cascade_t5=False)
    validate_text_config(SimpleNamespace(**model))
    checkpoint = config.setdefault("checkpoint", {})
    allowed = text_init_missing_keys(config)
    checkpoint["model_init_allowed_missing_keys"] = list(dict.fromkeys(
        [*checkpoint.get("model_init_allowed_missing_keys", []), *allowed]))
    logging.info("Online frozen Qwen3.5 identity=%s hidden=%d seq_len=%d", identity, hidden_dim, seq_len)


class Qwen35OnlinePrompter:
    """Non-nn.Module owner: Qwen is excluded from DMuon/FSDP/state_dict.

    Lazily loads on the consuming rank's device, after VAP initialization.
    No inference_mode tensors escape: projector backward needs its input saved.
    Equal-length grouping avoids padded tokens inside Qwen's recurrent layers.
    """
    manages_placement = True
    skip_text_encoder = True

    def __init__(self, config):
        self.config = config
        self.text_encoder = None
        self.tokenizer = None
        self.device = None

    def _load(self, device):
        if self.text_encoder is not None:
            if self.device != torch.device(device):
                raise ValueError("Online Qwen must remain on its rank-local device")
            return
        import transformers
        cfg = transformers.AutoConfig.from_pretrained(self.config.qwen35_model_path,
                                                      local_files_only=True)
        names = {"qwen3_5": "Qwen3_5Model", "qwen3_5_text": "Qwen3_5TextModel"}
        if cfg.model_type not in names:
            raise ValueError("Expected dense Qwen3.5")
        hidden = getattr(cfg, "text_config", cfg).hidden_size
        if hidden != self.config.text_feature_hidden_dim:
            raise ValueError("Online Qwen hidden size differs from config")
        model_class = getattr(transformers, names[cfg.model_type])
        self.device = torch.device(device)
        model, loading = model_class.from_pretrained(
            self.config.qwen35_model_path, local_files_only=True,
            dtype=torch.bfloat16, device_map={"": str(self.device)},
            attn_implementation="sdpa", output_loading_info=True,
        )
        unexpected = [k for k in loading.get("unexpected_keys", []) if k != "lm_head.weight"]
        if unexpected or loading.get("missing_keys") or loading.get("mismatched_keys"):
            raise ValueError(f"Incomplete Qwen base-model load: {loading}")
        self.text_encoder = model.eval().requires_grad_(False)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.qwen35_model_path, local_files_only=True)
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Qwen tokenizer must define EOS")
        logging.info("Loaded frozen online Qwen on %s; no T5/cache/LM head computation", device)

    @torch.no_grad()
    def encode_prompt(self, prompt, positive=True, device="cpu", **kwargs):
        del positive, kwargs
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompts or any(not isinstance(p, str) for p in prompts):
            raise ValueError("Expected a nonempty batch of exact prompt strings")
        self._load(device)
        self.text_encoder.eval()
        truncation = getattr(self.config, "qwen35_truncation", "none")
        if self.config.qwen35_text_recipe != qwen35_recipe_for_truncation(truncation):
            raise ValueError("Online Qwen recipe and truncation disagree")
        groups = {}
        for i, p in enumerate(prompts):
            # Same helper as training-cache generation: retain FIRST L-1
            # tokens and append EOS; never truncate the hidden states later.
            ids = qwen35_encode_ids(
                self.tokenizer, p, seq_len=self.config.text_feature_seq_len,
                eos_token_id=self.tokenizer.eos_token_id, truncation=truncation,
            )
            groups.setdefault(len(ids), []).append((i, ids))
        output = torch.zeros(len(prompts), self.config.text_feature_seq_len,
                             self.config.text_feature_hidden_dim, device=device,
                             dtype=torch.bfloat16)
        for length, rows in groups.items():
            for start in range(0, len(rows), self.config.qwen35_encode_micro_batch):
                block = rows[start:start + self.config.qwen35_encode_micro_batch]
                ids = torch.tensor([r[1] for r in block], device=device, dtype=torch.long)
                result = self.text_encoder(input_ids=ids, attention_mask=torch.ones_like(ids),
                                           use_cache=False, return_dict=True)
                hidden = result.last_hidden_state
                if hidden.shape != (len(block), length, self.config.text_feature_hidden_dim):
                    raise ValueError(f"Unexpected Qwen context shape {hidden.shape}")
                if not torch.isfinite(hidden).all():
                    raise ValueError("Nonfinite online Qwen context")
                for j, (i, _) in enumerate(block):
                    output[i, :length] = hidden[j].to(torch.bfloat16)
        return output
