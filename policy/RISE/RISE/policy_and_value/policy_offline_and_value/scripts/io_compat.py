# io_compat.py
# Unified loader/saver for PyTorch + safetensors with:
# - shape-safe partial loading (strict=False)
# - tied-weights recovery via safetensors metadata (e.g., embed_tokens <- lm_head)
# - optional rename rule for key migrations
from __future__ import annotations
import torch
from typing import Callable, Dict, Optional
from safetensors.torch import load_file, save_file
from safetensors import safe_open


def _apply_metadata_ties_(model: torch.nn.Module, ckpt_path: str) -> None:
    """
    Read safetensors metadata and copy tensors from source->target for tied weights.
    Example metadata:
      {'...embed_tokens.weight': '...lm_head.weight'}
    """
    try:
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            if not meta:
                return
            sd = model.state_dict()
            for tgt_key, src_key in meta.items():
                if tgt_key in sd and src_key in f.keys():
                    src = f.get_tensor(src_key)
                    if sd[tgt_key].shape == src.shape:
                        with torch.no_grad():
                            sd[tgt_key].copy_(src)
    except Exception as e:
        print(f"[compat] metadata ties skipped due to: {e}")


def load_model_compat(
    model: torch.nn.Module,
    weight_path: str,
    rename_fn: Optional[Callable[[str], str]] = None,
    extra_ties_fallback: Optional[Dict[str, str]] = None,
):
    """
    Shape-safe, metadata-aware loader.

    Steps:
    1) load_file -> (optional) key rename -> shape-equal filtering
    2) model.load_state_dict(..., strict=False)
    3) metadata-ties recovery (e.g., embed_tokens <- lm_head)
    4) fallback ties (explicit dict) if metadata not present

    Returns: IncompatibleKeys from torch.load_state_dict (for initial filtered load).
    """
    raw = load_file(weight_path)  # dict[str, Tensor]
    if rename_fn:
        raw = {rename_fn(k): v for k, v in raw.items()}

    sd = model.state_dict()
    filtered = {k: v for k, v in raw.items() if k in sd and sd[k].shape == v.shape}
    msg = model.load_state_dict(filtered, strict=False)

    # Step 3: metadata-based ties
    _apply_metadata_ties_(model, weight_path)

    # Step 4: explicit fallback ties (project specific)
    if extra_ties_fallback:
        with safe_open(weight_path, framework="pt", device="cpu") as f:
            for tgt_key, src_key in extra_ties_fallback.items():
                if tgt_key in sd and src_key in raw and sd[tgt_key].shape == raw[src_key].shape:
                    with torch.no_grad():
                        sd[tgt_key].copy_(raw[src_key])
                    print(f"[compat/fallback] {tgt_key} <- {src_key}")

    # Report (post-fix missing)
    still_missing = [k for k in model.load_state_dict({}, strict=False).missing_keys]
    if still_missing:
        print(f"[compat] load done. remaining missing={len(still_missing)}")
        for k in still_missing[:12]:
            print("  missing:", k)
        if len(still_missing) > 12:
            print(f"  ... (+{len(still_missing)-12} more)")
    else:
        print("[compat] load done. no missing keys.")
    if msg.unexpected_keys:
        print(f"[compat] unexpected={len(msg.unexpected_keys)} (ignored by strict=False)")
        for k in msg.unexpected_keys[:12]:
            print("  unexpected:", k)
        if len(msg.unexpected_keys) > 12:
            print(f"  ... (+{len(msg.unexpected_keys)-12} more)")
    return msg


def save_model_compat(
    model: torch.nn.Module,
    save_path: str,
    ties: Optional[Dict[str, str]] = None,
    write_full_state: bool = False,
):
    """
    Save with optional tied-weights metadata.
    - write_full_state=True: write full state_dict (max compatibility, larger file; ignores `ties`)
    - write_full_state=False & ties=None: write full state_dict, no metadata (default)
    - write_full_state=False & ties=dict: remove tgt keys and write metadata mapping tgt->src (lighter file)

    Example ties:
      {
        "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight":
        "paligemma_with_expert.paligemma.lm_head.weight"
      }
    """
    sd = model.state_dict()
    if write_full_state:
        save_file(sd, save_path, metadata=None)
        print("[compat] saved full state (no metadata).")
        return

    meta = None
    if ties:
        sd = sd.copy()
        for tgt, src in ties.items():
            if tgt in sd and src in sd:
                # drop target; loader will reconstruct from src using metadata
                del sd[tgt]
        meta = ties

    save_file(sd, save_path, metadata=meta or None)
    if meta:
        print(f"[compat] saved with metadata ties: {len(meta)}")
    else:
        print("[compat] saved full state (no ties).")
