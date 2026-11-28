"""Prompt tokenizer used by Focus-VLWA."""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

import numpy as np
import sentencepiece

_STATE_BINS = np.linspace(-1, 1, 257)[:-1]
_TOKENIZER_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"


def resolve_tokenizer_path(path: str | Path | None = None) -> Path:
    """Resolve or download the public tokenizer model."""
    configured = path or os.environ.get("FOCUS_VLWA_TOKENIZER_PATH")
    target = Path(configured).expanduser() if configured else Path.home() / ".cache/focus-vlwa/tokenizer.model"
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    urllib.request.urlretrieve(_TOKENIZER_URL, temporary)
    temporary.replace(target)
    return target


class FocusVLWATokenizer:
    """Encode a task and quantile-normalized robot state for the model prefix."""

    def __init__(self, max_len: int = 200, tokenizer_path: str | Path | None = None):
        self.max_len = max_len
        model = resolve_tokenizer_path(tokenizer_path).read_bytes()
        self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=model)

    def tokenize(self, prompt: str, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        state_bins = np.digitize(np.asarray(state), bins=_STATE_BINS) - 1
        state_text = " ".join(map(str, state_bins.reshape(-1)))
        full_prompt = f"Task: {cleaned}, State: {state_text};\nAction: "
        tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        if len(tokens) > self.max_len:
            logging.warning("Token length %s exceeds %s and will be truncated", len(tokens), self.max_len)
        tokens = tokens[: self.max_len]
        mask = [True] * len(tokens)
        padding = self.max_len - len(tokens)
        if padding:
            tokens.extend([0] * padding)
            mask.extend([False] * padding)
        return np.asarray(tokens, dtype=np.int64), np.asarray(mask, dtype=np.bool_)
