from __future__ import annotations

import hashlib
from pathlib import Path

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
TEXT_EMBED_CACHE_ENCODER_ID = "wan22ti2v5b"


def build_fastwam_prompt(task: str) -> str:
    return DEFAULT_PROMPT.format(task=task)


def build_text_embedding_cache_filename(prompt: str, context_len: int) -> str:
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"{hashed}.t5_len{int(context_len)}.{TEXT_EMBED_CACHE_ENCODER_ID}.pt"


def build_text_embedding_cache_path(cache_dir: str | Path, prompt: str, context_len: int) -> Path:
    return Path(cache_dir) / build_text_embedding_cache_filename(prompt, context_len)
