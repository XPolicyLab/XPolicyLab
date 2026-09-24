"""Tokenizer used by the WorldScape-owned Wan T5 encoder.

Adapted from
``groot.vla.model.dreamzero.transform.dreamzero_cotrain.HuggingfaceTokenizer``.
The tokenization and cleaning behavior is intentionally unchanged; optional
dependencies are imported only when an encoder is constructed.
"""

from __future__ import annotations

import html
import os
import re
from typing import Any


def basic_clean(text: str) -> str:
    try:
        import ftfy
    except ImportError as exc:  # pragma: no cover - depends on installation extras
        raise ImportError(
            "T5 text cleaning requires the optional dependency 'ftfy'."
        ) from exc
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class HuggingfaceTokenizer:
    def __init__(
        self,
        name: str,
        seq_len: int | None = None,
        clean: str | None = None,
        **kwargs: Any,
    ) -> None:
        if clean not in (None, "whitespace"):
            raise ValueError("clean must be None or 'whitespace'")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - installation dependent
            raise ImportError(
                "T5 tokenization requires the optional dependency 'transformers'. "
                "Install transformers to construct T5InstructionEncoder."
            ) from exc

        self.name = name
        self.seq_len = seq_len
        self.clean = clean
        load_kwargs = dict(kwargs)
        if os.path.isdir(name):
            load_kwargs.setdefault("local_files_only", True)
        self.tokenizer = AutoTokenizer.from_pretrained(name, **load_kwargs)
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence: str | list[str], **kwargs: Any):
        return_mask = kwargs.pop("return_mask", False)
        call_kwargs: dict[str, Any] = {"return_tensors": "pt"}
        if self.seq_len is not None:
            call_kwargs.update(
                {
                    "padding": "max_length",
                    "truncation": True,
                    "max_length": self.seq_len,
                }
            )
        call_kwargs.update(kwargs)
        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [self._clean(value) for value in sequence]
        ids = self.tokenizer(sequence, **call_kwargs)
        if return_mask:
            return ids.input_ids, ids.attention_mask
        return ids.input_ids

    def _clean(self, text: str) -> str:
        if self.clean == "whitespace":
            text = whitespace_clean(basic_clean(text))
        return text


__all__ = ["HuggingfaceTokenizer", "basic_clean", "whitespace_clean"]
