# coding=utf-8
"""Load a local tokenizer while retaining OpenPI's tokenization algorithm."""

from pathlib import Path

from openpi.models.tokenizer import PaligemmaTokenizer
import sentencepiece


class LocalPaligemmaTokenizer(PaligemmaTokenizer):
    def __init__(self, max_len: int, model_path: Path):
        # The pinned upstream constructor only supports its global download cache.
        # Its unchanged tokenize() method consumes these two instance attributes.
        self._max_len = max_len
        self._tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=str(model_path)
        )
