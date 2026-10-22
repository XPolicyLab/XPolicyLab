"""Lossless artifact storage selection."""

RAW_CODEC = "raw"


def normalize_value_codec(value: object) -> str:
    codec = str(value or RAW_CODEC).strip().lower()
    if codec != RAW_CODEC:
        raise ValueError(
            "Artifact caches require value_codec=raw to preserve encoder values."
        )
    return codec
