def _verify_norm_stats(path: str, expected_sha256: str) -> None:
    """Refuse to run this checkpoint with a different normalization file."""
    import hashlib
    from pathlib import Path
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f'Rule-based 312k EMA norm SHA256 mismatch: {path}: {actual}')
