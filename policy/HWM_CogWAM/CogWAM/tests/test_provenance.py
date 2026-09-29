"""UPSTREAM_SOURCES.json is the audit trail for the port; keep it honest.

Every file carried over from the upstream research tree records the SHA-256 of
the revision it came from and a prose note covering each semantic deviation.
That only stays useful if the file cannot silently rot -- a `target_path`
pointing at something that no longer exists, or two entries claiming the same
target, means the record has drifted from the tree it describes.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "UPSTREAM_SOURCES.json"

SHA256_LENGTH = 64


def _entries() -> list[dict]:
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert doc["schema_version"] == 1
    return doc["entries"]


def test_every_ported_target_exists() -> None:
    missing = [
        entry["target_path"]
        for entry in _entries()
        if entry.get("target_path") and not (REPO_ROOT / entry["target_path"]).exists()
    ]
    assert not missing, f"provenance points at files that are gone: {missing}"


def test_source_target_pairs_are_unique() -> None:
    """Several targets merge two upstream files, so uniqueness is on the pair."""
    pairs = [(entry.get("target_path"), entry["source_path"]) for entry in _entries()]
    duplicates = sorted({pair for pair in pairs if pairs.count(pair) > 1})
    assert not duplicates, f"the same source is recorded twice for one target: {duplicates}"


def test_every_entry_is_complete() -> None:
    """A null target means 'deliberately not ported' and still needs a reason."""
    for entry in _entries():
        source = entry.get("source_path")
        assert source, f"entry without source_path: {entry}"
        digest = entry.get("source_sha256", "")
        assert len(digest) == SHA256_LENGTH, f"{source}: source_sha256 is not a sha256"
        adaptation = entry.get("adaptation", "")
        assert len(adaptation) > 20, f"{source}: adaptation note is too thin to audit"
        if entry.get("target_path") is None:
            assert "NOT PORTED" in adaptation, f"{source}: null target must explain why"
