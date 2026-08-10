import sys
from pathlib import Path

POLICY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_DIR))

from verify_checkpoints import load_manifest  # noqa: E402


def test_manifest_is_complete_and_unique():
    entries = load_manifest(POLICY_DIR / "CSU-AI-0.sha256")
    assert len(entries) == 23
    assert len({entry.relative_path for entry in entries}) == 23
    assert any(entry.relative_path == Path("params/ocdbt.process_0/manifest.ocdbt") for entry in entries)
