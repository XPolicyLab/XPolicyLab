import sys
from pathlib import Path


POLICY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_DIR))

from download_checkpoints import build_url  # noqa: E402


def test_huggingface_dataset_url_preserves_repo_and_encodes_spaces():
    url = build_url(
        "https://huggingface.co",
        "owner/model",
        "master",
        "checkpoints/CSU-AI-0/path with spaces/file.bin",
    )
    assert url == (
        "https://huggingface.co/datasets/owner/model/resolve/master/"
        "checkpoints/CSU-AI-0/path%20with%20spaces/file.bin?download=true"
    )
