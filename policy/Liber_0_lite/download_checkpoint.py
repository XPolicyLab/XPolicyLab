"""Download and verify the pinned public checkpoint release."""
import argparse
import hashlib
import json
import os
from pathlib import Path

SFT_REVISION = '270e1146e9f278ed1122c1ea614e5600dcf33ec3'
PRETRAIN_REVISION = '697c4792180cd38fb2045fb94d70f7981d4638b2'


def verify(directory, files):
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name in files:
        digest = hashlib.sha256()
        with (directory / name).open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != manifest['sha256'][name]:
            raise ValueError(f'Checkpoint bundle checksum mismatch: {name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="LiberAI/Liber0-Lite-Robodojo")
    parser.add_argument("--revision")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    args = parser.parse_args()
    revision = args.revision or (PRETRAIN_REVISION if args.pretrained else SFT_REVISION)
    for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(key, None)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    from huggingface_hub import snapshot_download
    files = ('config.yaml', 'dataset_stats.json') if args.pretrained else ('model.pt', 'config.yaml', 'dataset_stats.json', 'README.md')
    extra = [f'pretrain/{name}' for name in ('model.pt', 'manifest.json', 'SHA256SUMS', 'README.md')] if args.pretrained else ['SHA256SUMS']
    snapshot_download(repo_id=args.repo_id, revision=revision,
                      local_dir=str(args.destination), allow_patterns=[*files, 'manifest.json', *extra])
    verify(args.destination, files)
    if args.pretrained:
        verify(args.destination / 'pretrain', ['model.pt'])
    print("Verified checkpoint bundle:", args.destination)
    snapshot_download(
        repo_id="HiDream-ai/HiDream-O1-Image",
        revision="0b0901d99f200389e138c61946af1185f5f49a13",
        local_dir=str(args.assets_dir),
        allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"],
    )
    print("Downloaded model assets:", args.assets_dir)


if __name__ == "__main__":
    main()
