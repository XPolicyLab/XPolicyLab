"""Download the public inference checkpoint."""
from pathlib import Path
import argparse
import os
import tempfile

POLICY_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPO = 'manifoldai-research/WorldScape-Policy-2-5'
DEFAULT_SUBFOLDER = 'wsp_2_5_preview_posttrain_robodojo'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default=os.environ.get('WSP_HF_REPO', DEFAULT_REPO))
    parser.add_argument('--revision', default='main')
    parser.add_argument('--subfolder', default=DEFAULT_SUBFOLDER)
    parser.add_argument('--output', type=Path, default=POLICY_DIR / 'checkpoints' / DEFAULT_SUBFOLDER)
    args = parser.parse_args()
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        parser.error(f'Destination already exists: {output}; use a new --output directory')
    folder = Path(args.subfolder)
    if folder.is_absolute() or '..' in folder.parts or '\\' in args.subfolder:
        parser.error('--subfolder must be a relative repository path')
    from huggingface_hub import snapshot_download
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.download-', dir=output.parent) as tmp:
        root = Path(tmp) / 'repository'
        snapshot_download(repo_id=args.repo, revision=args.revision, local_dir=root,
                          allow_patterns=[args.subfolder.rstrip('/') + '/**'] if args.subfolder else None)
        (root / folder).rename(output)
    print(f'Downloaded checkpoint: {output}')


if __name__ == '__main__':
    main()
