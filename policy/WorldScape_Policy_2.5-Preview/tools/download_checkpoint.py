"""Download and validate the public RoboDojo inference checkpoint."""
from pathlib import Path
import argparse
import os
import sys
import tempfile

POLICY_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPO = 'manifoldai-research/WorldScape-Policy-2-5'
DEFAULT_SUBFOLDER = 'wsp_2_5_preview_posttrain_robodojo'
MODEL_SHA256 = 'sha256:c713022e97279b8acaf113d7f1fa5d9889dc31884287f3d537391be4dd08cfa3'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default=os.environ.get('WSP_HF_REPO', DEFAULT_REPO))
    parser.add_argument('--revision', default='main')
    parser.add_argument('--subfolder', default=DEFAULT_SUBFOLDER,
                        help='Checkpoint folder inside the HF repo (pass an empty string for the repo root)')
    parser.add_argument('--output', type=Path, default=POLICY_DIR / 'checkpoints/wsp_2_5_preview_posttrain_robodojo')
    args = parser.parse_args()
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        parser.error(f'Destination already exists: {output}; use a new --output directory')
    folder = Path(args.subfolder)
    if folder.is_absolute() or '..' in folder.parts or '\\' in args.subfolder:
        parser.error('--subfolder must be a relative repository path')
    from huggingface_hub import snapshot_download
    sys.path.insert(0, str(POLICY_DIR / 'worldscape-policy/src'))
    from worldscape_policy.checkpoint.loader import validate_native_checkpoint_artifacts
    import json

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.download-', dir=output.parent) as tmp:
        root = Path(tmp) / 'repository'
        snapshot_download(repo_id=args.repo, revision=args.revision, local_dir=root,
                          allow_patterns=[args.subfolder.rstrip('/') + '/**'] if args.subfolder else None)
        checkpoint = root / folder
        validate_native_checkpoint_artifacts(checkpoint)
        manifest = json.loads((checkpoint / 'checkpoint_manifest.json').read_text())
        if manifest['model_sha256'] != MODEL_SHA256:
            raise ValueError('Downloaded weights differ from the validated RoboDojo release')
        checkpoint.rename(output)
    print(f'Validated checkpoint: {output}')


if __name__ == '__main__':
    main()
