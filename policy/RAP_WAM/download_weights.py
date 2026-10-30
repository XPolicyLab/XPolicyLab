"""Acquire a verified inference bundle from a local directory or a pinned HF revision."""
import argparse
import json
from pathlib import Path
import shutil
from .asset_storage import configure_workspace, require_work_path, staged_output


BUNDLE_FILES = ("model.pt", "inference_config.json", "task_context.pt", "task_strings.json",
                "model_tensors.json", "SHA256SUMS")
RELEASE_PATH = Path(__file__).with_name("PUBLIC_RELEASE.json")


def select_source(source=None, repo_id=None, revision=None):
    if source:
        if repo_id or revision:
            raise ValueError("--source cannot be combined with --repo-id or --revision")
        return Path(source).expanduser().resolve(), None, None
    if repo_id is None:
        if revision is not None:
            raise ValueError("--revision requires an explicit --repo-id")
        if not RELEASE_PATH.is_file():
            raise ValueError("No public release is configured; provide --source or --repo-id with --revision")
        release = json.loads(RELEASE_PATH.read_text())
        repo_id, revision = release.get("repo_id"), release.get("revision")
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("The release must specify a Hugging Face repo_id")
    if (not isinstance(revision, str) or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)):
        raise ValueError("Provide the immutable 40-character release commit with --revision")
    return None, repo_id, revision


def acquire_bundle(output, work, source, repo_id, revision, validate_bundle):
    output = require_work_path(output, work)
    # Hold one destination lock across network acquisition, copying and validation.
    with staged_output(output, work) as stage:
        if source is None:
            from huggingface_hub import snapshot_download
            cache = require_work_path(work / "cache/hf/hub", work, "Hugging Face cache")
            source = Path(snapshot_download(repo_id, revision=revision, cache_dir=str(cache),
                                           allow_patterns=list(BUNDLE_FILES)))
        validate_bundle(source)
        for name in BUNDLE_FILES:
            # HF snapshots use links into their verified blob cache. Copy their bytes,
            # but never copy arbitrary extra source files, subdirectories or links.
            shutil.copyfile(source / name, stage / name)
        validate_bundle(stage)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source", help="An existing exported inference-bundle directory")
    source.add_argument("--repo-id", help="Published Hugging Face bundle repository")
    parser.add_argument("--revision", help="Exact 40-character Hugging Face commit")
    parser.add_argument("--output", help="Destination inside RAP_WAM_WORK_DIR; defaults to assets/RAP-WAM-s742500")
    args = parser.parse_args()
    work = configure_workspace()
    output = require_work_path(args.output or work / "assets/RAP-WAM-s742500", work)
    source, repo_id, revision = select_source(args.source, args.repo_id, args.revision)
    # Torch/HF imports occur only after cache and temporary paths are constrained.
    from .assets import validate_bundle
    output = acquire_bundle(output, work, source, repo_id, revision, validate_bundle)
    print(output)


if __name__ == "__main__":
    main()
