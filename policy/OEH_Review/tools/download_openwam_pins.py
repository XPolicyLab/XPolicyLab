"""Download the released simulator checkpoints at fixed HF revisions.

This is asset preparation only. It does not launch a policy or an evaluation.
Next to each checkpoint directory it writes `<name>.observed.json`: every file's
observed sha256 plus the campaign identities (`checkpoint_sha256` = sha256 of the
concatenated hex digests of the sorted safetensors, `normalization_sha256`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from huggingface_hub import hf_hub_download


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observed_manifest(model: dict, local_dir: Path, observed: dict[str, str]) -> dict:
    weights = sorted(name for name in observed if name.endswith(".safetensors"))
    return {"repo": model["repo"], "revision": model["revision"], "local_dir": str(local_dir),
            "files_sha256": observed,
            "checkpoint_sha256": hashlib.sha256("".join(observed[name] for name in weights).encode()).hexdigest(),
            "normalization_sha256": observed.get("normalization_stats.npy")}


def main() -> None:
    pins = json.loads(Path(sys.argv[1]).read_text())
    root = Path(sys.argv[2]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for model in pins:
        local_dir = root / model["repo"].split("/")[-1]
        observed_files: dict[str, str] = {}
        for entry in model["files"]:
            print(f"Downloading {model['repo']}@{model['revision']} {entry['path']}", flush=True)
            path = Path(hf_hub_download(
                repo_id=model["repo"], revision=model["revision"],
                filename=entry["path"], local_dir=local_dir,
            ))
            observed = sha256_file(path)
            if entry["sha256"] is not None and observed != entry["sha256"]:
                raise ValueError(f"checksum mismatch for {path}: {observed}")
            if path.stat().st_size != entry["size"]:
                raise ValueError(f"size mismatch for {path}")
            print(f"Verified {path} sha256={observed}", flush=True)
            observed_files[entry["path"]] = observed
        manifest = local_dir.with_name(local_dir.name + ".observed.json")
        manifest.write_text(json.dumps(observed_manifest(model, local_dir, observed_files), indent=2, sort_keys=True) + "\n")
        print(f"Wrote {manifest}", flush=True)


if __name__ == "__main__":
    main()
