#!/usr/bin/env python3
"""Download and restore the pinned, public GO-3 Nano Preview evidence data."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request
import zipfile


def sha256(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "data_manifest.json").read_text())
    url = (
        f"https://huggingface.co/datasets/{manifest['dataset_repo']}/resolve/"
        f"{manifest['revision']}/{manifest['archive']}"
    )
    expected = {item["path"]: item for item in manifest["files"]}
    with tempfile.TemporaryDirectory(prefix="go3-evidence-") as temporary:
        archive = Path(temporary) / manifest["archive"]
        print(f"Downloading evidence data from {manifest['dataset_repo']}...", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "GO-3-Nano-Preview"})
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        with archive.open("rb") as stream:
            if sha256(stream) != manifest["archive_sha256"]:
                raise RuntimeError("Downloaded archive SHA-256 does not match the manifest")

        with zipfile.ZipFile(archive) as bundle:
            if len(bundle.infolist()) != len(expected) or set(bundle.namelist()) != set(expected):
                raise RuntimeError("Archive file list does not match the manifest")
            missing = []
            for name, item in expected.items():
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError(f"Invalid archive path: {name}")
                target = root.joinpath(*relative.parts)
                if not target.resolve().is_relative_to(root):
                    raise RuntimeError(f"Archive path leaves evidence directory: {name}")
                with bundle.open(name) as stream:
                    if sha256(stream) != item["sha256"]:
                        raise RuntimeError(f"File SHA-256 mismatch: {name}")
                if target.exists():
                    with target.open("rb") as stream:
                        if sha256(stream) != item["sha256"]:
                            raise RuntimeError(f"Existing file differs; refusing to overwrite: {target}")
                else:
                    missing.append((name, target))

            for name, target in missing:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    print(f"Verified {len(expected)} evidence files; restored {len(missing)} to {root}.")


if __name__ == "__main__":
    main()
