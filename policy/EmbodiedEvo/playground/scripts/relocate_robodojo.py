"""Finish a local conda clone and source snapshot; never modify the original."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-env", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    target = repo / "vendor/robodojo"
    prefix = repo / ".venv-robodojo"
    # Editable finder metadata points to source trees, not conda's prefix.
    # Rebind the cloned metadata to our byte-for-byte local snapshot. Compiled
    # CuRobo extensions stay in the cloned environment and are not rebuilt.
    changed = []
    for site in (prefix / "lib").glob("python*/site-packages"):
        paths = (
            set(site.glob("__editable__*")) | set(site.glob("*.pth")) | set(site.glob("*.egg-link"))
        )
        for path in sorted(paths):
            if not path.is_file() or path.suffix not in {".py", ".pth", ".egg-link"}:
                continue
            original = path.read_text()
            updated = original.replace(str(args.source_root.resolve()), str(target))
            if updated != original:
                path.write_text(updated)
                changed.append(str(path.relative_to(prefix)))
    subprocess.run(
        [str(prefix / "bin/python"), "utils/update_embodiment_config_path.py"],
        cwd=target,
        check=True,
    )

    def git(*argv):
        return subprocess.check_output(
            ["git", "-C", str(args.source_root), *argv], text=True
        ).strip()

    patch_text = git("diff", "--", "env", "src", "task", "utils")
    (target / ".embodiedevo-source.patch").write_text(patch_text + "\n")
    manifest = {
        "source_checkout": str(args.source_root.resolve()),
        "source_environment": str(args.source_env.resolve()),
        "commit": git("rev-parse", "HEAD"),
        "submodules": git("submodule", "status"),
        "local_source_patch_sha256": hashlib.sha256(patch_text.encode()).hexdigest(),
        "relocated_editable_metadata": changed,
        "generated_asset_paths": "utils/update_embodiment_config_path.py",
        "assets": "Copied from supplied installation; no symlinks to another checkout",
    }
    (target / ".embodiedevo-provisioned.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
