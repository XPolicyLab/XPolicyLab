"""Download an explicitly published, checksum-pinned frozen knowledge bundle.

No release URL is supplied by default: a real distributable bundle must exist
before leaderboard evaluation. Extraction rejects common credential filenames;
the publisher must still review the bundle for secrets before publishing it.
"""

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen

MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_EXTRACTED_BYTES = 1024 * 1024 * 1024
ALLOWED_ROOTS = {"protocol.json", "inputs", "frozen_harness"}


def extract_checked(archive, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Asset destination must be new; existing knowledge is preserved")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        total = 0
        for member in members:
            path = PurePosixPath(member.name)
            if (not path.parts or path.is_absolute() or ".." in path.parts
                    or path.parts[0] not in ALLOWED_ROOTS or "\\" in member.name
                    or not (member.isfile() or member.isdir())):
                raise ValueError(f"Unsafe or unexpected asset member: {member.name}")
            if any(part.lower() in {"auth.json", ".env", "credentials", "token.json", ".codex"}
                   for part in path.parts):
                raise ValueError("Credential-shaped files are forbidden in asset bundles")
            total += member.size
        if total > MAX_EXTRACTED_BYTES:
            raise ValueError("Extracted asset bundle exceeds the size limit")
        required = {"protocol.json", "inputs/test.md", "inputs/primitives/primitives.md"}
        names = {member.name for member in members if member.isfile()}
        if not required.issubset(names) or not any(name.startswith("frozen_harness/") for name in names):
            raise ValueError("Bundle lacks the required frozen source structure")
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                # Reject duplicate entries; do not let later entries overwrite earlier ones.
                with target.open("xb") as output, tar.extractfile(member) as source:
                    shutil.copyfileobj(source, output)
                target.chmod(0o600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Public HTTPS bundle URL, without credentials")
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parsed = urlsplit(args.url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query:
        parser.error("Use a public HTTPS URL without userinfo, signed query strings or credentials")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", args.sha256):
        parser.error("sha256 must be 64 hexadecimal characters")
    if args.output.exists():
        parser.error("output must not exist")
    with tempfile.TemporaryDirectory(prefix="embodiedevo-assets-") as temporary:
        archive = Path(temporary) / "frozen.tar.gz"
        digest = hashlib.sha256()
        with urlopen(args.url, timeout=60) as response, archive.open("xb") as output:
            if urlsplit(response.url).scheme != "https":
                raise ValueError("Asset download redirected outside HTTPS")
            count = 0
            while block := response.read(1024 * 1024):
                count += len(block)
                if count > MAX_ARCHIVE_BYTES:
                    raise ValueError("Asset download exceeds the size limit")
                digest.update(block)
                output.write(block)
        if digest.hexdigest() != args.sha256.lower():
            raise ValueError("Frozen asset SHA256 mismatch; nothing was extracted")
        extract_checked(archive, args.output)
    print("Downloaded frozen knowledge; native policy validation is still required before evaluation.")


if __name__ == "__main__":
    main()
