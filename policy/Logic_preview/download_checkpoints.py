"""Install the four Logic Preview checkpoint files and their nine official Wan assets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parent
BUNDLE = "joint2b_746_s010000"
WAN = "Wan2.1-I2V-14B-480P"
WAN_REV = "6b73f84e66371cdfe870c72acd6826e1d61cf279"
MS_REPO = "garlicba/logic_preview"
BLOCK = 4 * 1024**2


def read_sums(name):
    return {path.lstrip("*"): digest for digest, path in
            (line.split(maxsplit=1) for line in (ROOT / name).read_text().splitlines())}


def matches(path, expected):
    if not path.is_file():
        return False
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            h.update(block)
    return h.hexdigest() == expected


def fetch(url, path, digest, token=None):
    if matches(path, digest):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    cmd = ["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "4",
           "--connect-timeout", "30", "--output", str(partial)]
    header = None
    if token:
        cmd += ["--header", "@-"]
        header = ("Authorization: Bearer " + token + "\n").encode()
    print("Downloading", path.name, flush=True)
    if partial.exists():
        resume = subprocess.run(cmd + ["--continue-at", "-", url], input=header)
        if resume.returncode == 0 and matches(partial, digest):
            partial.replace(path)
            return
    subprocess.run(cmd + [url], input=header, check=True)
    if not matches(partial, digest):
        partial.unlink()
        raise ValueError(f"SHA-256 mismatch for {path.name} from {url}")
    partial.replace(path)


def restore_bundle(parts, volume_names, directory, expected):
    # Each compressed volume is itself a tar with one slice of the inner tar.
    # Let tar autodetect zstd on input. Writing only our four named members
    # avoids accepting arbitrary archive paths or links.
    with tempfile.TemporaryFile(dir=directory.parent) as joined:
        for index, name in enumerate(volume_names, start=1):
            subprocess.run(["tar", "-xOf", str(parts/name), f"{BUNDLE}.tar.part-{index:02}"],
                           stdout=joined, check=True)
        joined.seek(0)
        with tarfile.open(fileobj=joined, mode="r:") as tar:
            if len(tar.getmembers()) != len(expected) or set(tar.getnames()) != set(expected):
                raise ValueError("Unexpected checkpoint tar inventory")
            for item in tar:
                if not item.isfile():
                    raise ValueError(item.name)
                out = directory / Path(item.name).name
                with tar.extractfile(item) as source, out.open("xb") as dest:
                    shutil.copyfileobj(source, dest, BLOCK)
                if not matches(out, expected[item.name]):
                    raise ValueError(f"Invalid checkpoint: {item.name}")


def install(args):
    volume_hash = read_sums("VOLUME_SHA256SUMS")
    files = read_sums("SHA256SUMS")
    weights = args.weights_dir.resolve()
    weights.mkdir(parents=True, exist_ok=True)
    if not os.access(weights, os.W_OK):
        for name, digest in files.items():
            if not matches(weights/name, digest):
                raise ValueError(f"Missing or incorrect file in read-only weight cache: {weights/name}")
        print("Verified read-only weight cache:", weights, flush=True)
        return
    with (weights/".install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for group in (BUNDLE, WAN):
            required = {n: h for n, h in files.items() if n.startswith(group + "/")}
            if all(matches(weights/n, h) for n, h in required.items()):
                print("Verified installed files:", group, flush=True)
                continue
            with tempfile.TemporaryDirectory(dir=weights, prefix=".staging-") as work:
                directory = Path(work) / group
                directory.mkdir()
                if group == BUNDLE:
                    parts = args.parts_dir or (weights/".downloads")
                    def get_part(name):
                        if args.parts_dir:
                            if not matches(parts/name, volume_hash[name]):
                                raise ValueError(f"Incomplete or incorrect volume: {parts/name}")
                        else:
                            query = urlencode({"Revision": args.revision, "FilePath": name})
                            url = f"https://modelscope.cn/api/v1/models/{MS_REPO}/repo?{query}"
                            fetch(url, parts/name, volume_hash[name], os.getenv("MODELSCOPE_API_TOKEN"))
                    names = sorted(volume_hash)
                    with ThreadPoolExecutor(max_workers=5) as pool:
                        list(pool.map(get_part, names))
                    restore_bundle(parts, names, directory, required)
                else:
                    def get_wan(name):
                        rel = Path(name).relative_to(WAN)
                        out = directory/rel
                        out.parent.mkdir(parents=True, exist_ok=True)
                        if args.wan_dir:
                            src = args.wan_dir/rel
                            if not matches(src, required[name]):
                                raise ValueError(f"Incorrect official Wan asset: {src}")
                            shutil.copyfile(src, out)
                        else:
                            url = f"https://huggingface.co/Wan-AI/{WAN}/resolve/{WAN_REV}/{rel.as_posix()}"
                            cached = weights/".downloads"/WAN/rel
                            fetch(url, cached, required[name])
                            os.link(cached, out)
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        list(pool.map(get_wan, required))
                target = weights/group
                if not target.exists():
                    directory.rename(target)
                else:
                    for name in required:
                        (weights/name).parent.mkdir(parents=True, exist_ok=True)
                        (Path(work)/name).replace(weights/name)
                print("Installed and verified", target, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=os.getenv("LOGIC_PREVIEW_MODEL_REVISION", "master"),
                        help="ModelScope revision; payload SHA-256 is checked for every revision")
    parser.add_argument("--weights-dir", type=Path, default=os.getenv("LOGIC_PREVIEW_WEIGHTS_DIR") or ROOT/"weights")
    parser.add_argument("--parts-dir", type=Path, help="Use five previously downloaded volumes")
    parser.add_argument("--wan-dir", type=Path, help="Use a local copy of the official Wan assets")
    install(parser.parse_args())
