"""Download published policy/backbones; no private resources are required for eval."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest():
    return json.loads((HERE / "weights_manifest.json").read_text())


def verify_local(root, full_hash=True, include_policy=True):
    root = Path(root)
    for x in manifest()["files"]:
        if not include_policy and x["path"] == "policy.pt":
            continue
        p = root / x["path"]
        if not p.is_file() or p.stat().st_size != x["size"]:
            raise ValueError(
                f"Missing or incorrect weight: {p}; run download_checkpoint.sh"
            )
        if full_hash and sha256(p) != x["sha256"]:
            raise ValueError(f"Checksum mismatch: {p}")
    tok = root / manifest()["tokenizer"]["path"] / "google/umt5-xxl"
    if (
        not (tok / "tokenizer_config.json").is_file()
        or not (tok / "spiece.model").is_file()
    ):
        raise FileNotFoundError(f"Incomplete tokenizer: {tok}")


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("root", type=Path)
    a.add_argument("--verify-only", action="store_true")
    a.add_argument(
        "--pretrain-only",
        action="store_true",
        help="Download only the published Joint96 256013-step pretraining checkpoint",
    )
    a.add_argument(
        "--training-backbones",
        action="store_true",
        help="Also download original Wan DiT shards required by the training constructor",
    )
    a.add_argument(
        "--init-repo",
        help="Optional Hugging Face repository for training initialization",
    )
    a.add_argument("--init-file", default="policy.pt")
    a.add_argument("--init-revision", default="main")
    args = a.parse_args()
    root = args.root.expanduser().resolve()
    if args.pretrain_only and (args.training_backbones or args.init_repo):
        a.error("--pretrain-only cannot be combined with --training-backbones or --init-repo")
    entries = [manifest()["pretraining"]] if args.pretrain_only else manifest()["files"]
    if args.verify_only:
        if args.pretrain_only:
            x = entries[0]
            p = root / x["path"]
            if not p.is_file() or p.stat().st_size != x["size"]:
                raise ValueError(f"Missing or incorrect pretraining checkpoint: {p}")
            if sha256(p) != x["sha256"]:
                raise ValueError(f"Checksum mismatch: {p}")
        else:
            verify_local(root)
        print("All weights verified")
        return
    root.mkdir(parents=True, exist_ok=True)
    for x in entries:
        dest = root / x["path"]
        if dest.exists():
            if dest.stat().st_size != x["size"] or sha256(dest) != x["sha256"]:
                raise ValueError(f"Refusing to overwrite mismatched {dest}")
            continue
        print("Downloading", x["repo_id"], x["filename"], flush=True)
        if x["provider"] == "hf":
            from huggingface_hub import hf_hub_download

            cached = hf_hub_download(
                repo_id=x["repo_id"], filename=x["filename"], revision=x["revision"]
            )
        else:
            from modelscope.hub.file_download import model_file_download

            cached = model_file_download(
                model_id=x["repo_id"], file_path=x["filename"], revision=x["revision"]
            )
        if Path(cached).stat().st_size != x["size"] or sha256(cached) != x["sha256"]:
            raise ValueError("Downloaded file differs from release manifest")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".partial")
        shutil.copyfile(cached, tmp)
        os.replace(tmp, dest)
    if args.pretrain_only:
        print("Pretraining checkpoint verified:", root / entries[0]["path"])
        return
    from huggingface_hub import hf_hub_download, snapshot_download

    t = manifest()["tokenizer"]
    snapshot_download(
        repo_id=t["repo_id"],
        revision=t["revision"],
        allow_patterns=[t["pattern"]],
        local_dir=root / t["path"],
    )
    if args.training_backbones:
        snapshot_download(
            repo_id="Wan-AI/Wan2.2-TI2V-5B",
            revision="921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
            allow_patterns=[
                "diffusion_pytorch_model*.safetensors",
                "diffusion_pytorch_model*.json",
                "config.json",
            ],
            local_dir=root / "Wan-AI/Wan2.2-TI2V-5B",
        )
    if args.init_repo:
        dest = root / "initialization.pt"
        if dest.exists():
            raise FileExistsError(f"Initialization already exists: {dest}")
        src = hf_hub_download(
            repo_id=args.init_repo, filename=args.init_file, revision=args.init_revision
        )
        shutil.copyfile(src, dest)
        (root / "initialization.json").write_text(
            json.dumps(
                dict(
                    repo_id=args.init_repo,
                    filename=args.init_file,
                    revision=args.init_revision,
                    sha256=sha256(dest),
                ),
                indent=2,
            )
        )
    verify_local(root)
    print("All policy and backbone weights verified")


if __name__ == "__main__":
    main()
