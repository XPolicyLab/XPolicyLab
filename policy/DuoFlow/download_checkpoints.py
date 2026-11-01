"""下载固定版本的适配器、基础模型和 tokenizer。"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--adapter-repo", help="覆盖已发布的适配器 Hugging Face repo id")
    parser.add_argument("--adapter-revision", help="覆盖已发布的固定权重 commit")
    parser.add_argument("--adapter-only", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download

    root = Path(__file__).resolve().parent
    assets = args.assets_dir or Path(os.environ.get("DUOFLOW_ASSETS", root / "assets"))
    sources = json.loads((root / "ASSET_SOURCES.json").read_text())
    if not args.adapter_repo:
        released_adapter = sources.get("adapter", {})
        args.adapter_repo = released_adapter.get("repo_id")
        args.adapter_revision = args.adapter_revision or released_adapter.get("revision")
    if not args.adapter_only:
        for name in ["model", "paligemma2_tokenizer", "fast_tokenizer"]:
            item = sources[name]
            repo = item.get("repo_id", item.get("repository"))
            target = assets / ("backbone" if name == "model" else name)
            for filename in item["files"]:
                hf_hub_download(repo_id=repo, revision=item["revision"], filename=filename, local_dir=target)
    adapter_dir = root / "checkpoints/duoflow"
    if args.adapter_repo:
        if not args.adapter_revision:
            parser.error("--adapter-repo requires a pinned --adapter-revision")
        hf_hub_download(repo_id=args.adapter_repo, revision=args.adapter_revision,
                        filename="duoflow_adapter.safetensors", local_dir=adapter_dir)
    if not (adapter_dir / "duoflow_adapter.safetensors").is_file():
        raise SystemExit("Place the supplied adapter in checkpoints/duoflow, or pass its published repo and revision")
    print(f"Adapter: {adapter_dir / 'duoflow_adapter.safetensors'}")
    print(f"Assets: {assets.resolve()}")


if __name__ == "__main__":
    main()
