"""Download VPP2 evaluation weights or the his10k training initializer."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("eval", "train", "all"), default="eval")
    parser.add_argument("--source", choices=("huggingface", "modelscope"), default="huggingface")
    args = parser.parse_args()
    patterns = ["config.json", "checkpoints/Wan2.1-I2V-14B-480P/*"]
    if args.stage in ("eval", "all"):
        patterns.append("checkpoints/joint2b_s100000/*")
    if args.stage in ("train", "all"):
        patterns.append("checkpoints/initialization/robodojo_his10k.pt")
    destination = str(Path(__file__).resolve().parent)
    if args.source == "huggingface":
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id="Haodong082399/VPP2", local_dir=destination,
                          allow_patterns=patterns)
    else:
        from modelscope import snapshot_download

        snapshot_download(model_id="haodong123/VPP2", local_dir=destination,
                          allow_patterns=patterns)
    print(f"Downloaded VPP2 {args.stage} assets from {args.source}.")


if __name__ == "__main__":
    main()
