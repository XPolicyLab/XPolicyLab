"""Download reproducible assets without optimizer states or unrelated benchmarks."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent / "checkpoints"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=("robotwin", "training"))
    args = parser.parse_args()
    if args.asset == "robotwin":
        snapshot_download(
            "haofuly/SLA_benchmark_ckpts", revision="dcafbdb9c9af67c94eead433606fc4d4be572166",
            local_dir=ROOT, allow_patterns=["SLA_robotwin_ckpt/params/**", "SLA_robotwin_ckpt/assets/**"],
        )
    else:
        snapshot_download(
            "haofuly/SLA_base", revision="5bbdf3012ab643554aa2eb83ddeae9ace3dce476",
            local_dir=ROOT / "SLA_base", allow_patterns=["params/**", "assets/**"],
        )
        snapshot_download(
            "haofuly/SLA_training_component", revision="4a0917605798f1e8a645153ab8cdfb12aa453354",
            local_dir=ROOT, allow_patterns=["omnivggt/**", "dinov2/dinov2_vitl14_reg4_pretrain.pth"],
        )


if __name__ == "__main__":
    main()
