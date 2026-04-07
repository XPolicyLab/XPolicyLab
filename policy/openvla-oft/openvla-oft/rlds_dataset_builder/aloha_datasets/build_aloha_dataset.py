import argparse
import shutil
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from aloha_datasets.base_aloha_dataset_builder import make_aloha_builder_class


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True, help="TFDS dataset name to build.")
    parser.add_argument(
        "--preprocessed_dir",
        required=True,
        help="Path to the preprocessed ALOHA dataset root containing train/ and optional val/.",
    )
    parser.add_argument(
        "--tfds_data_dir",
        required=True,
        help="Output root for TFDS datasets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing dataset with the same name before rebuilding.",
    )
    parser.add_argument("--state_dim", type=int, default=14, help="State dimension.")
    parser.add_argument("--action_dim", type=int, default=14, help="Action dimension.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.tfds_data_dir).expanduser().resolve()
    preprocessed_dir = Path(args.preprocessed_dir).expanduser().resolve()

    if args.overwrite:
        shutil.rmtree(dataset_root / args.dataset_name, ignore_errors=True)

    dataset_root.mkdir(parents=True, exist_ok=True)

    builder_cls = make_aloha_builder_class(
        dataset_name=args.dataset_name,
        preprocessed_dir=str(preprocessed_dir),
        state_dim=args.state_dim,
        action_dim=args.action_dim,
    )
    builder = builder_cls(data_dir=str(dataset_root))
    builder.download_and_prepare()

    print(f"Built TFDS dataset `{args.dataset_name}`")
    print(f"Preprocessed ALOHA dir: {preprocessed_dir}")
    print(f"TFDS output dir: {dataset_root}")


if __name__ == "__main__":
    main()
