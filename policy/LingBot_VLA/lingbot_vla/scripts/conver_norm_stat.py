#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


STAT_KEYS = ("mean", "std", "q01", "q99", "q02", "q98")
SOURCE_KEYS = ("action", "observation.state")
DEFAULT_DIMS = (6, 1, 6, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert legacy norm_stats format into split "
            "arm.position / effector.position format."
        )
    )
    parser.add_argument("input_path", help="Path to the source JSON file.")
    parser.add_argument("output_path", help="Path to write the converted JSON file.")
    parser.add_argument(
        "dims",
        nargs="*",
        type=int,
        metavar="DIM",
        help=(
            "Dimension layout of the original vector. Default: "
            "`6 1 6 1` means [joint(6), gripper(1), joint(6), gripper(1)]."
        ),
    )
    args = parser.parse_args()
    if not args.dims:
        args.dims = list(DEFAULT_DIMS)
    elif len(args.dims) != 4:
        parser.error(
            f"expected 0 or 4 dimension values, got {len(args.dims)}: {args.dims}"
        )
    return args


def validate_dims(dims: list[int]) -> None:
    if any(dim < 0 for dim in dims):
        raise ValueError(f"Dimensions must be non-negative, got: {dims}")
    if sum(dims) <= 0:
        raise ValueError(f"At least one dimension must be positive, got: {dims}")


def split_indices(dims: list[int]) -> tuple[list[int], list[int]]:
    joint_1, gripper_1, joint_2, gripper_2 = dims

    boundaries = [0]
    for dim in dims:
        boundaries.append(boundaries[-1] + dim)

    joint_indices = list(range(boundaries[0], boundaries[1])) + list(
        range(boundaries[2], boundaries[3])
    )
    effector_indices = list(range(boundaries[1], boundaries[2])) + list(
        range(boundaries[3], boundaries[4])
    )
    return joint_indices, effector_indices


def validate_block(name: str, block: dict, expected_dim: int) -> None:
    missing = [key for key in STAT_KEYS if key not in block]
    if missing:
        raise KeyError(f"`{name}` is missing stat keys: {missing}")

    lengths = {key: len(block[key]) for key in STAT_KEYS}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"`{name}` has inconsistent stat lengths: {lengths}")

    (actual_dim,) = unique_lengths
    if actual_dim != expected_dim:
        raise ValueError(
            f"`{name}` dim mismatch: expected {expected_dim} from args, got {actual_dim}"
        )


def slice_block(block: dict, indices: list[int]) -> dict:
    return {key: [block[key][i] for i in indices] for key in STAT_KEYS}


def convert(data: dict, dims: list[int]) -> dict:
    validate_dims(dims)
    expected_dim = sum(dims)
    joint_indices, effector_indices = split_indices(dims)

    norm_stats = data.get("norm_stats")
    if not isinstance(norm_stats, dict):
        raise KeyError("Input JSON must contain a `norm_stats` object.")

    output = {"norm_stats": {}, "count": data.get("count")}
    for source_key in SOURCE_KEYS:
        if source_key not in norm_stats:
            raise KeyError(f"Input JSON missing `norm_stats.{source_key}`")

        block = norm_stats[source_key]
        validate_block(source_key, block, expected_dim)
        output["norm_stats"][f"{source_key}.arm.position"] = slice_block(
            block, joint_indices
        )
        output["norm_stats"][f"{source_key}.effector.position"] = slice_block(
            block, effector_indices
        )

    return output


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    dims = list(args.dims)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    converted = convert(data, dims)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        "Converted successfully:",
        f"input_dim={sum(dims)}",
        f"arm_dim={dims[0] + dims[2]}",
        f"effector_dim={dims[1] + dims[3]}",
        f"output={output_path}",
    )


if __name__ == "__main__":
    main()
