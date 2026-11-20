#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_ROBOTWIN_ROOT = "/path/to/RoboTwin-LeRobot-v30"
DEFAULT_OUTPUT_FILE = "outputs/WSA_Large/_repo_id_files/robotwin.txt"
DEFAULT_CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
BASE_REQUIRED_FEATURES = ("observation.state", "action")


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _iter_info_paths(root: Path, *, follow_symlinks: bool) -> Iterable[Path]:
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        if Path(dirpath).name == "meta" and "info.json" in filenames:
            yield Path(dirpath) / "info.json"


def _required_features(
    *,
    require_three_cameras: bool,
    camera_keys: list[str],
    extra_features: list[str],
) -> set[str]:
    required = set(BASE_REQUIRED_FEATURES)
    required.update(feature for feature in extra_features if feature)
    if require_three_cameras:
        required.update(f"observation.images.{key}" for key in camera_keys)
    return required


def discover_robotwin_repos(
    root: Path,
    *,
    required_features: set[str],
    require_media_dir: bool,
    follow_symlinks: bool,
) -> tuple[list[str], dict[str, int]]:
    counters = {
        "info_files": 0,
        "valid": 0,
        "missing_media": 0,
        "missing_features": 0,
        "json_errors": 0,
    }
    repos: list[str] = []

    for info_path in _iter_info_paths(root, follow_symlinks=follow_symlinks):
        counters["info_files"] += 1
        dataset_dir = info_path.parent.parent
        if require_media_dir and not ((dataset_dir / "data").is_dir() or (dataset_dir / "videos").is_dir()):
            counters["missing_media"] += 1
            continue

        try:
            with info_path.open("r", encoding="utf-8") as handle:
                info = json.load(handle)
        except (OSError, json.JSONDecodeError):
            counters["json_errors"] += 1
            continue

        features = set(info.get("features", {}).keys())
        if not required_features.issubset(features):
            counters["missing_features"] += 1
            continue

        repos.append(str(dataset_dir))

    deduped = sorted(set(repos))
    counters["valid"] = len(deduped)
    return deduped, counters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover RoboTwin LeRobot repo directories for WSA_Large multi-repo training."
    )
    parser.add_argument(
        "--robotwin-root",
        default=os.environ.get("ROBOTWIN_ROOT", DEFAULT_ROBOTWIN_ROOT),
        help="Root directory containing many RoboTwin LeRobot repos.",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="Text file to write, one dataset directory per line.",
    )
    parser.add_argument(
        "--require-three-cameras",
        type=_parse_bool,
        default=True,
        help="Require observation.images.<camera-key> features by default.",
    )
    parser.add_argument(
        "--camera-keys",
        nargs="+",
        default=list(DEFAULT_CAMERA_KEYS),
        help="Camera feature suffixes used when --require-three-cameras=true.",
    )
    parser.add_argument(
        "--require-feature",
        action="append",
        default=[],
        help="Additional feature key to require. Can be passed multiple times.",
    )
    parser.add_argument(
        "--require-media-dir",
        type=_parse_bool,
        default=True,
        help="Require each repo to contain either data/ or videos/.",
    )
    parser.add_argument(
        "--follow-symlinks",
        type=_parse_bool,
        default=True,
        help="Follow symlinks while scanning, matching `find -L` in the launch script.",
    )
    parser.add_argument(
        "--print-repos",
        type=_parse_bool,
        default=True,
        help="Print discovered repo paths after writing the file.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=200,
        help="Maximum number of repo paths to print when --print-repos=true.",
    )
    parser.add_argument(
        "--fail-if-empty",
        type=_parse_bool,
        default=True,
        help="Exit with a non-zero status if no valid repos are found.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.robotwin_root).expanduser()
    output_file = Path(args.output_file).expanduser()

    if not root.is_dir():
        raise FileNotFoundError(f"ROBOTWIN root does not exist or is not a directory: {root}")

    required_features = _required_features(
        require_three_cameras=bool(args.require_three_cameras),
        camera_keys=[str(key) for key in args.camera_keys],
        extra_features=[str(feature) for feature in args.require_feature],
    )
    repos, counters = discover_robotwin_repos(
        root,
        required_features=required_features,
        require_media_dir=bool(args.require_media_dir),
        follow_symlinks=bool(args.follow_symlinks),
    )

    if not repos and bool(args.fail_if_empty):
        required = ", ".join(sorted(required_features))
        raise RuntimeError(
            f"No valid RoboTwin LeRobot repos found under {root}. Required features: {required}."
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(repos) + ("\n" if repos else ""), encoding="utf-8")

    print(f"robotwin_root: {root}")
    print(f"output_file: {output_file}")
    print(f"required_features: {sorted(required_features)}")
    print(
        "summary: "
        f"info_files={counters['info_files']}, valid={counters['valid']}, "
        f"missing_media={counters['missing_media']}, "
        f"missing_features={counters['missing_features']}, "
        f"json_errors={counters['json_errors']}"
    )

    if bool(args.print_repos):
        max_print = max(int(args.max_print), 0)
        for repo in repos[:max_print]:
            print(repo)
        if len(repos) > max_print:
            print(f"... {len(repos) - max_print} more repos omitted")


if __name__ == "__main__":
    main()
