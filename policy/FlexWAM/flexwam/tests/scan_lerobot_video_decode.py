from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path


def _load_video_keys(dataset_root: Path) -> list[str]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info.get("features", {})
    video_keys = [key for key, value in features.items() if value.get("dtype") == "video"]
    if not video_keys:
        raise ValueError(f"No video features found in {dataset_root / 'meta/info.json'}")
    return video_keys


def _load_episodes(dataset_root: Path) -> list[dict]:
    episodes = []
    with (dataset_root / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    if not episodes:
        raise ValueError(f"No episodes found in {dataset_root / 'meta/episodes.jsonl'}")
    return episodes


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return int(episode_index) // int(chunks_size)


def _video_path(dataset_root: Path, video_key: str, episode_index: int, chunks_size: int) -> Path:
    chunk = _episode_chunk(episode_index, chunks_size)
    return dataset_root / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"


def _scan_torchcodec(video_path: Path, num_frames: int, batch_size: int) -> list[str]:
    from torchcodec.decoders import VideoDecoder

    errors: list[str] = []
    decoder = VideoDecoder(str(video_path), device="cpu", seek_mode="approximate")
    for start in range(0, num_frames, batch_size):
        end = min(num_frames, start + batch_size)
        indices = list(range(start, end))
        try:
            decoder.get_frames_at(indices=indices)
        except Exception as exc:
            errors.append(f"frames[{start}:{end}] {type(exc).__name__}: {exc}")
    return errors


def _scan_training_windows(
    dataset_root: Path,
    video_path: Path,
    num_frames: int,
    fps: float,
    window_size: int,
    stride: int,
    backend: str,
    max_errors: int,
) -> list[str]:
    sys.path.insert(0, str(dataset_root.parents[4] / "World-Model" / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from flexwam.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames

    errors: list[str] = []
    last_start = max(0, num_frames - window_size)
    for start in range(0, last_start + 1, stride):
        timestamps = [(start + i) / fps for i in range(window_size)]
        try:
            decode_video_frames(video_path, timestamps, tolerance_s=1e-4, backend=backend)
        except Exception as exc:
            errors.append(f"window_start={start} backend={backend} {type(exc).__name__}: {exc}")
            if len(errors) >= max_errors:
                break
    return errors


def _scan_ffmpeg(video_path: Path) -> list[str]:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video_path), "-f", "null", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0 or proc.stderr.strip():
        return [f"ffmpeg returncode={proc.returncode}: {proc.stderr.strip()}"]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan LeRobot videos for decode failures.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--backend", default="torchcodec", choices=["torchcodec", "pyav", "both", "ffmpeg"])
    parser.add_argument("--mode", default="frames", choices=["frames", "windows", "ffmpeg"])
    parser.add_argument("--window-size", type=int, default=33)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-errors-per-video", type=int, default=20)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    chunks_size = int(info.get("chunks_size", 1000))
    video_keys = _load_video_keys(dataset_root)
    episodes = _load_episodes(dataset_root)
    output = Path(args.output).expanduser().resolve() if args.output else Path("/tmp") / f"{dataset_root.name}_decode_scan.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    total = len(episodes) * len(video_keys)
    bad_count = 0
    checked = 0
    print(f"[scan] dataset={dataset_root}")
    print(f"[scan] episodes={len(episodes)} video_keys={video_keys} total_videos={total}")
    print(f"[scan] mode={args.mode} backend={args.backend} output={output}")

    with output.open("w", encoding="utf-8") as f:
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            num_frames = int(episode["length"])
            for video_key in video_keys:
                checked += 1
                path = _video_path(dataset_root, video_key, episode_index, chunks_size)
                errors: list[str] = []
                if not path.exists():
                    errors = [f"missing video: {path}"]
                else:
                    try:
                        if args.mode == "ffmpeg" or args.backend == "ffmpeg":
                            errors = _scan_ffmpeg(path)
                        elif args.mode == "frames":
                            if args.backend in {"torchcodec", "both"}:
                                errors.extend(f"torchcodec {err}" for err in _scan_torchcodec(path, num_frames, args.batch_size))
                            if args.backend in {"pyav", "both"}:
                                errors.extend(
                                    f"pyav {err}"
                                    for err in _scan_training_windows(
                                        dataset_root,
                                        path,
                                        num_frames,
                                        fps,
                                        window_size=args.window_size,
                                        stride=max(args.window_size, args.stride),
                                        backend="pyav",
                                        max_errors=args.max_errors_per_video,
                                    )
                                )
                        else:
                            backends = ["torchcodec", "pyav"] if args.backend == "both" else [args.backend]
                            for backend in backends:
                                errors.extend(
                                    _scan_training_windows(
                                        dataset_root,
                                        path,
                                        num_frames,
                                        fps,
                                        window_size=args.window_size,
                                        stride=args.stride,
                                        backend=backend,
                                        max_errors=args.max_errors_per_video,
                                    )
                                )
                    except Exception as exc:
                        errors.append(f"scanner_crash {type(exc).__name__}: {exc}\n{traceback.format_exc()}")

                if errors:
                    bad_count += 1
                    record = {
                        "episode_index": episode_index,
                        "video_key": video_key,
                        "path": str(path),
                        "num_frames": num_frames,
                        "errors": errors[: args.max_errors_per_video],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"[bad] ep={episode_index:06d} key={video_key} errors={len(errors)} path={path}")
                if checked % 25 == 0 or checked == total:
                    print(f"[scan] checked={checked}/{total} bad_videos={bad_count}", flush=True)

    print(f"[scan] done checked={checked} bad_videos={bad_count} output={output}")


if __name__ == "__main__":
    main()
