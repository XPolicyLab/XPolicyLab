#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


REQUIRED_KEYS = {
    "latent",
    "latent_num_frames",
    "latent_height",
    "latent_width",
    "video_num_frames",
    "video_height",
    "video_width",
    "text_emb",
    "text",
    "frame_ids",
    "start_frame",
    "end_frame",
    "fps",
    "ori_fps",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate LingBot-VA latent files.")
    parser.add_argument("--dataset-root", required=True, help="Root of the LeRobot dataset.")
    parser.add_argument(
        "--obs-cam-keys",
        nargs="*",
        default=None,
        help="Override video keys. Default: infer from meta/info.json",
    )
    parser.add_argument(
        "--episode-indices",
        nargs="*",
        type=int,
        default=None,
        help="Only validate selected episodes.",
    )
    return parser.parse_args()


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def get_video_keys(info, override_keys):
    if override_keys:
        return override_keys
    keys = []
    for name, spec in info["features"].items():
        if spec.get("dtype") == "video":
            keys.append(name)
    return keys


def check_payload(payload, expected_start, expected_end, expected_fps, expected_ori_fps, path):
    missing = REQUIRED_KEYS - payload.keys()
    if missing:
        raise ValueError(f"{path}: missing keys: {sorted(missing)}")

    latent = payload["latent"]
    text_emb = payload["text_emb"]
    frame_ids = payload["frame_ids"]

    if not isinstance(latent, torch.Tensor):
        raise TypeError(f"{path}: latent is not a torch.Tensor")
    if latent.ndim != 2:
        raise ValueError(f"{path}: latent.ndim={latent.ndim}, expected 2")
    if latent.shape[0] <= 0 or latent.shape[1] <= 0:
        raise ValueError(f"{path}: invalid latent shape {tuple(latent.shape)}")

    if not isinstance(text_emb, torch.Tensor):
        raise TypeError(f"{path}: text_emb is not a torch.Tensor")
    if text_emb.ndim != 2:
        raise ValueError(f"{path}: text_emb.ndim={text_emb.ndim}, expected 2")

    if latent.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"{path}: unexpected latent dtype {latent.dtype}")
    if text_emb.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"{path}: unexpected text_emb dtype {text_emb.dtype}")

    latent_num_frames = int(payload["latent_num_frames"])
    latent_height = int(payload["latent_height"])
    latent_width = int(payload["latent_width"])
    video_num_frames = int(payload["video_num_frames"])
    start_frame = int(payload["start_frame"])
    end_frame = int(payload["end_frame"])
    fps = int(payload["fps"])
    ori_fps = int(payload["ori_fps"])

    if latent_num_frames <= 0 or latent_height <= 0 or latent_width <= 0:
        raise ValueError(
            f"{path}: invalid latent grid ({latent_num_frames}, {latent_height}, {latent_width})"
        )

    expected_latent_tokens = latent_num_frames * latent_height * latent_width
    if latent.shape[0] != expected_latent_tokens:
        raise ValueError(
            f"{path}: latent token count mismatch, got {latent.shape[0]}, expected {expected_latent_tokens}"
        )

    if not isinstance(frame_ids, list) or len(frame_ids) == 0:
        raise ValueError(f"{path}: frame_ids is empty or not a list")

    if len(frame_ids) != video_num_frames:
        raise ValueError(
            f"{path}: len(frame_ids)={len(frame_ids)} != video_num_frames={video_num_frames}"
        )

    if start_frame != expected_start or end_frame != expected_end:
        raise ValueError(
            f"{path}: start/end mismatch, got [{start_frame},{end_frame}), "
            f"expected [{expected_start},{expected_end})"
        )

    if fps != expected_fps:
        raise ValueError(f"{path}: fps={fps}, expected {expected_fps}")
    if ori_fps != expected_ori_fps:
        raise ValueError(f"{path}: ori_fps={ori_fps}, expected {expected_ori_fps}")

    if any(not isinstance(x, int) for x in frame_ids):
        raise ValueError(f"{path}: frame_ids contains non-int values")

    if sorted(frame_ids) != frame_ids:
        raise ValueError(f"{path}: frame_ids is not sorted ascending")

    if frame_ids[0] < start_frame or frame_ids[-1] >= end_frame:
        raise ValueError(
            f"{path}: frame_ids range [{frame_ids[0]}, {frame_ids[-1]}] not inside "
            f"[{start_frame}, {end_frame})"
        )

    text = payload["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{path}: text is empty")

    return {
        "latent_shape": tuple(latent.shape),
        "text_emb_shape": tuple(text_emb.shape),
        "latent_grid": (latent_num_frames, latent_height, latent_width),
        "video_num_frames": video_num_frames,
    }


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()

    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    latents_root = dataset_root / "latents"

    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing episodes.jsonl: {episodes_path}")
    if not latents_root.exists():
        raise FileNotFoundError(f"Missing latents directory: {latents_root}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes = load_jsonl(episodes_path)
    ori_fps = int(info["fps"])
    video_keys = get_video_keys(info, args.obs_cam_keys)

    wanted_episode_indices = set(args.episode_indices) if args.episode_indices else None

    total_segments = 0
    total_files = 0

    for episode_record in episodes:
        episode_index = int(episode_record["episode_index"])
        if wanted_episode_indices is not None and episode_index not in wanted_episode_indices:
            continue

        action_config = episode_record.get("action_config")
        if not isinstance(action_config, list) or not action_config:
            raise ValueError(f"Episode {episode_index} missing action_config")

        episode_chunk = episode_index // int(info.get("chunks_size", 1000))

        for segment in action_config:
            total_segments += 1
            start_frame = int(segment["start_frame"])
            end_frame = int(segment["end_frame"])

            for video_key in video_keys:
                latent_path = (
                    latents_root
                    / f"chunk-{episode_chunk:03d}"
                    / video_key
                    / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
                )

                if not latent_path.exists():
                    raise FileNotFoundError(f"Missing latent file: {latent_path}")

                payload = torch.load(latent_path, weights_only=False)
                stats = check_payload(
                    payload,
                    expected_start=start_frame,
                    expected_end=end_frame,
                    expected_fps=int(payload["fps"]),
                    expected_ori_fps=ori_fps,
                    path=latent_path,
                )

                total_files += 1
                print(
                    f"[OK] ep={episode_index} key={video_key} "
                    f"segment=[{start_frame},{end_frame}) "
                    f"latent={stats['latent_shape']} "
                    f"grid={stats['latent_grid']} "
                    f"text_emb={stats['text_emb_shape']} "
                    f"video_frames={stats['video_num_frames']}"
                )

    print()
    print(f"Validation passed.")
    print(f"Segments checked: {total_segments}")
    print(f"Latent files checked: {total_files}")


if __name__ == "__main__":
    main()
