#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
from diffusers import AutoencoderKLWan
from PIL import Image
from transformers import T5TokenizerFast, UMT5EncoderModel


DEFAULT_VIDEO_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract LingBot-VA latent .pth files from a LeRobot dataset with Wan2.2."
    )
    parser.add_argument("--dataset-root", required=True, help="Root of the LeRobot dataset.")
    parser.add_argument("--model-root", required=True, help="Wan2.2-TI2V-5B-Diffusers directory.")
    parser.add_argument("--target-fps", type=int, default=10, help="Target FPS for latent extraction.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--obs-cam-keys",
        nargs="*",
        default=None,
        help="Override video keys. Default: infer all dtype=video keys from meta/info.json",
    )
    parser.add_argument(
        "--episode-indices",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of episode indices to process.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--min-sampled-frames", type=int, default=5)
    return parser.parse_args()


def to_dtype(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


@torch.no_grad()
def encode_text(tokenizer, text_encoder, text, max_length, device, dtype):
    text_inputs = tokenizer(
        [text],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    seq_len = int(attention_mask.gt(0).sum(dim=1).item())

    hidden = text_encoder(input_ids, attention_mask).last_hidden_state[0]
    hidden = hidden[:seq_len].to(dtype=dtype)
    padded = torch.cat(
        [
            hidden,
            torch.zeros(max_length - seq_len, hidden.shape[-1], dtype=dtype, device=hidden.device),
        ],
        dim=0,
    )
    return padded.cpu()


def normalize_latents(latents, latents_mean, latents_std):
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
    return ((latents.float() - latents_mean) * latents_std).to(latents.dtype)


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def get_video_keys(info, override_keys):
    if override_keys:
        return override_keys
    keys = []
    for name, spec in info["features"].items():
        if spec.get("dtype") == "video":
            keys.append(name)
    return keys or DEFAULT_VIDEO_KEYS


def build_video_path(dataset_root: Path, info: dict, episode_index: int, video_key: str):
    episode_chunk = episode_index // int(info.get("chunks_size", 1000))
    relative = info["video_path"].format(
        episode_chunk=episode_chunk,
        video_key=video_key,
        episode_index=episode_index,
    )
    return dataset_root / relative


def sample_frame_ids(start_frame: int, end_frame: int, ori_fps: int, target_fps: int):
    if end_frame <= start_frame:
        raise ValueError(f"Invalid segment: start={start_frame}, end={end_frame}")

    stride = max(1, int(round(ori_fps / target_fps)))
    frame_ids = list(range(start_frame, end_frame, stride))

    if len(frame_ids) < 2 and (end_frame - start_frame) >= 2:
        frame_ids = [start_frame, min(end_frame - 1, start_frame + 1)]

    if len(frame_ids) < 2:
        raise ValueError(
            f"Segment [{start_frame}, {end_frame}) is too short after sampling; "
            f"got {len(frame_ids)} frames with stride={stride}."
        )

    return frame_ids


def adjust_frame_ids_for_wan(frame_ids, min_frames=5):
    valid_len = ((len(frame_ids) - 1) // 4) * 4 + 1
    frame_ids = frame_ids[:valid_len]

    if len(frame_ids) < min_frames:
        raise ValueError(
            f"Need at least {min_frames} sampled frames after 4k+1 trimming, got {len(frame_ids)}"
        )

    return frame_ids


def decode_sampled_frames(video_path: Path, frame_ids, resize_hw):
    wanted = set(frame_ids)
    frames = {}

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                image = frame.to_ndarray(format="rgb24")
                if image.shape[:2] != resize_hw:
                    image = np.asarray(
                        Image.fromarray(image).resize((resize_hw[1], resize_hw[0]), Image.BILINEAR),
                        dtype=np.uint8,
                    )
                frames[index] = image
            if len(frames) == len(wanted):
                break

    missing = [idx for idx in frame_ids if idx not in frames]
    if missing:
        raise RuntimeError(f"Missing frames {missing[:10]} from {video_path}")

    return np.stack([frames[idx] for idx in frame_ids], axis=0)


@torch.no_grad()
def encode_video_latent(vae, video_frames, dtype, device):
    # video_frames: [T, H, W, 3]
    video = torch.from_numpy(video_frames).permute(3, 0, 1, 2).unsqueeze(0).float()
    video = video / 255.0 * 2.0 - 1.0
    video = video.to(device=device, dtype=dtype)

    encode_output = vae.encode(video)
    if hasattr(encode_output, "latent_dist"):
        mu = encode_output.latent_dist.mean
    elif isinstance(encode_output, tuple):
        mu = encode_output[0]
    else:
        raise RuntimeError(f"Unexpected VAE encode output type: {type(encode_output)}")

    latents_mean = torch.tensor(vae.config.latents_mean, device=mu.device)
    latents_std = torch.tensor(vae.config.latents_std, device=mu.device)
    mu_norm = normalize_latents(mu, latents_mean, 1.0 / latents_std)

    latent = mu_norm[0].permute(1, 2, 3, 0).contiguous()
    latent_num_frames, latent_height, latent_width, channels = latent.shape
    flattened = latent.view(latent_num_frames * latent_height * latent_width, channels)
    return flattened.cpu(), latent_num_frames, latent_height, latent_width


def main():
    args = parse_args()

    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError(f"height/width must be multiples of 16, got {args.height}x{args.width}")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    model_root = Path(args.model_root).expanduser().resolve()
    dtype = to_dtype(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing episodes.jsonl: {episodes_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes = load_jsonl(episodes_path)
    video_keys = get_video_keys(info, args.obs_cam_keys)
    ori_fps = int(info["fps"])

    tokenizer = T5TokenizerFast.from_pretrained(model_root / "tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_root / "text_encoder",
        torch_dtype=dtype,
    ).to(device)
    text_encoder.eval()

    vae = AutoencoderKLWan.from_pretrained(
        model_root / "vae",
        torch_dtype=dtype,
    ).to(device)
    vae.eval()

    wanted_episode_indices = set(args.episode_indices) if args.episode_indices else None

    for episode_record in episodes:
        episode_index = int(episode_record["episode_index"])
        if wanted_episode_indices is not None and episode_index not in wanted_episode_indices:
            continue

        action_config = episode_record.get("action_config")
        if not isinstance(action_config, list) or not action_config:
            raise KeyError(f"Episode {episode_index} is missing action_config")

        for segment in action_config:
            start_frame = int(segment["start_frame"])
            end_frame = int(segment["end_frame"])
            text = str(segment.get("action_text", "")).strip()
            if not text:
                raise ValueError(
                    f"Episode {episode_index} segment [{start_frame}, {end_frame}) has empty action_text"
                )

            frame_ids = sample_frame_ids(start_frame, end_frame, ori_fps, args.target_fps)
            frame_ids = adjust_frame_ids_for_wan(frame_ids, min_frames=args.min_sampled_frames)
            text_emb = encode_text(tokenizer, text_encoder, text, args.max_length, device, dtype)

            for video_key in video_keys:
                video_path = build_video_path(dataset_root, info, episode_index, video_key)
                if not video_path.exists():
                    raise FileNotFoundError(f"Missing video file: {video_path}")

                episode_chunk = episode_index // int(info.get("chunks_size", 1000))
                output_dir = dataset_root / "latents" / f"chunk-{episode_chunk:03d}" / video_key
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"

                if args.skip_existing and output_path.exists():
                    print(f"Skip existing: {output_path}")
                    continue

                print(
                    f"[latent] ep={episode_index} key={video_key} "
                    f"seg=[{start_frame},{end_frame}) sampled={len(frame_ids)} "
                    f"frames=({frame_ids[0]}->{frame_ids[-1]})"
                )

                video_frames = decode_sampled_frames(video_path, frame_ids, (args.height, args.width))
                latent, latent_num_frames, latent_height, latent_width = encode_video_latent(
                    vae,
                    video_frames,
                    dtype,
                    device,
                )

                payload = {
                    "latent": latent.to(torch.bfloat16),
                    "latent_num_frames": int(latent_num_frames),
                    "latent_height": int(latent_height),
                    "latent_width": int(latent_width),
                    "video_num_frames": int(len(frame_ids)),
                    "video_height": int(info["features"][video_key]["info"]["video.height"]),
                    "video_width": int(info["features"][video_key]["info"]["video.width"]),
                    "text_emb": text_emb.to(torch.bfloat16),
                    "text": text,
                    "frame_ids": frame_ids,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "fps": int(args.target_fps),
                    "ori_fps": ori_fps,
                }
                torch.save(payload, output_path)
                print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()