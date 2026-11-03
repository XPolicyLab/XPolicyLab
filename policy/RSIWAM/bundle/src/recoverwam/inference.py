"""Standalone image-to-video prediction with the repository's Wan components.

Run ``bash scripts/infer_video.sh --help`` from a checkout. This single-image
path uses bidirectional video attention, latent-space zero padding, and the
reference discrete flow schedule (999 -> 0), independently of policy sampling.
Prompts are passed through verbatim; no robot-specific template is added.
"""

import argparse
import json
import math
import os
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DIT_OVERRIDES = dict(video_attention_mask_mode="bidirectional", use_gradient_checkpointing=False)


def load_inputs(args, *, allow_directories=False):
    """Return physical line numbers, prompts and absolute input paths."""
    if args.prompt_file is None:
        if not args.prompt or not args.prompt.strip():
            raise ValueError("--image requires a non-empty --prompt")
        items = [(1, args.prompt, args.image.expanduser().resolve())]
    else:
        if args.prompt is not None:
            raise ValueError("Use --prompt with --image, or use --prompt-file")
        manifest = args.prompt_file.expanduser().resolve()
        items = []
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            prompt, separator, path = line.partition("@@")
            if not separator or not prompt.strip() or not path.strip():
                raise ValueError(f"{manifest}:{line_number}: expected prompt@@image_or_video_path")
            source = Path(path.strip()).expanduser()
            if not source.is_absolute():
                source = manifest.parent / source
            items.append((line_number, prompt.strip(), source.resolve()))
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise ValueError("No prediction inputs")
    for _, _, source in items:
        if not (source.is_file() or (allow_directories and source.is_dir())):
            raise FileNotFoundError(source)
    return items


def read_image(path, frame_index):
    from PIL import Image

    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        with Image.open(path) as image:
            return image.convert("RGB")
    import imageio.v2 as imageio

    with imageio.get_reader(str(path)) as reader:
        try:
            return Image.fromarray(reader.get_data(frame_index)).convert("RGB")
        except (IndexError, EOFError) as exc:
            raise ValueError(f"Cannot read frame {frame_index} from {path}") from exc


def output_size(image_size, max_area, height=None, width=None):
    """Match reference aspect-ratio resizing, aligned to VAE/DiT stride 16."""
    if height is not None:
        return height, width
    source_w, source_h = image_size
    ratio = source_h / source_w
    height = int(math.sqrt(max_area * ratio) // 16) * 16
    width = int(math.sqrt(max_area / ratio) // 16) * 16
    if min(height, width) < 16:
        raise ValueError("Image aspect ratio is too extreme for --max-area; set --height/--width")
    return height, width


def sampling_timesteps(steps, shift, device):
    import numpy as np
    import torch

    sigma = np.arange(1.0, 0, -1.0 / steps)
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return torch.from_numpy(np.round(sigma * 1000 - 1).astype(np.int64)).to(device)


def flow_step(prediction, sample, timestep, next_timestep):
    """Keep the reference flow interpolation and float32 latent arithmetic."""
    a = 1 - timestep / 1000
    b = timestep / 1000
    clean = (sample - b * prediction) / (a + b)
    noise = (sample + a * prediction) / (a + b)
    return (1 - next_timestep / 1000) * clean + (next_timestep / 1000) * noise


def check_checkpoint(checkpoint, model_dir):
    """Check every DiT key/shape on meta without allocating tensor storage."""
    import torch
    from .models.wan21_14b.helpers.io import load_state_dict
    from .models.wan21_14b.helpers.loader import _load_json_config, _validate_dit_config
    from .models.wan21_14b.wan_video_dit import WanVideoDiT

    config = _validate_dit_config({**_load_json_config(model_dir), **DIT_OVERRIDES})
    if (config.get("in_dim", 36), config.get("out_dim", 16), config.get("text_len", 512)) != (
        36, 16, 512
    ):
        raise ValueError("Expected Wan2.1 I2V with 36 input channels, 16 latents and text_len 512")
    state = load_state_dict(str(checkpoint), device="meta")
    for key in ("dit", "state_dict", "module", "model_state"):
        if isinstance(state.get(key), dict):
            state = state[key]
            break
    with torch.device("meta"):
        model = WanVideoDiT(**config)
    model.load_state_dict(state, strict=True, assign=True)
    return {"tensors": len(state), "parameters": sum(t.numel() for t in state.values())}


def predict_video(model, image, prompt, args, seed):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    device = model.device
    dtype = model.torch_dtype
    height, width = output_size(image.size, args.max_area, args.height, args.width)
    image_tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
    image_tensor = image_tensor.to(device=device, dtype=torch.float32).div(255).sub(0.5).div(0.5)

    def encode_prompt(text):
        with torch.autocast(device.type, dtype=dtype):
            # Wan21VideoModel zero-pads T5 outputs and attends to all 512 positions.
            context, _ = model.encode_prompt(text)
        return context.float()

    with torch.inference_mode():
        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(
            (1, 16, (args.num_frames - 1) // 4 + 1, height // 8, width // 8),
            dtype=torch.float32, device=device, generator=generator,
        )
        use_cfg = not math.isclose(args.guidance_scale, 1.0)
        context = encode_prompt(prompt)
        if use_cfg:
            context = torch.cat([context, encode_prompt(args.negative_prompt)])
        # CLIP sees the native image; only the VAE input uses output-size resizing.
        with torch.autocast(device.type, dtype=torch.float16):
            clip_features = model._encode_clip_fea_from_image(image_tensor[None])
        resized = F.interpolate(
            image_tensor[None], size=(height, width), mode="bicubic", align_corners=False
        )
        condition = model._encode_input_image_latents_tensor(resized)
        if use_cfg:
            clip_features = torch.cat([clip_features, clip_features])
            condition = torch.cat([condition, condition])
        timesteps = sampling_timesteps(args.sampling_steps, args.sigma_shift, device)
        for index, timestep in enumerate(tqdm(timesteps, desc="Video denoising")):
            sample = torch.cat([latents, latents]) if use_cfg else latents
            with torch.autocast(device.type, dtype=dtype):
                prediction = model.dit(
                    x=sample.to(dtype),
                    timestep=timestep.expand(sample.shape[0]),
                    context=context.to(dtype),
                    clip_fea=clip_features.to(dtype),
                    condition_latents=condition.to(dtype),
                ).float()
            if use_cfg:
                positive, negative = prediction.chunk(2)
                prediction = negative + (positive - negative) * args.guidance_scale
            following = timesteps[index + 1] if index + 1 < len(timesteps) else 0
            latents = flow_step(prediction, latents, timestep, following)
        return model._decode_latents(latents)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--backbone-weight", type=Path, required=True,
                        help="Raw backbone.pth or a standalone Video checkpoint")
    parser.add_argument("--wan-model-dir", type=Path,
                        default=os.environ.get("RecoverWAM_WAN_ROOT", "weights/Wan2.1-I2V-14B-480P"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Condition image, or a video to extract one frame")
    source.add_argument("--prompt-file", type=Path, help="UTF-8 lines: prompt@@image_or_video_path")
    parser.add_argument("--prompt", help="Full positive prompt for --image, used verbatim")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/video_prediction"),
                        help="New output directory; existing directories are refused")
    parser.add_argument("--caption-prompt", action="store_true",
                        help="Render the full instruction below the video; keep originals in raw/")
    parser.add_argument("--caption-font", type=Path, help="Optional TrueType font for instructions")
    parser.add_argument("--caption-font-size", type=int, default=14)
    parser.add_argument("--num-frames", type=int, default=49, help="Total output frames, including condition")
    parser.add_argument("--sampling-steps", type=int, default=30)
    parser.add_argument("--sigma-shift", type=float, default=3.0)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--max-area", type=int, default=99841)
    parser.add_argument("--height", type=int, help="Set together with --width, both multiples of 16")
    parser.add_argument("--width", type=int)
    parser.add_argument("--frame-index", type=int, default=24, help="Zero-based input-video frame index")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=3402)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, help="Use the first N nonempty prompt-file entries")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check inputs, assets and checkpoint tensor shapes without using a GPU")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_frames < 1 or (args.num_frames - 1) % 4:
        parser.error("--num-frames must be positive and of the form 4n+1")
    if not 1 <= args.sampling_steps <= 1000:
        parser.error("--sampling-steps must be in [1, 1000]")
    if not math.isfinite(args.sigma_shift) or args.sigma_shift <= 0:
        parser.error("--sigma-shift must be finite and positive")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 0:
        parser.error("--guidance-scale must be finite and non-negative")
    if args.max_area < 256 or args.fps < 1 or args.frame_index < 0:
        parser.error("Require --max-area >= 256, --fps >= 1 and --frame-index >= 0")
    if (args.height is None) != (args.width is None):
        parser.error("Set both --height and --width")
    if args.height is not None and any(n < 16 or n % 16 for n in (args.height, args.width)):
        parser.error("--height and --width must be positive multiples of 16")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.caption_font_size < 8:
        parser.error("--caption-font-size must be at least 8")
    items = load_inputs(args)
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.wan_model_dir = args.wan_model_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    for path in [args.checkpoint, *(args.wan_model_dir / name for name in (
        "Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    ))]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (args.wan_model_dir / "google/umt5-xxl").is_dir():
        raise FileNotFoundError(args.wan_model_dir / "google/umt5-xxl")
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new --output-dir: {args.output_dir}")
    os.environ.setdefault("RecoverWAM_TORCH_LOAD_MMAP", "1")
    metadata = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "checkpoint": check_checkpoint(args.checkpoint, args.wan_model_dir), "videos": []}
    for index, (line, prompt, path) in enumerate(items):
        image = read_image(path, args.frame_index)
        height, width = output_size(image.size, args.max_area, args.height, args.width)
        metadata["videos"].append(dict(
            input=str(path), prompt=prompt, seed=args.seed + index,
            height=height, width=width, output=f"{line:06d}.mp4",
        ))
        if args.caption_prompt:
            from .utils.video_caption import make_caption_panel

            panel = make_caption_panel(width, prompt, args.caption_font_size, args.caption_font)
            metadata["videos"][-1].update(
                raw_output=f"raw/{line:06d}.mp4", rendered_height=height + panel.height,
            )
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    import torch
    from .models.wan21_14b.wan21 import Wan21VideoModel
    from .utils.video_io import save_mp4

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Video generation requires CUDA; use --dry-run for CPU validation")
    if device.index is None:
        device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    model = Wan21VideoModel.from_wan21_14b_pretrained(
        device=str(device), torch_dtype=dtype, model_id=str(args.wan_model_dir),
        dit_checkpoint_path=str(args.checkpoint), tokenizer_max_len=512,
        tokenizer_model_id=str(args.wan_model_dir / "google/umt5-xxl"),
        video_dit_config=DIT_OVERRIDES, load_text_encoder=True, load_clip_encoder=True,
        video_scheduler=dict(
            train_shift=3.0, infer_shift=args.sigma_shift, num_train_timesteps=1000,
            train_sampling_strategy="shifted_uniform", beta_alpha=7.0, beta_beta=1.0,
        ),
    )
    model.eval().requires_grad_(False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for entry in metadata["videos"]:
        image = read_image(Path(entry["input"]), args.frame_index)
        frames = predict_video(model, image, entry["prompt"], args, entry["seed"])
        destination = args.output_dir / entry["output"]
        if args.caption_prompt:
            from .utils.video_caption import append_caption, make_caption_panel

            save_mp4(frames, str(args.output_dir / entry["raw_output"]), fps=args.fps)
            panel = make_caption_panel(
                entry["width"], entry["prompt"], args.caption_font_size, args.caption_font,
            )
            save_mp4(append_caption(frames, panel), str(destination), fps=args.fps)
            print(f"Saved {len(frames)} captioned frames to {destination}", flush=True)
            continue
        save_mp4(frames, str(destination), fps=args.fps)
        print(f"Saved {len(frames)} frames to {destination}", flush=True)


if __name__ == "__main__":
    main()
