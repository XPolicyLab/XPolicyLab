from typing import Optional

import torch
import tyro
from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video


def inference(pretrained: str, prompt: str, save_path: str, device: str = 'cuda', transformer_model_path: Optional[str] = None) -> None:
    """Run text-to-video generation and save as an mp4 file.

    Args:
        pretrained (str): Path or repo id of the base WAN diffusers weights.
        prompt (str): Positive text prompt for generation.
        save_path (str): Destination video path (e.g., "/tmp/out.mp4").
        device (str): Device to run on (e.g., 'cuda' or 'cpu').
        transformer_model_path (str | None): Optional fine-tuned transformer
            checkpoint to override the base one.
    """
    vae = AutoencoderKLWan.from_pretrained(pretrained, subfolder='vae', torch_dtype=torch.float32)
    kwargs = dict(vae=vae, torch_dtype=torch.bfloat16)
    if transformer_model_path is not None:
        kwargs['transformer'] = WanTransformer3DModel.from_pretrained(transformer_model_path, torch_dtype=torch.bfloat16)
    pipe = WanPipeline.from_pretrained(pretrained, **kwargs)
    pipe.to(device)
    negative_prompt = (
        '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
    )
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=704,
        width=1280,
        num_frames=121,
        num_inference_steps=50,
        guidance_scale=5.0,
    ).frames[0]
    export_to_video(output, save_path, fps=24)


if __name__ == '__main__':
    tyro.cli(inference)
