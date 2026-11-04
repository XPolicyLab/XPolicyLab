from __future__ import annotations

import argparse

import torch

from flexwam.models.backbone.lingbot_video import load_lingbot_components


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test LingBot Dense transformer/VAE loading and forward shapes.")
    parser.add_argument("--model-path", default="lingbot-video")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--text-len", type=int, default=8)
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device)
    components = load_lingbot_components(args.model_path, torch_dtype=dtype, device=device)
    transformer = components.transformer

    # Test the transformer on latent shape directly. VAE encode is intentionally not
    # run by default here because full Wan VAE encode is much heavier than a DiT smoke test.
    latent_t = (args.frames - 1) // 4 + 1
    latent_h = args.height // 8
    latent_w = args.width // 8
    hidden_states = torch.randn((1, 16, latent_t, latent_h, latent_w), device=device, dtype=dtype)
    context = torch.randn((1, args.text_len, 2560), device=device, dtype=dtype)
    context_mask = torch.ones((1, args.text_len), device=device, dtype=torch.bool)
    timestep = torch.tensor([500], device=device, dtype=torch.long)

    with torch.no_grad():
        out = transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=context,
            encoder_attention_mask=context_mask,
        ).sample

    print("transformer_path:", components.transformer_path)
    print("vae_path:", components.vae_path)
    print("input:", tuple(hidden_states.shape), hidden_states.dtype, hidden_states.device)
    print("context:", tuple(context.shape))
    print("output:", tuple(out.shape), out.dtype, out.device)


if __name__ == "__main__":
    main()
