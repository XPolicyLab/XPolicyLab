import argparse
import os
import time
import types
import torch

from giga_models.sockets import RobotInferenceServer

from world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer
from world_action_model.pipeline.utils import (
    add_state_to_action,
    build_ref_image,
    denormalize_action,
    extract_normalization_tensors,
    load_stats,
    load_t5_embedding_from_pkl,
    normalize_state,
)
from world_action_model.pipeline.wa_pipeline import WAPipeline
from diffusers.models import AutoencoderKLWan


def _parse_bool_list(value: str) -> list[bool]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if part in ("1", "true", "True"):
            out.append(True)
            continue
        if part in ("0", "false", "False"):
            out.append(False)
            continue
        raise ValueError(f"invalid mask element: {part!r} (expected 0/1/true/false)")
    return out


def get_policy(args: argparse.Namespace):
    device = torch.device(args.device)
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"unknown dtype: {args.dtype}")

    t5_embedding = load_t5_embedding_from_pkl(args.t5_embedding_pkl, target_len=int(args.t5_len)).to(
        device=device, dtype=torch.float32
    )
    stats = load_stats(args.stats_path)
    norm = extract_normalization_tensors(stats, device=device, state_dim=args.state_dim, action_dim=args.action_dim)
    delta_mask = torch.tensor(_parse_bool_list(args.delta_mask), device=device, dtype=torch.bool)
    if delta_mask.numel() != args.action_dim:
        raise ValueError(
            f"--delta_mask length ({delta_mask.numel()}) must match --action_dim ({args.action_dim})"
        )

    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.bfloat16)
    transformer = CasualWorldActionTransformer.from_pretrained(args.transformer_path).to(dtype)
    pipe = WAPipeline.from_pretrained(args.model_id, vae=vae, transformer=transformer, torch_dtype=dtype)
    pipe.to(device)

    @torch.no_grad()
    def inference(self, observation):
        state = observation["observation.state"]
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.to(device=device, dtype=torch.float32)

        images = {
            "observation.images.cam_high": observation["observation.images.cam_high"],
            "observation.images.cam_left_wrist": observation["observation.images.cam_left_wrist"],
            "observation.images.cam_right_wrist": observation["observation.images.cam_right_wrist"],
        }
        for k, v in list(images.items()):
            if not isinstance(v, torch.Tensor):
                images[k] = torch.as_tensor(v)
            else:
                images[k] = v

        ref_image = build_ref_image(
            images=images,
            dst_size=(args.dst_width, args.dst_height),
            crop_mode=args.crop_mode,
        )

        norm_state = normalize_state(state, norm, mode=args.norm_mode).to(device=device, dtype=dtype)
        imgs, action = pipe(
            height=args.dst_height,
            width=args.dst_width,
            action_chunk=args.action_chunk,
            state=norm_state,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            image=ref_image,
            action_only=not args.return_images,
            return_dict=False,
            prompt_embeds=t5_embedding.unsqueeze(0).to(device=device, dtype=torch.float32),
        )

        action = denormalize_action(action[0].float(), norm, mode=args.norm_mode)
        action = add_state_to_action(action, state[0].float(), action_chunk=args.action_chunk, mask=delta_mask)
        if args.return_images and args.vis_dir:
            try:
                import torchvision

                os.makedirs(args.vis_dir, exist_ok=True)
                if not hasattr(self, "_vis_idx"):
                    self._vis_idx = 0
                vis_idx = int(self._vis_idx)
                self._vis_idx = vis_idx + 1
                out_path = os.path.join(args.vis_dir, f"pred_{vis_idx:06d}_{int(time.time())}.mp4")

                vid = imgs[0].detach()
                if vid.ndim == 4 and vid.shape[0] in (1, 3):
                    vid = vid.permute(1, 2, 3, 0)
                elif vid.ndim == 3 and vid.shape[0] in (1, 3):
                    vid = vid.permute(1, 2, 0).unsqueeze(0)
                if vid.dtype.is_floating_point:
                    vid = ((vid + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8)
                else:
                    vid = vid.to(torch.uint8)
                torchvision.io.write_video(out_path, vid.cpu(), fps=int(args.vis_fps))
                print(out_path)
            except Exception as e:
                print(f"visualize failed: {e}")
        return action.cpu()

    pipe.inference = types.MethodType(inference, pipe)
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--transformer_path", type=str, required=True)
    parser.add_argument("--stats_path", type=str, required=True)
    parser.add_argument("--t5_embedding_pkl", type=str, required=True)
    parser.add_argument("--t5_len", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--dst_width", type=int, default=768)
    parser.add_argument("--dst_height", type=int, default=192)
    parser.add_argument("--action_chunk", type=int, default=48)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--norm_mode", type=str, default="zscore", choices=["minmax", "zscore"])
    parser.add_argument("--crop_mode", type=str, default="center", choices=["center", "random"])
    parser.add_argument("--return_images", action="store_true")
    parser.add_argument("--vis_dir", type=str, default="./vis")
    parser.add_argument("--vis_fps", type=int, default=5)
    parser.add_argument("--state_dim", type=int, default=14)
    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument(
        "--delta_mask",
        type=str,
        default="1,1,1,1,1,1,0,1,1,1,1,1,1,0",
    )
    args = parser.parse_args()

    policy = get_policy(args)
    server = RobotInferenceServer(policy, host=args.host, port=args.port)
    server.run()


if __name__ == "__main__":
    main()
