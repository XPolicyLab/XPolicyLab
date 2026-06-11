"""GigaWorldPolicy inference utilities for XPolicyLab evaluation.

Loads a trained CasualWATrainer checkpoint and serves action predictions
via a websocket interface compatible with the openpi client protocol.
"""

import argparse
import asyncio
import copy
import http
import json
import logging
import os
import random
import time
import traceback
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight msgpack helpers (compatible with openpi_client.msgpack_numpy)
# ---------------------------------------------------------------------------
import functools
import msgpack


def _pack_array(obj):
    """Serialize numpy arrays in the same format as openpi_client."""
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    """Deserialize numpy arrays from openpi_client format."""
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class msgpack_numpy:
    Packer = functools.partial(msgpack.Packer, default=_pack_array)
    packb = staticmethod(functools.partial(msgpack.packb, default=_pack_array))
    unpackb = staticmethod(functools.partial(msgpack.unpackb, object_hook=_unpack_array))


# ---------------------------------------------------------------------------
# Model builder (mirrors CasualWATrainer.get_models but inference-only)
# ---------------------------------------------------------------------------

def build_model(
    pretrained_path: str,
    checkpoint_path: str,
    action_dim: int = 12,
    state_dim: int = 16,
    flow_shift: float = 5.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    from diffusers.models import AutoencoderKLWan
    from world_action_model.models.transformer_wa_casual import (
        CasualWorldActionTransformer,
        WanRotaryPosEmbed1D,
    )
    from world_action_model.trainers.wa_trainer import get_model_path, process_transformer
    from diffusers.video_processor import VideoProcessor

    pretrained = get_model_path(pretrained_path)

    # VAE
    vae = AutoencoderKLWan.from_pretrained(os.path.join(pretrained, "vae"))
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device, dtype=dtype)

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype=dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype=dtype)

    # MoT transformer: Wan video expert from base weights + compact action/state expert
    from world_action_model.models.transformer_wa_mot import MoTWorldActionTransformer

    transformer = MoTWorldActionTransformer.from_pretrained_video(
        transformer_pretrained=os.path.join(pretrained, "transformer"),
        torch_dtype=dtype,
        action_dim=action_dim,
        state_dim=state_dim,
        action_expert={"hidden_dim": 1024, "ffn_dim": 4096},
        mot_checkpoint_mixed_attn=False,
        video_attention_mask_mode="gwp_casual",
    )
    process_transformer(transformer.video_expert, {})
    transformer.to(device, dtype=dtype)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    # MoT checkpoints are required; old single-transformer GWP checkpoints are rejected.
    keys = tuple(state_dict.keys())
    if not any(k.startswith("transformer.mot.") or k.startswith("mot.") or ".mot." in k for k in keys):
        raise ValueError(
            "This server expects GigaWorldPolicy checkpoints. "
            f"Checkpoint has no MoT keys: {checkpoint_path}"
        )

    tf_prefix = "transformer."
    tf_state = {}
    for k, v in state_dict.items():
        if k.startswith(tf_prefix):
            tf_state[k[len(tf_prefix):]] = v
        else:
            tf_state[k] = v
    missing, unexpected = transformer.load_state_dict(tf_state, strict=False)
    print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"  Missing (first 5): {missing[:5]}")
    if unexpected:
        print(f"  Unexpected (first 5): {unexpected[:5]}")

    transformer.eval()

    video_processor = VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial)

    return {
        "vae": vae,
        "transformer": transformer,
        "latents_mean": latents_mean,
        "latents_std": latents_std,
        "video_processor": video_processor,
        "flow_shift": flow_shift,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "device": device,
        "dtype": dtype,
    }


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def load_norm_stats(stats_path: str, state_dim: int, action_dim: int, device: str):
    with open(stats_path, "r") as f:
        stats = json.load(f)
    ns = stats["norm_stats"]

    def _to_tensor(arr, dim):
        t = torch.tensor(arr, dtype=torch.float32, device=device).flatten()[:dim]
        if t.numel() < dim:
            t = torch.nn.functional.pad(t, (0, dim - t.numel()), value=0.0)
        return t

    return {
        "state_mean": _to_tensor(ns["observation.state"]["mean"], state_dim),
        "state_std": _to_tensor(ns["observation.state"]["std"], state_dim),
        "action_mean": _to_tensor(ns["action"]["mean"], action_dim),
        "action_std": _to_tensor(ns["action"]["std"], action_dim),
    }


def normalize_state(state: torch.Tensor, norm: dict) -> torch.Tensor:
    eps = 1e-8
    return (state - norm["state_mean"]) / norm["state_std"].clamp_min(eps)


def denormalize_action(action: torch.Tensor, norm: dict) -> torch.Tensor:
    eps = 1e-8
    return action * norm["action_std"].clamp_min(eps) + norm["action_mean"]


# ---------------------------------------------------------------------------
# Image preprocessing (matches training transform)
# ---------------------------------------------------------------------------

def preprocess_image(img_uint8: np.ndarray, dst_size=(256, 256)) -> torch.Tensor:
    """Convert HWC uint8 image to normalized CHW float tensor."""
    dst_w, dst_h = dst_size
    img = Image.fromarray(img_uint8)
    w, h = img.size
    if float(dst_h) / h < float(dst_w) / w:
        new_h = int(round(float(dst_w) / w * h))
        new_w = dst_w
    else:
        new_h = dst_h
        new_w = int(round(float(dst_h) / h * w))
    img_t = TF.to_tensor(img).unsqueeze(0)  # 1,C,H,W
    img_t = TF.resize(img_t, (new_h, new_w), InterpolationMode.BILINEAR)
    # Center crop
    x1 = (new_w - dst_w) // 2
    y1 = (new_h - dst_h) // 2
    img_t = TF.crop(img_t, y1, x1, dst_h, dst_w)
    # Normalize to [-1, 1]
    img_t = img_t * 2.0 - 1.0
    return img_t  # 1,C,H,W


# ---------------------------------------------------------------------------
# Flow matching sampler (Euler)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_action(
    model_dict: dict,
    ref_latents: torch.Tensor,
    noisy_latents_shape: tuple,
    prompt_embeds: torch.Tensor,
    state: torch.Tensor,
    num_steps: int = 10,
    action_chunk: int = 48,
    action_only: bool = True,
):
    """Sample action using FlowMatchEulerDiscreteScheduler (aligned with openloop_eval)."""
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    device = model_dict["device"]
    dtype = model_dict["dtype"]
    transformer = model_dict["transformer"]
    flow_shift = model_dict["flow_shift"]
    action_dim = model_dict["action_dim"]
    bs = ref_latents.shape[0]

    # Prefix cache is valid only within one conditioning window. PyTorch may
    # reuse tensor data_ptrs across server requests, so clear stale prefixes
    # before sampling a new action chunk while preserving intra-sample caching.
    if hasattr(transformer, "clear_action_only_cache"):
        transformer.clear_action_only_cache()

    # Use the same scheduler as openloop_eval
    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps


    rng = model_dict.get("rng", None)
    noisy_action = torch.randn(
        bs, action_chunk, action_dim, device=device, dtype=dtype, generator=rng,
    )
    noisy_latents = torch.randn(
        noisy_latents_shape, device=device, dtype=dtype, generator=rng,
    )


    for i, t in enumerate(timesteps):
        # Build timestep tensor
        num_state_tokens = state.shape[1]
        latent_h = ref_latents.shape[-2]
        latent_w = ref_latents.shape[-1]
        frame_per_tokens = latent_h * latent_w // 4
        num_ref_latent_tokens = frame_per_tokens
        num_noisy_latent_tokens = frame_per_tokens * (noisy_latents.shape[2])
        total_tokens = num_state_tokens + action_chunk + num_ref_latent_tokens + num_noisy_latent_tokens

        timestep = torch.zeros(bs, total_tokens, device=device, dtype=dtype)
        # State tokens: t=0, ref tokens: t=0, action+noisy: t=current
        noise_t = t.float()
        timestep[:, num_state_tokens + num_ref_latent_tokens:] = noise_t

        if action_only:
            action_pred = transformer(
                ref_latents=ref_latents,
                noisy_latents=noisy_latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
                action=noisy_action,
                state=state,
                action_only=True,
            )
        else:
            visual_pred, action_pred = transformer(
                ref_latents=ref_latents,
                noisy_latents=noisy_latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
                action=noisy_action,
                state=state,
            )
            noisy_latents = scheduler.step(visual_pred, t, noisy_latents, return_dict=False)[0]

        noisy_action = scheduler.step(action_pred, t, noisy_action, return_dict=False)[0]

    return noisy_action


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------

class GWPPolicy:
    def __init__(
        self,
        model_dict: dict,
        norm_stats: dict,
        t5_embedding: torch.Tensor,
        tokenizer=None,
        text_encoder=None,
        prompt_max_length: int = 512,
        prompt_cache_size: int = 256,
        dst_size: tuple = (256, 256),
        num_steps: int = 10,
        num_frames: int = 48,
        action_chunk: int = 48,
        action_only: bool = True,
        # Robocasa-specific post-processing
        zero_action_dims: list = None,       # dims to force to 0 (e.g. [3] for ee_rot_rx)
        ctrl_mode_dim: int = None,           # dim index for control_mode (e.g. 11)
        ctrl_mode_threshold: float = 0.0,    # threshold: >0 → +1, ≤0 → -1
        view_keys: list = None,
        skip_action_denorm: bool = False,    # skip action denormalization (when trained with skip_action_norm)
        tshape: bool = False,                # T-shape layout: head full size + wrist half size
        tshape_head_index: int = 2,          # which view is the head (full size)
    ):
        self.model_dict = model_dict
        self.norm_stats = norm_stats
        self.t5_embedding = t5_embedding
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.prompt_max_length = prompt_max_length
        self.prompt_cache_size = prompt_cache_size
        self.prompt_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.dst_size = dst_size
        self.num_steps = num_steps
        self.num_frames = num_frames
        self.action_chunk = action_chunk
        self.action_only = action_only
        self.zero_action_dims = zero_action_dims or []
        self.ctrl_mode_dim = ctrl_mode_dim
        self.ctrl_mode_threshold = ctrl_mode_threshold
        self.skip_action_denorm = skip_action_denorm
        self.tshape = tshape
        self.tshape_head_index = tshape_head_index
        self.view_keys = view_keys or [
            "observation.images.robot0_agentview_left",
            "observation.images.robot0_eye_in_hand",
            "observation.images.robot0_agentview_right",
        ]
        self.infer_count = 0

    @torch.no_grad()
    def _get_prompt_embeds(self, obs: dict, device: str, dtype: torch.dtype) -> torch.Tensor:
        if self.tokenizer is None or self.text_encoder is None:
            return self.t5_embedding.unsqueeze(0).to(device, dtype=dtype)

        prompt = obs.get("prompt", "")
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8", errors="ignore")
        prompt = str(prompt)

        cached = self.prompt_cache.get(prompt)
        if cached is not None:
            # LRU refresh
            self.prompt_cache.move_to_end(prompt)
            return cached.unsqueeze(0).to(device, dtype=dtype)

        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.prompt_max_length,
        ).to(device)
        outputs = self.text_encoder(**inputs)
        length = int(inputs.attention_mask[0].sum().item())
        emb = outputs.last_hidden_state[0, :length].detach().cpu().float()
        emb = emb[:64]
        if emb.shape[0] < 64:
            emb = torch.nn.functional.pad(emb, (0, 0, 0, 64 - emb.shape[0]))

        is_zero = (emb.abs().sum().item() == 0)
        logger.info(
            f"T5 prompt embed: prompt={prompt!r}, shape={tuple(emb.shape)}, "
            f"mean={emb.mean().item():.6f}, std={emb.std().item():.6f}, "
            f"abs_max={emb.abs().max().item():.6f}, all_zero={is_zero}"
        )

        self.prompt_cache[prompt] = emb
        if len(self.prompt_cache) > self.prompt_cache_size:
            self.prompt_cache.popitem(last=False)
        return emb.unsqueeze(0).to(device, dtype=dtype)

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        device = self.model_dict["device"]
        dtype = self.model_dict["dtype"]
        vae = self.model_dict["vae"]
        latents_mean = self.model_dict["latents_mean"]
        latents_std = self.model_dict["latents_std"]

        start = time.monotonic()
        self.infer_count += 1

        prompt_preview = str(obs.get("prompt", "")).replace("\n", " ")[:160]
        image_summaries = []
        for key in ("observation/image", "observation/wrist_image", "observation/right_image"):
            if key not in obs:
                continue
            arr = np.asarray(obs[key])
            value_range = ""
            if arr.size:
                try:
                    value_range = f", range=({int(arr.min())},{int(arr.max())})"
                except Exception:
                    value_range = ""
            image_summaries.append(f"{key}:shape={tuple(arr.shape)},dtype={arr.dtype}{value_range}")
        raw_state = np.asarray(obs.get("observation/state", []), dtype=np.float32)
        state_preview = np.array2string(raw_state.reshape(-1)[:6], precision=4, separator=",")
        prompt_mode = "dynamic_t5" if self.tokenizer is not None and self.text_encoder is not None else "static_t5"
        print(
            "[GWPPolicy] infer request",
            f"#{self.infer_count}",
            f"prompt_mode={prompt_mode}",
            f"prompt={prompt_preview!r}",
            f"images=[{'; '.join(image_summaries) if image_summaries else 'none'}]",
            f"state_shape={tuple(raw_state.shape)}",
            f"state_first6={state_preview}",
            flush=True,
        )

        # 1. Process images from observation
        # The client sends: observation/image (agentview_left), observation/wrist_image (eye_in_hand)
        # We also need agentview_right if available
        image_start = time.monotonic()
        images = []
        if "observation/image" in obs:
            img = np.asarray(obs["observation/image"], dtype=np.uint8)
            images.append(img)
        if "observation/wrist_image" in obs:
            img = np.asarray(obs["observation/wrist_image"], dtype=np.uint8)
            images.append(img)
        if "observation/right_image" in obs:
            img = np.asarray(obs["observation/right_image"], dtype=np.uint8)
            images.append(img)

        if len(images) == 0:
            raise ValueError("No images found in observation")

        if self.tshape and len(images) > 1:
            # T-shape layout: head view at full dst_size, others at half size
            dst_w, dst_h = self.dst_size
            head = preprocess_image(images[self.tshape_head_index], (dst_w, dst_h))
            half_w, half_h = dst_w // 2, dst_h // 2
            others = []
            for i, img in enumerate(images):
                if i == self.tshape_head_index:
                    continue
                others.append(preprocess_image(img, (half_w, half_h)))
            wrist_row = torch.cat(others, dim=-1)  # 1,C,half_h,half_w*N
            # Pad/crop wrist_row width to match head width
            if wrist_row.shape[-1] < head.shape[-1]:
                wrist_row = torch.nn.functional.pad(wrist_row, (0, head.shape[-1] - wrist_row.shape[-1]))
            elif wrist_row.shape[-1] > head.shape[-1]:
                wrist_row = wrist_row[..., :head.shape[-1]]
            ref_image = torch.cat([head, wrist_row], dim=-2)  # 1,C,H*1.5,W — head on top
        else:
            processed = [preprocess_image(img, self.dst_size) for img in images]
            # Concatenate views along width (matching training)
            ref_image = torch.cat(processed, dim=-1)  # 1,C,H,W*num_views
        ref_image = ref_image.to(device, dtype=dtype)
        image_ms = (time.monotonic() - image_start) * 1000

        # Encode reference image through VAE
        # ref_image: 1,C,H,W -> 1,C,1,H,W for VAE (single frame)
        vae_start = time.monotonic()
        ref_image_5d = ref_image.unsqueeze(2)  # 1,C,1,H,W
        ref_latents = vae.encode(ref_image_5d).latent_dist.mode()
        ref_latents = (ref_latents - latents_mean) * latents_std
        vae_ms = (time.monotonic() - vae_start) * 1000

        # 2. Process state
        state_start = time.monotonic()
        state = np.asarray(obs["observation/state"], dtype=np.float32)
        state = torch.from_numpy(state).to(device, dtype=dtype)
        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(0)  # 1,1,D
        elif state.dim() == 2:
            state = state.unsqueeze(0)  # 1,1,D

        # Pad/truncate state
        sd = self.model_dict["state_dim"]
        if state.shape[-1] > sd:
            state = state[..., :sd]
        elif state.shape[-1] < sd:
            state = torch.nn.functional.pad(state, (0, sd - state.shape[-1]))

        # Normalize state
        state_f32 = state.float()
        state_f32 = normalize_state(state_f32.squeeze(0), self.norm_stats).unsqueeze(0)
        state = state_f32.to(dtype=dtype)
        state_ms = (time.monotonic() - state_start) * 1000

        # 3. Prompt embeddings (dynamic from obs["prompt"] if tokenizer+text_encoder are loaded)
        prompt_start = time.monotonic()
        prompt_embeds = self._get_prompt_embeds(obs, device=device, dtype=dtype)
        prompt_ms = (time.monotonic() - prompt_start) * 1000

        # 4. Prepare noisy latents shape (for visual branch)
        # num_frames - 1 future frames in latent space (matches training)
        noisy_latents_shape = (
            1,
            ref_latents.shape[1],
            (self.num_frames // 4) - 1,  # vae temporal factor = 4, minus ref frame
            ref_latents.shape[3],
            ref_latents.shape[4],
        )

        # 5. Sample actions (always generate full num_frames steps, then truncate)
        sample_start = time.monotonic()
        pred_actions = sample_action(
            self.model_dict,
            ref_latents=ref_latents,
            noisy_latents_shape=noisy_latents_shape,
            prompt_embeds=prompt_embeds,
            state=state,
            num_steps=self.num_steps,
            action_chunk=self.num_frames,
            action_only=self.action_only,
        )
        sample_ms = (time.monotonic() - sample_start) * 1000

        # 6. Denormalize actions
        post_start = time.monotonic()
        pred_actions = pred_actions.float().squeeze(0)  # T, action_dim
        if not self.skip_action_denorm:
            pred_actions = denormalize_action(pred_actions, self.norm_stats)
        actions_np = pred_actions.cpu().numpy()

        # 7. Truncate to requested action_chunk
        actions_np = actions_np[:self.action_chunk]

        # 8. Post-processing for the configured action layout
        # Force zero-std dims to 0
        for d in self.zero_action_dims:
            if d < actions_np.shape[-1]:
                actions_np[:, d] = 0.0
        # Threshold ctrl_mode to discrete {-1, +1}
        if self.ctrl_mode_dim is not None and self.ctrl_mode_dim < actions_np.shape[-1]:
            actions_np[:, self.ctrl_mode_dim] = np.where(
                actions_np[:, self.ctrl_mode_dim] > self.ctrl_mode_threshold, 1.0, -1.0)
        post_ms = (time.monotonic() - post_start) * 1000

        elapsed = time.monotonic() - start
        logger.debug(f"Inference took {elapsed*1000:.1f}ms, action shape: {actions_np.shape}, "
                      f"action[0]={np.array2string(actions_np[0], precision=4, separator=', ')}, "
                      f"action_abs_mean={np.abs(actions_np).mean(axis=0).tolist()}")

        return {"actions": actions_np}


# ---------------------------------------------------------------------------
# Websocket server
# ---------------------------------------------------------------------------

async def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


class InferenceServer:
    def __init__(self, policy: GWPPolicy, host: str = "0.0.0.0", port: int = 18055):
        self.policy = policy
        self.host = host
        self.port = port

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        import websockets.asyncio.server as ws_server

        async with ws_server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
            ping_interval=120,
            ping_timeout=600,
        ) as server:
            logger.info(f"Server listening on {self.host}:{self.port}")
            await server.serve_forever()

    async def _handler(self, websocket):
        logger.info(f"Connection from {websocket.remote_address}")
        packer = msgpack_numpy.Packer()

        # Send metadata
        await websocket.send(packer.pack({"model": "gwp-xpolicylab"}))

        loop = asyncio.get_running_loop()
        prev_total_time = None
        while True:
            try:
                start = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_start = time.monotonic()
                # Run inference in a thread so async ping/pong is not blocked
                result = await loop.run_in_executor(None, self.policy.infer, obs)
                infer_time = time.monotonic() - infer_start

                result["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    result["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(result))
                prev_total_time = time.monotonic() - start

            except Exception as e:
                if "ConnectionClosed" in type(e).__name__:
                    logger.info(f"Connection closed: {websocket.remote_address}")
                    break
                logger.error(traceback.format_exc())
                try:
                    await websocket.send(traceback.format_exc())
                    await websocket.close()
                except Exception:
                    pass
                break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="GigaWorldPolicy XPolicyLab inference server")
    parser.add_argument("--model_id", type=str,
                        default=os.environ.get("GIGAWORLD_PRETRAINED_PATH", os.environ.get("WAN22_DIFFUSERS_PATH", "")),
                        help="Path to pretrained Wan model. Defaults to "
                             "$GIGAWORLD_PRETRAINED_PATH / $WAN22_DIFFUSERS_PATH.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to a MoT model.pt checkpoint")
    parser.add_argument("--stats_path", type=str, required=True,
                        help="Path to normalization stats JSON")
    parser.add_argument("--t5_embedding_path", type=str, default=None,
                        help="Path to T5 embedding .pt file (optional, zeros if not provided)")
    parser.add_argument("--disable_dynamic_prompt", action="store_true",
                        help="Disable online prompt encoding and use static --t5_embedding_path/zeros only")
    parser.add_argument("--prompt_max_length", type=int, default=512,
                        help="Tokenizer max length for dynamic prompt encoding")
    parser.add_argument("--prompt_cache_size", type=int, default=256,
                        help="LRU cache size for encoded prompt embeddings")
    parser.add_argument("--action_dim", type=int, default=12)
    parser.add_argument("--state_dim", type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=24,
                        help="Training num_frames (controls noisy latent temporal dim & full action length)")
    parser.add_argument("--action_chunk", type=int, default=24,
                        help="Number of action steps to return (truncates from num_frames)")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of denoising steps")
    parser.add_argument("--action_only", action="store_true", default=True,
                        help="Only predict actions (skip video generation)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18055)
    parser.add_argument("--dst_size", type=int, nargs=2, default=[320, 256], help="Image dst_size (W H)")
    # Robocasa-specific post-processing
    parser.add_argument("--zero_action_dims", type=int, nargs="*", default=None,
                        help="Action dims to force to 0, if any")
    parser.add_argument("--ctrl_mode_dim", type=int, default=None,
                        help="Action dim index for control_mode, if used")
    parser.add_argument("--ctrl_mode_threshold", type=float, default=0.0,
                        help="Threshold for ctrl_mode binarization")
    parser.add_argument("--skip_action_denorm", action="store_true", default=False,
                        help="Skip action denormalization (use when trained with skip_action_norm=True)")
    parser.add_argument("--tshape", action="store_true", default=True,
                        help="Compatibility flag; this server uses the T-shape layout")
    parser.add_argument("--tshape_head_index", type=int, default=2,
                        help="Index of head view in image list (2=agentview_right)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample noise (torch.randn in flow matching). "
                        "All server replicas should use the same seed for reproducibility.")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    print("=" * 60)
    print("GigaWorldPolicy XPolicyLab Inference Server")
    print("=" * 60)
    print(f"  Model:      {args.model_id}")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Stats:      {args.stats_path}")
    print(f"  Action dim: {args.action_dim}, State dim: {args.state_dim}")
    print(f"  Chunk size: {args.action_chunk}, Num frames: {args.num_frames}, Steps: {args.num_steps}")
    print(f"  Port:       {args.port}")
    print(f"  Seed:       {args.seed}")
    print("=" * 60)

    # Set global RNG seeds for reproducibility of sample noise
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Build model
    model_dict = build_model(
        pretrained_path=args.model_id,
        checkpoint_path=args.checkpoint_path,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Dedicated CUDA generator for sample_action noise so that the noise sequence
    # is fully determined by --seed and not affected by other RNG-consuming ops.
    model_dict["rng"] = torch.Generator(device=model_dict["device"]).manual_seed(args.seed)

    # Load norm stats
    norm_stats = load_norm_stats(args.stats_path, args.state_dim, args.action_dim, "cuda")

    # Load fallback static T5 embedding
    if args.t5_embedding_path and os.path.exists(args.t5_embedding_path):
        t5_emb = torch.load(args.t5_embedding_path, map_location="cpu")
        if not isinstance(t5_emb, torch.Tensor):
            t5_emb = torch.as_tensor(t5_emb)
        t5_emb = t5_emb[:64]
        if t5_emb.shape[0] < 64:
            t5_emb = torch.nn.functional.pad(t5_emb, (0, 0, 0, 64 - t5_emb.shape[0]))
    else:
        t5_emb = torch.zeros(64, 4096, dtype=torch.float32)
        print("No T5 embedding provided, using zeros.")

    # Load tokenizer + text encoder for dynamic prompt embedding
    tokenizer = None
    text_encoder = None
    if not args.disable_dynamic_prompt:
        try:
            from transformers import AutoTokenizer, UMT5EncoderModel

            pretrained = args.model_id
            if os.path.isdir(pretrained):
                tok_path = os.path.join(pretrained, "tokenizer")
                te_path = os.path.join(pretrained, "text_encoder")
            else:
                from world_action_model.trainers.wa_trainer import get_model_path
                pretrained = get_model_path(args.model_id)
                tok_path = os.path.join(pretrained, "tokenizer")
                te_path = os.path.join(pretrained, "text_encoder")

            tokenizer = AutoTokenizer.from_pretrained(tok_path)
            text_encoder = UMT5EncoderModel.from_pretrained(te_path, torch_dtype=torch.float16).to("cuda")
            text_encoder.eval()
            print(f"Dynamic prompt encoding enabled. tokenizer={tok_path}, text_encoder={te_path}")
        except Exception as e:
            print(f"WARNING: failed to load dynamic prompt encoder, fallback to static embedding: {e}")

    # Build policy
    policy = GWPPolicy(
        model_dict=model_dict,
        norm_stats=norm_stats,
        t5_embedding=t5_emb,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt_max_length=args.prompt_max_length,
        prompt_cache_size=args.prompt_cache_size,
        dst_size=tuple(args.dst_size),
        num_steps=args.num_steps,
        num_frames=args.num_frames,
        action_chunk=args.action_chunk,
        action_only=args.action_only,
        zero_action_dims=args.zero_action_dims,
        ctrl_mode_dim=args.ctrl_mode_dim,
        ctrl_mode_threshold=args.ctrl_mode_threshold,
        skip_action_denorm=args.skip_action_denorm,
        tshape=args.tshape,
        tshape_head_index=args.tshape_head_index,
    )

    # Start server
    server = InferenceServer(policy, host=args.host, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
