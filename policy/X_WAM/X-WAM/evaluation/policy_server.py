"""
Policy Server — connects to Broker backend via DEALER socket.

Lifecycle
--------
1. After loading the model, send b"READY" to Broker to register as available.
2. Receive WORK messages (single request) from Broker.
3. Run inference on the single data item, send RESULT message.
4. Send b"READY" again, wait for next task.

Message Format (DEALER <-> Broker ROUTER backend)
  send READY  : [b"READY"]
  recv WORK   : [b"WORK", client_id, data_pkl]
  send RESULT : [b"RESULT", client_id, result_pkl]
"""

import os
import sys
import time
import hashlib
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import zmq
import tyro
import pickle
import numpy as np
from pprint import pprint
from einops import rearrange
from dataclasses import dataclass

import torch
from omegaconf import OmegaConf
import lightning as L
import torchvision.transforms.functional as TF

from runners.xwam_runner import XWAMRunner


def resize_and_center_crop_tensor(tensor, resized_shape, crop_ratio, depth=False):
    # tensor: [B, V, C, H, W]
    B, V, _, _, _ = tensor.shape
    tensor = tensor.flatten(0, 1)
    tensor = TF.resize(tensor, size=resized_shape, interpolation=TF.InterpolationMode.BILINEAR, antialias=False)
    tensor = tensor.unflatten(0, (B, V))
    H, W = resized_shape
    crop_h = int(H * crop_ratio)
    crop_w = int(W * crop_ratio)
    top = (H - crop_h) // 2
    left = (W - crop_w) // 2
    out = []
    for b in range(B):
        out_b = []
        for v in range(V):
            img = tensor[b, v]
            img_cropped = TF.crop(img, top, left, crop_h, crop_w)
            if depth:
                img_cropped = TF.resize(
                    img_cropped, [H, W], interpolation=TF.InterpolationMode.NEAREST_EXACT, antialias=False
                )
            else:
                img_cropped = TF.resize(
                    img_cropped, [H, W], interpolation=TF.InterpolationMode.BILINEAR, antialias=False
                )
            out_b.append(img_cropped)
        out_b = torch.stack(out_b, dim=0)
        out.append(out_b)
    out = torch.stack(out, dim=0)
    return out


def compute_seed(env_rank, rollout_id, step_id):
    """Deterministically derive a uint32 seed from (env_rank, rollout_id, step_id)."""
    key = f"{env_rank}_{rollout_id}_{step_id}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


@dataclass
class Args:
    exp_path: str
    steps: str = "last"
    wan_checkpoint_dir: str = None
    broker_addr: str = "localhost"
    broker_port: int = 10087
    denoise_steps: int = 50
    action_denoise_steps: int = 10


def build_statistics(config):
    """Build quantile normalization arrays from config.dataset.statistics.

    Supports both single-arm and dual-arm configs. Dimensions without
    explicit stats default to q01=-1.0, q99=1.0 (identity normalization).
    """
    stats = config.dataset.statistics
    has_right_arm = "proprio_right_ee_xyz" in stats.q01

    # --- States [16 dims] ---
    # [left_xyz(3), left_quat(4), left_gripper(1), right_xyz(3), right_quat(4), right_gripper(1)]
    state_q01 = list(stats.q01.proprio_left_ee_xyz) + [-1.0] * 4 + list(stats.q01.gripper_pos)
    state_q99 = list(stats.q99.proprio_left_ee_xyz) + [1.0] * 4 + list(stats.q99.gripper_pos)
    if has_right_arm:
        state_q01 += list(stats.q01.proprio_right_ee_xyz) + [-1.0] * 4 + list(stats.q01.gripper_pos)
        state_q99 += list(stats.q99.proprio_right_ee_xyz) + [1.0] * 4 + list(stats.q99.gripper_pos)
    else:
        state_q01 += [-1.0] * 8
        state_q99 += [1.0] * 8

    # --- Actions ---
    # Single-arm [7]: [left_xyz(3), left_aa(3), gripper(1)]
    # Dual-arm  [14]: [left_xyz(3), left_aa(3), left_grip(1), right_xyz(3), right_aa(3), right_grip(1)]
    action_q01 = (
        list(stats.q01.action_left_ee_xyz) + list(stats.q01.action_left_ee_axisangle) + list(stats.q01.gripper_action)
    )
    action_q99 = (
        list(stats.q99.action_left_ee_xyz) + list(stats.q99.action_left_ee_axisangle) + list(stats.q99.gripper_action)
    )
    if has_right_arm:
        action_q01 += (
            list(stats.q01.action_right_ee_xyz)
            + list(stats.q01.action_right_ee_axisangle)
            + list(stats.q01.gripper_action)
        )
        action_q99 += (
            list(stats.q99.action_right_ee_xyz)
            + list(stats.q99.action_right_ee_axisangle)
            + list(stats.q99.gripper_action)
        )

    return np.array(state_q01), np.array(state_q99), np.array(action_q01), np.array(action_q99), has_right_arm


def main(args: Args):
    config = OmegaConf.load(os.path.join(args.exp_path, "config.yaml"))
    config.sample_steps = args.denoise_steps
    config.use_decoupled_inference = args.action_denoise_steps > 0
    config.action_denoise_steps = args.action_denoise_steps
    config.action_num = config.dataset.frame_skip // config.dataset.action_skip

    if args.wan_checkpoint_dir is not None:
        config.wan_checkpoint_dir = args.wan_checkpoint_dir
    if config.get("wan_checkpoint_dir") is None:
        raise ValueError("Wan2.2-TI2V-5B checkpoint directory must be specified in config or via --wan_checkpoint_dir")
    pprint(OmegaConf.to_container(config))

    state_q01, state_q99, action_q01, action_q99, has_right_arm = build_statistics(config)
    action_dim = len(action_q01)

    L.seed_everything(config.seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = XWAMRunner(config=config).cuda().bfloat16()
    ckpt = torch.load(
        os.path.join(
            args.exp_path,
            f"checkpoints/{args.steps}.ckpt/checkpoint/mp_rank_00_model_states.pt",
        )
    )
    model.load_state_dict(ckpt["module"])
    model.eval()

    model.model = torch.compile(model.model)

    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.connect(f"tcp://{args.broker_addr}:{args.broker_port}")
    logging.info(f"Policy server connected to broker at " f"tcp://{args.broker_addr}:{args.broker_port}")

    socket.send_multipart([b"READY"])

    t_ready = time.time()  # Record the time of the first READY
    while True:
        frames = socket.recv_multipart()
        t_recv = time.time()
        # frames: [b"WORK", client_id, data_pkl]
        assert frames[0] == b"WORK", f"Unexpected message type: {frames[0]}"
        client_id = frames[1]
        data: dict = pickle.loads(frames[2])

        logging.info(f"Waited {t_recv - t_ready:.2f}s for next task")

        seed = compute_seed(data["env_rank"], data["rollout_id"], data["step_id"])

        rgb = torch.from_numpy(data["video"]).bfloat16().unsqueeze(0).cuda()
        rgb = rearrange(rgb, "b v h w c -> b v c h w")
        rgb = resize_and_center_crop_tensor(rgb, config.dataset.video_size, 0.95)
        prompt = data["prompt"]
        cfg = data.get("cfg", 0.0)

        # Normalize proprios
        proprio_raw = data["proprios"]
        proprio_norm = 2 * (proprio_raw - state_q01) / (state_q99 - state_q01) - 1
        if not has_right_arm:
            proprio_norm[8:] = 0.0
        proprio = torch.from_numpy(proprio_norm).bfloat16().unsqueeze(0).cuda()

        t0 = time.time()
        with torch.inference_mode():
            # pred_videos and xt_depth_latents are only returned when early_stop=True
            # xt_proprios are only returned when run_depth=True
            # cfg=0 means no classifier-free guidance
            pred_videos, xt_actions, xt_proprios, xt_depth_latents = model.generate(
                rgb, proprio, prompt, seeds=[seed], early_stop=True, cfg=cfg, run_depth=False
            )
        logging.info(f"Inferred in {time.time() - t0:.2f}s")

        # Denormalize outputs
        xt_actions_np = xt_actions[0].cpu().numpy()
        xt_proprios_np = xt_proprios[0].cpu().numpy()

        xt_proprios_np = (xt_proprios_np + 1) / 2 * (state_q99 - state_q01) + state_q01
        xt_actions_np = (xt_actions_np[:, :action_dim] + 1) / 2 * (action_q99 - action_q01) + action_q01
        if not has_right_arm:
            xt_actions_np[:, 6] *= -1.0  # invert gripper for single-arm (robocasa)

        result = {
            "proprios": xt_proprios_np,
            "actions": xt_actions_np,
        }
        socket.send_multipart([b"RESULT", client_id, pickle.dumps(result)])

        socket.send_multipart([b"READY"])
        t_ready = time.time()  # Update to the time of this READY


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(tyro.cli(Args))
