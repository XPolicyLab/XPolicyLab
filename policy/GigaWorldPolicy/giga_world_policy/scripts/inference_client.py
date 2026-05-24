import argparse
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from giga_datasets import load_dataset
from giga_models.sockets import RobotInferenceClient
from PIL import Image
from tqdm import tqdm


def read_video(video_path: str):
    from torchvision.io import VideoReader as TorchVideoReader

    reader = TorchVideoReader(video_path, "video")
    frames = []
    for frame in reader:
        frame_np = frame["data"].cpu().numpy()
        frame_np = np.transpose(frame_np, (1, 2, 0))
        frames.append(frame_np)
    return frames


def to_chw01(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def save_action_plot(gt_action: np.ndarray, pred_action: np.ndarray, save_path: str, mse: Optional[float] = None):
    import matplotlib.pyplot as plt

    t = gt_action.shape[0]
    plt.figure(figsize=(20, 10))
    if mse is not None:
        plt.suptitle(f"MSE: {mse:.6f}")
    for i in range(gt_action.shape[1]):
        plt.subplot(4, 4, i + 1)
        plt.plot(range(t), gt_action[:, i], label="gt")
        plt.plot(range(t), pred_action[:, i], label="pred")
        plt.title(f"dim {i}")
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compute_mse(gt_action: np.ndarray, pred_action: np.ndarray) -> float:
    return float(np.mean((gt_action - pred_action) ** 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_paths", type=str, nargs="+", required=True)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--action_chunk", type=int, default=48)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--max_rollouts", type=int, default=None)
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()

    dataset = load_dataset(list(args.dataset_paths))
    data_dict: Dict[str, Any] = dataset[args.episode_idx]

    front_frames = read_video(data_dict["front_video_path"])
    left_frames = read_video(data_dict["left_video_path"])
    right_frames = read_video(data_dict["right_video_path"])

    all_state = data_dict["state"]
    all_action = data_dict["action"]

    rollout_num = len(front_frames) // args.action_chunk
    if args.max_rollouts is not None:
        rollout_num = min(rollout_num, args.max_rollouts)

    client = RobotInferenceClient(host=args.host, port=args.port)

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    all_pred_actions = []
    all_gt_actions = []
    all_pred_frames = []
    all_gt_delta = []
    all_pred_delta = []
    all_mse = []

    for r in tqdm(range(rollout_num)):
        start_frame = r * args.action_chunk
        end_frame = start_frame + args.action_chunk

        state = torch.from_numpy(all_state[start_frame]).float().unsqueeze(0)
        gt_action = all_action[start_frame:end_frame]

        img_front = Image.fromarray(front_frames[start_frame])
        img_left = Image.fromarray(left_frames[start_frame])
        img_right = Image.fromarray(right_frames[start_frame])

        observation = {
            "observation.state": state,
            "observation.images.cam_high": to_chw01(img_front),
            "observation.images.cam_left_wrist": to_chw01(img_left),
            "observation.images.cam_right_wrist": to_chw01(img_right),
        }
        pred_action = client.inference(observation)
        pred_action = pred_action.numpy()

        all_pred_actions.append(pred_action)
        all_gt_actions.append(gt_action)

        mse = compute_mse(gt_action, pred_action)
        all_mse.append(mse)

        state_np = state.repeat(args.action_chunk, 1).cpu().numpy()
        gt_delta = gt_action - state_np
        pred_delta = pred_action - state_np
        all_gt_delta.append(gt_delta)
        all_pred_delta.append(pred_delta)

    if args.save_dir is not None:
        all_gt_actions_arr = np.concatenate(all_gt_actions, axis=0)
        all_pred_actions_arr = np.concatenate(all_pred_actions, axis=0)
        save_action_plot(all_gt_actions_arr, all_pred_actions_arr, os.path.join(args.save_dir, "action_all.png"))


    total_mse = float(np.mean(all_mse)) if len(all_mse) > 0 else float("nan")
    print(f"avg chunk MSE: {total_mse:.6f}")
    if args.save_dir is not None and len(all_gt_delta) > 0:
        all_gt_delta_arr = np.concatenate(all_gt_delta, axis=0)
        all_pred_delta_arr = np.concatenate(all_pred_delta, axis=0)
        save_action_plot(all_gt_delta_arr, all_pred_delta_arr, os.path.join(args.save_dir, "delta_all.png"), mse=compute_mse(all_gt_delta_arr, all_pred_delta_arr))


if __name__ == "__main__":
    main()
