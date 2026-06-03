import dataclasses
import os
import cv2
import numpy as np
import tqdm
import matplotlib
import io
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tyro

import logging
import torch
import safetensors.torch
from openpi_value.training import config as _config
from openpi_value.shared import download
import openpi_value.training.data_loader as _data
from openpi_value.models_pytorch.pi0_pytorch import PI0Pytorch
import openpi_value.models.tokenizer as _tokenizer
from types import SimpleNamespace
from openpi_value.shared import image_tools

all_pred = []
all_tgt  = []


def write_episode_video(
    episode_id: int,
    value_list: list[float],
    img_list: list[list[np.ndarray]],  # now list of image lists
    output_dir: str,
    fig_w: float = 12.0,
    fig_h: float = 4.0,
    dpi: int = 100,
    fps: int = 30,
    metric_only: bool = False,  # * Skip visualizations
):
    """Writes a video for a single episode visualizing predicted values and image frames."""
    if not value_list:
        return 
    
    n_frames = len(value_list)
    tgt_progress = np.linspace(0.0, 1.0, n_frames)
    val_pred     = np.array(value_list, dtype=np.float32)

    global all_pred
    global all_tgt

    all_pred.extend(val_pred.tolist())
    all_tgt.extend(tgt_progress.tolist())

    if metric_only:
        return

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"episode_{episode_id:03d}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # fourcc = cv2.VideoWriter_fourcc(*'h264')

    width_px = int(fig_w * dpi)
    height_px = int(fig_h * dpi)
    video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width_px, height_px))

    logging.info(f"Writing video for episode {episode_id} with {n_frames} frames...")

    for idx in tqdm.tqdm(range(n_frames), desc=f"Episode {episode_id} Video"):
        fig, axes = plt.subplots(1, 1 + len(img_list[idx]), figsize=(fig_w, fig_h), dpi=dpi)

        ax_plot = axes[0]
        # Plotting
        x = np.arange(idx + 1)
        y = np.array(value_list[: idx + 1], dtype=np.float32)
        ax_plot.plot(x, y, linewidth=2, color="tab:blue")
        ax_plot.set_xlim(0, n_frames)
        ax_plot.set_ylim(0.0, 1.0)
        ax_plot.set_xlabel("Frame")
        ax_plot.set_ylabel("Predicted Value")
        ax_plot.set_title("Value Prediction Over Time")
        ax_plot.grid(True)

        # Images: e.g., base, wrist_left, wrist_right
        views = img_list[idx]
        titles = ["Base Frame", "Wrist Left", "Wrist Right"]
        
        titles = titles[:len(views)]  # Adjust titles to match number of views

        for ax_img, view, title in zip(axes[1:], views, titles):
            # ax_img.imshow(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
            ax_img.imshow(view)
            
            ax_img.set_title(title)
            ax_img.axis("off")

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        png_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        plt.close(fig)

        img_bgr = cv2.imdecode(png_bytes, cv2.IMREAD_COLOR)
        h_b, w_b = img_bgr.shape[:2]
        if (w_b, h_b) != (width_px, height_px):
            img_bgr = cv2.resize(img_bgr, (width_px, height_px), interpolation=cv2.INTER_LINEAR)
        video_writer.write(img_bgr)

    video_writer.release()

    # Encode with H.264 for size/speed
    new_out_path = out_path.replace(".mp4", "_new.mp4")
    os.system(f"ffmpeg -y -i {out_path} -c:v libx264 -crf 18 -preset veryfast {new_out_path} > /dev/null 2>&1")
    logging.info(f"=> Episode {episode_id} generated to: {new_out_path}")
    os.remove(out_path)

def process_and_convert(img: np.ndarray, target_size: int) -> torch.Tensor:
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(rgb_img).float() / 255.0
    tensor = tensor * 2.0 - 1.0  # Normalize to [-1, 1]

    tensor = image_tools.resize_with_pad_torch(tensor, 224, 224)  # * Keep ratio

    # HWC to CHW
    tensor = tensor.permute(2, 0, 1) 

    # Batch Dimension: Change C, H, W to 1, C, H, W (the final shape)
    tensor = tensor.unsqueeze(0)
    
    # Final shape check
    return tensor


def main(
    config_name: str,
    ckpt_dir: str,
    split: str = "val_tasks",
    metric_only: bool = False,
    output_video_dir: str = "./visualizations",
    # headview_only: bool = False,
    # n_batches: int = 20000,
    # fps_downsample: int = 1,
):
    """Main function to run value prediction and visualization."""
    
    global all_pred
    global all_tgt

    # --- Config and Checkpoint Setup ---
    config = _config.get_config(config_name)
    checkpoint_dir = download.maybe_download(ckpt_dir)


    # =============== Load Model =============================
    new_model = config.model.__class__(**{**config.model.__dict__,
                                            'p_mask_ego_state': 1,
                                        })
    
    config = dataclasses.replace(config, model=new_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PI0Pytorch(new_model).to(device)
    
    model.sample_values = torch.compile(model.sample_values, mode="reduce-overhead")
    model.eval()  # Set model to evaluation mode
    model_path = os.path.join(checkpoint_dir, "model.safetensors")
    logging.info(f"Loading weights from: {model_path}")
        
    safetensors.torch.load_model(model, model_path, strict=True)

    logging.info(f"Loaded PyTorch weights successfully.")

    tokenizer = _tokenizer.PaligemmaTokenizer(new_model.max_token_len)
    # =============== Load Model finished ======================
    
    # --- Modify Config for Inference ---
    assert split in ['all', 'val_tasks', 'heldout_tasks']
    config = dataclasses.replace(
        config,
        batch_size=1, 
        is_train=False,
        num_workers=8, 
        split=split, 
        preceding_skipping_ratio=0,
    )
    # --- Data Loader ---

    # --- Helper for Image Processing ---
    def process_view_torch(torch_arr):
        """Converts a single image tensor [-1, 1] to a numpy array [0, 255]."""
        arr = torch_arr.cpu().numpy()
        img = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        # Assuming CHW -> HWC
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        return img

    value_frames = []  # type: list[float]
    raw_frames = []    # type: list[list[np.ndarray]]
    
    fps = 30

    # Setup output directory
    output_dir = os.path.join(output_video_dir, config_name, os.path.basename(checkpoint_dir) + f"_{split}")
    print(f">>>>>> Generating episode videos to: {output_dir} ...")
    os.makedirs(output_dir, exist_ok=True)



    # # * Load one video and draw visualization based on that
    image_folder_root = "path/to/your/images"  # * Change this to your image folder root
    left_folder = os.path.join(image_folder_root, 'hand_left')
    right_folder = os.path.join(image_folder_root, 'hand_right')
    top_folder = os.path.join(image_folder_root, 'top_head')

    left_images = sorted([os.path.join(left_folder, f) for f in os.listdir(left_folder) if f.endswith('.jpg')])
    right_images = sorted([os.path.join(right_folder, f) for f in os.listdir(right_folder) if f.endswith('.jpg')])
    top_images = sorted([os.path.join(top_folder, f) for f in os.listdir(top_folder) if f.endswith('.jpg')])

    n_frames = min(len(left_images), len(right_images), len(top_images))

    left_img_0 = cv2.imread(left_images[0])
    right_img_0 = cv2.imread(right_images[0])
    top_img_0 = cv2.imread(top_images[0])

    top_torch_0 = process_and_convert(top_img_0, 224)
    left_torch_0 = process_and_convert(left_img_0, 224)
    right_torch_0 = process_and_convert(right_img_0, 224)

    for i in range(1, n_frames):
        left_img = cv2.imread(left_images[i])
        right_img = cv2.imread(right_images[i])
        top_img = cv2.imread(top_images[i])
      
        top_torch = process_and_convert(top_img, 224)
        left_torch = process_and_convert(left_img, 224)
        right_torch = process_and_convert(right_img, 224)
        
        prompt = "Insert the memory stick."
        observation = {
            "state": torch.zeros((1, 32), dtype=torch.float32).to(device),
            "images": {
                "base_-100_rgb": top_torch_0.to(device),
                "left_wrist_-100_rgb": left_torch_0.to(device),
                "right_wrist_-100_rgb": right_torch_0.to(device),

                "base_0_rgb": top_torch.to(device),
                "left_wrist_0_rgb": left_torch.to(device),
                "right_wrist_0_rgb": right_torch.to(device),
            },
            "image_masks":{}  # * Set empty
        }

        #  * ['base_-100_rgb', 'left_wrist_-100_rgb', 'right_wrist_-100_rgb', 'base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']


        tokens, token_masks = tokenizer.tokenize(prompt, state=None)
        tokens = tokens[np.newaxis, :]  # Add batch dim  
        token_masks = token_masks[np.newaxis, :]  # Add batch dim

        observation = {**observation, 
                       "tokenized_prompt": torch.from_numpy(tokens).to(device), 
                       "tokenized_prompt_mask": torch.from_numpy(token_masks).to(device)}

        observation = SimpleNamespace(**observation)

        with torch.no_grad():
            # Predict value
            val_arr = model.sample_values(device, observation)  # Shape=(1, 1)
            val = float(val_arr[0, 0].item())

            value_frames.append(val)

        base_torch = observation.images["base_0_rgb"][0]
        img_base = process_view_torch(base_torch)
        current_views = [img_base] # Start with base view
        raw_frames.append(current_views)

    logging.info(f"Writing final episode {0}.")
    write_episode_video(
        episode_id=0,
        value_list=value_frames,
        img_list=raw_frames,
        output_dir=output_dir,
        fig_w=12.0,
        fig_h=4.0,
        dpi=100,
        fps=fps,
        metric_only=metric_only,
    )





if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
