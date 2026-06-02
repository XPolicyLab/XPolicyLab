import dataclasses
import os
import cv2
import numpy as np
import tqdm
import matplotlib
import io
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
import torch
import safetensors.torch
from types import SimpleNamespace
from openpi_value.models.model import Observation


from openpi_value.training import config as _config
from openpi_value.shared import download
import openpi_value.training.data_loader as _data
from openpi_value.models_pytorch.pi0_pytorch import PI0Pytorch
import openpi_value.models.tokenizer as _tokenizer
from openpi_value.shared import image_tools

import pandas as pd
import numpy as np
import torch
from typing import Union, Tuple
from PIL import Image
# Import PyTorch standard libraries for image processing
import torchvision.transforms.functional as F_tv # For resizing (on PIL Images or Tensors)
import torch.nn.functional as F_nn # For padding (on Tensors)


# --- Helper function for type conversion (if needed) ---
def convert_to_uint8(img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Converts an image to uint8 if it is a float image."""
    if isinstance(img, np.ndarray):
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
    elif isinstance(img, torch.Tensor):
        if img.is_floating_point():
            # Clamp tensor to [0, 1] before scaling, if necessary
            img = torch.clamp(img, 0., 1.)
            img = (255 * img).to(torch.uint8)
    return img


# --- Helper function for a single tensor (Internal use) ---
def _resize_with_pad_torch_single(
    image: torch.Tensor,
    height: int,
    width: int,
    interpolation: F_tv.InterpolationMode = F_tv.InterpolationMode.BILINEAR
) -> torch.Tensor:
    """Replicates tf.image.resize_with_pad for one image tensor using PyTorch.

    Args:
        image: A single image tensor in [H, W, C] or [C, H, W] format.
        height: The target height.
        width: The target width.
        interpolation: The interpolation method.

    Returns:
        The resized and padded image tensor.
    """
    # Assuming the tensor is in [H, W, C] or [C, H, W]
    # We must determine the dimensions. Standard in PyTorch is [C, H, W].

    if image.dim() != 3:
        raise ValueError("Input tensor must be 3-dimensional [H, W, C] or [C, H, W].")

    # Determine format: Check which dim is smallest (assuming color channel is C < H, W)
    # The torchvision/PyTorch resize function expects [C, H, W] for efficiency, so we convert.
    h, w, c = -1, -1, -1
    
    # Try to infer layout, assume [H, W, C] if the last dimension is 3 or 1 (common for custom code)
    if image.shape[-1] <= 4 and image.shape[-1] > 0:
        # Assumed [H, W, C] -> convert to [C, H, W]
        img_c_h_w = image.permute(2, 0, 1)
        c, h, w = img_c_h_w.shape
    else:
        # Assumed [C, H, W]
        img_c_h_w = image
        c, h, w = img_c_h_w.shape


    if w == width and h == height:
        return image.permute(1, 2, 0) if c == image.shape[-1] else image # Return to original layout

    # 1. Calculate ratio and new size
    ratio = max(w / width, h / height)
    resized_height = int(h / ratio)
    resized_width = int(w / ratio)

    # 2. Resize
    # F_tv.resize expects [C, H, W] or a PIL Image
    resized_tensor = F_tv.resize(
        img_c_h_w,
        size=(resized_height, resized_width),
        interpolation=interpolation,
        antialias=True # Better quality
    )

    # 3. Calculate padding
    pad_top = max(0, (height - resized_height) // 2)
    pad_bottom = max(0, height - resized_height - pad_top)
    pad_left = max(0, (width - resized_width) // 2)
    pad_right = max(0, width - resized_width - pad_left)

    # 4. Pad - F_nn.pad expects [N, C, H, W], so we unsqueeze the batch dimension
    # Padding order is (left, right, top, bottom)
    padded_tensor_c_h_w = F_nn.pad(
        resized_tensor.unsqueeze(0),
        (pad_left, pad_right, pad_top, pad_bottom),
        mode='constant',
        # value=0
        value=-1.0 # Assuming float image in [-1, 1]
    ).squeeze(0) # Remove batch dimension

    # Verify the final size
    final_c, final_h, final_w = padded_tensor_c_h_w.shape
    assert final_h == height and final_w == width

    # Return in the original layout (if the input was [H, W, C])
    if c == image.shape[-1]:
        return padded_tensor_c_h_w.permute(1, 2, 0)
    else:
        return padded_tensor_c_h_w


# --- Main function for a batch of tensors ---
def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    interpolation: F_tv.InterpolationMode = F_tv.InterpolationMode.BILINEAR,
    out_mode: str = 'HWC'
) -> torch.Tensor:
    """Replicates tf.image.resize_with_pad for multiple images using PyTorch.

    Resizes a batch of images to a target height and width without distortion by padding with zeros.
    The input images are expected to be in [..., H, W, C] format, matching the original NumPy/tf logic,
    but the internal processing will convert to [C, H, W] for efficiency.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        interpolation: The interpolation method to use (e.g., F_tv.InterpolationMode.BILINEAR).

    Returns:
        The resized images in [..., height, width, channel].
    """

    # if images.shape[-3] == 3:
    #     # Input is in [..., C, H, W] format, permute
    #     images = images.permute(*range(images.dim() - 3), -2, -1, -3)  # [..., H, W, C]

    if images.shape[-3] == 3:
        images = images.permute(0, 2, 3, 1)  # [B, H, W, C]

    if images.shape[-3:-1] == (height, width):
        return images # Already the correct size

    original_shape = images.shape  # * torch.Size([8, 192, 256, 3])

    # Flatten the batch dimensions into a single dimension for iteration
    num_images = images.shape[0] if images.dim() >= 3 else 1
    
    # Reshape to [N, H, W, C] where N is the flattened batch size
    flat_images = images.reshape(-1, *original_shape[-3:])

    resized_tensors = [
        _resize_with_pad_torch_single(im, height, width, interpolation=interpolation)
        for im in flat_images
    ]

    # Stack the results and reshape back to the original batch dimensions
    resized_batch = torch.stack(resized_tensors)

    # * [8, 224, 224, 3]
    assert out_mode in ['HWC', 'CHW'], "out_mode must be either 'HWC' or 'CHW'"

    if out_mode == 'CHW':
        resized_batch = resized_batch.permute(0, 3, 1, 2)

    return resized_batch


def write_episode_video_rm(
    episode_id: int,
    value_list: list[float],
    img_list: list[list[np.ndarray]],  # now list of image lists
    output_dir: str,
    fig_w: float = 12.0,
    fig_h: float = 4.0,
    dpi: int = 100,
    fps: int = 30,
    state_mode: str = None,
    advantage: float = None,
    ori_reward: float = None,
):
    """Writes a video for a single episode visualizing predicted values and image frames."""
    if not value_list:
        return 
    
    n_frames = len(value_list)
    
    os.makedirs(output_dir, exist_ok=True)

    # add date
    # current_time = pd.Timestamp.now().strftime("%d_%H%M%S")
    current_time = pd.Timestamp.now().strftime("%m%d_%H%M%S")

    out_path = os.path.join(output_dir, f"episode_{episode_id:03d}_{current_time}.mp4")

    print("!!!!!!Generating video to:", out_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')

    width_px = int(fig_w * dpi)
    height_px = int(fig_h * dpi)
    video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width_px, height_px))

    logging.info(f"Writing video for episode {episode_id} with {n_frames} frames...")

    for idx in tqdm.tqdm(range(n_frames), desc=f"Episode {episode_id} Video"):
        fig, axes = plt.subplots(1, 1 + len(img_list[idx]), figsize=(fig_w, fig_h), dpi=dpi)

        # Add main title showing state_mode and advantage
        title_parts = []
        if state_mode is not None:
            title_parts.append(f"State Mode: {state_mode}")
        if advantage is not None:
            title_parts.append(f"Advantage: {advantage:.4f}")
        if ori_reward is not None:
            title_parts.append(f"Ori Reward: {ori_reward:.4f}")
        
        if title_parts:
            main_title = " | ".join(title_parts)
            fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.98)

        ax_plot = axes[0]
        # Plotting
        x = np.arange(idx + 1)
        y = np.array(value_list[: idx + 1], dtype=np.float32)
        ax_plot.plot(x, y, linewidth=2, color="tab:blue")
        ax_plot.set_xlim(0, n_frames)
        
        # ax_plot.set_ylim(0.0, 1.0)
        ax_plot.set_ylim(-1.0, 1.0)
        
        ax_plot.set_xlabel("Frame")
        ax_plot.set_ylabel("Predicted Value")
        ax_plot.set_title("Value Prediction Over Time")
        ax_plot.grid(True)

        # Images: e.g., base, wrist_left, wrist_right
        views = img_list[idx]

        if len(views) > 3:
            titles = ["Start base", "Start Left", "Start Right", "Predicted base", "Predicted Left", "Predicted Right"]
        else:
            titles = ["Base Frame", "Wrist Left", "Wrist Right"]
        
        titles = titles[:len(views)]  # Adjust titles to match number of views

        for ax_img, view, title in zip(axes[1:], views, titles):
            ax_img.imshow(view)
            ax_img.set_title(title)
            ax_img.axis("off")

        # Adjust layout to make room for the main title
        if title_parts:
            plt.tight_layout(rect=[0, 0, 1, 0.95])
        else:
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
    # if os.path.exists(out_path):
    new_out_path = out_path.replace(".mp4", "_new.mp4")
    os.system(f"ffmpeg -y -i {out_path} -c:v libx264 -crf 18 -preset veryfast {new_out_path} > /dev/null 2>&1")
    logging.info(f"=> Episode {episode_id} generated to: {new_out_path}")

    if os.path.exists(out_path):
        os.remove(out_path)


def process_view_torch_rm(torch_arr):
    """
        Converts a single image tensor [-1, 1] to a NumPy array [0, 255].

        Args:

            torch_arr: Input tensor

        Returns:

            Converted NumPy array

    """
    arr = torch_arr.cpu().float().numpy()
    img = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    # CHW -> HWC
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    return img




def build_model_only(config_name: str, ckpt_dir: str):
    """Loads the training configuration based on the provided name."""
    config = _config.get_config(config_name)
    checkpoint_dir = download.maybe_download(ckpt_dir)


    # =============== Load Model =============================
    new_model = config.model.__class__(**{**config.model.__dict__,
                                            'p_mask_ego_state': 1,
                                            'value_TD_learning': False,
                                        })
    
    config = dataclasses.replace(config, model=new_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PI0Pytorch(new_model).to(device)
    
    model.eval()  # Set model to evaluation mode
    model_path = os.path.join(checkpoint_dir, "model.safetensors")
    logging.info(f"Loading weights from: {model_path}")
    try:
        safetensors.torch.load_model(model, model_path, strict=False)
    except FileNotFoundError:
        logging.error(f"Could not find model weights at {model_path}")
        # Fallback to config path if specified
        if config.pytorch_weight_path:
            model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
            logging.info(f"Trying fallback path: {model_path}")
            safetensors.torch.load_model(model, model_path, strict=False)
        else:
            raise

    logging.info(f"Loaded PyTorch weights successfully.")
    # =============== Load Model finished ======================
    config = dataclasses.replace(
        config,
        batch_size=1, 
        is_train=False,
        num_workers=8,
        split='all',
        
        use_suboptimal_progress=False,
        
        suboptimal_progress_multiplier=1,
        suboptimal_progress_offset=0,
        
        # preceding_skipping_ratio=0,
        # preceding_skipping_ratio=0.2,   # * Skip preceding 20%.
        preceding_skipping_ratio=0.,   # * Skip preceding 20%.
    )
   
    return model, config



class RewardModel:
    def __init__(self, config_name: str, ckpt_dir: str, split: str = "val_tasks", metric_only: bool = False, output_video_dir: str = "./visualizations"):
        """
            Initialize the RewardModel class

            Args:

                config_name: Configuration name

                ckpt_dir: Checkpoint directory

                split: Dataset split, defaults to "val_tasks"

                metric_only: Whether to calculate only metrics without generating videos, defaults to False

                output_video_dir: Video output directory, defaults to "./visualizations"
        """
        self.config_name = config_name
        self.ckpt_dir = ckpt_dir
        self.split = split
        self.metric_only = metric_only
        self.output_video_dir = output_video_dir
        
        self.all_pred = []
        self.all_tgt = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.model, self.config = build_model_only(config_name, ckpt_dir)
        
        # self.config = self._load_config()
        # self.model = self._load_model()
        
        self.tokenizer = _tokenizer.PaligemmaTokenizer(self.config.model.max_token_len)

    # * For a paired observation
    def predict_reward(self, observation, prompts):
        """
            Predicted values ​​(batch processing supported)

            Args:

                observation: Dictionary of observation data, which can be a single observation or a batch of observations.

                prompts: Prompt text, which can be a single string or a list of strings.

            Returns:

                Predicted values, which can be a single value or a list of values.
        """
        # Check the input type to determine if it is batch processing.
        is_batch = self._check_batch_input(observation, prompts)
        
        # Processing prompts
        if is_batch:
            # Batch processing: Convert the prompts into batch tokens
            batch_size = len(prompts)
            tokens_list = []
            token_masks_list = []
            
            for prompt in prompts:
                tokens, token_masks = self.tokenizer.tokenize(prompt, state=None)
                tokens_list.append(tokens)
                token_masks_list.append(token_masks)
            
            tokens = np.stack(tokens_list, axis=0)
            token_masks = np.stack(token_masks_list, axis=0)
            
        else:
            # Single sample processing
            batch_size = 1
            prompts = prompts if isinstance(prompts, str) else prompts[0]
            tokens, token_masks = self.tokenizer.tokenize(prompts, state=None)
            tokens = tokens[np.newaxis, :]  # Add batch dimension
            token_masks = token_masks[np.newaxis, :]  # Add batch dimension

        # Convert observation data to SimpleNamespace
        if is_batch:
            # Batch processing: Ensure all observations have the correct batch dimension
            observation = self._prepare_batch_observation(observation, batch_size)
        
        observation = {**observation, 
                       "tokenized_prompt": torch.from_numpy(tokens).to(self.device), 
                       "tokenized_prompt_mask": torch.from_numpy(token_masks).to(self.device)}
        observation = SimpleNamespace(**observation)
        

        with torch.no_grad():
            
            val_arr = self.model.sample_values(self.device, observation)  # shape=(batch_size, 1)

            if is_batch:
                values = [float(val.item()) for val in val_arr[:, 0]]
            else:
                values = [float(val_arr[0, 0].item())]
        
        return values
    
    def _check_batch_input(self, observation, prompts):
        """
            Check if the input is batch input.

            Args:

                observation: Observed data

                prompts: Prompt text

            Returns:

                Return True if it is batch input, otherwise return False.
        """
        # If prompts is a list with a length greater than 1, it is considered batch input.
        if isinstance(prompts, list) and len(prompts) > 1:
            return True
        
        # If the state in the observation has a batch dimension greater than 1, it is also considered as batch input.
        if "state" in observation and isinstance(observation["state"], torch.Tensor):
            if observation["state"].shape[0] > 1:
                return True
        
        return False
    
    def _prepare_batch_observation(self, observation, batch_size):
        """
            Prepare batch observation data, ensuring all tensors have the correct batch dimension.

            Args:

                observation: Observation data

                batch_size: Batch size

            Returns:

                Prepared batch observation data
        """
        batch_observation = {}
        
        if "state" in observation:
            state = observation["state"]
            if state.shape[0] != batch_size:
                raise ValueError(f"State batch size {state.shape[0]} does not match prompt batch size {batch_size}")
            batch_observation["state"] = state.to(self.device)
        
        if "images" in observation:
            batch_observation["images"] = {}
            for img_key, img_tensor in observation["images"].items():
                if img_tensor.shape[0] != batch_size:
                    raise ValueError(f"Image {img_key} batch size {img_tensor.shape[0]} does not match prompt batch size {batch_size}")
                batch_observation["images"][img_key] = img_tensor.to(self.device)
        
        if "image_masks" in observation:
            batch_observation["image_masks"] = {}
            for mask_key, mask_tensor in observation["image_masks"].items():
                if mask_tensor is not None and isinstance(mask_tensor, torch.Tensor):
                    if mask_tensor.shape[0] != batch_size:
                        raise ValueError(f"Image mask {mask_key} batch size {mask_tensor.shape[0]} does not match prompt batch size {batch_size}")
                    batch_observation["image_masks"][mask_key] = mask_tensor.to(self.device)
                else:
                    batch_observation["image_masks"][mask_key] = mask_tensor
        
        return batch_observation
    
  
    
    def get_metrics(self):

        if not self.all_pred or not self.all_tgt:
            return None
        
        return {
            "predictions": self.all_pred,
            "targets": self.all_tgt,
            "mse": np.mean((np.array(self.all_pred) - np.array(self.all_tgt)) ** 2)
        }

def extract_view_tensors(video_path):
    cap = cv2.VideoCapture(video_path)
    

    list_left, list_mid, list_right = [], [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. OpenCV defaults to BGR, convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. Segment the image into width, width, and channel.
        h, w, c = frame.shape
        w_single = w // 3

        img_l = frame[:, :w_single, :]
        img_m = frame[:, w_single : 2*w_single, :]
        img_r = frame[:, 2*w_single :, :]

        list_left.append(img_l)
        list_mid.append(img_m)
        list_right.append(img_r)

    cap.release()

    # 3. Converts a list into a Tensor of (Time, Channel, Height, Width).
    def list_to_tensor(frame_list):
        # (T, H, W, C)
        np_frames = np.stack(frame_list)
        # convert to Tensor
        tensor = torch.from_numpy(np_frames)
        tensor = (tensor.float() / 127.5) - 1.0
        # (T, H, W, C) -> (T, C, H, W)
        return tensor.permute(0, 3, 1, 2)

    return list_to_tensor(list_left), list_to_tensor(list_mid), list_to_tensor(list_right)

def tensor_to_numpy_image(tensor):
    """
        Convert a (C, H, W) Tensor of [-1, 1] to a (H, W, C) BGR Numpy image of [0, 255].
        For saving video using OpenCV.
    """
    # 1. [-1, 1] -> [0, 1]
    img = (tensor.detach().cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0
    # 2. Mapped to [0, 255]
    img = (img * 255).clip(0, 255).astype(np.uint8)
    # 3. RGB -> BGR (OpenCV needs BGR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def plot_reward_curve(rewards, frames_indices, target_height):
    """
        Plot curves using Matplotlib and convert them to OpenCV images.
    """
    fig = plt.figure(figsize=(6, 4), dpi=100)
    plt.plot(frames_indices, rewards, color='red', linewidth=2, marker='o', markersize=4)
    plt.title(f"Current Reward: {rewards[-1]:.4f}")
    plt.xlabel("Frame")
    plt.ylabel("Reward Value")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Convert a Matplotlib canvas to an RGB image array
    fig.canvas.draw()
    img_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8) # uv pip install matplotlib==3.7.0
    img_plot = img_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    # RGB -> BGR
    img_plot = cv2.cvtColor(img_plot, cv2.COLOR_RGB2BGR)
    
    # Resize to match video height
    h, w, _ = img_plot.shape
    scale = target_height / h
    new_w = int(w * scale)
    img_plot_resized = cv2.resize(img_plot, (new_w, target_height))
    
    return img_plot_resized

    
    
