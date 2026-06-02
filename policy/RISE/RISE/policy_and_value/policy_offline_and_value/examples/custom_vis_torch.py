import dataclasses
import os
import cv2
import jax
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
from openpi_value.policies import policy_config
from openpi_value.shared import download
import openpi_value.training.data_loader as _data
from openpi_value.models_pytorch.pi0_pytorch import PI0Pytorch

all_pred = []
all_tgt  = []


# TODO: Viusalize one sample case. (Two frames.)

# * Visualize the dataset
def build_datasets(config: _config.TrainConfig, shuffle: bool = True):
    """Builds the data loader with configurable shuffling."""
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(
        config, 
        framework="pytorch", 
        shuffle=shuffle,
        skip_norm_stats=config.skip_norm_stats,
    )
    return data_loader, data_loader.data_config()


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

    # --- Start Modification: New Advantage Formulation ---
    # Formula: Mean of (V(t-k) - V(t-50)) for k in 0..49
    # This is equivalent to: Mean(V(t)...V(t-49)) - V(t-50)
    # When t < 50, we only use available values (no padding with V(0))
    
    adv_list = []

    ADVANTAGE_SCALAR = 5.   # * Conveyor
    # ADVANTAGE_SCALAR = 10.  # * Backpack and Box.

    # MODE in ["v1", "v2"]
    # MODE = "v2"  
    MODE = "v1"  # * Let's try v1
    
    # MODE = "v1"  # * Let's roll back to v1 --> v1 seems flattener than v2
    
    if MODE == "v1":
        # * mean(t+1,...,t+50) - v(t)
        window_size = 50
        for t in range(n_frames):
            # 1. Identify the reference value V(t-50)
            # When t < window_size, use V(0) as reference
            ref_idx = max(0, t - window_size)
            ref_val = val_pred[ref_idx]

            # 2. Identify the window values V(t), V(t-1) ... V(t-49)
            # Only use indices that are >= 0 (no negative indices)
            # When t < window_size, we'll have fewer than 50 values
            window_start_idx = max(0, t - window_size + 1)
            window_indices = list(range(t, window_start_idx - 1, -1))  # [t, t-1, ..., max(0, t-49)]
            window_vals = val_pred[window_indices]

            # 3. Calculate Mean
            # Mathematical simplification: Mean(Window - Ref) = Mean(Window) - Ref
            avg_window = np.mean(window_vals)
            adv_raw = avg_window - ref_val
            
            adv_raw = adv_raw * ADVANTAGE_SCALAR

            # 4. Clamp to [-1, 1]
            adv = max(-1.0, min(adv_raw, 1.0))
            # adv = min(adv_raw, 1.0)
            
            
            adv_list.append(adv)

    elif MODE == "v2":
        # for a chunk [t, t+1, ...., t+50]
        # use the mean of last 5 frames to subtract the mean of the first 5 frames
        chunk_size = 50  # 50 frames
        n_first = 5  # First 5 frames
        n_last = 5   # Last 5 frames
        
        for t in range(n_frames):
            # Define chunk starting at t: [t, t+1, ..., t+chunk_size-1] (50 frames total)
            chunk_end = min(t + chunk_size, n_frames)
            
            # Get first 5 frames: [t, t+1, t+2, t+3, t+4]
            first_end = min(t + n_first, chunk_end)
            first_vals = val_pred[t:first_end]
            
            # Get last 5 frames: [chunk_end-5, chunk_end-4, ..., chunk_end-1]
            last_start = max(t, chunk_end - n_last)
            last_vals = val_pred[last_start:chunk_end]
            
            # Calculate advantage: mean(last_5) - mean(first_5)
            if len(first_vals) > 0 and len(last_vals) > 0:
                avg_first = np.mean(first_vals)
                avg_last = np.mean(last_vals)
                adv_raw = avg_last - avg_first
            else:
                # Edge case: not enough frames, set advantage to 0
                adv_raw = 0.0
            
            
            adv_raw = adv_raw * ADVANTAGE_SCALAR
            
            # Clamp to [-1, 1]
            adv = max(-1.0, min(adv_raw, 1.0))
            
            adv_list.append(adv)

        
    
    adv_pred = np.array(adv_list, dtype=np.float32)
    
    DUMP = False
    if DUMP:
        # * TODO: dump adv_list into local file
        adv_dump_dir = os.path.join(output_dir, "adv_dumps")
        os.makedirs(adv_dump_dir, exist_ok=True)
        adv_dump_path = os.path.join(adv_dump_dir, f"episode_{episode_id:03d}_adv.npy")
        np.save(adv_dump_path, np.array(adv_list, dtype=np.float32))
        # dump val_list
        np.save(os.path.join(adv_dump_dir, f"episode_{episode_id:03d}_val.npy"), np.array(val_pred, dtype=np.float32))

    # --- End Modification ---

    global all_pred
    global all_tgt

    all_pred.extend(val_pred.tolist())
    all_tgt.extend(tgt_progress.tolist())

    if metric_only:
        return

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"episode_{episode_id:03d}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width_px = int(fig_w * dpi)
    height_px = int(fig_h * dpi)
    video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width_px, height_px))

    logging.info(f"Writing video for episode {episode_id} with {n_frames} frames...")

    for idx in tqdm.tqdm(range(n_frames), desc=f"Episode {episode_id} Video"):
        fig, axes = plt.subplots(1, 1 + len(img_list[idx]), figsize=(fig_w, fig_h), dpi=dpi)

        ax_plot = axes[0]
        # Plotting
        x = np.arange(idx + 1)
        
        # 1. Plot Value (Blue)
        y_val = val_pred[: idx + 1]
        ax_plot.plot(x, y_val, linewidth=2, color="tab:blue", label="Value")

        # 2. Plot Advantage (Orange)
        y_adv = adv_pred[: idx + 1]
        ax_plot.plot(x, y_adv, linewidth=2, color="tab:orange", label="Advantage")

        ax_plot.set_xlim(0, n_frames)
        ax_plot.set_ylim(-1., 1.)
        
        ax_plot.set_xlabel("Frame")
        ax_plot.set_ylabel("Predicted Value / Advantage")
        ax_plot.set_title("Value & Advantage Over Time")
        ax_plot.grid(True)
        ax_plot.legend(loc="upper left", fontsize='small')

        # Images: e.g., base, wrist_left, wrist_right
        views = img_list[idx]
        titles = ["Base Frame", "Wrist Left", "Wrist Right"]
        titles = titles[:len(views)]  # Adjust titles to match number of views

        for ax_img, view, title in zip(axes[1:], views, titles):
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
    
    model.sample_values = torch.compile(model.sample_values, mode="reduce-overhead")
    
    # except:
    #     pass  # torch.compile may fail in some environments
    
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
        split=split,
        
        use_suboptimal_progress=False,
        
        suboptimal_progress_multiplier=1,
        suboptimal_progress_offset=0,
        
        preceding_skipping_ratio=0,
    )
    return model, config
    
    

def build_model_and_dataset(config_name: str, checkpoint_dir:str, device=None):
    """Loads the training configuration based on the provided name."""
    config = _config.get_config(config_name)
    # checkpoint_dir = download.maybe_download(ckpt_dir)


    # =============== Load Model =============================
    new_model = config.model.__class__(**{**config.model.__dict__,
                                            'p_mask_ego_state': 1,
                                            'value_TD_learning': False,
                                        })
    
    config = dataclasses.replace(config, model=new_model)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PI0Pytorch(new_model).to(device)
    
    model.sample_values = torch.compile(model.sample_values, mode="reduce-overhead")
    
    # except:
    #     pass  # torch.compile may fail in some environments
    
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
    
    # --- Modify Config for Inference ---
    # assert split in ['all', 'val_tasks', 'heldout_tasks']
    config = dataclasses.replace(
        config,
        batch_size=1, 
        is_train=False,
        num_workers=8,
        # split='val_tasks',
        split='all',
        
        use_suboptimal_progress=False,
        
        suboptimal_progress_multiplier=1,
        suboptimal_progress_offset=0,
        
        preceding_skipping_ratio=0,
    )
    # --- Data Loader ---
    # Must use shuffle=False to get contiguous episodes
    loader, _ = build_datasets(config, shuffle=False)
    data_iter = iter(loader)
    return model, data_iter
    

def main(
    config_name: str,
    ckpt_dir: str,
    split: str = "val_tasks",
    metric_only: bool = False,
    output_video_dir: str = "./visualizations",
    headview_only: bool = False,
    n_batches: int = 20000,
    fps_downsample: int = 1,
):
    """Main function to run value prediction and visualization."""
    
    global all_pred
    global all_tgt
    

    # # --- Config and Checkpoint Setup ---
    # config = _config.get_config(config_name)
    checkpoint_dir = download.maybe_download(ckpt_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    model, data_iter = build_model_and_dataset(config_name, checkpoint_dir, device=device)


    # # =============== Load Model =============================
    # new_model = config.model.__class__(**{**config.model.__dict__,
    #                                         'p_mask_ego_state': 1,
    #                                         'value_TD_learning': False,
    #                                     })
    
    # config = dataclasses.replace(config, model=new_model)
    # model = PI0Pytorch(new_model).to(device)
    
    # model.sample_values = torch.compile(model.sample_values, mode="reduce-overhead")
    
    # # except:
    # #     pass  # torch.compile may fail in some environments
    
    # model.eval()  # Set model to evaluation mode
    # model_path = os.path.join(checkpoint_dir, "model.safetensors")
    # logging.info(f"Loading weights from: {model_path}")
    # try:
    #     safetensors.torch.load_model(model, model_path, strict=False)
    # except FileNotFoundError:
    #     logging.error(f"Could not find model weights at {model_path}")
    #     # Fallback to config path if specified
    #     if config.pytorch_weight_path:
    #         model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
    #         logging.info(f"Trying fallback path: {model_path}")
    #         safetensors.torch.load_model(model, model_path, strict=False)
    #     else:
    #         raise

    # logging.info(f"Loaded PyTorch weights successfully.")
    # # =============== Load Model finished ======================
    
    # # --- Modify Config for Inference ---
    # assert split in ['all', 'val_tasks', 'heldout_tasks']
    # config = dataclasses.replace(
    #     config,
    #     batch_size=1, 
    #     is_train=False,
    #     num_workers=8,
    #     split=split,
        
    #     use_suboptimal_progress=False,
        
    #     suboptimal_progress_multiplier=1,
    #     suboptimal_progress_offset=0,
        
    #     # preceding_skipping_ratio=0,
    #     # preceding_skipping_ratio=0.2,   # * Skip preceding 20%.
    #     preceding_skipping_ratio=0.,   # * Skip preceding 20%.
    # )
    
    #     config = dataclasses.replace(config, 
                                     
    #                                  with_episode_start=True,  # * Always use start frame as first frame in a tuple
    

    # # --- Data Loader ---
    # # Must use shuffle=False to get contiguous episodes
    # loader, _ = build_datasets(config, shuffle=False)
    # data_iter = iter(loader)


    # --- Helper for Image Processing ---
    def process_view_torch(torch_arr):
        """Converts a single image tensor [-1, 1] to a numpy array [0, 255]."""
        arr = torch_arr.cpu().float().numpy()
        img = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        # Assuming CHW -> HWC
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        return img

    observation0, _ = next(data_iter)
    prev_frame_index = int(observation0.frame_index.item())

    episode_counter = 0
    value_frames = []  # type: list[float]
    raw_frames = []    # type: list[list[np.ndarray]]
    
    batch_cnt = 0
    fps = 30

    # Setup output directory
    output_dir = os.path.join(output_video_dir, config_name, os.path.basename(checkpoint_dir) + f"_{split}")
    print(f">>>>>> Generating episode videos to: {output_dir} ...")
    os.makedirs(output_dir, exist_ok=True)

    # --- Main Episode Loop ---
    for observation, actions in tqdm.tqdm(data_iter, total=n_batches, desc="Processing batches"):
        
        batch_cnt += 1
        if batch_cnt > n_batches:
            break

        if not metric_only and batch_cnt % fps_downsample != 0:
            continue

        cur_frame_idx = int(observation.frame_index.item())

        # New episode boundary
        if cur_frame_idx <= prev_frame_index:
            logging.info(f"Detected episode boundary. Writing video for episode {episode_counter}.")
            write_episode_video(
                episode_id=episode_counter,
                value_list=value_frames,
                img_list=raw_frames,
                output_dir=output_dir,
                fig_w=12.0,
                fig_h=4.0,
                dpi=100,
                fps=fps,
                metric_only=metric_only,
            )
            episode_counter += 1
            value_frames.clear()
            raw_frames.clear()

        # Run model inference
        with torch.no_grad():
            observation_dev = jax.tree.map(lambda x: x.to(device), observation)

            # Predict value
            val_arr = model.sample_values(device, observation_dev)  # Shape=(1, 1)
            val = float(val_arr[0, 0].item())

        # Store value
        value_frames.append(val)

        # Store images for video
        if not metric_only:
            base_torch = observation.images["base_0_rgb"][0]
            img_base = process_view_torch(base_torch)
            current_views = [img_base] # Start with base view

            if not headview_only:
                if "left_wrist_0_rgb" in observation.images:
                    left_torch = observation.images["left_wrist_0_rgb"][0]
                    img_left = process_view_torch(left_torch)
                    current_views.append(img_left)
                
                if "right_wrist_0_rgb" in observation.images:
                    right_torch = observation.images["right_wrist_0_rgb"][0]
                    img_right = process_view_torch(right_torch)
                    current_views.append(img_right)
            
            raw_frames.append(current_views)

        prev_frame_index = cur_frame_idx

    # Write the very last episode
    if len(value_frames) != 0:
        logging.info(f"Writing final episode {episode_counter}.")
        write_episode_video(
            episode_id=episode_counter,
            value_list=value_frames,
            img_list=raw_frames,
            output_dir=output_dir,
            fig_w=12.0,
            fig_h=4.0,
            dpi=100,
            fps=fps,
            metric_only=metric_only,
        )




    # --- Final Metric Calculation ---
    all_pred = np.array(all_pred)
    all_tgt = np.array(all_tgt)
    if len(all_pred) > 0:
        avg_mae = np.mean(np.abs(all_pred - all_tgt))
        print("\n" + "="*50)
        print(f"Config/Iter: {config_name}/{os.path.basename(checkpoint_dir)}")
        print(f"Split: {split}")
        print(f"Average Value Prediction MAE: {avg_mae:.4f}")
        print("="*50 + "\n")
    else:
        print("No predictions were made.")

    print(f"<<<<<< Finished generating visualizations to: {output_dir}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
