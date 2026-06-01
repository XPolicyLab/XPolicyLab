import dataclasses
import os
import cv2
import jax
import numpy as np
import tqdm
import matplotlib
import io
import shutil  # Added for cleanup
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
from openpi_value.training.custom_lerobot_dataset import LeRobotDatasetMetadata

import pyarrow as pa
import pyarrow.parquet as pq
import json

matplotlib.use("Agg")

all_pred = []
all_tgt  = []

# TODO: Single frame version

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

    # --- Start Modification: Calculate Advantage ---
    # Advantage(t) = min((V(t) - V(t-50)) * 10, 1)
    # We treat V(t-50) as V(0) if t < 50 (clamping to start)
    adv_list = []
    for t in range(n_frames):
        prev_val = val_pred[max(0, t - 50)]
        curr_val = val_pred[t]
        adv = min((curr_val - prev_val) * 10.0, 1.0)
        adv_list.append(adv)
    
    adv_pred = np.array(adv_list, dtype=np.float32)
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
        
        # 1. Plot Value (Original Blue)
        y_val = val_pred[: idx + 1]
        ax_plot.plot(x, y_val, linewidth=2, color="tab:blue", label="Value")

        # 2. Plot Advantage (New Orange)
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

def get_all_parquet_files(data_root: str) -> list[str]:
    """Recursively finds all .parquet files within the given data_root."""
    parquet_files = []
    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    return parquet_files

# * Changed: Reads from disk (.npy) instead of global dict to prevent OOM
def add_column_from_disk(data_root: str, 
                         add_column_name: str,
                         temp_value_dir: str,  # Path to temp npy files
                         output_root: str):
    """
    Iterates over all Parquet files, loads corresponding values from disk, 
    and adds the new column.
    """
    parquet_files = get_all_parquet_files(data_root)
    # * sort 
    parquet_files.sort()

    os.makedirs(output_root, exist_ok=True)

    for file_path in tqdm.tqdm(parquet_files, desc="Processing parquet files"):
        relative_path = os.path.relpath(file_path, data_root)
        output_file_path = os.path.join(output_root, relative_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        try:
            # 1. Load the original parquet table
            parquet_table = pq.read_table(file_path)

            # Get Episode Index (assuming ..._episode_X.parquet format)
            try:
                ep_ind = int(file_path.split('.')[-2].split('_')[-1])
            except:
                # Fallback for complex filenames
                import re
                ep_ind = int(re.search(r'\d+', os.path.basename(file_path)).group())
            
            # Load values from temporary disk storage
            npy_path = os.path.join(temp_value_dir, f"ep_{ep_ind}.npy")
            
            if os.path.exists(npy_path):
                ep_values_np = np.load(npy_path)
                
                # Check length consistency
                if len(ep_values_np) != parquet_table.num_rows:
                     # Truncate to safe length
                    min_len = min(len(ep_values_np), parquet_table.num_rows)
                    ep_values_np = ep_values_np[:min_len]
                    parquet_table = parquet_table.slice(0, min_len)

                ep_values = pa.array(ep_values_np)

                # 2. Append Column
                new_table = parquet_table.append_column(
                    add_column_name,
                    ep_values
                )

                # 4. Write to temp
                temp_output_path = output_file_path + ".tmp"
                pq.write_table(new_table, temp_output_path)
                
                # 5. Atomic replace
                os.replace(temp_output_path, output_file_path)
            else:
                # If no predictions found (e.g. filtered by split), just copy the file? 
                # Or skip. Here we skip modifying if no data exists.
                pass
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")


def smooth_value_and_compute_advantage(
    data_root: str, output_root: str, n_smooth: int = 3, chunk_size: int = 50, advantage_scaler: float = 1.0
):
    """
    Iterates over all Parquet files, loads the 'frame_value' column, 
    computes 'frame_value_smooth' and 'action_advantage', and saves 
    the modified table to the output_root.
    """
    parquet_files = get_all_parquet_files(data_root)
    os.makedirs(output_root, exist_ok=True)
    
    # Check for even N to handle smoothing window correctly (N must be odd for center-based average)
    if n_smooth % 2 == 0:
        tqdm.write(f"Warning: Smoothing window N={n_smooth} should ideally be odd. Using N={n_smooth}.")

    for file_path in tqdm.tqdm(parquet_files, desc="Processing parquet files"):
        relative_path = os.path.relpath(file_path, data_root)
        output_file_path = os.path.join(output_root, relative_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        try:
            # 1. Load the original parquet table and get 'frame_value'
            parquet_table = pq.read_table(file_path)
            
            # Convert PyArrow array to NumPy array for efficient calculation
            if 'frame_value' not in parquet_table.column_names:
                tqdm.write(f"Skipping {file_path}: 'frame_value' column not found.")
                continue
                
            frame_values_pa = parquet_table['frame_value']
            # Cast to float64 for calculation stability
            frame_values_np = frame_values_pa.to_numpy(zero_copy_only=False).astype(np.float64)
            num_rows = len(frame_values_np)
            
            # --- Step 1: Smooth per-frame value ---
            # frame_value_smooth(t) = avg(frame_value[t - floor(N/2) : t + floor(N/2) + 1])
            
            # Initialize smoothed array
            WITH_SMOOTH = False
            
            if WITH_SMOOTH:
                frame_value_smooth = np.zeros(num_rows, dtype=np.float64)
                half_n = n_smooth // 2
                
                for t in range(num_rows):
                    # Define the window boundaries, clamped to the array size
                    start_idx = max(0, t - half_n)
                    end_idx = min(num_rows, t + half_n + 1)
                    
                    # Calculate the average over the valid window
                    frame_value_smooth[t] = np.mean(frame_values_np[start_idx:end_idx])
            else:
                frame_value_smooth = frame_values_np.copy()

            # --- Step 2: Compute action_advantage ---
            # action_advantage(t) = frame_value_smooth(t + K) - frame_value_smooth(t)
            
            action_advantage = np.zeros(num_rows, dtype=np.float64)
            
            # For time steps t where t + K is within bounds
            valid_range = num_rows - chunk_size

            MODE = "v1"  # in ["v1", "v2"]   # * V1 is always better?
            
            assert MODE == "v1"
            
            if MODE == "v1":
                # Process ALL frames, including incomplete chunks at the end
                for t in range(num_rows):
                    # Calculate the sum of differences for the current timestep t
                    # For incomplete chunks, use available frames up to the end
                    available_steps = min(chunk_size, num_rows - t - 1)
                    
                    if available_steps > 0:
                        advantage = 0
                        for k in range(1, available_steps + 1):  # t+1 to t+available_steps
                            advantage += frame_value_smooth[t+k] - frame_value_smooth[t]
                        
                        # Normalize by available_steps (not chunk_size) to maintain scale consistency
                        action_advantage[t] = advantage / available_steps
                    else:
                        # No future frames available, set advantage to 0
                        action_advantage[t] = 0.0
            
            elif MODE == "v2":
                # * use the average value for last few frames to subtract the average value for first few frames
                # For a chunk [t, t+1, ..., t+chunk_size-1], calculate: mean(last_few) - mean(first_few)
                n_first = n_last = 5
                
                # Process ALL frames, including incomplete chunks at the end
                for t in range(num_rows):
                    # Define chunk starting at t: [t, t+1, ..., t+chunk_size-1] (chunk_size frames total)
                    # Ensure chunk_end doesn't exceed num_rows
                    chunk_end = min(t + chunk_size, num_rows)
                    chunk_length = chunk_end - t
                    
                    if chunk_length <= 1:
                        # Only 0 or 1 frame available, cannot compute advantage
                        action_advantage[t] = 0.0
                        continue
                    
                    # For incomplete chunks, adapt window sizes based on available frames
                    # We want to use equal-sized windows for first and last, ensuring they don't overlap
                    # Use at most half the chunk length for each window
                    max_window_size = min(n_first, n_last, chunk_length // 2)
                    
                    if max_window_size < 1:
                        # Very short chunk (2 frames), use simple difference
                        action_advantage[t] = frame_value_smooth[chunk_end - 1] - frame_value_smooth[t]
                    else:
                        # Get first few frames: [t, t+1, ..., t+max_window_size-1]
                        first_vals = frame_value_smooth[t:t + max_window_size]
                        
                        # Get last few frames: [chunk_end - max_window_size, ..., chunk_end - 1]
                        last_vals = frame_value_smooth[chunk_end - max_window_size:chunk_end]
                        
                        # Calculate advantage: mean(last_few) - mean(first_few)
                        avg_first = np.mean(first_vals)
                        avg_last = np.mean(last_vals)
                        action_advantage[t] = avg_last - avg_first

            action_advantage = action_advantage * advantage_scaler

            # * clamp action into range [-1, 1]
            
            WIHT_CLIP = False
            # ! clip is less convinient for post processing
            if WIHT_CLIP:
                action_advantage = np.clip(action_advantage, -1.0, 1.0)

            
            # The last K frames cannot look K steps ahead, so their advantage remains 0
            # (or some other defined padding value, 0 is common for terminal states/chunks)


            # 3. Convert NumPy arrays back to PyArrow arrays
            # Use pa.float64() for the new columns
            smooth_column_array = pa.array(frame_value_smooth, type=pa.float64())
            advantage_column_array = pa.array(action_advantage, type=pa.float64())
            
            # 4. Add the new columns
            new_table = parquet_table.append_column(
                'frame_value_smooth',
                smooth_column_array
            )
            new_table = new_table.append_column(
                'action_advantage',
                advantage_column_array
            )

            # 5. Write and atomically replace the file
            temp_output_path = output_file_path + ".tmp"
            pq.write_table(new_table, temp_output_path)
            os.replace(temp_output_path, output_file_path)
            
        except Exception as e:
            tqdm.write(f"Error processing file {file_path}: {e}")



def deal_mata(data_root: str, output_root: str):
    
    
    os.system(f"cp -r {os.path.join(data_root, 'meta')} {output_root}")
    
    # * append advantage into meta/info.json
    with open(os.path.join(output_root, 'meta', 'info.json'), 'r') as f:
        meta_info = json.load(f)
        meta_info['features']['action_advantage'] = {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": None
        }
    # * dump back to file
    
    # os.makedirs(os.path.join(output_root, 'meta'), exist_ok=True)
    with open(os.path.join(output_root, 'meta', 'info.json'), 'w') as f:
        json.dump(meta_info, f, indent=4)


def soft_link_video(data_root: str, output_root: str):
    video_dir = os.path.join(data_root, "videos")
    
    # remove videos suffix if exists
    if output_root.endswith('/videos'):
        output_root = output_root.replace('/videos', '')
    
    try:
        os.system(f"ln -s {video_dir} {output_root}")
    except:
        pass


def main(
    config_name: str,
    ckpt_dir: str,
    split: str = "val_tasks",
    metric_only: bool = False,
    output_video_dir: str = "./visualizations",
    headview_only: bool = False,
    with_vis: bool = True,
    batch_size: int = 64, # * New: Increased batch size
    max_vis_episodes: int = 3, # * New: Limit visualizations to save time
):
    """Main function to run value prediction and visualization."""

    # * Removed frame_to_value dict to save memory
    temp_val_dir = f"./temp_values_{config_name}"
    
    
    if os.path.exists(temp_val_dir):
        shutil.rmtree(temp_val_dir)
    os.makedirs(temp_val_dir, exist_ok=True)
    
    global all_pred
    global all_tgt

    # --- Config and Checkpoint Setup ---
    config = _config.get_config(config_name)
    checkpoint_dir = download.maybe_download(ckpt_dir)


    # * =============== Load Model =============================
    new_model = config.model.__class__(**{**config.model.__dict__,
                                            'p_mask_ego_state': 1,
                                            'value_TD_learning': False,
                                        })
    
    config = dataclasses.replace(config, model=new_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PI0Pytorch(new_model).to(device)
    
    model.sample_values = torch.compile(model.sample_values, mode="reduce-overhead")
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
    # * =============== Load Model finished ======================
    
    # * =============== Initialize Data =============================
    assert split in ['all', 'val_tasks', 'heldout_tasks']
    config_for_data = dataclasses.replace(
        config,
        batch_size=batch_size, # * Changed from 1 to user arg
        is_train=False,
        num_workers=8, 
        split=split,
        preceding_skipping_ratio=0,

        use_suboptimal_progress=False,

        drop_last=False, # * Ensure we do not drop last batch
        
        suboptimal_progress_multiplier=1,
        suboptimal_progress_offset=0,
        
        # preceding_skipping_ratio=0,
        # preceding_skipping_ratio=0.2,   # * Skip preceding 20%.
    )
    # --- Data Loader ---
    # Must use shuffle=False to get contiguous episodes
    loader, data_config = build_datasets(config_for_data, shuffle=False)

    # * =============== Initialize Data Finished =============================


    # --- Helper for Image Processing ---
    def process_view_torch(torch_arr):
        """Converts a single image tensor [-1, 1] to a numpy array [0, 255]."""
        arr = torch_arr.cpu().float().numpy()
        img = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        # Assuming CHW -> HWC
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        return img


    prev_frame_index = None

    value_frames = []  # type: list[float]
    raw_frames = []    # type: list[list[np.ndarray]]
    
    fps = 30
    vis_count = 0 # * Track number of visualized episodes

    # Setup output directory
    output_dir = os.path.join(output_video_dir, config_name, os.path.basename(checkpoint_dir) + f"_{split}")
    print(f">>>>>> Generating episode videos to: {output_dir} ...")
    os.makedirs(output_dir, exist_ok=True)

    repo_id = data_config.repo_id
    if isinstance(repo_id, list):
        assert len(repo_id) == 1, "Multiple repo_ids not supported in this script."
        repo_id = repo_id[0]
    
    meta = LeRobotDatasetMetadata(repo_id)
    num_frames = meta.total_frames

    num_batches = (num_frames + batch_size - 1) // batch_size


    i_bs = 0
    print(num_batches)
    
    for batch_observation, batch_actions in tqdm.tqdm(loader, total=num_batches, desc="Processing batches"):

        print(f"Processing batch {i_bs+1}/{num_batches}...")
        i_bs += 1

        if i_bs > num_batches:
            break
        
        # * Optimized: Inference on the whole batch at once
        with torch.no_grad():
            observation_dev = jax.tree.map(lambda x: x.to(device, non_blocking=True), batch_observation)
            # Predict value [B, 1]
            val_batch = model.sample_values(device, observation_dev)
            val_batch_np = val_batch.cpu().numpy().flatten()
            
            # Metadata [B]
            frame_indices = batch_observation.frame_index.numpy().flatten()
            episode_indices = batch_observation.episode_index.numpy().flatten()
            
        # * Iterate over the batch locally to preserve logic
        current_batch_size = len(frame_indices)
        
        for i in range(current_batch_size):
            cur_frame_idx = int(frame_indices[i])
            ep_index = int(episode_indices[i])
            val = float(val_batch_np[i])

            if prev_frame_index is None:
                prev_frame_index = cur_frame_idx

            # New episode boundary
            # * Logic preserved: if frame index resets, we finished an episode
            if prev_frame_index > 0 and cur_frame_idx <= prev_frame_index:
                
                # * Save to DISK instead of memory dict
                # The `value_frames` list contains the FULL episode sequence now
                if len(value_frames) > 0:
                    prev_ep_id = int(episode_indices[i-1]) if i > 0 else ep_index - 1 # Approximation for ID
                    # Or better, track 'current_episode_id' variable. 
                    # Assuming contiguous, the finished episode is the one just before this frame.
                    
                    # * Save Values to .npy
                    np.save(os.path.join(temp_val_dir, f"ep_{prev_ep_id}.npy"), np.array(value_frames))
                
                    logging.info(f"Detected episode boundary. Finished Episode {prev_ep_id}.")

                    # * Visualization (Limited by count)
                    if with_vis and not metric_only:
                        if vis_count < max_vis_episodes and len(raw_frames) > 0:
                            write_episode_video(
                                episode_id=prev_ep_id,
                                value_list=value_frames,
                                img_list=raw_frames,
                                output_dir=output_dir,
                                fig_w=12.0,
                                fig_h=4.0,
                                dpi=100,
                                fps=fps,
                                metric_only=metric_only,
                            )
                            vis_count += 1
                
                # Clear buffers
                value_frames.clear()
                raw_frames.clear()

            # Store value (Buffer)
            value_frames.append(val)

            # Store images for video (Buffer)
            # * Optimization: Only process images if we are under the visualization limit
            should_vis = (not metric_only) and (with_vis) and (vis_count < max_vis_episodes)
            
            if should_vis:
                # Extract specific index from batch tensors
                base_torch = batch_observation.images["base_0_rgb"][i]
                img_base = process_view_torch(base_torch)
                current_views = [img_base] 

                if not headview_only:
                    if "left_wrist_0_rgb" in batch_observation.images:
                        left_torch = batch_observation.images["left_wrist_0_rgb"][i]
                        img_left = process_view_torch(left_torch)
                        current_views.append(img_left)
                    
                    if "right_wrist_0_rgb" in batch_observation.images:
                        right_torch = batch_observation.images["right_wrist_0_rgb"][i]
                        img_right = process_view_torch(right_torch)
                        current_views.append(img_right)
                
                raw_frames.append(current_views)

            prev_frame_index = cur_frame_idx


    # Write the very last episode
    if len(value_frames) != 0:
        print(f"Writing final episode {ep_index}.")
        np.save(os.path.join(temp_val_dir, f"ep_{ep_index}.npy"), np.array(value_frames))
        
        if with_vis and vis_count < max_vis_episodes:
            write_episode_video(
                episode_id=ep_index,
                value_list=value_frames,
                img_list=raw_frames,
                output_dir=output_dir,
                fig_w=12.0,
                fig_h=4.0,
                dpi=100,
                fps=fps,
                metric_only=metric_only,
            )



    print(">>>>>> Finished processing all batches. Now adding value column to dataset...")
    # * Changed: Add columns using the disk-based function
    
    start_root = repo_id
    
    
    output_root_value = repo_id + "_with_frame_value_v1"
    add_column_from_disk(
        data_root=repo_id,
        add_column_name="frame_value",
        temp_value_dir=temp_val_dir, # Pass the temp dir
        output_root=output_root_value,
    )
    
    print(">>>>>> Finished adding value column to dataset.>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    
    
    # * Cleanup Temp Dir
    shutil.rmtree(temp_val_dir)

    print(f"<<<<<< Finished generating visualizations to: {output_dir}")
    
    
    # * Next: Convert to advantage
    print(">>>>>> Now calculating advantage...>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    
    N_SMOOTH = 3        # * Number of adjacent frames for smoothing (t-1, t, t+1)
    CHUNK_SIZE = 50     # * Lookahead steps for advantage calculation (t+25)  ---> Align with RLinf
    ADVANTAGE_SCALER = 5.  # * Advantage scaler --> numerically progress difference is too small.
    
    
    data_root = output_root_value
    output_dir = data_root.split('_with_')[0] + f"_w_adv"


    
    smooth_value_and_compute_advantage(data_root, 
                                       output_dir, 
                                       n_smooth=N_SMOOTH,
                                       chunk_size=CHUNK_SIZE, 
                                       advantage_scaler=ADVANTAGE_SCALER
                                    )
    print(f"✅ Finished processing. New files are saved to: **{output_dir}**")
    
    shutil.rmtree(output_root_value)
    deal_mata(start_root, output_dir)
    soft_link_video(start_root,output_dir)
    print("✅ Finished updating metadata and linking videos.")
    
    print("✅ ✅ ✅  Task Done")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
