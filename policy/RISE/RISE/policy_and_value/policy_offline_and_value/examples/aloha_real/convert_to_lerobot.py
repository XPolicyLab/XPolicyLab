import os
os.environ["SVT_LOG"] = "1"

from typing import List, Dict, Tuple
from pathlib import Path
import shutil
import logging
import time
from functools import partial
import subprocess

import numpy as np
import tyro
import av
import h5py

from mini_lerobot.builder import LeRobotDatasetBuilder


# * Example usage:
# * python examples/aloha_real/convert_aloha_data_to_lerobot_kai.py --data-dir /cpfs01/user/yangjiazhi/workspace/VLA/Datasets/KAI/rl_yjz/rl_init_pick_place_2/ --repo-ids aloha_mobile_dummy  --prompt ""Pick and sort bricks on the conveyor." --save-dir /cpfs01/user/yangjiazhi/workspace/VLA/Datasets/KAI/huggingface/kai_convert --save_repoid rl_yjz_init_pick_place_2 


OPTIONAL_FEATURES = ("noise", "inferred_action")

FEATURES = {
    "observation.images.top_head": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.hand_left": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.hand_right": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (14,),
    },
    "action": {
        "dtype": "float32",
        "shape": (14,),
    },


    # * Optional
    # * for offline PPO
    "noise":{
        "dtype": "float32",
        "shape": (700, ),
    },

    "inferred_action": {
        "dtype": "float32",
        "shape": (700,),
    }
}


def resolve_output_features(valid_files: List[Path]) -> tuple[Dict, set[str]]:
    enabled_optional_features = set(OPTIONAL_FEATURES)
    for file in valid_files:
        with h5py.File(file, "r") as f:
            enabled_optional_features &= {key for key in OPTIONAL_FEATURES if key in f}
        if not enabled_optional_features:
            break

    resolved_features = {
        key: value
        for key, value in FEATURES.items()
        if key not in OPTIONAL_FEATURES or key in enabled_optional_features
    }
    return resolved_features, enabled_optional_features

def lazy_load_hdf5_dataset(
    episode_path: str | Path,
) -> Tuple[Dict, h5py.File]:
    """Load hdf5 dataset and return a dict with observations and actions"""
    f = h5py.File(episode_path, 'r')

    state_qpos = np.array(f["observations/qpos"])
    
    epi_len = state_qpos.shape[0]

    if "inferred_action" in f.keys():
        inferred_action = np.array(f["inferred_action"]).astype(np.float32)
        inferred_action = inferred_action.reshape((epi_len, -1))  # [ep_len, 14]
        


    if "noise" in f.keys():
        noise = np.array(f["noise"]).astype(np.float32)
        noise = noise.reshape((epi_len, -1))  # [epi_len, 1, 50, 14] -> [epi_len, 50 * 14]
        
        
        

    # epi_len = state_qpos.shape[0]
    episode = {
        "observation.state": state_qpos.reshape((epi_len, -1)),
        "observation.images.top_head": None,
        "observation.images.hand_left": None,
        "observation.images.hand_right": None,
        "action": state_qpos.reshape((epi_len, -1)),
        "epi_len": epi_len
    }

    if "inferred_action" in f.keys():
        episode["inferred_action"] = inferred_action
    if "noise" in f.keys():
        episode["noise"] = noise

    return episode, f


def encode_video_frames(
        images: np.ndarray, 
        dst: Path,
        fps: int,
        vcodec: str = "libsvtav1",
        pix_fmt: str = "yuv420p",
        g: int | None = 2,
        crf: int | None = 30,
        fast_decode: int = 0,
        log_level: int | None = av.logging.ERROR,
        overwrite: bool = False,
) -> bytes:
    """More info on ffmpeg arguments tuning on `benchmark/video/README.md`"""
    # Check encoder availability
    if vcodec not in ["h264", "hevc", "libsvtav1"]:
        raise ValueError(f"Unsupported video codec: {vcodec}. Supported codecs are: h264, hevc, libsvtav1.")

    video_path = Path(dst)

    video_path.parent.mkdir(parents=True, exist_ok=overwrite)

    # Encoders/pixel formats incompatibility check
    if (vcodec == "libsvtav1" or vcodec == "hevc") and pix_fmt == "yuv444p":
        print(
            f"Incompatible pixel format 'yuv444p' for codec {vcodec}, auto-selecting format 'yuv420p'"
        )
        pix_fmt = "yuv420p"

    # Define video output frame size (assuming all input frames are the same size)

    dummy_image = images[0]
    height, width, _ = dummy_image.shape

    # Define video codec options
    video_options = {}

    if g is not None:
        video_options["g"] = str(g)

    if crf is not None:
        video_options["crf"] = str(crf)

    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        video_options[key] = value

    # Set logging level
    if log_level is not None:
        # "While less efficient, it is generally preferable to modify logging with Python’s logging"
        logging.getLogger("libav").setLevel(log_level)

    # Create and open output file (overwrite by default)
    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        # Loop through input frames and encode them
        for input_image in images:
            # input_image = Image.open(input_data).convert("RGB")
            # input_frame = av.VideoFrame.from_image(input_image)
            input_frame = av.VideoFrame.from_ndarray(input_image, format="rgb24", channel_last=True)
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)

        # Flush the encoder
        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    # Reset logging level
    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")


def produce_episode(
    video_map: dict[str, Path],
    log_dir: Path,
    prompt: str,
    enabled_optional_features: set[str] | None = None,
):
    episode_start_time = time.time()
    enabled_optional_features = enabled_optional_features or set()
    
    camera_mapping = {
        "top_head": "cam_high",
        "hand_left": "cam_left_wrist", 
        "hand_right": "cam_right_wrist"
    }
    
    try:
        episode, f = lazy_load_hdf5_dataset(log_dir)

        epi_len = episode.pop("epi_len")
        tasks = [prompt] * epi_len

        feature_data = {
            "observation.state": episode["observation.state"],
            "action": episode["action"]
        }

        if "inferred_action" in episode.keys() and "inferred_action" in enabled_optional_features:
            feature_data["inferred_action"] = episode["inferred_action"]
        if "noise" in episode.keys() and "noise" in enabled_optional_features:
            feature_data["noise"] = episode["noise"]
        
        camera_list = [key for key in episode.keys() if key.startswith("observation.images.")]
        
        for camera_key in camera_list:
            camera_start_time = time.time()
            video_dst = video_map[camera_key]
            hdf5_camera_name = camera_key.replace("observation.images.", "")
            
            video_dir_name = camera_mapping.get(hdf5_camera_name, hdf5_camera_name)
            
            data_dir = log_dir.parent
            episode_name = log_dir.stem
            
            existing_video_path = data_dir / "video" / video_dir_name / f"{episode_name}.mp4"
            
            if existing_video_path.exists():
                # copy_start_time = time.time()
                video_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(existing_video_path, video_dst)
                # copy_time = time.time() - copy_start_time
                # copy_count += 1
            else:
                # encode_start_time = time.time()
                images = np.array(episode[camera_key])
                encode_video_frames(images, dst=video_dst, fps=30, overwrite=True)
        
        f.close()
        return feature_data, tasks
        
    except Exception as e:
        error_time = time.time() - episode_start_time
        print(f"  ❌ processing episode {log_dir.name} failed: {e}, Ignoring this episode.")
        print(f"  ⏱️  Error timestamp: {error_time:.2f}s")
        # raise

def main(
    data_dir: Path | str,
    save_dir: Path | str,
    repo_ids: List[str] | str,
    prompt: str | None = None,
    save_repoid: str | None = None,
    max_workers: int = 8,
    *,
    overwrite: bool = False,
    upload: bool = False,
    only_sync: bool = False,
):
    
    data_dir = Path(data_dir)
    if type(repo_ids) is str:
        repo_ids = [repo_ids]
    
    task = data_dir.name.split('_')[0]
    
    if save_repoid is None:
        repoid = data_dir.name.split('_')
        # task = repoid[0]
        save_repoid = '_'.join(repoid[1: -1]) + '_lerobot'
        print(f"save_repoid will be set according to repo_ids: {save_repoid}")

    log_files: List[Path] = []
    for repo_id in repo_ids:
        repo_path = data_dir / repo_id
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository path {repo_path} does not exist.")
        found_files = sorted(d for d in repo_path.iterdir() if not d.is_dir() and d.suffix == '.hdf5')
        log_files.extend(found_files)
    # filter invalid hdf5 files
    valid_files = []
    for file in log_files:
        data_dir = file.parent
        episode_name = file.stem
        all_videos_exist = True
        for video_dir_name in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
            existing_video_path = data_dir / "video" / video_dir_name / f"{episode_name}.mp4"
            if not existing_video_path.exists():
                print(f"  ⚠️  {existing_video_path} not found, skipping this file.")
                all_videos_exist = False
                break
        if not all_videos_exist:
            continue
        try:
            with h5py.File(file, 'r') as f:
                pass
            valid_files.append(file)
        except Exception as e:
            print(f"  ❌ Invalid {file}, error: {e}, Ignoring this file.")

    # output_path = Path(save_dir) / task / save_repoid
    output_path = Path(save_dir) / save_repoid


    if not only_sync:
        if output_path.exists():
            if overwrite:
                shutil.rmtree(output_path)
            else:
                raise FileExistsError(f"Output path {output_path} already exists. Use --overwrite to overwrite.")
        
        resolved_features, enabled_optional_features = resolve_output_features(valid_files)
        disabled_optional_features = sorted(set(OPTIONAL_FEATURES) - enabled_optional_features)
        if disabled_optional_features:
            print(
                "Excluding optional features not present in every valid episode: "
                + ", ".join(disabled_optional_features)
            )

        builder = LeRobotDatasetBuilder(
            repo_id=save_repoid,
            fps=30,
            features=resolved_features,
            robot_type='agilex',
            root=output_path,
        )
        if prompt is None:
            if task == 'iros':
                prompt = "fold the cloth"
            else:
                prompt = f"{task} the cloth"
        builder.add_episodes(
            partial(produce_episode, prompt=prompt, enabled_optional_features=enabled_optional_features),
            valid_files,
            max_workers=max_workers,
        )
        
        builder.flush()

    if upload:
        # begin upload data
        assert task in ['hang', 'fold', 'flat', 'iros'], f"Unknown task: {task}, expected one of ['hang', 'fold', 'flat', 'iros']"
        print(f"  ⏫ Starting upload...")
        if task == 'iros':
            remote_path = f"oss://oss-pai-d8dbg42zb0rplbe70e-cn-wulanchabu/data/fold_cloth/{task}/{save_repoid}"
        else:
            remote_path = f"oss://oss-pai-d8dbg42zb0rplbe70e-cn-wulanchabu/data/{task}_cloth/{save_repoid}"
        cmd = [
            "ossutil", "cp",
            "-r",                                
            str(output_path) + "/",
            remote_path,                         
            "-j", "100",                         
            "-u",                                             
        ]
        try:
            result = subprocess.run(cmd, text=True, check=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"{e.stderr}")

def add_episode(

):
    pass

if __name__ == "__main__":
    st = time.time()
    tyro.cli(main)
    print(f"Time taken: {time.time() - st} seconds")
