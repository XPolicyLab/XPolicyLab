"""
将 HDF5 格式数据转换为 LeRobot v2.0 数据集格式（不依赖 lerobot 库）。

LeRobot v2.0 使用 MP4 视频存储图像数据，每个 episode 每个相机一个 MP4 文件。

生成的目录结构:
    {targetDir}/
    ├── meta/
    │   ├── info.json
    │   ├── episodes.jsonl
    │   └── tasks.jsonl
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       ├── episode_000001.parquet
    │       └── ...
    └── videos/
        └── chunk-000/
            └── observation.images.{camera}/
                ├── episode_000000.mp4
                ├── episode_000001.mp4
                └── ...

用法:
    python hdf5_to_lerobot_v2.py \
        --datasetDir /path/to/hdf5 \
        --type our_exp_bicam \
        --targetDir /path/to/output/lerobot
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import yaml

CODEBASE_VERSION = "v2.0"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_PARQUET_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def get_episode_chunk(episode_index: int, chunks_size: int = DEFAULT_CHUNK_SIZE) -> int:
    return episode_index // chunks_size


def encode_video_frames(
    frames: list[np.ndarray],
    output_path: Path,
    fps: int,
    codec: str = "libx264",
    pix_fmt: str = "yuv420p",
):
    """
    将 RGB 帧序列编码为 MP4 视频。
    使用 ffmpeg 命令行进行编码，确保输出兼容性。
    frames: list of HWC RGB numpy arrays (uint8)
    """
    if not frames:
        raise ValueError("帧列表为空，无法编码视频")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, frame in enumerate(frames):
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(tmp_dir, f"frame_{i:06d}.png"), bgr)

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "frame_%06d.png"),
            "-vcodec", codec,
            "-pix_fmt", pix_fmt,
            "-g", str(2),       # GOP size = 2，方便按帧解码
            "-crf", "20",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg 编码失败: {result.stderr}")


def get_video_info(video_path: Path) -> dict:
    """使用 ffprobe 获取视频元信息。"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {}

    probe = json.loads(result.stdout)
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            r_fps = stream.get("r_frame_rate", "30/1")
            num, den = r_fps.split("/")
            return {
                "video.fps": float(num) / float(den),
                "video.codec": stream.get("codec_name", "h264"),
                "video.pix_fmt": stream.get("pix_fmt", "yuv420p"),
                "video.is_depth_map": False,
                "has_audio": False,
            }
    return {}


def build_features(args):
    """根据配置构建 features 字典，包含用户数据特征和默认元数据特征。"""
    states_names = []
    actions_names = []
    for i, name in enumerate(args.armJointStateNames):
        if "puppet" in name:
            for j in range(args.armJointStateDims[i]):
                states_names.append(f"arm.jointStatePosition.{name}.joint{j}")
        if "master" in name:
            for j in range(args.armJointStateDims[i]):
                actions_names.append(f"arm.jointStatePosition.{name}.joint{j}")

    features = OrderedDict()

    features["observation.state"] = {
        "dtype": "float64",
        "shape": (len(states_names),),
        "names": [states_names],
    }
    features["action"] = {
        "dtype": "float64",
        "shape": (len(actions_names),),
        "names": [actions_names],
    }

    for camera in args.cameraColorNames:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    if args.useCameraPointCloud:
        for camera in args.cameraPointCloudNames:
            features[f"observation.pointClouds.{camera}"] = {
                "dtype": "float64",
                "shape": (args.pointNum * 6,),
            }

    default_features = {
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
    }
    features.update(default_features)

    return features


def create_info(args, features):
    """创建 info.json 内容。"""
    features_serialized = {}
    for k, v in features.items():
        features_serialized[k] = {}
        for kk, vv in v.items():
            features_serialized[k][kk] = list(vv) if isinstance(vv, tuple) else vv

    return {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": args.robotType,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": 0,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": args.fps,
        "splits": {},
        "data_path": DEFAULT_PARQUET_PATH,
        "video_path": DEFAULT_VIDEO_PATH,
        "features": features_serialized,
    }


def load_episode_data(args, episode_path: Path):
    """从 HDF5 文件加载单个 episode 的数据。"""
    with h5py.File(episode_path, "r") as episode:
        puppet_data = [
            episode[f"arm/jointStatePosition/{name}"][()]
            for name in args.armJointStateNames if "puppet" in name
        ]
        if not puppet_data:
            raise ValueError(f"没有找到 puppet 机械臂数据: {episode_path}")
        states = np.concatenate(puppet_data, axis=1) if len(puppet_data) > 1 else puppet_data[0]

        master_data = [
            episode[f"arm/jointStatePosition/{name}"][()]
            for name in args.armJointStateNames if "master" in name
        ]
        if not master_data:
            raise ValueError(f"没有找到 master 机械臂数据: {episode_path}")
        actions = np.concatenate(master_data, axis=1) if len(master_data) > 1 else master_data[0]

        colors = {}
        for camera in args.cameraColorNames:
            cam_frames = []
            for i in range(episode[f"camera/color/{camera}"].shape[0]):
                frame = cv2.cvtColor(episode[f"camera/color/{camera}"][i], cv2.COLOR_BGR2RGB)
                cam_frames.append(frame)
            colors[camera] = cam_frames

        pointclouds = {}
        if args.useCameraPointCloud:
            for camera in args.cameraPointCloudNames:
                pc_frames = []
                for i in range(episode[f"camera/pointCloud/{camera}"].shape[0]):
                    pc_frames.append(np.load(
                        os.path.join(str(episode_path.resolve())[:-9],
                                     episode[f"camera/color/{camera}"][i].decode("utf-8"))))
                pointclouds[camera] = pc_frames

        instruction = None
        if "instruction" in episode:
            instruction_data = episode["instruction"][()]
            if isinstance(instruction_data, bytes):
                instruction = instruction_data.decode("utf-8")
            elif isinstance(instruction_data, np.ndarray):
                instruction = str(instruction_data)
            else:
                instruction = str(instruction_data)

        return colors, pointclouds, states, actions, instruction


def build_parquet_table(episode_buffer: dict, features: dict):
    """将 episode buffer 构建为 pyarrow Table。"""
    columns = {}

    for key, ft in features.items():
        if ft["dtype"] == "video":
            # 仅保存 timestamp，路径由 info.json 中的模板推导
            columns[key] = pa.array(episode_buffer["timestamp"], type=pa.float32())
        elif ft["dtype"] == "float32":
            columns[key] = pa.array([float(v) for v in episode_buffer[key]], type=pa.float32())
        elif ft["dtype"] == "float64":
            arr = np.array(episode_buffer[key])
            if arr.ndim == 2:
                columns[key] = pa.array([row.tolist() for row in arr], type=pa.list_(pa.float64()))
            else:
                # 标量 float64 也存成长度为 1 的数组
                data = [[v] for v in arr.tolist()]
                columns[key] = pa.array(data, type=pa.list_(pa.float64()))
        elif ft["dtype"] == "int64":
            columns[key] = pa.array([int(v) for v in episode_buffer[key]], type=pa.int64())
        elif ft["dtype"] == "bool":
            columns[key] = pa.array([bool(v) for v in episode_buffer[key]], type=pa.bool_())
        else:
            arr = np.array(episode_buffer[key])
            if arr.ndim == 2:
                columns[key] = pa.array([row.tolist() for row in arr], type=pa.list_(pa.float64()))
            else:
                columns[key] = pa.array(arr.tolist())

    return pa.table(columns)


def process(args):
    dataset_dir = Path(args.datasetDir)
    target_dir = Path(args.targetDir)

    if not dataset_dir.exists():
        raise ValueError(f"datasetDir 不存在: {dataset_dir}")
    if target_dir.exists():
        print(f"目标目录已存在，删除: {target_dir}")
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "meta").mkdir(parents=True, exist_ok=True)

    hdf5_files = sorted(dataset_dir.glob("**/episode*.hdf5"))
    if not hdf5_files:
        raise ValueError(f"未找到 HDF5 文件: {dataset_dir}")
    print(f"找到 {len(hdf5_files)} 个 HDF5 文件")

    features = build_features(args)
    info = create_info(args, features)
    video_keys = [k for k, v in features.items() if v["dtype"] == "video"]

    print("\n" + "=" * 80)
    print("Features 列表:")
    print("=" * 80)
    for key in features:
        print(f"  - {key}: dtype={features[key]['dtype']}, shape={features[key].get('shape')}")
    print("=" * 80 + "\n")

    task_to_index: dict[str, int] = {}
    episodes_meta = []
    total_frames = 0
    total_videos = 0
    error_files = []
    video_info_recorded = False

    for ep_idx in tqdm.tqdm(range(len(hdf5_files)), desc="转换 episodes"):
        episode_path = hdf5_files[ep_idx]

        try:
            colors, pointclouds, states, actions, instruction = load_episode_data(args, episode_path)
        except Exception as e:
            print(f"\n读取失败 {episode_path}: {e}")
            error_files.append(episode_path)
            continue

        num_frames = states.shape[0]
        if num_frames == 0:
            print(f"\n警告: {episode_path.name} 数据为空，跳过")
            error_files.append(episode_path)
            continue

        current_task = instruction if instruction else args.instruction
        if current_task not in task_to_index:
            task_to_index[current_task] = len(task_to_index)
        task_index = task_to_index[current_task]

        chunk = get_episode_chunk(ep_idx)

        # 编码每个相机的视频
        video_paths_map = {}
        for vid_key in video_keys:
            camera = vid_key.replace("observation.images.", "")
            video_rel = DEFAULT_VIDEO_PATH.format(
                episode_chunk=chunk, video_key=vid_key, episode_index=ep_idx
            )
            video_abs = target_dir / video_rel
            encode_video_frames(colors[camera], video_abs, fps=args.fps)
            video_paths_map[vid_key] = video_rel
            total_videos += 1

            if not video_info_recorded:
                vi = get_video_info(video_abs)
                if vi:
                    info["features"][vid_key]["info"] = vi
                video_info_recorded = True

        # 构建 episode buffer
        episode_buffer: dict[str, list] = {key: [] for key in features}

        for i in range(num_frames):
            global_index = total_frames + i
            timestamp = i / args.fps

            episode_buffer["observation.state"].append(states[i])
            episode_buffer["action"].append(actions[i])
            episode_buffer["timestamp"].append(timestamp)
            episode_buffer["frame_index"].append(i)
            episode_buffer["episode_index"].append(ep_idx)
            episode_buffer["index"].append(global_index)
            episode_buffer["task_index"].append(task_index)

            for vid_key in video_keys:
                episode_buffer[vid_key].append(video_paths_map[vid_key])

            if args.useCameraPointCloud:
                for camera in args.cameraPointCloudNames:
                    pc_key = f"observation.pointClouds.{camera}"
                    episode_buffer[pc_key].append(pointclouds[camera][i])

        # 写 parquet
        parquet_rel = DEFAULT_PARQUET_PATH.format(episode_chunk=chunk, episode_index=ep_idx)
        parquet_path = target_dir / parquet_rel
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        table = build_parquet_table(episode_buffer, features)
        pq.write_table(table, parquet_path)

        episodes_meta.append({
            "episode_index": ep_idx,
            "tasks": [current_task],
            "length": num_frames,
        })

        total_frames += num_frames

    # ========== 写元数据 ==========
    info["total_episodes"] = len(episodes_meta)
    info["total_frames"] = total_frames
    info["total_tasks"] = len(task_to_index)
    info["total_videos"] = total_videos
    info["total_chunks"] = (len(episodes_meta) - 1) // DEFAULT_CHUNK_SIZE + 1 if episodes_meta else 0
    info["splits"] = {"train": f"0:{len(episodes_meta)}"}

    with open(target_dir / "meta" / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    with open(target_dir / "meta" / "episodes.jsonl", "w", encoding="utf-8") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    with open(target_dir / "meta" / "tasks.jsonl", "w", encoding="utf-8") as f:
        for task_text, idx in sorted(task_to_index.items(), key=lambda x: x[1]):
            task_entry = OrderedDict([("task_index", idx), ("task", task_text)])
            f.write(json.dumps(task_entry, ensure_ascii=False) + "\n")

    # ========== 打印汇总 ==========
    print("\n" + "=" * 80)
    print("转换完成!")
    print("=" * 80)
    print(f"  总 episodes: {info['total_episodes']}")
    print(f"  总 frames:   {info['total_frames']}")
    print(f"  总 tasks:    {info['total_tasks']}")
    print(f"  总 videos:   {info['total_videos']}")
    print(f"  总 chunks:   {info['total_chunks']}")
    print(f"  输出目录:    {target_dir}")
    print("\nTask 映射表:")
    for task_text, idx in sorted(task_to_index.items(), key=lambda x: x[1]):
        print(f"  task_index {idx}: {task_text}")

    if error_files:
        print(f"\n错误文件 ({len(error_files)}):")
        for ef in error_files:
            print(f"  - {ef}")
    print("=" * 80)


def get_arguments():
    parser = argparse.ArgumentParser(
        description="将 HDF5 数据转换为 LeRobot v2.0 格式（MP4 视频，不依赖 lerobot 库）"
    )
    parser.add_argument("--datasetDir", type=str, required=True, help="HDF5 数据所在目录")
    parser.add_argument("--datasetName", type=str, default="real_piper", help="数据集名称")
    parser.add_argument("--type", type=str, default="aloha", help="配置类型（用于加载 YAML）")
    parser.add_argument("--instruction", type=str, default="null", help="默认任务指令")
    parser.add_argument("--targetDir", type=str, required=True, help="输出目录")
    parser.add_argument("--robotType", type=str, default="cobot_magic", help="机器人类型")
    parser.add_argument("--fps", type=int, default=30, help="帧率")
    parser.add_argument("--useCameraPointCloud", type=bool, default=False, help="是否使用点云")
    parser.add_argument("--pointNum", type=int, default=5000, help="点云点数")
    parser.add_argument("--cameraColorNames", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--cameraDepthNames", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--cameraPointCloudNames", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--armJointStateNames", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--armJointStateDims", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--armEndPoseNames", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--armEndPoseDims", default=[], help=argparse.SUPPRESS)

    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent / f"{args.type}_data_params.yaml"
    if not config_path.exists():
        config_path = Path(f"{args.type}_data_params.yaml")

    with open(config_path, "r") as file:
        yaml_data = yaml.safe_load(file)
        args.cameraColorNames = yaml_data["dataInfo"]["camera"]["color"]["names"]
        args.cameraDepthNames = yaml_data["dataInfo"]["camera"]["depth"]["names"]
        args.cameraPointCloudNames = yaml_data["dataInfo"]["camera"]["pointCloud"]["names"]
        args.armJointStateNames = yaml_data["dataInfo"]["arm"]["jointState"]["names"]
        default_joint_dim = yaml_data["dataInfo"]["arm"]["jointState"].get("dim", 7)
        args.armJointStateDims = [default_joint_dim for _ in range(len(args.armJointStateNames))]
        args.armEndPoseNames = yaml_data["dataInfo"]["arm"]["endPose"]["names"]
        default_endpose_dim = yaml_data["dataInfo"]["arm"]["endPose"].get("dim", 7)
        args.armEndPoseDims = [default_endpose_dim for _ in range(len(args.armEndPoseNames))]

    return args


def main():
    args = get_arguments()
    process(args)


if __name__ == "__main__":
    main()
