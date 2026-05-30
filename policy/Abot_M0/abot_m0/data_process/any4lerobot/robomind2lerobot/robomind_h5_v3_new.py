import argparse
import gc
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import ray
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import flatten_dict, validate_episode_buffer, write_info, write_stats
from lerobot.datasets.video_utils import get_safe_default_codec
from ray.runtime_env import RuntimeEnv
from robomind_uitls.configs import ROBOMIND_CONFIG
from robomind_uitls.lerobot_uitls import compute_episode_stats, generate_features_from_config
from robomind_uitls.robomind_uitls import load_local_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class RoboMINDDatasetMetadata(LeRobotDatasetMetadata):
    def save_episode(
        self,
        split,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
        episode_metadata: dict,
    ) -> None:
        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        episode_dict.update(episode_metadata)
        episode_dict.update(flatten_dict({"stats": episode_stats}))
        self._save_episode_metadata(episode_dict)

        # Update info
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length
        self.info["total_tasks"] = len(self.tasks)
        if split == "train":
            self.info["splits"]["train"] = f"0:{self.info['total_episodes']}"
            self.train_count = self.info["total_episodes"]
        elif "val" in split:
            self.info["splits"]["validation"] = f"{self.train_count}:{self.info['total_episodes']}"

        write_info(self.info, self.root)

        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats is not None else episode_stats
        write_stats(self.stats, self.root)


class RoboMINDDataset(LeRobotDataset):
    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        obj = cls.__new__(cls)
        obj.meta = RoboMINDDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None
        obj.batch_encoding_size = batch_encoding_size
        obj.episodes_since_last_encoding = 0

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()

        obj.writer = None
        obj.latest_episode = None
        obj._current_file_start_frame = None
        # Initialize tracking for incremental recording
        obj._lazy_loading = False
        obj._recorded_frames = 0
        obj._writer_closed_for_reading = False

        return obj

    def save_episode(self, split, action_config: dict, episode_data: dict | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        
        episode_buffer = episode_data if episode_data is not None else self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Update tasks and task indices with new tasks if any
        self.meta.save_episode_tasks(episode_tasks)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key]).squeeze()

        self._wait_image_writer()

        ep_stats = compute_episode_stats(episode_buffer, self.features)

        ep_metadata = self._save_episode_data(episode_buffer)
        has_video_keys = len(self.meta.video_keys) > 0
        use_batched_encoding = self.batch_encoding_size > 1

        if has_video_keys and not use_batched_encoding:
            for video_key in self.meta.video_keys:
                ep_metadata.update(self._save_episode_video(video_key, episode_index))

        # `meta.save_episode` be executed after encoding the videos
        ep_metadata.update({"action_config": action_config})
        self.meta.save_episode(split, episode_index, episode_length, episode_tasks, ep_stats, ep_metadata)

        if has_video_keys and use_batched_encoding:
            # Check if we should trigger batch encoding
            self.episodes_since_last_encoding += 1
            if self.episodes_since_last_encoding == self.batch_encoding_size:
                start_ep = self.num_episodes - self.batch_encoding_size
                end_ep = self.num_episodes
                self._batch_save_episode_video(start_ep, end_ep)
                self.episodes_since_last_encoding = 0

        if not episode_data:
            # Reset episode buffer and clean up temporary images (if not already deleted during video encoding)
            self.clear_episode_buffer(delete_images=len(self.meta.image_keys) > 0)


def get_all_tasks(src_path: Path, output_path: Path, embodiment: str):
    output_path = output_path / src_path.name / embodiment
    src_path = src_path / f"h5_{embodiment}"

    if src_path.exists():
        df = pd.read_csv(src_path.parent.parent / "RoboMIND_v1_2_instr.csv", index_col=0).drop_duplicates()
        instruction_dict = df.set_index("task")["instruction"].to_dict()
        for task_type in src_path.iterdir():

            if ".tar.gz" in task_type.name:
                continue

            yield (
                task_type.name,
                {"train": task_type / "success_episodes" / "train", "val": task_type / "success_episodes" / "val"},
                (output_path / task_type.name).resolve(),
                instruction_dict[task_type.name],
            )


def save_as_lerobot_dataset(task: tuple[dict, Path, str], src_path, benchmark, embodiment, save_depth, save_images: bool = True):
    task_type, splits, local_dir, task_instruction = task

    config = ROBOMIND_CONFIG[embodiment]
    features = generate_features_from_config(config)

    # [HACK]: franka and ur image is bgr...
    bgr2rgb = False
    if embodiment in ["franka_1rgb", "franka_3rgb", "franka_fr3_dual", "ur_1rgb"]:
        bgr2rgb = True

    if local_dir.exists():
        shutil.rmtree(local_dir)

    if not save_depth:
        features = dict(filter(lambda item: "depth" not in item[0], features.items()))
    
    # 如果不保存图片，从features中移除image类型的项（保留video类型用于视频编码）
    if not save_images:
        features = {k: v for k, v in features.items() if v.get("dtype") != "image"}
   
    dataset: RoboMINDDataset = RoboMINDDataset.create(
        repo_id=f"{embodiment}/{local_dir.name}",
        root=local_dir,
        fps=30,
        robot_type=embodiment,
        features=features,
    )
   
    logging.info(f"start processing for {benchmark}, {embodiment}, {task_type}, saving to {local_dir}")
    for split, path in splits.items():
        action_config_path = src_path / "language_description_annotation_json" / f"h5_{embodiment}.json"
        
        if action_config_path.exists():
            action_config = json.load(open(action_config_path))
            action_config = {
                Path(config["id"]).parent.name: config["response"]
                for config in action_config
                if local_dir.name in config["id"] and split in config["id"]
            }
        else:
            action_config = {}

        # 将路径拆分为部件列表
        parts = list(path.parts)
        parts.insert(-2, parts[-3])
        # 重组为新路径
        path = Path(*parts)
        print("1111111111111",path)

       
        for episode_path in path.glob("**/trajectory.hdf5"):
            status, raw_dataset, err = load_local_dataset(episode_path, config, save_depth, bgr2rgb)
            if status and len(raw_dataset) >= 50:
                try:
                    
                    for frame_data in raw_dataset:
                        frame_data["task"] = task_instruction
                        # 如果不保存图片，从frame_data中移除image类型的数据（保留video类型用于视频编码）
                        if not save_images:
                            frame_data_filtered = {k: v for k, v in frame_data.items() 
                                                 if k not in features or features.get(k, {}).get("dtype") != "image"}
                            dataset.add_frame(frame_data_filtered)
                        else:
                            dataset.add_frame(frame_data)
                  
                    dataset.save_episode(
                        split, action_config.get(episode_path.parent.parent.name, {"task_summary": None, "steps": None})
                    )
                    logging.info(f"process done for {path}, len {len(raw_dataset)}")
                except Exception:
                    logging.exception(
                        "Error saving episode for %s (split=%s, episode_path=%s)",
                        task_type,
                        split,
                        episode_path,
                    )
                    # [HACK]: not consistent image shape...
                    if config["images"]["camera_top"]["shape"] == (720, 1280, 3):
                        config["images"]["camera_top"]["shape"] = (480, 640, 3)
                        config["images"]["camera_top_depth"]["shape"] = (480, 640, 1)
                    else:
                        config["images"]["camera_top"]["shape"] = (720, 1280, 3)
                        config["images"]["camera_top_depth"]["shape"] = (720, 1280, 1)
                    save_as_lerobot_dataset(task, src_path, benchmark, embodiment, save_depth, save_images)
                    return
            else:
                logging.warning(f"Skipped {episode_path}: len of dataset:{len(raw_dataset)} or {str(err)}")
            gc.collect()

    if dataset.meta.total_episodes == 0:
        shutil.rmtree(local_dir)
    del dataset


def main(
    src_path: Path,
    output_path: Path,
    benchmark: str,
    embodiments: list[str],
    cpus_per_task: int,
    save_depth: bool,
    save_images: bool = True,
    debug: bool = False,
    log_path: str = "robomind_conversion",
):
    if debug:
        tasks = get_all_tasks(src_path / benchmark, output_path, embodiments[0])
        save_as_lerobot_dataset(next(tasks), src_path, benchmark, embodiments[0], save_depth, save_images)
    else:
        runtime_env = RuntimeEnv(
            env_vars={"HDF5_USE_FILE_LOCKING": "FALSE", "HF_DATASETS_DISABLE_PROGRESS_BARS": "TRUE"}
        )
        ray.init(runtime_env=runtime_env)
        resources = ray.available_resources()
        cpus = int(resources["CPU"])

        logging.info(f"Available CPUs: {cpus}, num_cpus_per_task: {cpus_per_task}")
        remote_task = ray.remote(save_as_lerobot_dataset).options(num_cpus=cpus_per_task)

        # 已完成任务的记录文件
        completed_tasks_file = Path(output_path) / f".completed_tasks_{log_path}.txt"
        print(completed_tasks_file)
      
        completed_tasks = set()
        
        # 读取已完成的任务列表（支持断点续传）
        if completed_tasks_file.exists():
            try:
                with open(completed_tasks_file, 'r') as f:
                    completed_tasks = {line.strip() for line in f if line.strip()}
                logging.info(f"从文件读取到 {len(completed_tasks)} 个已完成的任务")
            except Exception as e:
                logging.warning(f"读取已完成任务列表失败 ({str(e)})，将重新开始")

        # 收集所有任务
        all_tasks = []
        for embodiment in embodiments:
            tasks = get_all_tasks(src_path / benchmark, output_path, embodiment)
            for task in tasks:
                task_type, splits, local_dir, task_instruction = task
                task_type = "2024_09_20_close_cabinet"
                # 使用 task_type 作为唯一标识（格式：{embodiment}/{task_type}）
                task_id = f"{embodiment}/{task_type}"
                all_tasks.append((task_id, task, embodiment))
        
        total_tasks = len(all_tasks)
        logging.info(f"Total tasks to process: {total_tasks}")
        
        # 过滤掉已完成的任务
        remaining_tasks = [
            (task_id, task, embodiment) 
            for task_id, task, embodiment in all_tasks 
            if task_id not in completed_tasks
        ]
        skipped_tasks = total_tasks - len(remaining_tasks)
        if skipped_tasks > 0:
            logging.info(f"跳过 {skipped_tasks} 个已完成的任务，剩余 {len(remaining_tasks)} 个任务需要处理")
        
        # 提交任务
        futures = []
        for task_id, task, embodiment in remaining_tasks:
            task_type, splits, local_dir, task_instruction = task
            future = remote_task.remote(task, src_path, benchmark, embodiment, save_depth, save_images)
            futures.append((task_id, splits, future))
        
        logging.info(f"Submitted {len(futures)} tasks")
        
        
        # 处理任务结果
        for task_id, task_path, future in futures:
            try:
                ray.get(future)
                logging.info(f"Completed task: {task_id}")
                # 记录成功完成的任务
                completed_tasks.add(task_id)
                # 确保目录存在
                completed_tasks_file.parent.mkdir(parents=True, exist_ok=True)
                # 追加写入到文件（每次完成后立即写入，避免丢失）
                try:
                    with open(completed_tasks_file, 'a') as f:
                        f.write(f"{task_id}\n")
                        f.flush()  # 立即刷新到磁盘
                        os.fsync(f.fileno())  # 强制同步到磁盘
                except Exception as e:
                    logging.warning(f"警告：写入已完成任务记录文件失败 ({str(e)})，但任务已成功完成")
            except Exception as e:
                logging.error(f"Exception occurred for {task_id}: {str(e)}")
                with open(f"{log_path}.txt", "a") as f:
                    f.write(f"{task_id}, exception details: {str(e)}\n")
        
        # 打印最终统计信息
        logging.info(f"\n{'='*60}")
        logging.info(f"任务处理完成！")
        logging.info(f"总共处理任务数: {len(remaining_tasks)}")
        logging.info(f"成功完成任务数: {len(completed_tasks)}")
        logging.info(f"已完成任务记录文件: {completed_tasks_file}")
        
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-path",
        type=Path,
        default=Path("/mnt/nas-data-4/gaowo.cyz/RoboMIND"),
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["benchmark1_0_compressed", "benchmark1_1_compressed", "benchmark1_2_compressed"],
        default="benchmark1_0_compressed",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("/mnt/xlab-nas-2/vla_dataset/lerobot/robomind_10_test_trash"),
    )
    parser.add_argument(
        "--embodiments",
        type=str,
        nargs="+",
        help=str(
            [
                "sim_tienkung_1rgb"
                "agilex_3rgb",
                "franka_1rgb",
                "franka_3rgb",
                "franka_fr3_dual",
                "tienkung_gello_1rgb",
                "tienkung_prod1_gello_1rgb",
                "tienkung_xsens_1rgb",
                "ur_1rgb",
            ]
        ),
        default=[
            "franka_3rgb",
            # "franka_fr3_dual",
            # "sim_tienkung_1rgb",
            # "tienkung_prod1_gello_1rgb",
        ],
    )
    parser.add_argument("--cpus-per-task", type=int, default=100)
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument(
        "--save-images",
        action="store_true",
        default=False,
        help="Save images to disk (default: False). Use --no-save-images to disable.",
    )
    parser.add_argument("--no-save-images", dest="save_images", action="store_false", help="Disable saving images to disk")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--log-path",
        type=str,
        default="robomind_conversion",
        help="Log file path prefix for completed tasks tracking",
    )
    args = parser.parse_args()

    main(**vars(args))
