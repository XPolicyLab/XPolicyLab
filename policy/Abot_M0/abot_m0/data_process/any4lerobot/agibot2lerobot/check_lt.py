import argparse
import contextlib
import gc
import os
import shutil
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import ray
import torch
from tqdm import tqdm

from agibot_utils.agibot_utils import get_task_info, load_local_dataset
from agibot_utils.config import AgiBotWorld_TASK_TYPE
from agibot_utils.lerobot_utils import compute_episode_stats, generate_features_from_config
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import validate_episode_buffer, validate_frame
from ray.runtime_env import RuntimeEnv


SUCCESS_FILE = "_SUCCESS"


# ----------------------------
# Low-level FD suppression (captures ffmpeg/libav logs that bypass Python redirect)
# ----------------------------
@contextlib.contextmanager
def suppress_fds(enabled: bool = True):
    """Suppress low-level writes to fd=1/2 (stdout/stderr)."""
    if not enabled:
        yield
        return
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old1 = os.dup(1)
    old2 = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(old1, 1)
        os.dup2(old2, 2)
        os.close(old1)
        os.close(old2)
        os.close(devnull_fd)


def _prod(shape: Tuple[int, ...]) -> int:
    p = 1
    for x in shape:
        p *= int(x)
    return int(p)


def normalize_feature_shapes(features: dict) -> dict:
    """Always normalize feature['shape'] list->tuple, int->(int,)."""
    out = {}
    for k, ft in features.items():
        ft = dict(ft)
        shp = ft.get("shape", None)
        if isinstance(shp, list):
            ft["shape"] = tuple(shp)
        elif isinstance(shp, int):
            ft["shape"] = (shp,)
        out[k] = ft
    return out


def flatten_bimanual_features_by_shape(features: dict, enabled: bool) -> dict:
    """
    双臂拉平：任何 shape 的第0维为2（如 (2,4),(2,3),(2,)）-> 统一拉平为一维 (2*...,)
    例：
      (2,4) -> (8,)
      (2,3) -> (6,)
      (2,)  -> (2,)
    """
    if not enabled:
        return features
    out = {}
    for k, ft in features.items():
        ft = dict(ft)
        shp = ft.get("shape", None)
        if isinstance(shp, (tuple, list)):
            shp = tuple(shp)
            if len(shp) >= 1 and shp[0] == 2:
                ft["shape"] = (_prod(shp),)
        out[k] = ft
    return out


def auto_reshape_frame_by_features(frame: dict, features: dict) -> dict:
    """
    让 frame[k] 自动匹配 features[k]['shape']：
    - 如果元素总数一致，则 reshape
    - 先 squeeze 再比对
    """
    for k, ft in features.items():
        if k not in frame:
            continue
        expected = ft.get("shape", None)
        if expected is None or not isinstance(expected, (tuple, list)):
            continue
        expected = tuple(expected)

        v = frame[k]
        arr = np.asarray(v)
        if arr.shape == ():
            continue
        if arr.shape == expected:
            continue

        arr2 = np.asarray(v).squeeze()
        if arr2.shape == expected:
            frame[k] = arr2
            continue

        if arr2.size == _prod(expected):
            frame[k] = arr2.reshape(expected)
            continue

    return frame


def dump_features_once(log_path: Path, features: dict, task_key: str):
    """写 features 到每个 task 的 debug_shapes.log（不会破坏 tqdm）"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n==== FEATURES DUMP ({task_key}) @ {datetime.now().isoformat()} ====\n")
        for k, ft in features.items():
            shp = ft.get("shape", None)
            f.write(
                f"  - {k}: dtype={ft.get('dtype', None)}, "
                f"shape={repr(shp)} (type={type(shp).__name__})\n"
            )


# ----------------------------
# Tracker actor: NO printing, NO tqdm (prevents '(Actor pid=...)' prefix breaking bars)
# ----------------------------
@ray.remote(num_cpus=0)
class ProgressTrackerActor:
    def __init__(self):
        # task_key -> {total, done, finished, ok, note}
        self.state: Dict[str, Dict] = {}

    def init_tasks(self, task_keys: List[str]):
        for k in task_keys:
            if k not in self.state:
                self.state[k] = {"total": 0, "done": 0, "finished": False, "ok": False, "note": ""}

    def register(self, task_key: str, total_eps: int):
        if task_key not in self.state:
            self.state[task_key] = {"total": 0, "done": 0, "finished": False, "ok": False, "note": ""}
        self.state[task_key]["total"] = int(total_eps)

    def inc(self, task_key: str, n: int = 1):
        if task_key not in self.state:
            self.state[task_key] = {"total": 0, "done": 0, "finished": False, "ok": False, "note": ""}
        self.state[task_key]["done"] += int(n)

    def finish(self, task_key: str, ok: bool, note: str = ""):
        if task_key not in self.state:
            self.state[task_key] = {"total": 0, "done": 0, "finished": False, "ok": False, "note": ""}
        self.state[task_key]["finished"] = True
        self.state[task_key]["ok"] = bool(ok)
        self.state[task_key]["note"] = str(note)

    def snapshot(self) -> Dict[str, Dict]:
        return self.state


# ----------------------------
# Dataset
# ----------------------------
class AgiBotDataset(LeRobotDataset):
    def _append_debug_log(self, msg: str):
        log_path = getattr(self, "debug_log_path", None)
        if log_path is None:
            return
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _dump_validate_debug(self, err: Exception, frame: dict, features: dict):
        header = f"\n==== validate_frame FAILED @ {datetime.now().isoformat()} ====\n"
        header += f"Exception: {type(err).__name__}: {err}\n"
        try:
            fi = self.episode_buffer["size"] if self.episode_buffer is not None else None
            header += f"episode_buffer.size (next frame_index): {fi}\n"
        except Exception:
            pass
        self._append_debug_log(header)

        self._append_debug_log("[FEATURES expected shapes]")
        for k, ft in features.items():
            shp = ft.get("shape", None)
            dt = ft.get("dtype", None)
            self._append_debug_log(
                f"  - {k}: expected_shape={repr(shp)} (type={type(shp).__name__}), dtype={dt}"
            )

        self._append_debug_log("\n[FRAME actual values]")
        for k, v in frame.items():
            if isinstance(v, np.ndarray):
                self._append_debug_log(
                    f"  - {k}: np.ndarray shape={v.shape} dtype={v.dtype} (shape_type={type(v.shape).__name__})"
                )
            else:
                try:
                    arr = np.asarray(v)
                    self._append_debug_log(
                        f"  - {k}: type={type(v).__name__}, asarray.shape={arr.shape}, asarray.dtype={arr.dtype}"
                    )
                except Exception:
                    self._append_debug_log(f"  - {k}: type={type(v).__name__}, value_repr={repr(v)[:200]}")

    def add_frame(self, frame: dict) -> None:
        # torch -> numpy
        for name in list(frame.keys()):
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        # only validate hf_features keys
        features = {key: value for key, value in self.features.items() if key in self.hf_features}
        features = normalize_feature_shapes(features)

        # auto reshape/flatten to expected
        frame = auto_reshape_frame_by_features(frame, features)

        try:
            validate_frame(frame, features)
        except Exception as e:
            self._dump_validate_debug(e, frame, features)
            raise

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(frame.pop("task"))

        for key, value in frame.items():
            if key not in self.features:
                raise ValueError(f"Frame element not in features: '{key}' not in '{self.features.keys()}'.")
            self.episode_buffer[key].append(value)

        self.episode_buffer["size"] += 1

    def save_episode(self, videos: dict, action_config: list, episode_data: Optional[dict] = None) -> None:
        episode_buffer = episode_data if episode_data is not None else self.episode_buffer
        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        self.meta.save_episode_tasks(episode_tasks)
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key]).squeeze()

        for key in self.meta.video_keys:
            episode_buffer[key] = str(videos[key])

        ep_stats = compute_episode_stats(episode_buffer, self.features)
        ep_metadata = self._save_episode_data(episode_buffer)

        has_video_keys = len(self.meta.video_keys) > 0
        use_batched_encoding = self.batch_encoding_size > 1

        self.current_videos = videos
        if has_video_keys and not use_batched_encoding:
            for video_key in self.meta.video_keys:
                ep_metadata.update(self._save_episode_video(video_key, episode_index))

        ep_metadata.update({"action_config": action_config})
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, ep_metadata)

        if has_video_keys and use_batched_encoding:
            self.episodes_since_last_encoding += 1
            if self.episodes_since_last_encoding == self.batch_encoding_size:
                start_ep = self.num_episodes - self.batch_encoding_size
                end_ep = self.num_episodes
                self._batch_save_episode_video(start_ep, end_ep)
                self.episodes_since_last_encoding = 0

        if episode_data is None:
            self.clear_episode_buffer(delete_images=len(self.meta.image_keys) > 0)

    def _encode_temporary_episode_video(self, video_key: str, episode_index: int) -> Path:
        temp_path = Path(tempfile.mkdtemp(dir=self.root)) / f"{video_key}_{episode_index:03d}.mp4"
        shutil.copy(self.current_videos[video_key], temp_path)
        return temp_path


# ----------------------------
# Tasks
# ----------------------------
def get_all_tasks(src_path: Path, output_path: Path):
    json_files = src_path.glob("task_info/*.json")
    for json_file in json_files:
        local_dir = output_path / "agibotworld" / json_file.stem
        yield (json_file, local_dir.resolve())


@dataclass
class TaskResult:
    task_key: str
    ok: bool
    note: str = ""


def save_as_lerobot_dataset(
    agibot_world_config,
    task: Tuple[Path, Path],
    save_depth: bool,
    tracker,
    flatten_bimanual: bool,
    strict_integrity: bool,
) -> TaskResult:
    warnings.filterwarnings("ignore", category=UserWarning)

    json_file, local_dir = task
    task_key = json_file.stem
    success_path = local_dir / SUCCESS_FILE

    try:
        # -------------------------
        # Resume
        # -------------------------
        if success_path.exists():
            tracker.register.remote(task_key, 1)
            tracker.inc.remote(task_key, 1)
            tracker.finish.remote(task_key, True, "skipped (_SUCCESS exists)")
            return TaskResult(task_key, True, "skipped (_SUCCESS exists)")

        if local_dir.exists() and not success_path.exists():
            shutil.rmtree(local_dir, ignore_errors=True)

        # -------------------------
        # Task info / features
        # -------------------------
        src_path = json_file.parent.parent
        task_info_list = get_task_info(json_file)
        task_name = task_info_list[0]["task_name"]
        task_init_scene = task_info_list[0]["init_scene_text"]
        task_instruction = f"{task_name} | {task_init_scene}"
        task_id = json_file.stem.split("_")[-1]
        task_info = {ep["episode_id"]: ep for ep in task_info_list}

        features = generate_features_from_config(agibot_world_config)
        if not save_depth:
            features.pop("observation.images.head_depth", None)

        # normalize + optional bimanual flatten
        features = normalize_feature_shapes(features)
        features = flatten_bimanual_features_by_shape(features, enabled=flatten_bimanual)

        dataset: AgiBotDataset = AgiBotDataset.create(
            repo_id=json_file.stem,
            root=local_dir,
            fps=30,
            robot_type="a2d",
            features=features,
        )

        # per-task debug log
        dataset.debug_log_path = local_dir / "debug_shapes.log"
        dump_features_once(dataset.debug_log_path, features, task_key)

        # -------------------------
        # Enumerate episodes
        # -------------------------
        all_subdir = [f.as_posix() for f in src_path.glob(f"observations/{task_id}/*") if f.is_dir()]
        all_subdir_eids = sorted([int(Path(path).name) for path in all_subdir])

        expected_eids = [eid for eid in all_subdir_eids if eid in task_info]
        tracker.register.remote(task_key, len(expected_eids))

        saved_ok = 0
        missing_videos = 0
        corrupted_mp4 = 0

        for eid in expected_eids:
            action_config = task_info[eid]["label_info"]["action_config"]

            raw_dataset = load_local_dataset(
                eid,
                src_path=src_path,
                task_id=task_id,
                save_depth=save_depth,
                AgiBotWorld_CONFIG=agibot_world_config,
            )
            _, frames, videos = raw_dataset

            if not all([Path(v).exists() for v in videos.values()]):
                missing_videos += 1
                tracker.inc.remote(task_key, 1)
                continue

            for frame_data in frames:
                frame_data["task"] = task_instruction
                dataset.add_frame(frame_data)

            try:
                # ✅只静音这一段（ffmpeg/libav 输出）
                with suppress_fds(True):
                    dataset.save_episode(videos=videos, action_config=action_config)
                saved_ok += 1
            except Exception as e:
                corrupted_mp4 += 1
                dataset.episode_buffer = None
            finally:
                gc.collect()
                tracker.inc.remote(task_key, 1)

        # -------------------------
        # Integrity + _SUCCESS
        # -------------------------
        expected = len(expected_eids) if strict_integrity else max(len(expected_eids) - missing_videos - corrupted_mp4, 0)

        meta_total = getattr(dataset.meta, "total_episodes", None)
        if meta_total is None:
            meta_total = getattr(dataset, "num_episodes", None)

        ok = (meta_total == expected)
        note = (
            f"saved={saved_ok}/{len(expected_eids)}, "
            f"missing_videos={missing_videos}, corrupted={corrupted_mp4}, "
            f"meta.total_episodes={meta_total}, expected={expected}, "
            f"flatten_bimanual={flatten_bimanual}"
        )

        if ok:
            success_path.write_text(note + "\n", encoding="utf-8")

        tracker.finish.remote(task_key, ok, note)
        return TaskResult(task_key, ok, note)

    except Exception as e:
        err = f"exception: {type(e).__name__}: {e}"
        tracker.finish.remote(task_key, False, err)
        return TaskResult(task_key, False, err)


# ----------------------------
# Driver-side tqdm renderer (clean, no Ray pid prefixes)
# ----------------------------
def render_progress_driver(tracker, task_keys: List[str], poll_interval: float = 0.2):
    # fixed positions: global=0, each task idx+1
    pos_map = {k: i + 1 for i, k in enumerate(task_keys)}

    global_bar = tqdm(total=len(task_keys), position=0, desc="All tasks", dynamic_ncols=True)
    task_bars: Dict[str, tqdm] = {}
    finished = set()

    try:
        while True:
            st = ray.get(tracker.snapshot.remote())

            for k in task_keys:
                info = st.get(k, None)
                if info is None:
                    continue

                if k not in task_bars:
                    task_bars[k] = tqdm(
                        total=max(int(info["total"]), 0),
                        position=pos_map[k],
                        desc=k,
                        dynamic_ncols=True,
                        leave=True,
                    )

                bar = task_bars[k]

                # update total (worker may register later)
                new_total = int(info["total"])
                if new_total != bar.total:
                    bar.total = new_total
                    bar.refresh()

                # update done
                done = int(info["done"])
                delta = done - bar.n
                if delta > 0:
                    bar.update(delta)

                # handle finished
                if info["finished"] and k not in finished:
                    finished.add(k)
                    global_bar.update(1)
                    note = info.get("note", "")
                    ok = bool(info.get("ok", False))
                    if note:
                        bar.set_postfix_str(("OK " if ok else "FAIL ") + note)

            if len(finished) == len(task_keys):
                break

            time.sleep(poll_interval)

    finally:
        for b in task_bars.values():
            try:
                b.close()
            except Exception:
                pass
        try:
            global_bar.close()
        except Exception:
            pass


# ----------------------------
# Main
# ----------------------------
def main(
    src_path: str,
    output_path: str,
    eef_type: str,
    task_ids: list,
    cpus_per_task: int,
    save_depth: bool,
    debug: bool = False,
    flatten_bimanual: bool = False,
    strict_integrity: bool = True,
):
    tasks_iter = get_all_tasks(Path(src_path), Path(output_path))

    agibot_world_config, type_task_ids = (
        AgiBotWorld_TASK_TYPE[eef_type]["task_config"],
        AgiBotWorld_TASK_TYPE[eef_type]["task_ids"],
    )

    if eef_type == "gripper":
        remaining_ids = AgiBotWorld_TASK_TYPE["dexhand"]["task_ids"] + AgiBotWorld_TASK_TYPE["tactile"]["task_ids"]
        tasks_iter = filter(lambda task: task[0].stem not in remaining_ids, tasks_iter)
    else:
        tasks_iter = filter(lambda task: task[0].stem in type_task_ids, tasks_iter)

    if task_ids:
        tasks_iter = filter(lambda task: task[0].stem in task_ids, tasks_iter)

    tasks = list(tasks_iter)

    if debug:
        # single-process debug
        json_file, local_dir = tasks[0]
        print(f"[DEBUG] processing {json_file.stem} -> {local_dir}")
        print(f"[DEBUG] features dump -> {local_dir / 'debug_shapes.log'}")

        class DummyTracker:
            def register(self, *a, **k): pass
            def inc(self, *a, **k): pass
            def finish(self, *a, **k): pass

        res = save_as_lerobot_dataset(
            agibot_world_config,
            (json_file, local_dir),
            save_depth,
            tracker=DummyTracker(),
            flatten_bimanual=flatten_bimanual,
            strict_integrity=strict_integrity,
        )
        print(res)
        return

    runtime_env = RuntimeEnv(
        env_vars={
            "HDF5_USE_FILE_LOCKING": "FALSE",
            "HF_DATASETS_DISABLE_PROGRESS_BARS": "TRUE",
            "PYTHONWARNINGS": "ignore::UserWarning",
            "AV_LOG_LEVEL": "quiet",
            "RAY_DEDUP_LOGS": "1",
        }
    )
    ray.init(runtime_env=runtime_env)

    task_keys = [json_file.stem for (json_file, _) in tasks]

    tracker = ProgressTrackerActor.remote()
    ray.get(tracker.init_tasks.remote(task_keys))

    resources = ray.available_resources()
    cpus = int(resources.get("CPU", 1))
    print(f"Available CPUs: {cpus}, num_cpus_per_task: {cpus_per_task}, total_tasks: {len(tasks)}")

    remote_task = ray.remote(save_as_lerobot_dataset).options(num_cpus=cpus_per_task)

    futures = []
    for (json_file, local_dir) in tasks:
        futures.append(
            remote_task.remote(
                agibot_world_config,
                (json_file, local_dir),
                save_depth,
                tracker,
                flatten_bimanual,
                strict_integrity,
            )
        )

    # ✅ clean tqdm rendering in driver (no '(Actor pid=...)' noise)
    render_progress_driver(tracker, task_keys, poll_interval=0.2)

    results: List[TaskResult] = ray.get(futures)

    failed = [r for r in results if not r.ok]
    if failed:
        with open("output.txt", "a", encoding="utf-8") as f:
            for r in failed:
                f.write(f"{r.task_key}, {r.note}\n")

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--eef-type", type=str, choices=["gripper", "dexhand", "tactile"], default="gripper")
    parser.add_argument("--task-ids", type=str, nargs="+", help="task_327 task_351 ...", default=[])
    parser.add_argument("--cpus-per-task", type=int, default=3)
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--debug", action="store_true")

    # ✅双臂拉平成一维（按 shape 第0维=2）
    parser.add_argument("--flatten-bimanual", action="store_true")

    # strict integrity by default; set this to be best-effort
    parser.add_argument("--non-strict-integrity", action="store_true")

    args = parser.parse_args()

    main(
        src_path=str(args.src_path),
        output_path=str(args.output_path),
        eef_type=args.eef_type,
        task_ids=args.task_ids,
        cpus_per_task=args.cpus_per_task,
        save_depth=args.save_depth,
        debug=args.debug,
        flatten_bimanual=args.flatten_bimanual,
        strict_integrity=not args.non_strict_integrity,
    )