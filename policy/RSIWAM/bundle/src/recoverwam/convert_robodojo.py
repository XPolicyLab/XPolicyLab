"""Convert the official RoboDojo EE16 LeRobot v3 export to RecoverWAM episodes.

Inputs are standard MP4s and Parquet tables, not XPolicyLab JPEG/HDF5 buffers.
Keep source action/state values and native timing; never shift targets again.
"""

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .data import DEFAULT_PROMPT

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
EE_NAMES = ["l_x", "l_y", "l_z", "l_w", "l_wx", "l_wy", "l_wz", "l_g",
            "r_x", "r_y", "r_z", "r_w", "r_wx", "r_wy", "r_wz", "r_g"]


def inside(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_episodes(source, tasks):
    info = json.loads((source / "meta/info.json").read_text())
    if (info["codebase_version"], info["fps"], info["total_episodes"],
            info["total_frames"]) != ("v3.0", 25, 3500, 1856102):
        raise ValueError("Expected the official shift-fixed EE16 v3 export: "
                         "3500 episodes / 1856102 frames / 25 Hz")
    for name in ("action", "observation.state"):
        feature = info["features"][name]
        names = feature["names"]
        if names and isinstance(names[0], list):
            names = names[0]
        if feature["dtype"] != "float32" or feature["shape"] != [16] or names != EE_NAMES:
            raise ValueError(f"{name}: expected absolute EE16 [xyz, qwxyz, gripper] x 2")
    with Path(tasks).open(newline="") as f:
        task_rows = list(csv.DictReader(f))
    task_map = {r["instruction"]: r for r in task_rows}
    paths = sorted((source / "meta/episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(source / "meta/episodes")
    rows = sorted(pa.concat_tables([pq.read_table(p) for p in paths]).to_pylist(),
                  key=lambda r: r["episode_index"])
    if [r["episode_index"] for r in rows] != list(range(3500)):
        raise ValueError("Expected unique episode IDs 0..3499")
    if sum(r["length"] for r in rows) != info["total_frames"]:
        raise ValueError("Episode lengths disagree with meta/info.json")
    episodes = []
    for row in rows:
        index, length = int(row["episode_index"]), int(row["length"])
        instructions = row["tasks"]
        if len(instructions) != 1 or instructions[0].strip() not in task_map:
            raise ValueError(f"Episode {index}: unknown instruction {instructions!r}")
        task = task_map[instructions[0].strip()]
        if not int(task["first_episode"]) <= index < int(task["first_episode"]) + 100:
            raise ValueError(f"Episode {index}: task order differs from the reference split")
        start, end = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        if length <= 0 or end - start != length:
            raise ValueError(f"Episode {index}: invalid data bounds")
        data = inside(source, info["data_path"].format(
            chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]))
        cameras = []
        for camera in CAMERAS:
            key = f"observation.images.{camera}"
            prefix = f"videos/{key}"
            a, b = float(row[f"{prefix}/from_timestamp"]), float(row[f"{prefix}/to_timestamp"])
            # A small number of source streams lack the last frame. Repeat only
            # that final frame, within this episode; never read the next episode.
            if not np.isfinite([a, b]).all() or a < 0 or b <= a or abs((b-a)*25-length) > 1.01:
                raise ValueError(f"Episode {index}/{camera}: invalid camera time range")
            video = inside(source, info["video_path"].format(
                video_key=key, chunk_index=row[f"{prefix}/chunk_index"],
                file_index=row[f"{prefix}/file_index"]))
            cameras.append((video, a, b))
        episodes.append(dict(index=index, length=length, start=start, data=data,
                             cameras=cameras, task=task))
    return episodes


def episode_table(source_table, episode):
    index, length = episode["index"], episode["length"]
    offset = episode["start"] - int(source_table["index"][0].as_py())
    if offset < 0 or offset + length > len(source_table):
        raise ValueError(f"Episode {index}: data bounds exceed the Parquet shard")
    table = source_table.slice(offset, length)
    for column, expected in (
        ("index", np.arange(episode["start"], episode["start"] + length)),
        ("episode_index", np.full(length, index)),
        ("frame_index", np.arange(length)),
    ):
        if not np.array_equal(table[column].to_numpy(), expected):
            raise ValueError(f"Episode {index}: invalid {column} alignment")
    values = {}
    for name in ("action", "observation.state"):
        values[name] = table[name].combine_chunks().cast(pa.list_(pa.float32()))
        array = np.asarray(values[name].to_pylist(), dtype=np.float32)
        if array.shape != (length, 16) or not np.isfinite(array).all():
            raise ValueError(f"Episode {index}: invalid {name}")
    return pa.table({
        "frame_index": pa.array(np.arange(length)),
        "raw_frame_index": pa.array(np.arange(length)),
        **values,
        "instruction": [episode["task"]["instruction"]] * length,
    })


def render_video(episode, output):
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
               "-filter_complex_threads", "1"]
    filters = []
    for i, (path, start, end) in enumerate(episode["cameras"]):
        with av.open(str(path)) as video:
            if video.streams.video[0].average_rate != 25:
                raise ValueError(f"Expected a 25 Hz source video: {path}")
        command += ["-threads", "1", "-ss", f"{start:.6f}", "-t", f"{end-start:.6f}",
                    "-i", str(path)]
        scale = ("scale=640:480" if i == 0 else
                 "crop=w='min(iw,ih*4/3)':h='min(ih,iw*3/4)',scale=240:240")
        filters.append(f"[{i}:v]fps=25,tpad=stop_mode=clone:stop_duration=0.04,"
                       f"trim=end_frame={episode['length']},{scale},setsar=1,"
                       f"setpts=N/(25*TB)[v{i}]")
    filters += ["[v1][v2]vstack=inputs=2[side]", "[v0][side]hstack=inputs=2[v]"]
    temporary = output.with_suffix(".partial.mp4")
    command += ["-filter_complex", ";".join(filters), "-map", "[v]", "-an", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-frames:v", str(episode["length"]),
                "-threads", "1", str(temporary)]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Episode {episode['index']}: ffmpeg failed: {result.stderr}")
        with av.open(str(temporary)) as video:
            stream = video.streams.video[0]
            if (stream.width, stream.height, stream.average_rate, stream.frames) != (
                    880, 480, 25, episode["length"]):
                raise ValueError(f"Invalid output frame count/size/rate: {temporary}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def convert(source, output, tasks, workers=4, limit=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError("workers and limit must be positive")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Install ffmpeg with libx264 support first")
    episodes = load_episodes(source, tasks)
    if limit is not None:
        episodes = episodes[:limit]
    output.mkdir(parents=True)
    (output / "data").mkdir()
    (output / "videos_original").mkdir()
    rows = []
    previous, table = None, None
    # Read each packed data shard once. Frame/action checks precede video work.
    for episode in episodes:
        if episode["data"] != previous:
            previous = episode["data"]
            table = pq.read_table(previous, columns=[
                "index", "episode_index", "frame_index", "action", "observation.state"])
        stem = f"episode_{episode['index']:06d}"
        parquet, video = f"data/{stem}.parquet", f"videos_original/{stem}.mp4"
        pq.write_table(episode_table(table, episode), output / parquet, compression="zstd")
        task = episode["task"]
        rows.append(dict(episode_index=episode["index"], task_name=task["task_name"],
                         dimension=task["dimension"], video_path=video, parquet_path=parquet,
                         fps=25, trim_start=0, trim_end=episode["length"],
                         num_frames=episode["length"], high_level_instruction=task["instruction"],
                         training_prompt=DEFAULT_PROMPT.format(task=task["instruction"])))
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        jobs = pool.map(render_video, episodes, [output / r["video_path"] for r in rows])
        for completed, _ in enumerate(jobs, 1):
            if completed % 25 == 0 or completed == len(rows):
                print(f"Converted {completed}/{len(rows)} episodes", flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    # Publish metadata only after every episode has passed conversion.
    with (output / "full_episode_metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(episodes=len(rows), frames=sum(r["num_frames"] for r in rows),
                   fps=25, video_size=[480, 880], source=str(source),
                   complete_reference=len(rows) == 3500)
    (output / "conversion.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
