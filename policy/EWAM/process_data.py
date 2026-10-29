"""Prepare EWAM training data for RoboTwin or RoboDojo.

RoboTwin -> the upstream ``robotwin_dim16`` layout:
    <out>/<clean|randomized>/<task>/videos/<i>.mp4    T-shape 3-view video, 320x360, 30 fps
    <out>/<clean|randomized>/<task>/qpos/<i>.pt       (T, 14) joint actions [L6, LG, R6, RG]
    <out>/<clean|randomized>/<task>/metas/<i>.txt     scene prefix + instruction, one per line
    <out>/<clean|randomized>/<task>/umt5_wan/<i>.pt   list of UMT5-XXL embeddings, one per line
  Inputs: XPolicyLab-standard HDF5 (vision/cam_*/colors, action|state/*_joint_states,
  instruction(s)) or the RoboTwin 2.0 release (observation/*_camera/rgb,
  joint_action/vector, ../instructions/<episode>.json).

RoboDojo -> read directly by the ``robodojo`` loader:
    <out>/episodes/<task>/episode_*.hdf5   links to the raw episodes
    <out>/t5_cache/<sha1>.pt               UMT5-XXL embedding of each distinct instruction, (seq, 4096)
"""

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

POLICY_DIR = Path(__file__).resolve().parent
# Parent of this checkout: `import XPolicyLab` and `<parent>/data/<bench>`.
WORKSPACE_DIR = POLICY_DIR.parents[2]
EWAM_ROOT = POLICY_DIR / "EWAM"
for path in (WORKSPACE_DIR, EWAM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from XPolicyLab.utils.process_data import decode_image_bit, get_robot_action_dim_info  # noqa: E402

from data.robodojo.robodojo_dataset import instruction_key, read_instruction  # noqa: E402

ROBOTWIN_SCENE_PREFIX = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
    "The aloha robot is currently performing the following task: "
)
VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS = 320, 360, 30
QPOS_ABS_LIMIT = 4.0
STANDARD_CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
NATIVE_CAMERAS = ("head_camera", "left_camera", "right_camera")


def find_episodes(task_dir: Path, env_cfg_type: str) -> list[Path]:
    for data_dir in (task_dir / env_cfg_type / "data", task_dir / "data", task_dir / "demo_clean" / "data", task_dir):
        files = sorted(data_dir.glob("*.hdf5"))
        if files:
            return files
    return []


def resolve_task_dirs(raw_root: Path, raw_task_dirs: str) -> list[Path]:
    if raw_task_dirs == "all":
        dirs = sorted(p for p in raw_root.iterdir() if p.is_dir())
    else:
        dirs = [raw_root / name.strip() for name in raw_task_dirs.split(",") if name.strip()]
    missing = [str(d) for d in dirs if not d.is_dir()]
    if missing:
        raise FileNotFoundError(f"raw task dirs not found: {missing}")
    return dirs


def load_t5_encoder(wan_path: Path, device: str):
    from wan.modules.t5 import T5EncoderModel

    return T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=torch.device(device),
        checkpoint_path=str(wan_path / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(wan_path / "google" / "umt5-xxl"),
    )


def encode_texts(encoder, texts: list[str], device: str, batch_size: int = 16) -> list[torch.Tensor]:
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            out += [emb.cpu() for emb in encoder(texts[i:i + batch_size], torch.device(device))]
    return out


# ----------------------------------------------------------------------------- RoboTwin


def _robotwin_instructions(f: h5py.File, hdf5_path: Path) -> list[str]:
    if "instructions" in f:
        raw = f["instructions"][()]
        raw = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = [raw]
            return [str(x) for x in (parsed if isinstance(parsed, list) else [parsed])]
        return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in np.asarray(raw).reshape(-1)]
    if "instruction" in f:
        return [read_instruction(f)]
    for candidate in (hdf5_path.parent / "instructions", hdf5_path.parent.parent / "instructions"):
        json_path = candidate / f"{hdf5_path.stem}.json"
        if json_path.is_file():
            return [str(x) for x in json.loads(json_path.read_text(encoding="utf-8"))["seen"]]
    return []


def _robotwin_episode(hdf5_path: Path) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    with h5py.File(hdf5_path, "r") as f:
        if "vision" in f:
            frames = [decode_image_bit(f[f"vision/{cam}/colors"][()]) for cam in STANDARD_CAMERAS]
            group = "action" if "action/left_arm_joint_states" in f else "state"
            n = len(frames[0])
            parts = []
            for side in ("left", "right"):
                parts.append(f[f"{group}/{side}_arm_joint_states"][()].reshape(n, -1)[:, :6])
                parts.append(f[f"{group}/{side}_ee_joint_states"][()].reshape(n, -1)[:, :1])
            qpos = np.concatenate(parts, axis=1)
        else:
            frames = [decode_image_bit(f[f"observation/{cam}/rgb"][()]) for cam in NATIVE_CAMERAS]
            qpos = f["joint_action/vector"][()]
        instructions = _robotwin_instructions(f, hdf5_path)
    return frames, np.asarray(qpos, dtype=np.float32), instructions


def _t_layout(head: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h, w = head.shape[:2]
    bottom = np.hstack([cv2.resize(left, (w // 2, h // 2)), cv2.resize(right, (w // 2, h // 2))])
    frame = np.vstack([head, bottom])
    if frame.shape[:2] != (VIDEO_HEIGHT, VIDEO_WIDTH):
        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
    return frame


def convert_robotwin_episode(job: tuple[Path, Path, int]) -> str:
    hdf5_path, task_out, episode_id = job
    paths = {name: task_out / name / f"{episode_id}.{ext}"
             for name, ext in (("videos", "mp4"), ("qpos", "pt"), ("metas", "txt"))}
    if all(p.is_file() for p in paths.values()):
        return "skipped"
    frames, qpos, instructions = _robotwin_episode(hdf5_path)
    if qpos.ndim != 2 or qpos.shape[1] != 14:
        return f"bad qpos shape {qpos.shape}: {hdf5_path}"
    if np.abs(qpos).max() > QPOS_ABS_LIMIT:
        return f"rejected (|qpos| > {QPOS_ABS_LIMIT}): {hdf5_path}"
    if len({len(x) for x in frames}) != 1:
        return f"camera frame counts differ: {hdf5_path}"

    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(paths["videos"]), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS,
                             (VIDEO_WIDTH, VIDEO_HEIGHT))
    for head, left, right in zip(*frames):
        writer.write(cv2.cvtColor(_t_layout(head, left, right), cv2.COLOR_RGB2BGR))
    writer.release()
    torch.save(torch.from_numpy(qpos), paths["qpos"])
    lines = instructions or [""]
    paths["metas"].write_text("".join(f"{ROBOTWIN_SCENE_PREFIX}{text}\n" for text in lines), encoding="utf-8")
    return "ok"


def prepare_robotwin(args, task_dirs: list[Path], out_dir: Path) -> None:
    jobs, next_id = [], {}
    for task_dir in task_dirs:
        split = "randomized" if "randomized" in str(task_dir.relative_to(args.raw_root)) else "clean"
        task = task_dir.relative_to(args.raw_root).parts[0]
        files = find_episodes(task_dir, args.env_cfg_type)
        if args.expert_data_num:
            files = files[: args.expert_data_num]
        start = next_id.get((split, task), 0)
        jobs += [(p, out_dir / split / task, start + i) for i, p in enumerate(files)]
        next_id[(split, task)] = start + len(files)
        print(f"[EWAM] {task_dir} -> {split}/{task}: {len(files)} episodes")
    if not jobs:
        raise RuntimeError("no RoboTwin episodes found")

    with mp.Pool(args.workers) as pool:
        for status in pool.imap_unordered(convert_robotwin_episode, jobs):
            if status not in ("ok", "skipped"):
                print(f"[EWAM] {status}")

    metas = sorted(out_dir.glob("*/*/metas/*.txt"))
    pending = [m for m in metas if not (m.parent.parent / "umt5_wan" / f"{m.stem}.pt").is_file()]
    print(f"[EWAM] T5: {len(pending)} / {len(metas)} episodes to encode")
    if pending:
        encoder = load_t5_encoder(args.wan_path, args.device)
        for meta in pending:
            prompts = [line for line in meta.read_text(encoding="utf-8").split("\n") if line.strip()]
            target = meta.parent.parent / "umt5_wan" / f"{meta.stem}.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(encode_texts(encoder, prompts, args.device), target)


# ----------------------------------------------------------------------------- RoboDojo


def prepare_robodojo(args, task_dirs: list[Path], out_dir: Path) -> None:
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    instructions = set()
    for task_dir in task_dirs:
        files = find_episodes(task_dir, args.env_cfg_type)
        if args.expert_data_num:
            files = files[: args.expert_data_num]
        if not files:
            raise RuntimeError(f"no episodes under {task_dir}")
        link_dir = episodes_dir / task_dir.name
        link_dir.mkdir(exist_ok=True)
        for p in files:
            link = link_dir / p.name
            if not link.exists():
                link.symlink_to(p.resolve())
            with h5py.File(p, "r") as f:
                instructions.add(read_instruction(f))
        print(f"[EWAM] {task_dir}: {len(files)} episodes")

    cache_dir = out_dir / "t5_cache"
    cache_dir.mkdir(exist_ok=True)
    pending = sorted(t for t in instructions if not (cache_dir / f"{instruction_key(t)}.pt").is_file())
    print(f"[EWAM] T5: {len(pending)} / {len(instructions)} instructions to encode")
    if pending:
        encoder = load_t5_encoder(args.wan_path, args.device)
        for text, emb in zip(pending, encode_texts(encoder, pending, args.device)):
            torch.save(emb, cache_dir / f"{instruction_key(text)}.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bench_name", choices=["RoboTwin", "RoboDojo"])
    parser.add_argument("ckpt_name", help="run name; also the default raw task dir")
    parser.add_argument("env_cfg_type")
    parser.add_argument("action_type", choices=["joint"])
    parser.add_argument("expert_data_num", nargs="?", default=None,
                        help="first N episodes per task dir; a non-numeric value is taken as raw_task_dirs")
    parser.add_argument("raw_task_dirs", nargs="?", default=None,
                        help="comma-separated dirs under raw_root, or 'all' (default: ckpt_name)")
    parser.add_argument("--raw_root", type=Path, default=None, help="default: <parent>/data/<bench_name>")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="default: policy/EWAM/data/<bench>-<ckpt>-<env>-<action>")
    parser.add_argument("--wan_path", type=Path, default=POLICY_DIR / "checkpoints" / "Wan2.2-TI2V-5B",
                        help="Wan2.2-TI2V-5B dir holding the UMT5-XXL encoder and tokenizer")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    raw_task_dirs = args.raw_task_dirs
    if args.expert_data_num is not None and not args.expert_data_num.isdigit():
        if raw_task_dirs is not None:
            parser.error("raw_task_dirs was provided twice")
        raw_task_dirs, args.expert_data_num = args.expert_data_num, None
    args.expert_data_num = int(args.expert_data_num) if args.expert_data_num else 0
    args.raw_root = args.raw_root or WORKSPACE_DIR / "data" / args.bench_name
    out_dir = args.out_dir or (
        POLICY_DIR / "data" / f"{args.bench_name}-{args.ckpt_name}-{args.env_cfg_type}-{args.action_type}")

    dims = get_robot_action_dim_info(args.env_cfg_type)
    if list(dims["arm_dim"]) != [6, 6] or list(dims["ee_dim"]) != [1, 1]:
        raise ValueError(f"EWAM expects a dual 6+1 arm, got {args.env_cfg_type}: {dims}")

    task_dirs = resolve_task_dirs(args.raw_root, raw_task_dirs or args.ckpt_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.bench_name == "RoboTwin":
        prepare_robotwin(args, task_dirs, out_dir)
    else:
        prepare_robodojo(args, task_dirs, out_dir)
    print(f"[EWAM] data ready: {out_dir}")


if __name__ == "__main__":
    main()
