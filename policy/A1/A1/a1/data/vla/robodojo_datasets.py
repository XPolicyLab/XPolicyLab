"""RoboTwin 3.0 / RoboDojo 数据读取器。

数据布局（每个任务一个目录）：

    <task_dir>/arx_x5/data/episode_0000000.hdf5
    <task_dir>/arx_x5/data/episode_0000001.hdf5
    ...

每个 episode 的 HDF5（data_format_version=v1.0, 25Hz, ARX5 双臂）结构：

    instruction                              -> str（该 episode 的语言指令）
    additional_info/frequency                -> int
    action/{left,right}_arm_joint_states     -> (T, 6)   关节角
    action/{left,right}_ee_joint_states      -> (T, 1)   夹爪
    state/{left,right}_arm_joint_states      -> (T, 6)
    state/{left,right}_ee_joint_states       -> (T, 1)
    state/{left,right}_ee_poses              -> (T, 7)   xyz + 四元数(xyzw)
    state/{left,right}_delta_ee_poses        -> (T, 7)
    vision/<cam>/colors                      -> (T,) JPEG 字节串, 解码后 480x640x3 RGB
    vision/<cam>/{intrinsic,extrinsic}_matrix, shape

动作空间（与现有 maniparena joint_delta 配方一致，delta_mask=[6,-1,6,-1]）：

    joint: [左臂6 | 左爪1 | 右臂6 | 右爪1]  共 14 维
    ee   : [左ee_pose7 | 左爪1 | 右ee_pose7 | 右爪1] 共 16 维（由 state/*_ee_poses 构建）

本读取器为 map-style `Dataset`，被 `IterableDatasetWrapper` 包一层后随机访问。
为兼容多 worker（fork/spawn）与避免 h5py 句柄跨进程问题，初始化时只读取每个
episode 的帧数与指令，真正的 array / 图像在 `get()` 内按需打开文件读取。
"""

import io
import os
import glob
import json
import logging
from typing import Dict, List, Optional

import h5py
import numpy as np
from PIL import Image

from a1.data.dataset import Dataset
from a1.data.vla.utils import NormalizationType
from a1.data.vla.maniparena_datasets import normalize_action_and_proprio, make_bool_mask

log = logging.getLogger(__name__)


# 关节动作各分段在 action/state 组中的字段顺序（拼接后即 14 维向量）
_JOINT_FIELDS = [
    "left_arm_joint_states",   # 6
    "left_ee_joint_states",    # 1
    "right_arm_joint_states",  # 6
    "right_ee_joint_states",   # 1
]

# 默认相机顺序：主相机(head) + 左右腕部相机
_DEFAULT_CAMERA_KEYS = ["cam_head", "cam_left_wrist", "cam_right_wrist"]


def _decode_jpeg(raw) -> np.ndarray:
    """HDF5 colors 元素(定长字节串, 可能 null 填充) -> HWC uint8 RGB。"""
    b = raw.tobytes() if isinstance(raw, np.ndarray) else bytes(raw)
    try:
        img = Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        # 个别帧可能带尾部填充，去掉再试一次
        img = Image.open(io.BytesIO(b.rstrip(b"\x00"))).convert("RGB")
    # 用 np.array（而非 np.asarray）强制拷贝成可写数组，避免下游 torch.from_numpy
    # 对只读数组发出 "NumPy array is not writable" 警告（仅警告，无功能影响）。
    return np.array(img, dtype=np.uint8)


class RoboDojoDatasetReader(Dataset):
    def __init__(
        self,
        dataset_path: str,
        chunk_size: int = 50,
        fixed_action_dim: int = 32,
        normalization_type: Optional[NormalizationType] = None,
        norm_stats_path: Optional[str] = None,
        use_proprio: bool = True,
        use_wrist_image: bool = True,
        camera_keys: Optional[List[str]] = None,
        action_type: str = "joint",
        delta: bool = False,
        delta_mask: Optional[List[int]] = None,
        pad_action_and_proprio: bool = True,
        num_episodes: Optional[int] = None,
        embodiment: str = "arx_x5",
        clip_value: Optional[float] = None,
    ):
        assert action_type in ("joint", "ee"), f"未知 action_type: {action_type}"
        self.dataset_path = dataset_path
        self.chunk_size = chunk_size
        self.fixed_action_dim = fixed_action_dim
        self.normalization_type = normalization_type
        self.norm_stats_path = norm_stats_path
        self.use_proprio = use_proprio
        self.use_wrist_image = use_wrist_image
        self.camera_keys = camera_keys or _DEFAULT_CAMERA_KEYS
        self.action_type = action_type
        self.pad_action_and_proprio = pad_action_and_proprio
        self.embodiment = embodiment
        self.clip_value = clip_value

        self.delta = delta
        self.delta_mask = make_bool_mask(*delta_mask) if (delta and delta_mask is not None) else None
        if self.delta:
            assert self.delta_mask is not None, "delta=True 时必须提供 delta_mask"

        if self.normalization_type is not None:
            assert self.norm_stats_path is not None, "normalization_type 非空时必须提供 norm_stats_path"
            with open(self.norm_stats_path) as f:
                self.norm_stats = json.load(f)
        else:
            self.norm_stats = None

        # 发现 episode 文件
        data_dir = self._resolve_data_dir(dataset_path)
        files = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
        if num_episodes is not None:
            files = files[: int(num_episodes)]
        if not files:
            raise FileNotFoundError(f"在 {data_dir} 下未找到 episode_*.hdf5")

        # 仅读取每条 episode 的帧数与指令，构建 (episode_idx, frame_idx) 索引
        self._episodes: List[Dict] = []
        self._index: List[tuple] = []
        for ep_i, fp in enumerate(files):
            try:
                with h5py.File(fp, "r") as f:
                    T = int(f["action/left_arm_joint_states"].shape[0])
                    instr = f["instruction"][()]
                    instr = instr.decode() if isinstance(instr, (bytes, bytearray, np.bytes_)) else str(instr)
            except Exception as e:  # 跳过损坏文件
                log.warning(f"跳过无法读取的 episode {fp}: {e}")
                continue
            if T <= 1:
                continue
            self._episodes.append({"path": fp, "num_frames": T, "instruction": instr})
            for frame_idx in range(T):
                self._index.append((ep_i, frame_idx))

        if not self._index:
            raise ValueError(f"{data_dir} 下没有可用帧")
        log.info(
            f"RoboDojo[{os.path.basename(dataset_path)}]: "
            f"{len(self._episodes)} episodes, {len(self._index)} frames, "
            f"action_type={action_type}, cameras={self.camera_keys}"
        )

    @staticmethod
    def _resolve_data_dir(dataset_path: str) -> str:
        """允许传入任务目录、<task>/arx_x5、或直接的 data 目录。"""
        cand = [
            os.path.join(dataset_path, "arx_x5", "data"),
            os.path.join(dataset_path, "data"),
            dataset_path,
        ]
        for c in cand:
            if os.path.isdir(c) and glob.glob(os.path.join(c, "episode_*.hdf5")):
                return c
        # 兜底：自动在两层内搜索包含 episode_*.hdf5 的目录
        for c in cand:
            if os.path.isdir(c):
                hits = glob.glob(os.path.join(c, "**", "episode_*.hdf5"), recursive=True)
                if hits:
                    return os.path.dirname(hits[0])
        return cand[0]

    # ---- array 组装 ----
    @staticmethod
    def _stack_group(grp) -> np.ndarray:
        return np.concatenate([np.asarray(grp[k]) for k in _JOINT_FIELDS], axis=-1).astype(np.float32)

    def _read_arrays(self, f) -> tuple:
        """返回 (action_seq, state_seq)，形状均为 (T, D)。"""
        if self.action_type == "joint":
            act = self._stack_group(f["action"])   # (T,14)
            st = self._stack_group(f["state"])      # (T,14)
        else:  # ee：action 组无 ee_pose，用 state 的 ee_poses 构建 16 维
            s = f["state"]
            def ee(side):
                return np.concatenate(
                    [np.asarray(s[f"{side}_ee_poses"]), np.asarray(s[f"{side}_ee_joint_states"])],
                    axis=-1,
                )  # (T,8) = xyz3+quat4+grip1
            st = np.concatenate([ee("left"), ee("right")], axis=-1).astype(np.float32)  # (T,16)
            act = st.copy()  # ee 监督即未来 ee 位姿轨迹
        return act, st

    def __len__(self) -> int:
        return len(self._index)

    def get(self, item, rng=None):
        if not (isinstance(item, (int, np.integer)) and 0 <= int(item) < len(self._index)):
            raise ValueError(f"非法 item: {item}")
        ep_i, frame_idx = self._index[int(item)]
        ep = self._episodes[ep_i]

        with h5py.File(ep["path"], "r") as f:
            act_seq, st_seq = self._read_arrays(f)
            images = self._load_images(f, frame_idx)

        T = act_seq.shape[0]
        frame_idx = min(frame_idx, T - 1)

        frame_state = st_seq[frame_idx: frame_idx + 1]                  # (1, D)
        frame_action = act_seq[frame_idx: frame_idx + self.chunk_size]  # (<=chunk, D)
        if frame_action.shape[0] < self.chunk_size:
            pad = np.repeat(frame_action[-1:], self.chunk_size - frame_action.shape[0], axis=0)
            frame_action = np.concatenate([frame_action, pad], axis=0)

        if self.clip_value is not None:
            np.clip(frame_state, -self.clip_value, self.clip_value, out=frame_state)
            np.clip(frame_action, -self.clip_value, self.clip_value, out=frame_action)

        # delta：动作减去当前帧状态（仅在 mask=True 的维度上）
        if self.delta:
            mask = np.asarray(self.delta_mask)
            dims = mask.shape[-1]
            state_form = np.where(mask, frame_state[..., :dims], 0.0)   # (1, dims)
            frame_action[..., :dims] = frame_action[..., :dims] - state_form

        # 归一化（与训练侧 maniparena 完全一致的实现）
        if self.normalization_type is not None:
            norm = normalize_action_and_proprio(
                {"actions": frame_action, "state": frame_state},
                self.norm_stats,
                {"actions": "actions", "state": "state"},
                self.normalization_type,
            )
            frame_action, frame_state = norm["actions"], norm["state"]

        # pad 到模型固定动作维度
        if self.pad_action_and_proprio:
            if frame_action.shape[-1] < self.fixed_action_dim:
                frame_action = np.pad(frame_action, ((0, 0), (0, self.fixed_action_dim - frame_action.shape[-1])))
            if frame_state.shape[-1] < self.fixed_action_dim:
                frame_state = np.pad(frame_state, ((0, 0), (0, self.fixed_action_dim - frame_state.shape[-1])))

        frame_action = frame_action.astype(np.float32)
        frame_state = frame_state.astype(np.float32)

        return_dict = {
            "question": ep["instruction"],
            "timestep": frame_idx / 25.0,
            "answer": "Action",
            "style": "action",
            "action": frame_action,
            "action_pad_mask": np.zeros_like(frame_action, dtype=bool),
            "proprio": frame_state if self.use_proprio else None,
            "metadata": {
                "frame_index": frame_idx,
                "episode_index": ep_i,
                "index": int(item),
                "task_index": 0,
                "task": ep["instruction"],
                "file_path": ep["path"],
                "num_frames": ep["num_frames"],
                "embodiment": self.embodiment,
            },
        }
        if self.use_wrist_image:
            return_dict["images"] = images
        else:
            return_dict["image"] = [images[0]]
        return return_dict

    def _load_images(self, f, frame_idx: int) -> List[np.ndarray]:
        imgs = []
        for cam in self.camera_keys:
            ds = f.get(f"vision/{cam}/colors")
            if ds is None:
                continue
            fi = min(frame_idx, ds.shape[0] - 1)
            imgs.append(_decode_jpeg(ds[fi]))
        if not imgs:
            raise ValueError(f"未读到任何相机图像，camera_keys={self.camera_keys}")
        return imgs

    @classmethod
    def download(cls, n_procs=1):
        raise NotImplementedError()
