import os
import json
# import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import h5py
import imageio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from a1.data.dataset import Dataset

from a1.data.vla.utils import quaternion_to_euler
# os.environ["VLA_CONFIG_YAML"] = "vla_config_realmachine.yaml"
from a1.vla.constants import PROPRIO_DIM, NUM_ACTIONS_CHUNK

from a1.vla.util import FIXED_ACTION_DIM


__all__ = ["AgiBotWorldAlphaDataset"]



def _concat_action_step(
    end_pos_t: Optional[np.ndarray],
    end_quat_t: Optional[np.ndarray],
    eff_pos_t: Optional[np.ndarray],
    prefer_right_first: bool = True,
) -> Optional[np.ndarray]:
    """
    将单步动作数据拼为 1D 向量，期望共 16 维：
    - 每只手 8 维: [end.position(xyz)=3, end.orientation(xyzw)=3, gripper=2]
    - 右手在前（默认），再左手。
    若任一关键数据缺失，返回 None。
    传入形状：
      end_pos_t: (2,3)  左手 [0,:] 右手 [1,:]
      end_quat_t: (2,3)
      eff_pos_t: (2,) 或 (2, k) 取 gripper 通道（若为 1 维 gripper 则直接使用）
    """
    if end_pos_t is None or end_quat_t is None or eff_pos_t is None:
        return None

    # 处理 gripper 打开度
    if eff_pos_t.ndim == 1 and eff_pos_t.shape[0] == 2:
        grip_left, grip_right = eff_pos_t[0], eff_pos_t[1]
    elif eff_pos_t.ndim == 2 and eff_pos_t.shape[0] == 2:
        # 若为多通道，取第 0 通道作为 gripper（保底）
        grip_left, grip_right = eff_pos_t[0, 0], eff_pos_t[1, 0]
    else:
        return None

    # 左右顺序
    left_vec = np.concatenate([end_pos_t[0], end_quat_t[0], np.array([grip_left], dtype=np.float32)], axis=0)
    right_vec = np.concatenate([end_pos_t[1], end_quat_t[1], np.array([grip_right], dtype=np.float32)], axis=0)

    if prefer_right_first:
        out = np.concatenate([right_vec, left_vec], axis=0)
    else:
        out = np.concatenate([left_vec, right_vec], axis=0)

    return out.astype(np.float32)


def _concat_state_step(
    end_pos_t: Optional[np.ndarray],
    end_quat_t: Optional[np.ndarray],
    eff_pos_t: Optional[np.ndarray],
    prefer_right_first: bool = True,
) -> Optional[np.ndarray]:
    """
    将单步本体状态数据拼为 1D 向量，目标维度 PROPRIO_DIM（通常为 16，与动作一致）。
    约定与动作相同：右手在前，再左手。
    """
    return _concat_action_step(end_pos_t, end_quat_t, eff_pos_t, prefer_right_first)


# -------------- Simple caches --------------

@lru_cache(maxsize=256)
def _get_video_reader_cached(video_path: str):
    # imageio reader is lightweight; keep cached per path
    return imageio.get_reader(video_path, format='ffmpeg')


@lru_cache(maxsize=512)
def _open_h5_cached(h5_path: str):
    # Cache h5py File handles for faster repeated random access
    return h5py.File(h5_path, "r")

@dataclass
class AgiBotWorldAlphaDataset(Dataset):
    """
    读取 AgiBotWorld-Alpha 数据集，并返回与 LeRobotDatasetWrapper / RLDSBatchTransform 一致的样本结构。

    返回字典字段（与 LeRobotDatasetWrapper 对齐）：
      - question: 文本指令（优先使用当前 action 的 action_text，否则使用 init_scene_text 或 task_name）
      - timestep: 当前帧索引（int）
      - answer: 固定 "Action"
      - style: 固定 "action"
      - action: (NUM_ACTIONS_CHUNK, ACTION_DIM) np.float32
      - proprio: (PROPRIO_DIM,) np.float32（当前帧）
      - images 或 image: 图像 numpy 数组，HWC uint8；当 use_wrist_image=True 时返回 images=[primary, wrist]
      - episode_index: episode_id（int）
      - metadata: 附加信息（set_id, episode_id, frame_index, instruction 等）
    """

    root_dir: str
    primary_camera: str = "head_color"
    right_camera: Optional[str] = "hand_right_color"
    left_camera: Optional[str] = "hand_left_color"
    use_proprio: bool = True
    use_wrist_image: bool = False
    pad_action_and_proprio: bool = True
    normalization_type: Optional[str] = None
    prefer_right_first: bool = False

    # 通过 Excel 指定可用帧范围：lfwj_range.py 生成的 filtered_frame_ranges.xlsx
    frame_ranges_excel: Optional[str] = None  # 指定路径；若 None 将尝试在 root_dir 或 CWD 自动发现

    # 性能相关配置（已移除缓存相关实现，保留最小读取路径）

    def __post_init__(self):
        self.observation_dir = os.path.join(self.root_dir, "observations")
        self.parameters_dir = os.path.join(self.root_dir, "parameters")
        self.proprio_dir = os.path.join(self.root_dir, "proprio_stats")
        # 有些仓库动作放在 Actiondata/proprio_stats
        alt_proprio_dir = os.path.join(self.root_dir, "Actiondata", "proprio_stats")
        if os.path.isdir(alt_proprio_dir):
            self.proprio_dir = alt_proprio_dir
        self.task_info_dir = os.path.join(self.root_dir, "task_info")

        self._task_info: Dict[str, List[Dict[str, Any]]] = self._load_task_info()

        # 加载通过 agibot_range.py 导出的帧范围 Excel
        self._frame_ranges_map: Optional[Dict[Tuple[str, int], Tuple[int, int]]] = self._load_frame_ranges_excel()

        # 预构建可用样本索引：[(set_id, episode_id, frame_idx)]
        self._index: List[Tuple[str, int, int]] = []
        self._build_index()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, item):
        return self.get(item, np.random)

    def _build_index(self):
        # generate index from self._frame_ranges_map, expand the range from start_frame to end_frame
        for (set_id, episode_id), (start_frame, end_frame) in self._frame_ranges_map.items():
            # consider the chunk size
            for frame_idx in range(start_frame, end_frame - NUM_ACTIONS_CHUNK):
                self._index.append((set_id, episode_id, frame_idx))

    # ---------------------------- Index & metadata ----------------------------
    def _load_task_info(self) -> Dict[str, Any]:
        """读取所有 task_info json 文件"""
        info = {}
        for fname in os.listdir(self.task_info_dir):
            if fname.endswith(".json"):
                set_id = fname.replace(".json", "").replace("task_", "")
                with open(os.path.join(self.task_info_dir, fname), "r") as f:
                    info[set_id] = json.load(f)
        return info

    def get_episode_info(self, set_id: str, episode_id: int) -> Optional[Dict[str, Any]]:
        """获取某个 episode 的任务描述信息"""
        episodes = self._task_info.get(set_id, [])
        for e in episodes:
            if e["episode_id"] == episode_id:
                return e
        return None

    def _video_path(self, set_id: str, episode_id: int, camera: str) -> str:
        return os.path.join(self.observation_dir, set_id, str(episode_id), "videos", f"{camera}.mp4")

    def get_frame_image(self, set_id: str, episode_id: int, frame_idx: int, camera: str = "head_color") -> np.ndarray:
        """
        直接使用 imageio 读取单帧（无缓存、无 decord）。
        """
        video_path = self._video_path(set_id, episode_id, camera)
        if not os.path.exists(video_path):
            return None
        try:
            reader = _get_video_reader_cached(video_path)
            frame = reader.get_data(frame_idx)
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            return np.asarray(frame)
        except Exception as e:
            print(f"读取视频帧失败: {video_path}, frame_idx={frame_idx}, 错误: {e}")
            return None

    def get_action_for_frame(self, set_id: str, episode_id: int, frame_idx: int) -> Optional[Dict[str, Any]]:
        """根据帧编号找到当前的 action 阶段"""
        ep_info = self.get_episode_info(set_id, episode_id)
        if ep_info is None:
            return None
        for action in ep_info["label_info"]["action_config"]:
            if action["start_frame"] <= frame_idx < action["end_frame"]:
                return action
        return None

    # ------------------------------ I/O helpers ------------------------------
    def _open_h5(self, set_id: str, episode_id: int) -> Optional[h5py.File]:
        h5_path = os.path.join(self.proprio_dir, str(set_id), str(episode_id), "proprio_stats.h5")
        if not os.path.exists(h5_path):
            return None
        try:
            return _open_h5_cached(h5_path)
        except Exception:
            return None

    # ------------------------------ Vectorization ------------------------------
    def _read_action_chunk(self, h5: h5py.File, start_idx: int, length: int) -> Optional[np.ndarray]:
        """
        批量切片 + 向量化计算，显著减少 HDF5 小块随机读与 Python 循环。
        返回 (chunk, action_pad_mask)
        """
        try:
            end_pos_ds = h5["action/end/position"]  # (N,2,3)
            end_quat_ds = h5["action/end/orientation"]  # (N,2,4)
        except Exception:
            return None

        eff_pos_ds = h5.get("action/effector/position", None)  # (N,2) or (N,2,k)
        if eff_pos_ds is None:
            return None

        n = end_pos_ds.shape[0]
        if start_idx + length > n:
            print(f"Error: start_idx + length > n, start_idx: {start_idx}, length: {length}, n: {n}")
            return None

        # 一次性切片到内存
        end_pos = end_pos_ds[start_idx:start_idx+length]          # (L,2,3)
        end_quat = end_quat_ds[start_idx:start_idx+length]        # (L,2,4)
        eff_pos = eff_pos_ds[start_idx:start_idx+length]          # (L,2) or (L,2,k)

        # quaternion -> euler，向量化
        # 输入 q: (L,2,4) [x,y,z,w]
        x = end_quat[..., 0]
        y = end_quat[..., 1]
        z = end_quat[..., 2]
        w = end_quat[..., 3]

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        sinp_clipped = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp_clipped)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        euler = np.stack([roll, pitch, yaw], axis=-1).astype(np.float32)  # (L,2,3)

        # 处理 gripper（取第0通道）
        if eff_pos.ndim == 2 and eff_pos.shape[-1] == 2:
            grip = eff_pos.astype(np.float32)  # (L,2)
        elif eff_pos.ndim == 3 and eff_pos.shape[-2] == 2:
            grip = eff_pos[..., 0].astype(np.float32)  # (L,2)
        else:
            return None

        # 拼接 per hand: pos(3)+euler(3)+grip(1) => 7 dims per hand
        left_vec = np.concatenate([end_pos[:, 0], euler[:, 0], grip[:, 0:1]], axis=-1)   # (L,7)
        right_vec = np.concatenate([end_pos[:, 1], euler[:, 1], grip[:, 1:1+1]], axis=-1)  # (L,7)

        if self.prefer_right_first:
            out = np.concatenate([right_vec, left_vec], axis=-1)  # (L,14)
        else:
            out = np.concatenate([left_vec, right_vec], axis=-1)  # (L,14)

        # 对齐到固定 ACTION 维度
        pad_len_action = max(0, FIXED_ACTION_DIM - out.shape[1])
        if self.pad_action_and_proprio:
            if out.shape[1] < FIXED_ACTION_DIM:
                out = np.pad(out, ((0, 0), (0, pad_len_action)), mode="constant")
            elif out.shape[1] > FIXED_ACTION_DIM:
                raise ValueError(f"Action dimension mismatch: {out.shape[1]} > {FIXED_ACTION_DIM}")

        action_pad_mask = np.zeros_like(out, dtype=bool)
        if pad_len_action > 0:
            action_pad_mask[:, -pad_len_action:] = True

        return out.astype(np.float32), action_pad_mask

    def _read_proprio(self, h5: h5py.File, idx: int) -> Optional[np.ndarray]:
        # 与动作一致的 16 维格式：end pos(3)+quat(4)+gripper(1) for right; 同 left；右前左后
        try:
            end_pos = h5["state/end/position"]  # (N,2,3)
            end_quat = h5["state/end/orientation"]  # (N,2,4)
        except Exception:
            return None

        eff_pos = h5.get("state/effector/position", None)
        if eff_pos is None:
            return None

        n = end_pos.shape[0]
        if idx < 0 or idx >= n:
            return None

        # Convert quaternion (2,4) -> euler (2,3) for consistency with action
        end_quat_t = end_quat[idx]  
        left_euler = quaternion_to_euler(end_quat_t[0][0], end_quat_t[0][1], end_quat_t[0][2], end_quat_t[0][3])
        right_euler = quaternion_to_euler(end_quat_t[1][0], end_quat_t[1][1], end_quat_t[1][2], end_quat_t[1][3])
        end_quat_euler = np.stack([np.asarray(left_euler, dtype=np.float32), np.asarray(right_euler, dtype=np.float32)], axis=0)

        vec = _concat_state_step(end_pos[idx], end_quat_euler, eff_pos[idx], self.prefer_right_first)
        if vec is None:
            return None

        if self.pad_action_and_proprio:
            if vec.shape[0] < PROPRIO_DIM:
                pad_len = PROPRIO_DIM - vec.shape[0]
                vec = np.pad(vec, (0, pad_len), mode="constant")
            elif vec.shape[0] > PROPRIO_DIM:
                # vec = vec[:PROPRIO_DIM]
                raise ValueError(f"Proprio dimension mismatch: {vec.shape[0]} > {PROPRIO_DIM}")
        
        return vec.astype(np.float32)

    # --------------------------------- Public ---------------------------------
    def get(self, item: int, rng) -> Dict[str, Any]:
        set_id, episode_id, frame_idx = self._index[item]

        # 读取指令文本
        ep_info = self.get_episode_info(set_id, episode_id)
        # instruction = self._build_instruction(ep_info, frame_idx)
        # s_time = time.time()
        action_text = self.get_action_for_frame(set_id, episode_id, frame_idx)['action_text']
        # print(f"get_action_for_frame time: {time.time() - s_time}")
        instruction = ep_info['task_name'] + ". " + action_text

        # 图像
        hand_left_img = None
        hand_right_img = None
        if self.use_wrist_image:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    "primary": executor.submit(self.get_frame_image, set_id, episode_id, frame_idx, self.primary_camera),
                    "left": executor.submit(self.get_frame_image, set_id, episode_id, frame_idx, self.left_camera) if self.left_camera else None,
                    "right": executor.submit(self.get_frame_image, set_id, episode_id, frame_idx, self.right_camera) if self.right_camera else None,
                }
                primary_future = futures["primary"]
                primary_img = primary_future.result()
                left_future = futures["left"]
                right_future = futures["right"]
                hand_left_img = left_future.result() if left_future is not None else None
                hand_right_img = right_future.result() if right_future is not None else None
            if primary_img is None:
                raise IndexError("Primary image missing for sample.")
        else:
            primary_img = self.get_frame_image(set_id, episode_id, frame_idx, self.primary_camera) 
            if primary_img is None:
                # 理论上索引阶段已筛掉，但此处再兜底
                raise IndexError("Primary image missing for sample.")

        # 动作 + 本体（无缓存，统一打开后关闭）
        h5 = self._open_h5(set_id, episode_id)
        if h5 is None:
            raise IndexError("Missing proprio_stats.h5 for sample.")
        try:
            action_chunk, action_pad_mask = self._read_action_chunk(h5, frame_idx, NUM_ACTIONS_CHUNK)
            if action_chunk is None:
                raise IndexError("Action chunk unavailable for sample.")
            proprio_vec = self._read_proprio(h5, frame_idx) if self.use_proprio else None
            proprio_vec = np.expand_dims(proprio_vec, axis=0)
        finally:
            # 不主动关闭，交由缓存管理；h5py 句柄会复用并在进程退出时释放
            pass
        

        sample: Dict[str, Any] = {
            "question": instruction,
            "timestep": int(frame_idx),
            "answer": "Action",
            "style": "action",
            "action": action_chunk,
            "action_pad_mask": action_pad_mask,
            "proprio": proprio_vec if self.use_proprio else None,
            "episode_index": int(episode_id),
            "metadata": {
                "dataset_name": "AgibotWorld-Alpha",
                "set_id": str(set_id),
                "episode_id": int(episode_id),
                "frame_index": int(frame_idx),
                "instruction": instruction,
            },
        }

        if self.use_wrist_image and hand_left_img is not None and hand_right_img is not None:
            sample["images"] = [primary_img, hand_left_img, hand_right_img]
        else:
            sample["image"] = primary_img

        return sample


    # ---------------------------- Excel ranges ----------------------------
    def _load_frame_ranges_excel(self) -> Optional[Dict[Tuple[str, int], Tuple[int, int]]]:
        """
        读取 lfwj_range.py 导出的 filtered_frame_ranges.xlsx（或用户指定 Excel），
        返回 {(set_id, episode_id): (start_frame, end_frame)} 的映射。
        自动发现顺序：用户指定 > root_dir/filtered_frame_ranges.xlsx > CWD/filtered_frame_ranges.xlsx。
        读取失败或文件不存在时返回 None。
        """
        try:
            import pandas as pd  # type: ignore
        except Exception:
            return None
        # 决定候选路径
        candidates: List[str] = []
        if self.frame_ranges_excel is not None:
            candidates.append(self.frame_ranges_excel)
        candidates.append(os.path.join(self.root_dir, "filtered_frame_ranges.xlsx"))
        candidates.append(os.path.join(os.getcwd(), "filtered_frame_ranges.xlsx"))

        excel_path: Optional[str] = None
        for p in candidates:
            if isinstance(p, str) and os.path.isfile(p):
                excel_path = p
                break

        df = pd.read_excel(excel_path)

        required_cols = {"set_id", "episode_id", "start_frame", "end_frame"}
        if not required_cols.issubset(set(df.columns)):
            return None

        frame_map: Dict[Tuple[str, int], Tuple[int, int]] = {}
        for _, row in df.iterrows():
            try:
                set_id = str(row["set_id"]).strip()
                # 跳过空 set_id
                if set_id == "" or set_id.lower() == "nan":
                    continue
                episode_id = int(row["episode_id"])  # type: ignore
                s = int(row["start_frame"])  # type: ignore
                e = int(row["end_frame"])  # type: ignore
                if e < s:
                    continue
                key = (set_id, episode_id)
                # 若存在重复，合并为最小 start / 最大 end
                if key in frame_map:
                    prev_s, prev_e = frame_map[key]
                    frame_map[key] = (min(prev_s, s), max(prev_e, e))
                else:
                    frame_map[key] = (s, e)
            except Exception:
                continue

        return frame_map if len(frame_map) > 0 else None

def test_agibot_dataset():

    ds = AgiBotWorldAlphaDataset(
        root_dir="/vast/users/xiaodan/zhangjian/datasets/AgiBotWorld-Alpha",
        primary_camera="head_color",
        right_camera="hand_right_color",
        left_camera="hand_left_color",
        use_proprio=True,
        use_wrist_image=True,
    )
    print('index length:',len(ds._index))
    print(ds._index[0],)
    print("_frame_ranges_map length:",len(ds._frame_ranges_map))
    print(next(iter(ds._frame_ranges_map.items())))

    print("dataset length:", len(ds))
    # if len(ds) == 0:
    #     print("No valid sample found after indexing. Try setting skip_corrupt_episodes=False or widening limits.")
    #     return
    for i in range(0,len(ds),100):
        # ex = ds[0]
        ex = ds[i]
        print(f"Item {i}/{len(ds)}")
        # print("keys:", ex.keys())
        print("question:", ex["question"])
        print("image:", ex["images"][0].shape,"length of images:",len(ex["images"]))
        print("action:", ex["action"].shape)
        print("proprio:", ex["proprio"].shape)
        break
    # print("question:", ex["question"])
    # if "images" in ex:
    #     print("image shapes:", [im.shape for im in ex["images"]])
    # else:
    #     print("image shape:", ex["image"].shape)
    # print("action shape:", ex["action"].shape)  # (NUM_ACTIONS_CHUNK, ACTION_DIM)
    # print("proprio shape:", None if ex["proprio"] is None else ex["proprio"].shape)

if __name__ == "__main__":
    test_agibot_dataset()
