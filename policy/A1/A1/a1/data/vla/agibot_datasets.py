# import h5py
# import json
# import numpy as np


# def print_tree(name, obj):
#     """打印树状结构"""
#     indent = "  " * (name.count("/") - 1)
#     if isinstance(obj, h5py.Group):
#         print(f"{indent}📂 {name}/")
#     elif isinstance(obj, h5py.Dataset):
#         print(f"{indent}📄 {name} (shape={obj.shape}, dtype={obj.dtype})")


# def h5_to_dict(obj):
#     """递归转换 HDF5 对象为 Python 字典"""
#     if isinstance(obj, h5py.Dataset):
#         data_info = {
#             "type": "dataset",
#             "shape": obj.shape,
#             "dtype": str(obj.dtype),
#         }
#         try:
#             # 小数据集直接保存
#             if obj.size <= 100:
#                 data_info["data"] = obj[()].tolist()
#             else:
#                 # 保存前10个元素/样本作为预览
#                 if obj.ndim > 0:
#                     data_info["preview"] = obj[0 : min(10, obj.shape[0])].tolist()
#                 else:
#                     data_info["preview"] = obj[()].tolist()
#         except Exception as e:
#             data_info["error"] = str(e)
#         return data_info

#     elif isinstance(obj, h5py.Group):
#         group_info = {"type": "group", "items": {}}
#         for key, val in obj.attrs.items():
#             group_info[f"attr:{key}"] = (
#                 val.tolist() if isinstance(val, np.ndarray) else val
#             )
#         for key, item in obj.items():
#             group_info["items"][key] = h5_to_dict(item)
#         return group_info


# def read_h5_to_json(h5_path, json_path="output.json"):
#     with h5py.File(h5_path, "r") as f:
#         print(f"\n=== HDF5 文件树状结构: {h5_path} ===\n")
#         f.visititems(print_tree)

#         print("\n=== 正在导出为 JSON... ===")
#         structure = {"/": h5_to_dict(f)}

#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(structure, f, ensure_ascii=False, indent=2)

#     print(f"\n✅ 已将内容写入 {json_path}")


# if __name__ == "__main__":
#     # 修改为你的 HDF5 文件路径
#     h5_file = "/mnt/data2/datasets/Real-robot/AgibotWorld-Alpha/Actiondata/proprio_stats/327/648642/proprio_stats.h5"
#     json_file = "output.json"
#     read_h5_to_json(h5_file, json_file)


import os
import json
import h5py
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Iterator
import imageio
import matplotlib.pyplot as plt
import pandas as pd


class RobotDatasetReader:
    def __init__(self, root_dir: str, xlsx: str, norm_stat_file: str, num_actions_chunk: int= 8):
        """
        root_dir: 数据集根目录，下面有 observations / parameters / proprio_stats / task_info
        """
        self.xlsx = xlsx
        self.root_dir = root_dir
        self.norm_stat_file = norm_stat_file
        with open(self.norm_stat_file, "r") as f:
            self.norm_stats = json.load(f)
        self.norm_stats = {k: {kk: np.array(v) for kk, v in vv.items()} for k, vv in self.norm_stats.items()}
        self.num_actions_chunk = num_actions_chunk
        self.observation_dir = os.path.join(root_dir, "observations")
        self.parameters_dir = os.path.join(root_dir, "parameters")
        self.proprio_dir = os.path.join(root_dir, "Actiondata/proprio_stats")
        self.task_info_dir = os.path.join(root_dir, "task_info")

        # 缓存 task_info
        self.task_info = self._load_task_info()

    def _load_task_info(self) -> Dict[str, Any]:
        """读取所有 task_info json 文件"""
        info = {}
        for fname in os.listdir(self.task_info_dir):
            if fname.endswith(".json"):
                set_id = fname.replace(".json", "").replace("task_", "")
                with open(os.path.join(self.task_info_dir, fname), "r") as f:
                    info[set_id] = json.load(f)
        self.frames = read_filtered_frame_ranges(self.xlsx,self.num_actions_chunk)
        return info

    def __iter__(self):
        for frame in self.frames:
            yield self.get_frame_data(frame["set_id"], frame["episode_id"], frame["frame_idx"])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.get_frame_data(self.frames[idx]["set_id"], self.frames[idx]["episode_id"], self.frames[idx]["frame_idx"])

    def get_episode_info(
        self, set_id: str, episode_id: int
    ) -> Optional[Dict[str, Any]]:
        """获取某个 episode 的任务描述信息"""
        episodes = self.task_info.get(set_id, [])
        for e in episodes:
            if e["episode_id"] == episode_id:
                return e
        return None

    def get_action_for_frame(
        self, set_id: str, episode_id: int, frame_idx: int
    ) -> Optional[Dict[str, Any]]:
        """根据帧编号找到当前的 action 阶段"""
        ep_info = self.get_episode_info(set_id, episode_id)
        if ep_info is None:
            return None
        for action in ep_info["label_info"]["action_config"]:
            if action["start_frame"] <= frame_idx < action["end_frame"]:
                return action
        return None

    def _get_state(
        self, set_id: str, episode_id: int, frame_idx: int
    ) -> Dict[str, Any]:
        """
        获取某一帧的 state 状态（只读取 state 下的数据）
        包含 joint / effector / end / head / robot / waist 等子模块
        """
        h5_path = os.path.join(
            self.proprio_dir, set_id, str(episode_id), "proprio_stats.h5"
        )
        state_data = {}
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"未找到文件: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            if "state" not in f:
                raise ValueError(f"文件中没有 state 数据: {h5_path}")

            # 遍历 state 下的所有 group
            for module_name in f["state"].keys():
                module_group = f["state"][module_name]
                module_data = {}

                for key in module_group.keys():
                    dataset = module_group[key]
                    if frame_idx < len(dataset):
                        module_data[key] = dataset[frame_idx]

                if module_data:
                    state_data[module_name] = module_data

            # timestamp 单独处理
            if "timestamp" in f and frame_idx < len(f["timestamp"]):
                state_data["timestamp"] = f["timestamp"][frame_idx]
        return state_data

    def get_frame_image(
        self, set_id: str, episode_id: int, frame_idx: int, camera: str = "head_color"
    ) -> np.ndarray:
        """
        从视频中抽取某一帧图像（支持 AV1 软件解码，不修改源数据集）
        camera: 视频文件名（不带扩展名），如 "head_color"、"head_left_fisheye_color"
        """
        video_path = os.path.join(
            self.observation_dir, set_id, str(episode_id), "videos", f"{camera}.mp4"
        )
        if not os.path.exists(video_path):
            return None

        try:
            # 使用 imageio + ffmpeg 读取视频帧
            import time
            st = time.time()
            reader = imageio.get_reader(video_path, format="ffmpeg")
            
            # total_frames = reader.count_frames()
            # print(f"读取视频帧成功: {video_path}, frame_idx={frame_idx}, 时间: {time.time() - st}")
            # if frame_idx >= total_frames:
            #     reader.close()
            #     return None
            
            frame = reader.get_data(frame_idx)
            reader.close()
            if frame.shape[2] == 4:  # 如果有 alpha 通道
                frame = frame[:, :, :3]
            return np.asarray(frame)

        except Exception as e:
            print(f"读取视频帧失败: {video_path}, frame_idx={frame_idx}, 错误: {e}")
            return None

    def get_neighbor_frames(
        self,
        set_id: str,
        episode_id: int,
        frame_idx: int,
        window: int = 5,
        camera: str = "head_color",
    ) -> List[np.ndarray]:
        """获取前后几帧图像"""
        imgs = []
        for i in range(frame_idx - window, frame_idx + window + 1):
            if i < 0:
                continue
            img = self.get_frame_image(set_id, episode_id, i, camera)
            if img is not None:
                imgs.append(img)
        return imgs

    def get_neighbor_frames_dict(
        self,
        set_id: str,
        episode_id: int,
        frame_idx: int,
        window: int = 5,
        camera: str = "head_color",
    ) -> dict:
        """
        获取当前帧前后 window 帧，返回字典形式
        key: 相对于当前帧的偏移 (-window,...,-1,1,...,window)
        value: 对应帧的 numpy 图像数组，如果超出范围返回 None
        """
        neighbor_frames = {}

        for i in range(1, window + 1):
            idx = frame_idx - i
            try:
                neighbor_frames[-i] = (
                    self.get_frame_image(set_id, episode_id, idx, camera)
                    if idx >= 0
                    else None
                )
            except Exception:
                neighbor_frames[-i] = None

        for i in range(1, window + 1):
            idx = frame_idx + i
            try:
                neighbor_frames[i] = self.get_frame_image(
                    set_id, episode_id, idx, camera
                )
            except Exception:
                neighbor_frames[i] = None

        return neighbor_frames

    def normalize_(self, value, norm_stat, quantize: bool = True):
        if quantize:
            value = (value - norm_stat['q01']) / (norm_stat['q99'] - norm_stat['q01'] + 1e-8)
        else:
            value = (value - norm_stat['mean']) / (norm_stat['std'] + 1e-8)
        return value

    def process_state(self, state, mode='states',quantize: bool = True):
        robot_state =[]
        effector = state['effector']
        end = state['end']
        robot_state.append(end['position'][0])
        robot_state.append(quaternion_to_rpy(end['orientation'][0]))
        robot_state.append(effector['position'][:1])
        robot_state.append(end['position'][1])
        robot_state.append(quaternion_to_rpy(end['orientation'][1]))
        robot_state.append(effector['position'][1:])
        robot_state = np.concatenate(robot_state)
        # assert robot_state.shape == (14,), f"robot_state shape: {robot_state.shape}, state: {state}"
        if robot_state.shape != (14,):
            robot_state = np.zeros(14)
        robot_state = self.normalize_(robot_state, self.norm_stats[mode], quantize)
        return robot_state

    def get_frame_data(
        self, set_id: str, episode_id: int, frame_idx: int, camera: str = "head_color"
    ) -> Dict[str, Any]:
        """
        高层封装：获取某一帧的完整信息
        - 总任务描述
        - 当前阶段任务
        - state 状态
        - 图像数据
        """
        ep_info = self.get_episode_info(set_id, episode_id)
        actions = []
        for i in range(frame_idx+1, frame_idx+self.num_actions_chunk+1):
            action = self._get_state(set_id, episode_id, i)
            action = self.process_state(action, mode='actions',quantize=True)
            actions.append(action)
        actions = np.array(actions)
        state = self._get_state(set_id, episode_id, frame_idx)
        state = self.process_state(state, mode='states',quantize=True)
        current_action = self.get_action_for_frame(set_id, episode_id, frame_idx)
        imgs = []
        imgs.append(self.get_frame_image(set_id, episode_id, frame_idx, 'head_color'))
        imgs.append(self.get_frame_image(set_id, episode_id, frame_idx, 'hand_left_color'))
        imgs.append(self.get_frame_image(set_id, episode_id, frame_idx, 'hand_right_color'))

        return {
            "episode_id": episode_id,
            "task_name": ep_info["task_name"] if ep_info else None,
            "init_scene_text": ep_info["init_scene_text"] if ep_info else None,
            'current_action': current_action['action_text'],
            "action": actions,
            "state": state,  # ✅ 现在包含 joint/robot/effector/head 等完整 state
            "image": imgs,
        }

    def get_frame_depth(
        self, set_id: str, episode_id: int, frame_idx: int, camera: str = "head_depth"
    ) -> Optional[np.ndarray]:
        """获取指定帧的深度图"""
        depth_dir = os.path.join(
            self.root_dir, "observations", str(set_id), str(episode_id), "depth"
        )
        depth_name = f"{camera}_{frame_idx:06d}.png"
        depth_path = os.path.join(depth_dir, depth_name)

        if not os.path.exists(depth_path):
            return None
        try:
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            return depth_img
        except Exception:
            return None

    def get_neighbor_depths(
        self,
        set_id: str,
        episode_id: int,
        frame_idx: int,
        window: int = 5,
        camera: str = "head_depth",
    ) -> Dict[int, np.ndarray]:
        """获取相邻帧的深度图"""
        depth_dict = {}
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            neighbor_idx = frame_idx + offset
            if neighbor_idx < 0:
                continue
            depth_img = self.get_frame_depth(set_id, episode_id, neighbor_idx, camera)
            if depth_img is not None:
                depth_dict[offset] = depth_img
        return depth_dict


def quaternion_to_rpy(q, order: str = "xyzw", degrees: bool = False):
    """将四元数转换为欧拉角 RPY (roll, pitch, yaw)。

    采用航空常用的 ZYX (yaw-pitch-roll) 组合，对应返回顺序为 (roll, pitch, yaw)。

    参数:
    - q: 长度为4的序列或 numpy 数组。
    - order: 四元数分量顺序，"wxyz" 或 "xyzw"。
    - degrees: True 则返回角度制，否则弧度制。
    """
    arr = np.asarray(q, dtype=np.float64)
    if arr.shape[-1] != 4:
        raise ValueError(f"四元数维度必须为4，收到: shape={arr.shape}")

    if order.lower() == "wxyz":
        w, x, y, z = arr
    elif order.lower() == "xyzw":
        x, y, z, w = arr
    else:
        raise ValueError("order 仅支持 'wxyz' 或 'xyzw'")

    # 归一化，避免数值漂移
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0:
        raise ValueError("四元数范数为0")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # roll (x轴旋转)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    # pitch (y轴旋转)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    # yaw (z轴旋转)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    if degrees:
        factor = 180.0 / np.pi
        return roll * factor, pitch * factor, yaw * factor
    return roll, pitch, yaw

def read_filtered_frame_ranges(xlsx_path: str, num_actions_chunk: int= 8) -> List[Dict[str, Any]]:
    """读取 filtered_frame_ranges.xlsx，返回每行的字典列表。

    期望列名：set_id, episode_id, start_frame, end_frame。
    """
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"未找到文件: {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    # 规范列名
    expected_cols = ["set_id", "episode_id", "start_frame", "end_frame"]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Excel缺少列: {missing_cols}; 实际列: {list(df.columns)}")

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            set_id = str(row["set_id"]).strip()
            episode_id = int(row["episode_id"]) if not pd.isna(row["episode_id"]) else None
            start_frame = int(row["start_frame"]) if not pd.isna(row["start_frame"]) else 0
            end_frame = int(row["end_frame"]) if not pd.isna(row["end_frame"]) else -1
        except Exception as e:
            raise ValueError(f"解析行失败: {row.to_dict()} 错误: {e}")

        if episode_id is None or end_frame < 0:
            continue
        
        for frame_idx in range(start_frame, end_frame-num_actions_chunk+1):
            records.append(
                {
                    "set_id": set_id,
                    "episode_id": episode_id,
                    "frame_idx": frame_idx,
                }
            )

    return records


if __name__ == "__main__":
    # print(quaternion_to_rpy([ 0.52416398, -0.22196564,  0.79680929,  0.20267791]))
    dataset = RobotDatasetReader("/mnt/data2/datasets/Real-robot/AgibotWorld-Alpha", "a1/data/vla/agibot/agibot_alpha_frame_ranges.xlsx","a1/data/vla/agibot/norm_stats.json",1)
    # print(len(dataset))
    # # 示例：从 Excel 读取帧区间并遍历前几条
    
    try:
        states = []
        from tqdm import tqdm
        for i,frame in enumerate(tqdm(dataset)):
            # assert frame["state"].shape == (14,), f"state shape: {frame['state'].shape}"
            print(frame["state"])
            print(frame["action"])
            # break
        #     if frame["state"].shape == (14,):
        #         states.append(frame["state"])
        #     # if i > 1000:
        #     #     break
        # # 拼接成一个数组
        # states = np.stack(states, axis=0)
        # actions = states
        # # 统计mean,std,max,min,q01,q99并写入norm_stats.json
        # norm_stats = {
        #     "states": {
        #         "mean": states.mean(0).tolist(),
        #         "std": states.std(0).tolist(),
        #         "max": states.max(0).tolist(),
        #         "min": states.min(0).tolist(),
        #         "q01": np.quantile(states, 0.01, axis=0).tolist(),
        #         "q99": np.quantile(states, 0.99, axis=0).tolist(),
        #     },
        #     "actions": {
        #         "mean": actions.mean(0).tolist(),
        #         "std": actions.std(0).tolist(),
        #         "max": actions.max(0).tolist(),
        #         "min": actions.min(0).tolist(),
        #         "q01": np.quantile(actions, 0.01, axis=0).tolist(),
        #         "q99": np.quantile(actions, 0.99, axis=0).tolist(),
        #     },
        # }
        # with open("norm_stats.json", "w") as f:
        #     json.dump(norm_stats, f)
        # print(states.shape)
        # print(actions.shape)
    except Exception as e:
        # 打印完整错误信息
        import traceback
        print(f"完整错误信息: {traceback.format_exc()}")
        print(f"读取或迭代失败: {e}")
