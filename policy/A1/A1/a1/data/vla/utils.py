from enum import Enum
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on

def quat_to_rotate6d(q: np.ndarray, scalar_first = False) -> np.ndarray:
    return R.from_quat(q, scalar_first = scalar_first).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

def euler_to_rotate6d(q: np.ndarray, pattern: str = "xyz") -> np.ndarray:
    return R.from_euler(pattern, q, degrees=False).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

def quaternion_to_euler(x, y, z, w):
    """
    四元数 (x, y, z, w) 转换为欧拉角 (roll, pitch, yaw)
    输出单位：弧度
    旋转顺序：ZYX (yaw-pitch-roll)
    """

    # 计算 roll (x轴旋转)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # 计算 pitch (y轴旋转)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # 超出范围时取 90°
    else:
        pitch = math.asin(sinp)

    # 计算 yaw (z轴旋转)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def quaternion_to_euler_numpy(quaternions):
        # quaternion -> euler，向量化
        # 输入 q: (L,2,4) [x,y,z,w]
        x = quaternions[..., 0]
        y = quaternions[..., 1]
        z = quaternions[..., 2]
        w = quaternions[..., 3]

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
        return euler