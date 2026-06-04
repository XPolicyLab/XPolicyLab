# HoloBrain × XPolicyLab — 新增本体集成指南

记录将一个新的 XPolicyLab 本体（embodiment）接入 HoloBrain 训练流程时**容易踩的坑**。先读 [`INSTALLATION.md`](INSTALLATION.md) 安装环境；本文聚焦于**数据 → 可视化对齐**这一步，因为这一步是新增本体后第一道排查关口。

如果 `process_data.sh` 跑完后用 `data_visualize.py` 投影出来的红点和真实机械臂位置对不上，往往是下面四类问题之一。

---

## 坑 1 ：相机外参的坐标约定（OpenGL vs OpenCV）

**症状**：所有关节红点都消失，或者全部投在画面之外/反过来。EE 看似在夹爪附近其实是**镜像**后的偶然重合。

**原因**：XPolicyLab 数据采集底层用的是 SAPIEN，HDF5 里 `vision/<cam>/extrinsic_matrix` 是**OpenGL/SAPIEN 约定的 `cam2world`**：

| 约定     | 相机本地轴方向                 | "看出去的方向" |
| -------- | ------------------------------ | -------------- |
| OpenCV   | X 右、Y 下、Z 前（入屏）       | +Z             |
| OpenGL / SAPIEN | X 右、Y 上、Z 后（出屏） | **-Z**         |

而 HoloBrain 的 `GetProjectionMat` / `_project_joints_to_2d` 用的是 OpenCV 内参（`fx, fy, cx, cy`），过滤条件 `depth[3] < 0.02` 假设"在相机前"的点 Z 为正。

如果 `process_data.py` 只做 `np.linalg.inv(extrinsic)` 把 cam2world 翻成 world2cam，但**不翻转相机本地 Y/Z 轴**，所有"在相机前"的点 Z 都会是负的 → 被过滤掉 → 红点全消失。

**修复**：[`process_data.py:_to_holobrain_world2cam`](RoboOrchardLab/projects/holobrain/process_data.py) 现在默认是 `cam2world_opengl`，会做：

```python
flip = np.diag([1.0, -1.0, -1.0, 1.0])
return flip @ np.linalg.inv(extrinsic)
```

**新增本体时**：先看上游数据流：
- SAPIEN / IsaacGym / Mujoco → 通常是 OpenGL（保持默认）
- OpenCV / ROS realsense → 通常已经是 OpenCV，需要 `export XPOLICY_HOLOBRAIN_EXTRINSIC_CONVENTION=cam2world`（或 `world2cam` 看具体存储）

**怎么验证约定对不对**：用 Python 取一帧 cam2world，把世界坐标系下的一个已知场景点（比如桌面中心 `(0, 0, 0.75)`）按四种约定（cam2world × OpenCV/OpenGL，world2cam × OpenCV/OpenGL）变换到相机帧，看**深度的绝对值**是否符合该相机到该点的物理距离。哪个对得上，就是哪个约定。

---

## 坑 2 ：URDF 基座位置 与 `T_base2world` 的双重坐标变换

**症状**：所有关节都偏离真实位置一个固定的（旋转+平移）量；EE 似乎勉强对得上是因为腕载相机自身跟着臂在动。

**原理**：HoloBrain 的可视化管线里，机器人状态做完 FK 后会再过 `T_base2world`：

```
link_pos_in_world = T_base2world @ FK(joint_state, urdf)
```

所以"URDF 里 base 的位置"和"`T_base2world`"是**两段坐标变换**，要么把所有差异放进 URDF（`T_base2world = I`），要么把所有差异放进 `T_base2world`（URDF 用上游单位 base）。**不能两边都做一半**，否则会变成"双重旋转/双重平移"。

本仓库选择的方案（与 RoboTwin2.0 上游一致）：URDF 沿用上游的单位坐标（如 `dual_x5_exact_from_x5a.urdf` 把 fl/fr 放在 `(0.2, ±0.3, 0.765) rpy=0`），通过 `config_robotwin_dataset.py` 里的 `T_base2world` 把它映射到 `env_cfg/robot/<robot>.yml` 描述的实际世界位置。例如 dual_x5：

```yaml
default_root_pos: [-0.3, -0.45, 0.765]      # 左臂世界位置
default_root_rot: [0.707, 0, 0, 0.707]      # 90° around Z
```

对应 `T_base2world`：

```python
T_base2world = [
    [0, -1, 0,  0    ],
    [1,  0, 0, -0.65 ],
    [0,  0, 1,  0    ],
    [0,  0, 0,  1    ],
]
# 校验：T_base2world @ (0.2, 0.3, 0.765, 1) = (-0.3, -0.45, 0.765, 1)  ✓
```

**新增本体时**：
1. 看 `env_cfg/robot/<robot>.yml` 列出的两臂 `default_root_pos` / `default_root_rot`。
2. 决定 URDF 用"上游单位坐标"还是"真实世界坐标"。
3. 在 `config_robotwin_dataset.py` 的 dataset_config 里加一条新 dataset key（如 `xpolicy_dual_x5`），设置匹配的 `T_base2world`。**不要直接复用 `robotwin2_0` 的 `T_base2world`** 除非两个本体几何完全相同。
4. **验证方法**：用零关节态做 FK，把 fl_link6 在世界坐标的位置和数据里 `cam_left_wrist` 第 0 帧的位置比对，差异应 < 1 mm。

---

## 坑 3 ：从单臂 URDF 拼成双臂 URDF 的注意事项

XPolicyLab 的本体往往只提供单臂 URDF（如 `embodiments/arx_x5/X5A.urdf`），需要包装成 HoloBrain `DualArmKinematics` 期望的双臂格式（`fl_*` / `fr_*` 前缀）。参考 `RoboOrchardLab/projects/holobrain/urdf/arx5/dual_x5_exact_from_x5a.urdf`。

### 3.1 关节索引必须落在 `[10-15]` 和 `[18-23]`

`DualArmKinematics` 默认：

```python
left_arm_joint_id  = [10, 11, 12, 13, 14, 15]   # fl_joint1..6
right_arm_joint_id = [18, 19, 20, 21, 22, 23]   # fr_joint1..6
```

这些索引指的是 `pytorch_kinematics.build_chain_from_urdf(...).get_joints()` 返回列表里的位置。pk **跳过 fixed 关节**，但保留 revolute/prismatic/continuous。

所以要让 `fl_joint1..6` 落在 `[10..15]`、`fr_joint1..6` 落在 `[18..23]`，需要：

- 在 fl 链之前插入 **10 个 dummy 非 fixed 关节**（continuous 即可，状态保持 0）。这就是 `dual_x5_exact_from_x5a.urdf` 里 `dummy_joint_0..9` 的作用。
- fl 部分内部的关节顺序：`fl_joint1..6` (6 个 revolute, 索引 10-15) → `fl_joint7, fl_joint8` (2 个 prismatic 夹爪, 索引 16-17)。
- fr 部分同理：`fr_base_joint` (fixed，**不计数**) → `fr_joint1..6` (索引 18-23) → `fr_joint7, fr_joint8` (索引 24-25)。

**验证**：脚手期 URDF 写好后，跑一下：

```bash
python -c "
import pytorch_kinematics as pk
chain = pk.build_chain_from_urdf(open('.../your_dual.urdf','rb').read())
for i,j in enumerate(chain.get_joints()): print(i, j.name)
"
```

确认 `fl_joint1..6` 是 10-15、`fr_joint1..6` 是 18-23。

### 3.2 link 命名必须匹配 `DualArmKinematics` 的默认 keys

可视化和 FK loss 需要按名字取出 link 位姿。默认 keys：

```python
left_arm_link_keys  = ["fl_link1", "fl_link2", "fl_link3", "fl_link4", "fl_link5", "fl_link6"]
right_arm_link_keys = ["fr_link1", "fr_link2", "fr_link3", "fr_link4", "fr_link5", "fr_link6"]
left_finger_keys    = ["fl_link7", "fl_link8"]
right_finger_keys   = ["fr_link7", "fr_link8"]
```

把单臂 URDF 里的所有 `<link name="...">` 和 `<joint><parent|child link="...">` 全部加 `fl_` / `fr_` 前缀，**包括 mesh 引用的不要改，路径保持原样**。

如果上游单臂 URDF 用了非默认 link 名（如 piper 用 `left_link1`），可以在 `dataset_config` 的 `kinematics_config` 里覆盖（参考 `robotwin2_0_ur5_wsg` 那一段）。

### 3.3 EE 视觉指针偏移（夹爪中心 vs 手指根部）

`DualArmKinematics` 的 EE 位置是 `(fl_link7 + fl_link8).mean(axis=0)` —— 也就是**两根手指 link 原点的平均**。这通常是手指**根部**（关节安装点），不是**指尖**或**夹爪几何中心**。

这就是 dual_x5 视频里 EE 坐标轴看起来"略微向前偏"的原因 —— URDF 里手指 link 原点定义在根部，但视觉上夹爪中心在指尖。这个偏差**不影响训练**（loss 用同一个定义），只影响可视化观感。

如果想让 EE 显示在真实指尖，可以在双臂 URDF 里把 `fl_link7` / `fl_link8` 的 `<joint><origin xyz="...">` 沿 link6 局部 X 轴往前平移（X5A 大约 +0.07 m 到达指尖），但这会改变 FK 输出 → 需要同步迁移训练数据 / loss 定义。**不推荐**改这个，留作可视化已知偏差即可。

### 3.4 不要忘了 collision、inertial、material

`pytorch_kinematics` 只关心 `joint` 树和 `origin`，但 ROS 工具链、Isaac 渲染、`urdfpy` 等会读 `<collision>` / `<inertial>` / `<material>`。拼双臂时全套复制保留，最容易避免后续兼容性问题。

---

## 坑 4 ：CAMERA_MAP 别忘了配齐

[`process_data.py:CAMERA_MAP`](RoboOrchardLab/projects/holobrain/process_data.py#L17) 把 HoloBrain 的 `front_camera / left_camera / right_camera / head_camera` 映射到 XPolicyLab HDF5 里的真实相机名（`cam_head / cam_left_wrist / cam_right_wrist / ...`）。

```python
CAMERA_MAP = {
    "front_camera":  "cam_head",
    "left_camera":   "cam_left_wrist",
    "right_camera":  "cam_right_wrist",
    "head_camera":   "cam_head",
}
```

注意当前 `front_camera` 和 `head_camera` **都指向 `cam_head`**（因为 XPolicyLab 单第三视角 + 单头部相机被复用了两次）。如果新本体提供独立的 front 相机（如 `cam_third_view`），把 `front_camera` 改为 `cam_third_view`，否则训练时两路视觉特征是重复的。

---

## 新增本体的快速排查清单

1. ☐ HDF5 里 `extrinsic_matrix` 是哪种约定？（默认假设 `cam2world_opengl`）
2. ☐ `env_cfg/robot/<robot>.yml` 里 `default_root_pos` / `default_root_rot` 是什么？
3. ☐ 拼好的双臂 URDF 跑 `pk.build_chain_from_urdf` 后，`fl_joint1..6` 是否在 `[10..15]`、`fr_joint1..6` 在 `[18..23]`？
4. ☐ link 名是否以 `fl_` / `fr_` 开头，且包含 `link1..6` 和 `link7..8`？
5. ☐ `config_robotwin_dataset.py` 里是否加了 `T_base2world`？零关节态 FK 出的 fl_link6 世界位置和数据帧 0 的 cam_left_wrist 位置差异是否 < 1 mm？
6. ☐ `CAMERA_MAP` 是否映射到了正确的 HDF5 相机名？
7. ☐ `data_visualize.py` 生成的 mp4 里，每个关节红点是否沿着臂体可见？EE 是否在夹爪附近？
