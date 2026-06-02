# Recorder 模块

Recorder 是 RealEnv GUI 的 episode 数据录制插件层。当前实现为一个中心采样线程加多个插件：`EpisodeRecorder` 每隔 `1 / fps` 调一次 `env.get_obs()`，再把同一帧 obs 分发给 video / trajectory 插件。

## 当前实现

Workbench 支持可选 recorder 参数：

```python
controller = WorkbenchController(env, recorder=recorder)
```

评测线程中：

```python
recorder.start(episode_dir)
env.eval_one_episode()
recorder.stop()
```

如果 GUI 终止、关闭或 retry 清理未提交 episode，workbench 会走：

```python
recorder.abort()
```

## 类与接口

`EpisodeRecorder`

- `start(episode_dir)`：启动中心采样线程，并通知所有插件开始录制。
- `record_obs(obs)`：把一帧 obs 分发给所有插件。
- `stop()`：停止采样线程，正常结束所有插件，返回插件输出信息。
- `abort()`：停止采样线程，终止所有插件并清理半截产物。

`VideoRecorderPlugin`

- `start(episode_dir)`：创建 `episode_dir/recorder/video/`，写 `recording` manifest。
- `record_obs(obs)`：从 `obs["vision"][camera_name]["color"]` 取图像帧，写入 ffmpeg。
- `stop()`：关闭 ffmpeg writer，写 `committed` manifest。
- `abort()`：关闭 writer，并删除 `episode_dir/recorder/video/`。

`TrajectoryRecorderPlugin`

- `start(episode_dir)`：创建 `episode_dir/recorder/trajectory/`，写 `recording` manifest。
- `record_obs(obs)`：缓存 vision color、state 和采样时间戳。
- `stop()`：写 `trajectory.hdf5` 和 `committed` manifest。
- `abort()`：删除 `episode_dir/recorder/trajectory/`。

## Xone 轨迹格式

Xone 数据采集链路中，`CollectAny.write()` 默认按 controller/sensor 分组写 HDF5；`X_spark_format_pipeline()` 会转成 Xone 统一结构：

```text
vision/
  cam_head/
    colors
    shape
  cam_left_wrist/
    colors
    shape
  cam_right_wrist/
    colors
    shape
state/
  left_arm_joint_states
  left_ee_joint_states
  left_ee_poses
  right_arm_joint_states
  right_ee_joint_states
  right_ee_poses
```

trajectory 插件沿用这个结构，并额外保存：

```text
timestamps
vision/{camera}/intrinsic_matrix
vision/{camera}/extrinsics_matrix
```

## 产物目录

当前 recorder 产物：

```text
episode_000000/
  recorder/
    video/
      manifest.json
      cam_head.mp4
      cam_left_wrist.mp4
      cam_right_wrist.mp4
    trajectory/
      manifest.json
      trajectory.hdf5
```

video manifest 的输出类型是 `rgb_video`；trajectory manifest 的输出类型是 `xone_hdf5_trajectory`。

## 清理规则

abort/retry 会删除：

```text
episode_000000/recorder/
```

同时仍兼容删除旧 recorder 直接写在 episode 目录下的：

```text
episode_000000/*.mp4
```

## 长期目标

Recorder 可以继续扩展为更多插件：

- 已有 obs schema 的其他视图或派生产物。
- robot state 的更细粒度日志。
- action log。
- 其他按需求新增的数据类型。

Recorder 不负责：

- 判断 episode 成功/失败。
- 推进 episode。
- 控制 robot。
- 修改 workbench 状态机。

## 未实现

- action 插件。
- 新数据类型插件扩展点。
- 临时文件提交机制。
