# Dev Real Env TODO

本文档只记录 `real_env/` 真机 client GUI 重构线的当前进度和后续任务。

## 文档结构

- `real_env/design.md`：统一设计，包含状态机、数据落盘、recorder 长期方向。
- `real_env/real_env.md`：RealEnv 模块职责和接口。
- `real_env/workbench.md`：Workbench / GUI controller 模块职责和状态行为。
- `real_env/recorder.md`：Recorder 模块当前状态和长期插件设计。
- `real_env/layout_shot.md`：layout 参考图拍摄工具。
- `real_env/testing_deployment.md`：本地测试、真机部署参数和调用方式。
- `real_env/TODO.md`：当前进度和下一步。

## 已完成

- [x] 固定 layout 资源约定：`XONE_ROOT/layouts/{task_name}/layout_{episode_idx:06d}.png`。
- [x] 保留最小 `RealEnv`：初始化 robot/model client、`get_obs()`、`take_action()`、`eval_one_episode()`、`reset()`。
- [x] 将 observation / action 转换拆到 `real_env/helpers.py`。
- [x] 支持 server 返回的 numpy-like action payload，包括 bytes/base64 变体。
- [x] 加入 `m_robot_lock`，保护 `get_obs()`、`take_action()`、`reset()` 的 robot 访问。
- [x] 加入 `RealEnv.request_stop()` / `clear_stop()`，只设置 stop flag，不直接停 robot。
- [x] `RealEnv.take_action()` 和 `is_episode_end()` 支持 GUI 中断。
- [x] `RealEnv.close()`：保留占位接口，供入口和 workbench 生命周期统一调用。
- [x] 新增 `WorkbenchController`：显式状态、episode index、layout 自动加载、摆放原图保存。
- [x] GUI 主路径：`TASK_INIT -> LAYOUT_READY -> PLACEMENT -> EVALUATING -> AWAIT_RESULT -> PLACEMENT/TASK_FINISHED`。
- [x] “摆放完成”后自动启动后台评测线程：调用 `env.reset()` 和 `env.eval_one_episode()`。
- [x] 成功/失败按钮：写入结果事件，推进 episode 或进入 `TASK_FINISHED`。
- [x] GUI 评测中断：`EVALUATING` 中发送 `env.request_stop()`，评测线程退出后进入 `EPISODE_ABORTED`。
- [x] GUI “重试本轮”：支持 `AWAIT_RESULT` / `EPISODE_ABORTED` 回到当前 episode 的 `PLACEMENT`。
- [x] abort/retry 清理当前未提交产物：`placement.png`、`placement_metadata.json`、`recorder/` 和旧 recorder 的 `*.mp4`。
- [x] recorder 中心采样线程：按 fps 采样 obs，分发给插件。
- [x] 视频 recorder：写入 `episode_dir/recorder/video/`。
- [x] 轨迹 recorder：按 Xone HDF5 结构写入 `episode_dir/recorder/trajectory/trajectory.hdf5`。
- [x] 单元测试覆盖 workbench 主路径、abort、retry、产物清理、GUI offscreen smoke。
- [x] layout 拍摄工具 `layout_shot.py`。
- [x] 评测过程中 GUI 预览 alpha 临时为 0。
- [x] `ckpt_setting` 作为唯一 client 侧 ckpt 相关字段（结果路径）；移除 `ckpt_dir`。
- [x] 模块 README 与 docs 整理（design / workbench / recorder / layout_shot / testing）。
- [x] 真机启动脚本 `real_env/run.sh`。

## 未完成

- [ ] 从 `result_events.jsonl` 恢复中断任务。
- [ ] Error 状态恢复策略：区分可恢复配置错误、评测线程错误、robot/hardware 错误。
- [ ] Recorder 临时文件提交机制。
- [ ] Recorder 新数据类型插件扩展点。
- [ ] Recorder action log 插件。
- [ ] `ResultStore`：统一封装 `result_events.jsonl` append、统计和恢复。
- [ ] 更完整的 task metadata：记录 policy、task、ckpt_setting、layout root、运行时间等。
- [ ] 真机 GUI 入口实际 smoke：确认 `run_real_env_workbench.py` 在 robot client 上能完成一轮人工流程。

## 暂不做

- 不让 `RealEnv` 管 GUI 成功/失败、retry、abort。
- 不在 `request_stop()` 中调用 robot stop/halt/emergency-stop。
- 暂不在 `RealEnv.close()` 中拆解 robot/model client 内部资源。
- 不引入 attempt 子目录。
- 不做已提交 episode 的编辑或回滚。
