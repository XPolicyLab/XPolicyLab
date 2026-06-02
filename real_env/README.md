# real_env

真机评测 client 模块。运行在 robot 侧，连接远端 policy server，提供 GUI 工作流完成 layout 对齐、摆放确认、policy 评测和结果提交。

## 模块结构

```text
real_env/
  README.md                 本文件
  constants.py              XONE_ROOT / XPOLICYLAB_ROOT
  helpers.py                obs/action 转换、YAML 加载
  real_env_client.py        RealEnv：robot + policy I/O
  workbench.py              WorkbenchController + PyQt GUI
  recorder.py               episode 视频 / 轨迹录制插件
  layout_shot.py            layout 参考图拍摄 GUI
  run_real_env_workbench.py 评测 GUI 入口
  run.sh                    真机启动脚本（设置 PYTHONPATH）
  docs/
    design.md               统一设计与状态机
    real_env.md             RealEnv 接口
    workbench.md            GUI workflow
    recorder.md             录制插件
    layout_shot.md          layout 拍摄
    testing_deployment.md   测试与部署
    TODO.md                 进度与待办
```

## 三层职责

| 层 | 文件 | 职责 |
| --- | --- | --- |
| 环境适配 | `real_env_client.py` | robot obs/action、policy server、eval 调用 |
| 工作流 GUI | `workbench.py` | 状态机、layout、placement、评测线程、结果事件 |
| 录制插件 | `recorder.py` | 可选 video / trajectory 写入 episode 目录 |

`RealEnv` 不管理 GUI 状态；`WorkbenchController` 是唯一 workflow 状态持有者。

## 目录约定

`constants.py` 中 `XONE_ROOT = XPolicyLab 的上一级目录`（pipeline 根目录）：

```text
{XONE_ROOT}/
  config/{base_cfg}.yml
  task_info/{task_name}.json
  layouts/{task_name}/layout_{episode:06d}.png
  eval_results/{policy_name}/{ckpt_setting}/{task_name}/
  src/                      robot 包
  XPolicyLab/               本仓库
```

`ckpt_setting` 仅用于 client 侧结果目录分段；模型权重由 policy server 加载。

## 快速启动

### 1. 拍摄 layout（评测前）

```bash
bash real_env/run.sh   # 见 run.sh 内示例，或：
PYTHONPATH=$PWD:$XONE_ROOT/src python -m real_env.layout_shot \
  --base_cfg x-one-piper-orbbec \
  --task_name stack_bowls \
  --layouts_count 10
```

### 2. 启动 policy server（server 机器）

按各 policy 的 `eval.sh` / `setup_policy_server.py` 在 GPU 机器启动 ModelServer。

### 3. 启动评测 GUI（robot client）

```bash
# 编辑 real_env/run.sh 中的参数后：
bash real_env/run.sh

# 或直接：
PYTHONPATH=$PWD:$XONE_ROOT/src python -m real_env.run_real_env_workbench \
  --base_cfg x-one-piper-orbbec \
  --task_name stack_bowls \
  --policy_name ACT \
  --ckpt_setting RoboDojo_real-stack_bowls-piper-200-joint \
  --host <policy_server_ip> \
  --port <port> \
  --eval_episode_num 10
```

GUI 不会启动 policy server；需先确认 server 已监听 `--host:--port`。

## 文档索引

- 设计与状态机：`docs/design.md`
- RealEnv API：`docs/real_env.md`
- GUI 按钮与产物：`docs/workbench.md`
- Recorder：`docs/recorder.md`
- Layout 拍摄：`docs/layout_shot.md`
- 本地测试 / 参数 / 排错：`docs/testing_deployment.md`
- 进度：`docs/TODO.md`

## 本地测试

```bash
conda run -n dev python -m unittest test.test_real_env_workbench test.test_real_env_recorder
```

虚拟假 robot 全流程（不污染主线）：`tmp/virtual_real_env/run.sh`
