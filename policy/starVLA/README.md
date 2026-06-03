# starVLA

遵循 `XPolicyLab/README.md` 中的统一参数语义与命名约定：

- 数据集子目录命名固定为 5 元组：
  `<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>`
- 训练产物子目录命名固定为 6 元组：
  `<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>`
- `task_name` 仅用于评测阶段，表示仿真器中要运行的任务
- `ckpt_name` 表示 checkpoint 标识；单任务通常与 `task_name` 相同，多任务共训建议显式写成 `cotrain` 或其他固定名称

## 安装

clone 之后先初始化 StarVLA 子模块，再安装依赖(先安装Xpolicy)

```bash
cd /path/to/XPolicyLab/policy/starVLA
git submodule update --init --recursive

conda activate XPolicyLab
bash install.sh
```

## 数采

命令（5 个参数）：

```bash
cd /path/to/XPolicyLab/policy/starVLA
bash process_data.sh ${dataset_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type}
```

例子：

```bash
cd /cpfs_infra/user/wangkaixuan/chengy/demo_env/XPolicyLab/policy/starVLA
conda activate XPolicyLab

bash process_data.sh RoboDojo stack_bowls arx_x5 3500 joint
```

说明：

- 处理后数据默认写入 `policy/starVLA/data/<5元组>`
- 当前 StarVLA 训练默认直接读取外部 LeRobot 数据集；`xpolicy_oft_vla.yaml` 中的 `datasets.vla_data.data_root_dir` 指向 `/cpfs_infra/user/wangkaixuan`
- 当前真实训练数据为 `/cpfs_infra/user/wangkaixuan/RoboDojo_sim_arx-x5_v30`
- StarVLA dataloader 统计缓存文件为 `meta/stats_xpolicy.json`

## 训练

命令（7 个参数，不含 `task_name`）：

```bash
cd /path/to/XPolicyLab/policy/starVLA
bash train.sh ${dataset_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${seed} ${gpu_id}
```

参数语义与总 README 保持一致：

- `dataset_name`: 数据集名称，如 `RoboDojo`
- `ckpt_name`: checkpoint 标识。单任务通常与 `task_name` 相同；多任务共训建议填 `cotrain`
- `env_cfg_type`: 环境配置 / 本体类型，如 `arx_x5`
- `expert_data_num`: 训练轨迹数；如果使用外部 LeRobot 数据集且目录已固定，可将其视为命名占位符，建议填与数据版本一致的固定值
- `action_type`: 动作类型，如 `joint`
- `seed`: 随机种子
- `gpu_id`: GPU 编号列表，如 `0,1,2,3`

### 默认多任务训练

```bash
conda activate XPolicyLab
cd /cpfs_infra/user/wangkaixuan/chengy/demo_env/XPolicyLab/policy/starVLA

bash train.sh RoboDojo cotrain arx_x5 3500 joint 0 0,1,2,3,4,5,6,7
```

训练输出目录：

```text
policy/starVLA/checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-0
```

说明：

- `train.sh` 默认入口为 `source_starvla/starVLA/training/train_starvla.py`
- 默认配置文件为 `xpolicy_oft_vla.yaml`
- 默认使用 StarVLA QwenOFT / MLP action head
- 数据集路径与 `data_mix` 固定由 `xpolicy_oft_vla.yaml` 控制
- 训练中间 checkpoint 默认保存在 `<run_dir>/checkpoints/steps_<step>_pytorch_model.pt`
- 若需要释放显存，可先查看进程：`ps -ef | grep python | grep -v grep`

## 推理

命令（11 个参数）：

```bash
cd /path/to/XPolicyLab/policy/starVLA
bash eval.sh ${dataset_name} ${task_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${seed} ${policy_gpu_id} ${env_gpu_id} ${policy_conda_env} ${eval_env_conda_env}
```

不指定 ckpt：

```bash
conda activate XPolicyLab
cd /cpfs_infra/user/wangkaixuan/chengy/demo_env/XPolicyLab/policy/starVLA

bash eval.sh RoboDojo stack_bowls cotrain arx_x5 3500 joint 0 0 1 XPolicyLab XPolicyLab
```

默认会按 6 元组查找：

```text
policy/starVLA/checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-0/final_model/pytorch_model.pt
```

指定 ckpt：

```bash
conda activate XPolicyLab
cd /cpfs_infra/user/wangkaixuan/chengy/demo_env/XPolicyLab/policy/starVLA

export STARVLA_CKPT_PATH=/cpfs_infra/user/wangkaixuan/chengy/demo_env/XPolicyLab/policy/starVLA/checkpoints/RoboDojo-stack_bowls-arx_x5-3500-joint-0/checkpoints/steps_60000_pytorch_model.pt
bash eval.sh RoboDojo stack_bowls cotrain arx_x5 3500 joint 0 0 1 XPolicyLab XPolicyLab
```
