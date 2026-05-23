XPolicyLab 接入 H_RDT 操作说明
================================

目录：
/vepfs-cnbje63de6fae220/mobile/chengy/xpolicy/demo_env/XPolicyLab/policy/H_RDT


1. 基本说明
-----------

本 policy 名称使用 H_RDT，外部源码位于：

H_RDT/

当前第一版只支持 joint 动作：

action_type=joint

暂不支持 ee，因为 H-RDT 原始 RobotWin2 示例是按 joint action 训练和推理的。


2. 激活环境
-----------

进入目录：

cd /vepfs-cnbje63de6fae220/mobile/chengy/xpolicy/demo_env/XPolicyLab/policy/H_RDT

激活 conda 环境：

conda activate XPolicyLab_chengy

建议确认缓存和临时目录走大盘：

export TMPDIR=/vepfs-cnbje63de6fae220/tmp
export TEMP=/vepfs-cnbje63de6fae220/tmp
export TMP=/vepfs-cnbje63de6fae220/tmp
export HF_HOME=/vepfs-cnbje63de6fae220/mobile/chengy/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub


3. 安装依赖
-----------

如果还没有安装 H-RDT 依赖，可以执行：

bash install.sh

install.sh 会做两件事：

1. 安装 H_RDT/requirements.txt
2. 回到 XPolicyLab 根目录执行 pip install -e .


4. T5 模型
----------

T5 是 H-RDT 这个 policy 的语言编码器依赖，不是 XPolicyLab 框架通用依赖。

H-RDT 配置中语言特征维度是 4096，因此需要 t5-v1_1-xxl 这类匹配的模型。

下载示例：

cd /vepfs-cnbje63de6fae220/xspark_shared/model_weights

huggingface-cli download google/t5-v1_1-xxl \
  --local-dir t5-v1_1-xxl \
  --resume-download

下载完成后设置：

export T5_MODEL_PATH=/vepfs-cnbje63de6fae220/xspark_shared/model_weights/t5-v1_1-xxl

如果已经提前生成了语言 embedding，也可以直接把文件放到：

H_RDT/datasets/robotwin2/lang_embeddings/${task_name}.pt


5. 数据处理
-----------

XPolicyLab 原始数据路径格式：

/vepfs-cnbje63de6fae220/mobile/chengy/xpolicy/demo_env/data/${dataset_name}/${task_name}/${env_cfg_type}/data

H_RDT 的 process_data.sh 会把 XPolicyLab 数据转换成 H-RDT 可训练的 RobotWin2 风格 HDF5。

命令格式：

bash process_data.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} joint

示例：

bash process_data.sh RoboDojo stack_bowls arx_x5 50 joint

输出目录：

data/${dataset_name}-${task_name}-${env_cfg_type}-${expert_data_num}-joint/

例如：

data/RoboDojo-stack_bowls-arx_x5-50-joint/


6. 启动训练
-----------

推荐直接用 train.sh。它会自动检查并调用 process_data.sh，然后启动 H-RDT 的 accelerate 训练。

命令格式：

bash train.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} joint ${seed} ${gpu_id} [pretrained_backbone_path]

示例，不加载 human pretrain backbone：

bash train.sh RoboDojo stack_bowls arx_x5 50 joint 0 0

示例，加载 H-RDT human pretrain backbone：

bash train.sh RoboDojo stack_bowls arx_x5 50 joint 0 0 /path/to/pretrained_backbone/pytorch_model.bin

参数含义：

dataset_name: 数据集名称，例如 RoboDojo
task_name: 任务名称，例如 stack_bowls
env_cfg_type: 环境配置名称，例如 arx_x5
expert_data_num: 使用轨迹数量，例如 50
action_type: 固定为 joint
seed: 随机种子，例如 0
gpu_id: 使用的 GPU，例如 0
pretrained_backbone_path: 可选，H-RDT 预训练 backbone 路径


7. 可选训练参数
---------------

train.sh 支持通过环境变量覆盖部分训练参数：

HRDT_TRAIN_BATCH_SIZE，默认 4
HRDT_SAMPLE_BATCH_SIZE，默认 4
HRDT_MAX_TRAIN_STEPS，默认 1000000
HRDT_CHECKPOINTING_PERIOD，默认 5000
HRDT_CHECKPOINTS_TOTAL_LIMIT，默认 40
HRDT_DATALOADER_NUM_WORKERS，默认 4
HRDT_LEARNING_RATE，默认 1e-4
HRDT_REPORT_TO，默认 tensorboard
HRDT_DEEPSPEED_CONFIG，默认 configs/zero1.json

示例，小规模 smoke test：

export HRDT_MAX_TRAIN_STEPS=10
export HRDT_CHECKPOINTING_PERIOD=5
export HRDT_TRAIN_BATCH_SIZE=1
export HRDT_SAMPLE_BATCH_SIZE=1

bash train.sh RoboDojo stack_bowls arx_x5 50 joint 0 0


8. 输出位置
-----------

训练 checkpoint 默认保存到：

checkpoints/${dataset_name}-${task_name}-${env_cfg_type}-${expert_data_num}-joint_seed${seed}/

示例：

checkpoints/RoboDojo-stack_bowls-arx_x5-50-joint_seed0/


9. 评测接口
-----------

当前已经适配了 XPolicyLab 的部署接口：

model.py
deploy.yml
eval.sh
setup_eval_policy_server.sh

评测第一版也只支持：

action_type=joint

eval.sh 参数格式：

bash eval.sh \
  ${dataset_name} \
  ${task_name} \
  ${ckpt_name} \
  ${env_cfg_type} \
  ${expert_data_num} \
  joint \
  ${seed} \
  ${policy_gpu_id} \
  ${env_gpu_id} \
  ${policy_conda_env} \
  ${eval_env_conda_env} \
  ${checkpoint_path} \
  ${config_path} \
  ${lang_embedding_path}

示例：

bash eval.sh \
  RoboDojo \
  stack_bowls \
  hrdt_ckpt \
  arx_x5 \
  50 \
  joint \
  0 \
  0 \
  0 \
  XPolicyLab_chengy \
  XPolicyLab_chengy \
  /path/to/checkpoint_dir \
  data/RoboDojo-stack_bowls-arx_x5-50-joint/hrdt_finetune_xpolicy.yaml \
  H_RDT/datasets/robotwin2/lang_embeddings/stack_bowls.pt


10. 常见问题
------------

1. 报缺少 T5：

设置：

export T5_MODEL_PATH=/path/to/t5-v1_1-xxl

2. 报缺少 language embedding：

确认是否存在：

H_RDT/datasets/robotwin2/lang_embeddings/${task_name}.pt

如果不存在，train.sh 会尝试用 T5 自动生成。

3. 报 action_type 不支持：

当前 H_RDT 只支持：

joint

4. 报 No space left on devices：

通常不是 /vepfs-cnbje63de6fae220 没空间，而是缓存或临时目录写到了 /root 或 /tmp。
请确认：

echo $TMPDIR
echo $HF_HOME
echo $TRANSFORMERS_CACHE
echo $HUGGINGFACE_HUB_CACHE

5. 报找不到数据：

确认原始数据目录存在：

/vepfs-cnbje63de6fae220/mobile/chengy/xpolicy/demo_env/data/${dataset_name}/${task_name}/${env_cfg_type}/data

例如：

/vepfs-cnbje63de6fae220/mobile/chengy/xpolicy/demo_env/data/RoboDojo/stack_bowls/arx_x5/data


11. 当前已改动的关键文件
------------------------

process_data.py
process_data.sh
train.sh
model.py
deploy.yml
eval.sh
setup_eval_policy_server.sh
install.sh
H_RDT/main.py
H_RDT/train/train.py
H_RDT/datasets/dataset.py
H_RDT/models/encoder/t5_encoder.py

