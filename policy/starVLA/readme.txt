clone之后  cd /cpfs_infra/user/wangkaixuan/chengy/demo_env/XPolicyLab/policy/starVLA 这个子文件下面 运行install.sh
bash ./eval.sh RoboDojo stack_bowls stack_bowls arx_x5 3500 joint 0 0 1 XPolicyLab XPolicyLab \
/cpfs_infra/user/wangkaixuan/chengy/demo_env/XPolicyLab/policy/starVLA/checkpoints/RoboDojo-stack_bowls-arx_x5-3500-joint-0/checkpoints/steps_60000_pytorch_model.pt
测试正常





使用真实训练数据：
/cpfs_infra/user/wangkaixuan/RoboDojo_sim_arx-x5_v30
StarVLA dataloader 统计缓存文件：meta/stats_xpolicy.json

旧的 50 条数据转换命令（不再作为默认训练数据）：
cd /cpfs_infra/user/wangkaixuan/chengy/demo_env/XPolicyLab/policy/starVLA
conda activate XPolicyLab

bash process_data.sh RoboDojo stack_bowls arx_x5 50 joint


训练（默认使用 train_starvla.py，只使用 vla_data；VLM 会通过机器人 action loss 更新）：
cd /cpfs_infra/user/wangkaixuan/chengy/demo_env/XPolicyLab/policy/starVLA
conda activate XPolicyLab

训练 StarVLA-alpha 默认模型（OFT / MLP action head）：
bash train.sh RoboDojo stack_bowls arx_x5 3500 joint 0 0,1,2,3,4,5,6,7

使用同一真实数据集做 10 step 保存测试：
bash train_10step_save.sh RoboDojo stack_bowls arx_x5 3500 joint 0 0,1,2,3,4,5,6,7
输出目录：checkpoints/RoboDojo-stack_bowls-arx_x5-3500-joint-0-smoke10

train.sh 参数：
dataset_name ckpt_name env_cfg_type expert_data_num action_type seed gpu_id

输出目录命名：
checkpoints/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>

数据集路径固定由 xpolicy_oft_vla.yaml 配置

释放缓存：
ps -ef | grep python | grep -v grep

仿真测试：
cd /cpfs_infra/user/wangkaixuan/chengy/demo_env/XPolicyLab/policy/starVLA

bash eval.sh \
  RoboDojo \
  stack_bowls \
  stack_bowls \
  arx_x5 \
  3500 \
  joint \
  0 \
  0 \
  1 \
  XPolicyLab \
  XPolicyLab \
  /cpfs_infra/user/wangkaixuan/chengy/demo_env/XPolicyLab/policy/starVLA/checkpoints/RoboDojo-stack_bowls-arx_x5-3500-joint-0-smoke10/final_model/pytorch_model.pt