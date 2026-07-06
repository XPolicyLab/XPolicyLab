# Removed Absolute Paths

This document records the hard-coded absolute paths that were removed from the
XPolicyLab `policy/` integration layer on 2026-07-07. These paths pointed at
internal cluster storage (`/mnt/xspark-data`, `/vepfs-cnbje63de6fae220`,
`/xspark-cache`, `/mnt/pfs/pg4hw0`, `/mnt/nfs`, `/mnt/petrelfs`, `/root/wan`)
and were not usable outside those machines.

Line numbers refer to the file content **before** the cleanup.

## Replacement Policy

| Original pattern | Replaced with |
| --- | --- |
| Dataset / model-weight defaults in docs and configs | Neutral `/path/to/...` placeholders |
| Shell env-var fallbacks (`${VAR:-/mnt/...}`) | Portable defaults (`${HOME}/.cache/huggingface/lerobot`, `/tmp`, `$(conda info --base)`, repo-relative dirs) or `/path/to/...` placeholders |
| `PRETRAINED_PATH`, `QWEN25_PATH`, etc. with no sane public default | Env-var override required (`${VAR:?...}`) or `/path/to/...` placeholder |
| Repo-internal paths (openpi `assets_dir`, Dexbotic data root, OpenVLA repo root) | Resolved relative to the file location at import/run time |
| `Abot_M0/model.py` stats fallback | `abot_m0/checkpoints/stats_gr00t.json` inside the adapter (still overridable via `ABOT_STATS_JSON`) |

Third-party upstream code vendored under `policy/<POLICY>/source_*` or deep
upstream trees (for example `starVLA/source_starvla`, `EventVLA/source_eventvla`,
`TinyVLA/tinyvla`, Motus upstream `/share/home/...` defaults) was intentionally
left untouched; those paths belong to the original authors' releases.

## A1

### `policy/A1/A1/configs/datasets/xpolicylab_runtime.yaml`

| Line | Removed absolute path |
| --- | --- |
| 8 | `/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v21` |

### `policy/A1/A1/train_config.yaml`

| Line | Removed absolute path |
| --- | --- |
| 5 | `/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v21` |

### `policy/A1/A1/xpolicylab_train.py`

| Line | Removed absolute path |
| --- | --- |
| 310 | `/mnt/xspark-data/xspark_shared/lerobot` |

## AHA_WAM

### `policy/AHA_WAM/AHAWAM/README.md`

| Line | Removed absolute path |
| --- | --- |
| 33 | `/mnt/petrelfs/caijisong/XPolicyLab/policy/AHA_WAM` |
| 35 | `/mnt/petrelfs/caijisong/dualWAM/checkpoints` |
| 46 | `/mnt/petrelfs/caijisong/XPolicyLab/policy/AHA_WAM` |
| 48 | `/mnt/petrelfs/caijisong/dualWAM/checkpoints` |
| 63 | `/mnt/petrelfs/caijisong/dualWAM/checkpoints` |
| 75 | `/mnt/petrelfs/caijisong/XPolicyLab/policy/AHA_WAM` |

### `policy/AHA_WAM/AHAWAM/configs/data/robodojo.yaml`

| Line | Removed absolute path |
| --- | --- |
| 4 | `/mnt/petrelfs/muyao/data/RoboDojo_lerobot_v21_video` |
| 31 | `/mnt/petrelfs/muyao/data/RoboDojo_lerobot_v21_video/dataset_stats.json` |
| 57 | `/mnt/petrelfs/muyao/data/RoboDojo_lerobot_v21_video/text_embeds_cache` |
| 64 | `/mnt/petrelfs/muyao/data/RoboDojo_lerobot_v21_video` |
| 73 | `/mnt/petrelfs/muyao/data/RoboDojo_lerobot_v21_video/dataset_stats.json` |
| 99 | `/mnt/petrelfs/muyao/data/RoboDojo_lerobot_v21_video/text_embeds_cache` |

### `policy/AHA_WAM/AHAWAM/configs/model/ahawam.yaml`

| Line | Removed absolute path |
| --- | --- |
| 9 | `/mnt/petrelfs/caijisong/dualWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` |

## Abot_M0

### `policy/Abot_M0/INSTALLATION.md`

| Line | Removed absolute path |
| --- | --- |
| 36 | `/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_v21_video_abot/meta/stats_gr00t.json` |

### `policy/Abot_M0/README.md`

| Line | Removed absolute path |
| --- | --- |
| 189 | `/mnt/xspark-data/xspark_shared/lerobot` |
| 196 | `/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_v21_video_abot/meta/stats_gr00t.json` |

### `policy/Abot_M0/model.py`

| Line | Removed absolute path |
| --- | --- |
| 16 | `/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_v21_video_abot/meta/stats_gr00t.json` |

### `policy/Abot_M0/abot_m0/INSTALLATION.md`

| Line | Removed absolute path |
| --- | --- |
| 40 | `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Qwen3-VL-4B-Instruct-Action` |
| 41 | `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/ABot-M0-Pretrain/checkpoints/ABot_M0_Pretrain.pt` |
| 65 | `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Qwen3-VL-4B-Instruct-Action` |
| 66 | `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/ABot-M0-Pretrain/checkpoints/ABot_M0_Pretrain.pt` |
| 179 | `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Qwen3-VL-4B-Instruct-Action` |
| 180 | `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/ABot-M0-Pretrain/checkpoints/ABot_M0_Pretrain.pt` |

### `policy/Abot_M0/abot_m0/examples/RoboDojo/train_files/ABot_RoboDojo.yaml`

| Line | Removed absolute path |
| --- | --- |
| 48 | `/xspark-cache/shared/lerobot/` |

### `policy/Abot_M0/abot_m0/examples/RoboDojo/train_files/run_RoboDojo_train.sh`

| Line | Removed absolute path |
| --- | --- |
| 7 | `/xspark-cache/shared/lerobot` |
| 19 | `/mnt/xspark-data/xspark_shared/model_weights` |

### `policy/Abot_M0/abot_m0/reproduce_minimal_load.py`

| Line | Removed absolute path |
| --- | --- |
| 7 | `/vepfs-cnbje63de6fae220/xspark_shared/lerobot` |
| 11 | `/vepfs-cnbje63de6fae220/niantian/ABot-Manipulation` |
| 23 | `/vepfs-cnbje63de6fae220/xspark_shared/lerobot` |

### `policy/Abot_M0/abot_m0/test_data_loading.py`

| Line | Removed absolute path |
| --- | --- |
| 7 | `/vepfs-cnbje63de6fae220/xspark_shared/lerobot` |
| 19 | `/vepfs-cnbje63de6fae220/xspark_shared/lerobot` |

### `policy/Abot_M0/abot_m0/train.sh`

| Line | Removed absolute path |
| --- | --- |
| 26 | `/mnt/xspark-data/xspark_shared/lerobot` |
| 44 | `/mnt/xspark-data/xspark_shared/model_weights` |

### `policy/Abot_M0/scripts/generate_deploy_stats.py`

| Line | Removed absolute path |
| --- | --- |
| 33 | `/mnt/xspark-data/xspark_shared/lerobot` |

## Being_H05

### `policy/Being_H05/Being-H/configs/dataset_info.py`

| Line | Removed absolute path |
| --- | --- |
| 287 | `/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v21` |

### `policy/Being_H05/Being-H/configs/posttrain/robodojo/robodojo_joint_arx_x5.yaml`

| Line | Removed absolute path |
| --- | --- |
| 2 | `/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v21` |

## Dexbotic_DM0

### `policy/Dexbotic_DM0/dexbotic/dexbotic/data/data_source/robodojo_RoboDojo-cotrain-arx_x5-3500-ee.py`

| Line | Removed absolute path |
| --- | --- |
| 7 | `/vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/Dexbotic_DM0/data/RoboDojo-cotrain-arx_x5-3500-ee/video` |
| 8 | `/vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/Dexbotic_DM0/data/RoboDojo-cotrain-arx_x5-3500-ee` |

## GO1

### `policy/GO1/AgiBot-World/go1/configs/go1_sft_robodojo_shared.py`

| Line | Removed absolute path |
| --- | --- |
| 15 | `/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v21` |
| 20 | `/mnt/pfs/pg4hw0/qiwei/models/GO-1` |
| 21 | `/mnt/pfs/pg4hw0/qiwei/models/GO-1` |

### `policy/GO1/AgiBot-World/go1/configs/go1_sft_xpolicylab.py`

| Line | Removed absolute path |
| --- | --- |
| 21 | `/mnt/pfs/pg4hw0/qiwei/models/GO-1` |
| 22 | `/mnt/pfs/pg4hw0/qiwei/models/GO-1` |

## H_RDT

### `policy/H_RDT/H_RDT/datasets/xpolicylab/setup_xpolicylab.sh`

| Line | Removed absolute path |
| --- | --- |
| 6 | `/vepfs-cnbje63de6fae220/hekun/datasets/RoboDojo/sim_cloud` |
| 10 | `/vepfs-cnbje63de6fae220/mobile/chengy/xpolicy/demo_env/XPolicyLab/policy/H_RDT/H_RDT/t5-v1_1-xxl` |

## InternVLA_A1

### `policy/InternVLA_A1/internvla_a1/launch/internvla_a1_3b_finetune.sh`

| Line | Removed absolute path |
| --- | --- |
| 7 | `/xspark-cache/shared` |
| 8 | `/xspark-cache/shared/lerobot` |
| 10 | `/mnt/xspark-data/xspark_shared/model_weights/Cosmos-Tokenizer-CI8x8` |
| 11 | `/mnt/xspark-data/xspark_shared/model_weights/Qwen3-VL-2B-Instruct` |
| 14 | `/mnt/nfs/miniconda3` |
| 69 | `/mnt/xspark-data/xspark_shared/model_weights/InternVLA-A1-3B` |

## LingBot_VA

### `policy/LingBot_VA/lingbot_va/wan_va/configs/va_robotwin30_train_cfg.py`

| Line | Removed absolute path |
| --- | --- |
| 15 | `/mnt/pfs/pg4hw0/niantian/lingbot-va/train_out/checkpoints/checkpoint_step_3600` |
| 122 | `/mnt/pfs/pg4hw0/niantian/lerobot/robotwin4tasks_ee_joint_gripper` |

## LingBot_VLA

### `policy/LingBot_VLA/README.md`

| Line | Removed absolute path |
| --- | --- |
| 128 | `/mnt/xspark-data/xspark_shared/model_weights/Qwen2.5-VL-3B-Instruct` |

### `policy/LingBot_VLA/lingbot_vla/configs/norm/robodojo_sim_arx_x5.yaml`

| Line | Removed absolute path |
| --- | --- |
| 7 | `/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v21` |

### `policy/LingBot_VLA/lingbot_vla/configs/vla/fold_clothes.yaml`

| Line | Removed absolute path |
| --- | --- |
| 2 | `/mnt/pfs/pg4hw0/niantian/model_weights/lingbot-vla-4b` |
| 3 | `/mnt/pfs/pg4hw0/niantian/model_weights/Qwen2.5-VL-3B-Instruct/` |
| 11 | `/mnt/pfs/pg4hw0/niantian/lerobot/fold_clothes` |
| 14 | `/mnt/pfs/pg4hw0/niantian/lingbot-vla/assets/norm_stats/fold_clothes_0.json` |
| 17 | `/mnt/pfs/pg4hw0/niantian/lingbot-vla/output/fold_clothes_01` |

### `policy/LingBot_VLA/lingbot_vla/train.sh`

| Line | Removed absolute path |
| --- | --- |
| 8 | `/mnt/xspark-data/zijian/.cache` |
| 12 | `/mnt/xspark-data/zijian/tmp` |

### `policy/LingBot_VLA/lingbot_vla/train_multinode_robodojo.sh`

| Line | Removed absolute path |
| --- | --- |
| 56 | `/mnt/nfs/miniconda3/etc/profile.d/conda.sh` |
| 60 | `/mnt/xspark-data/xspark_shared/model_weights/lingbot-vla-4b` |
| 61 | `/mnt/xspark-data/xspark_shared/model_weights/Qwen2.5-VL-3B-Instruct` |
| 62 | `/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v21` |
| 64 | `/mnt/xspark-data/zijian/XPolicyLab_main/policy/LingBot_VLA/checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-0` |
| 75 | `/mnt/xspark-data/xspark_shared/lerobot` |
| 76 | `/mnt/xspark-data/zijian/.cache` |
| 79 | `/mnt/xspark-data/zijian/tmp` |

### `policy/LingBot_VLA/setup_eval_policy_server.sh`

| Line | Removed absolute path |
| --- | --- |
| 25 | `/mnt/xspark-data/xspark_shared/model_weights/Qwen2.5-VL-3B-Instruct` |

## Motus

### `policy/Motus/motus/configs/lerobot_RoboDojo_sim.yaml`

| Line | Removed absolute path |
| --- | --- |
| 32 | `/xspark-cache/shared/lerobot/RoboDojo_sim_arx-x5_v21` |
| 39 | `/mnt/xspark-data/xspark_shared/model_weights` |
| 46 | `/mnt/xspark-data/xspark_shared/model_weights/Wan2.2-TI2V-5B` |
| 47 | `/mnt/xspark-data/xspark_shared/model_weights/Wan2.2-TI2V-5B` |
| 48 | `/mnt/xspark-data/xspark_shared/model_weights/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` |
| 51 | `/mnt/xspark-data/xspark_shared/model_weights/Qwen3-VL-2B-Instruct` |

### `policy/Motus/motus/configs/lerobot_sim_stack_bowls.yaml`

| Line | Removed absolute path |
| --- | --- |
| 32 | `/vepfs-cnbje63de6fae220/xspark_shared/lerobot/sim_stack_bowls_v21` |
| 39 | `/vepfs-cnbje63de6fae220/xspark_shared/model_weights` |
| 46 | `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Wan2.2-TI2V-5B` |
| 47 | `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Wan2.2-TI2V-5B` |
| 48 | `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` |
| 51 | `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Qwen3-VL-2B-Instruct` |
| 151 | `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Motus` |

### `policy/Motus/motus/configs/robotwin_xspark.yaml`

| Line | Removed absolute path |
| --- | --- |
| 12 | `/vepfs-cnbje63de6fae220/xspark_shared/robotwin_data/motus_processed` |
| 21 | `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Wan2.2-TI2V-5B` |
| 22 | `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Wan2.2-TI2V-5B` |
| 23 | `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` |
| 27 | `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Qwen3-VL-2B-Instruct` |
| 100 | `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Motus` |

### `policy/Motus/motus/data/robotwin2/robotwin_data_convert/config_xspark.yml`

| Line | Removed absolute path |
| --- | --- |
| 3 | `/vepfs-cnbje63de6fae220/xspark_shared/robotwin_data/raw` |
| 4 | `/vepfs-cnbje63de6fae220/xspark_shared/robotwin_data/motus_processed` |
| 18 | `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Wan2.2-TI2V-5B` |

### `policy/Motus/motus/data/robotwin2/robotwin_data_convert/convert_xspark_gpu4.sh`

| Line | Removed absolute path |
| --- | --- |
| 5 | `/vepfs-cnbje63de6fae220/xspark_shared/miniconda3/etc/profile.d/conda.sh` |

### `policy/Motus/motus/process_data.sh`

| Line | Removed absolute path |
| --- | --- |
| 5 | `/vepfs-cnbje63de6fae220/xspark_shared/lerobot/sim_stack_bowls_v21` |
| 6 | `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/` |

### `policy/Motus/motus/scripts/prepare_lerobot_t5_cache.sh`

| Line | Removed absolute path |
| --- | --- |
| 6 | `/vepfs-cnbje63de6fae220/xspark_shared/lerobot/robodojo_sim` |
| 6 | `/vepfs-cnbje63de6fae220/xspark_shared/model_weights` |
| 13 | `/vepfs-cnbje63de6fae220/xspark_shared/miniconda3/etc/profile.d/conda.sh` |
| 14 | `/vepfs-cnbje63de6fae220/xspark_shared/miniconda3/envs/motus/bin/python` |

### `policy/Motus/motus/train.sh`

| Line | Removed absolute path |
| --- | --- |
| 15 | `/mnt/xspark-data/xspark_shared/motus_ckpt/` |

### `policy/Motus/scripts/build_t5_cache.py`

| Line | Removed absolute path |
| --- | --- |
| 87 | `/mnt/xspark-data/final_data/RoboDojo_first100` |
| 97 | `/mnt/xspark-data/xspark_shared/model_weights/Wan2.2-TI2V-5B` |

## OpenVLA_OFT

### `policy/OpenVLA_OFT/openvla_oft/rlds_dataset_builder/aloha_datasets/README.md`

| Line | Removed absolute path |
| --- | --- |
| 26 | `/mnt/pfs/pg4hw0/niantian/openvla-oft` |
| 29 | `/mnt/pfs/pg4hw0/niantian/openvla-oft/data/aloha_preprocessed/aloha_put_back_block_200_demos` |
| 30 | `/mnt/pfs/pg4hw0/niantian/tensorflow_datasets` |

### `policy/OpenVLA_OFT/openvla_oft/scripts/build_tfds_aloha.sh`

| Line | Removed absolute path |
| --- | --- |
| 11 | `/mnt/pfs/pg4hw0/niantian/openvla-oft` |
| 12 | `/mnt/pfs/pg4hw0/niantian/tensorflow_datasets` |

### `policy/OpenVLA_OFT/openvla_oft/scripts/download_openvla.py`

| Line | Removed absolute path |
| --- | --- |
| 16 | `/mnt/pfs/pg4hw0/niantian/openvla_assets/models/openvla-7b` |

### `policy/OpenVLA_OFT/openvla_oft/scripts/finetune.sh`

| Line | Removed absolute path |
| --- | --- |
| 6 | `/mnt/xspark-data/xspark_shared/model_weights/openvla-7b` |
| 7 | `/mnt/xspark-data/xspark_shared/tensorflow_datasets` |
| 8 | `/mnt/xspark-data/xspark_shared/model_weights/.cache/huggingface` |

### `policy/OpenVLA_OFT/openvla_oft/scripts/register_aloha_dataset.py`

| Line | Removed absolute path |
| --- | --- |
| 36 | `/mnt/pfs/pg4hw0/niantian/openvla-oft` |

## Pi_0

### `policy/Pi_0/openpi/src/openpi/training/config.py`

| Line | Removed absolute path |
| --- | --- |
| 575 | `/vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/Pi_05/openpi/assets/RoboDojo_assets` |
| 610 | `/vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/Pi_05/openpi/assets/RoboDojo_assets` |
| 645 | `/vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/Pi_05/openpi/assets/RoboDojo_assets` |

## Pi_05

### `policy/Pi_05/openpi/src/openpi/training/config.py`

| Line | Removed absolute path |
| --- | --- |
| 575 | `/vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/Pi_05/openpi/assets/RoboDojo_assets` |
| 610 | `/vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/Pi_05/openpi/assets/RoboDojo_assets` |
| 645 | `/vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/Pi_05/openpi/assets/RoboDojo_assets` |

## Pi_0_Fast

### `policy/Pi_0_Fast/openpi/src/openpi/training/config.py`

| Line | Removed absolute path |
| --- | --- |
| 575 | `/mnt/nfs/niantian/RoboDojo_env/XPolicyLab/policy/Pi_0_Fast/openpi/assets/RoboDojo_assets/` |
| 610 | `/mnt/nfs/niantian/RoboDojo_env/XPolicyLab/policy/Pi_0_Fast/openpi/assets/RoboDojo_assets/` |
| 645 | `/mnt/nfs/niantian/RoboDojo_env/XPolicyLab/policy/Pi_0_Fast/openpi/assets/RoboDojo_assets/` |

## RDT_1B

### `policy/RDT_1B/rdt/finetune.sh`

| Line | Removed absolute path |
| --- | --- |
| 8 | `/mnt/xspark-data/xspark_shared/model_weights/t5-v1_1-xxl` |
| 9 | `/mnt/xspark-data/xspark_shared/model_weights/siglip-so400m-patch14-384` |
| 10 | `/mnt/nfs/niantian/RoboDojo_env/aloha_data/RoboDojo` |
| 32 | `/mnt/xspark-data/xspark_shared/model_weights/rdt-1b` |

## Spirit_v15

### `policy/Spirit_v15/spirit_v15/scripts/convert_robotwin_to_spirit.py`

| Line | Removed absolute path |
| --- | --- |
| 30 | `/vepfs-cnbje63de6fae220/xspark_shared/robotwin_data` |

### `policy/Spirit_v15/spirit_v15/scripts/run_robotwin_finetune.sh`

| Line | Removed absolute path |
| --- | --- |
| 25 | `/vepfs-cnbje63de6fae220/xspark_shared/robotwin_data` |
| 34 | `/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Spirit-v1.5` |

## X_VLA

### `policy/X_VLA/xvla/evaluation/RMBench/client.py`

| Line | Removed absolute path |
| --- | --- |
| 12 | `/mnt/pfs/pg4hw0/niantian/RoboTwin` |

## X_WAM

### `policy/X_WAM/transform_robodojo_to_xwam.py`

| Line | Removed absolute path |
| --- | --- |
| 28 | `/root/wan/bin/activate` |

## Xiaomi_Robotics_0

### `policy/Xiaomi_Robotics_0/xiaomi_robotics_0/xr0/assets/config.py`

| Line | Removed absolute path |
| --- | --- |
| 2052 | `/vepfs-cnbje63de6fae220/xspark_shared/xiaomi_datasets/json` |
| 2080 | `/vepfs-cnbje63de6fae220/xspark_shared/xiaomi_checkpoints/project_xr0/robodojo_sim` |
