# 🤖 A1: A Fully Transparent Open-Source, Adaptive and Efficient Truncated Vision-Language-Action Model


<p align="center">
  <a href="https://arxiv.org/abs/2604.05672">
    <img src="https://img.shields.io/badge/arXiv-2604.05672-red?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://github.com/ATeam-Research/A1">
    <img src="https://img.shields.io/badge/Code-GitHub-black?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="http://www.ateam.xin/#/research/A1">
    <img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=internet-explorer&logoColor=white" alt="Project Page">
  </a>
</p>

<!-- <p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10">
  <img src="https://img.shields.io/badge/PyTorch-2.6.0-orange?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.4-green?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA 12.4">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p> -->

---

## 📋 Table of Contents

- [🔧 Installation](#installation)
- [⚙️ Environment Setup](#environment-setup)
- [🚀 Deploy](#deploy)
- [📊 Evaluation](#evaluation)
- [🎓 Training](#training)
- [📦 Model Zoo & Datasets](#model-zoo--datasets)

---

## 🔧 Installation
```
conda create -n a1 python=3.10
conda activate a1
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .[all]
pip install --no-deps --force-reinstall git+https://github.com/moojink/dlimp_openvla
pip install -r requirements.txt 
```

## ⚙️ Environment Setup

### 🔐 1. Copy Environment Template
```bash
📋 cp .env.example .env.personal
```

### ✏️ 2. Configure Your Settings

Edit `.env.personal` with your personal settings:

```bash
# Example content
CONDA_ROOT=/path/to/conda
CONDA_ENV=a1
WANDB_ENTITY=your_entity
WANDB_PROJECT=your_project
```

> 🔒 This file is Git-ignored and won't be committed!

### 🔄 3. Load Environment
```bash
source .env.personal
```

> **⚠️ Security Note**: `.env.personal` contains sensitive information (paths, API keys, etc.). Do NOT commit it to Git.

---

## 🚀 Deploy

Start the API server for model inference:

```bash
🖥️ bash deploy/deploy.sh --weight /put/checkpoint/here --port <port>
```

**📋 Arguments:**
| Argument | Required | Description |
|:--------:|:--------:|:------------|
| `--weight` | ✅ | Path to model checkpoint |
| `--port` | ❌ | Server port (auto-selected if not provided) |
| `--norm` | ❌ | Enable normalization (0 or 1) |

**✨ Example:**
```bash
bash deploy/deploy.sh --weight ./model/checkpoints/pretrain --port 8000
```

---

## 📊 Evaluation

### 🎮 LIBERO Evaluation

#### 📦 1. LIBERO Installation
```bash
📥 git submodule update --init robot_experiments/libero/LIBERO
📥 pip install -e robot_experiments/libero/LIBERO
```

#### 🚀 2. Run Evaluation

| Mode | Command | Description |
|:----:|:--------|:------------|
| 🎯 **Standard** | `bash eval_libero.sh` | Standard evaluation |
| ⚡ **Early Exit** | `bash eval_libero_exit.sh` | Evaluates all 4 LIBERO task suites |



### 🧪 VLABench Evaluation

VLABench evaluation requires running both a server and a client in separate terminals.

**📋 Setup Steps:**

| Step | Action | Command |
|:----:|:-------|:--------|
| 1️⃣ | **Install VLABench** | `pip install -r ...` |
| 2️⃣ | **Download Assets** | `python scripts/download_assets.py` |
| 3️⃣ | **Start Server** 💻 | `bash deploy/deploy.sh ...` |
| 4️⃣ | **Run Client** 🎮 | `python eval_client.py` |

**🔧 Step 1: Install VLABench**
```bash
pip install -r robot_experiments/vlabench/VLABench/requirements.txt
pip install -e robot_experiments/vlabench/VLABench
```

**📥 Step 2: Download Assets** (if not already downloaded)
```bash
cd robot_experiments/vlabench/VLABench
python scripts/download_assets.py --choice all
```

**🖥️ Step 3: Start the Evaluation Server** (Terminal 1)
```bash
# Load environment and start the API server
bash deploy/deploy.sh --weight <path_to_checkpoint> --port 8000
```

**🎮 Step 4: Run the Evaluation Client** (Terminal 2)
```bash
cd robot_experiments/vlabench
python eval_client.py
```

> **⚠️ Note**: The server and client must run in separate terminals. The server loads the model and waits for client connections, while the client sends evaluation requests and receives results.

### 🏆 RoboChallenge Evaluation

RoboChallenge evaluation is executed through the `run_task.py` script, supporting two modes:

| 🎮 Mode | 📖 Description | 🎯 Purpose |
|:-------:|:--------------|:-----------|
| `mock` | Automatic evaluation using local pre-recorded data | Local testing, debugging |
| `real` | Connect to real robot and submit official evaluation | Official competition evaluation |

**🦾 Supported Robot Types:**
- `ALOHA` 🤖
- `ARX5` 🔧
- `UR5` ⚡
- `FRANKA` 🦿

#### 🧪 Mock Mode (Local Automatic Evaluation)

```bash
# 💻 Terminal 1: Deploy model
bash deploy/deploy.sh --weight <path_to_checkpoint> --port 8000

# 🎮 Terminal 2: Run mock evaluation
cd robot_experiments/RoboChallengeInference
python run_task.py \
    --task_name open_the_drawer \
    --test_type mock \
    --url http://localhost:8000
```

#### 🌍 Real Mode (Official Evaluation)

```bash
# 💻 Terminal 1: Deploy model
bash deploy/deploy.sh --weight <path_to_checkpoint> --port 8000

# 🤖 Terminal 2: Run real robot evaluation
cd robot_experiments/RoboChallengeInference
python run_task.py \
    --task_name open_the_drawer \
    --test_type real \
    --url http://localhost:8000 \
    --user_token <your_token> \
    --run_id <run_id> \
    --action_nums 30
```

> **💡 Tips:**
> - ✅ Mock mode automatically starts the mock server, no manual startup required
> - 🔑 Real mode requires valid `user_token` and `run_id` for official evaluation
> - 📋 `task_name` must be defined in `task_config.ROBO_CHALLENGE_TASKS`


---

## 🎓 Training

### 🌟 Pretraining

Pretraining trains the model from scratch using large-scale VLA datasets, supporting distributed training on Slurm clusters.

**📁 Configuration files:**
- 📄 `configs/experiments/pretrain.yaml` - Pretraining experiment configuration
- 📄 `configs/datasets/pretrain.yaml` - Pretraining dataset configuration
- 🖥️ `scripts/slurms/pretrain.sh` - Pretraining script (runs on Slurm cluster)

**🖥️ Slurm Cluster Training:**

1️⃣ **Configure Slurm submission script** `scripts/slurms/submit_job.sh`:
   - 🔧 `nnodes`: Number of nodes required (default 8 nodes)
   - 🔧 `gpus_per_node`: GPUs per node (default 8)
   - 🔧 `partition` and `quotatype`: Partition name and QOS type

2️⃣ **Submit pretraining job:**
```bash
🚀 bash scripts/slurms/submit_job.sh
```

**💻 Single-node multi-GPU training (non-Slurm):**
```bash
🚀 bash scripts/slurms/pretrain.sh
```

> **⚠️ Note:** Pretraining requires significant computational resources. Distributed training on Slurm clusters is recommended. Global batch size = 128 × number of nodes.

---

### 📚 LIBERO Training

LIBERO training fine-tunes on simulation data, supporting single-node multi-GPU training.

**📁 Configuration files:**
- 📄 `configs/experiments/libero_simulation.yaml` - LIBERO training configuration
- 📄 `configs/datasets/libero_4_tasks.yaml` - LIBERO 4-task dataset configuration

**🚀 Run training:**
```bash
bash train_libero.sh
```



---

### 🧪 VLAbench Training

VLAbench training fine-tunes in the VLAbench simulation environment.

**📁 Configuration files:**
- 📄 `configs/experiments/vlabench.yaml` - VLAbench training configuration
- 📄 `configs/datasets/vlabench.yaml` - VLAbench dataset configuration

**🚀 Run training:**
```bash
bash train_vlabench.sh
```


---

### 🏆 RoboChallenge Training

RoboChallenge training uses the `train_rc.sh` script to fine-tune on specific tasks (e.g., open_the_drawer, put_cup_on_coaster).

**📁 Configuration files:**
| File | Description |
|:----:|:------------|
| `configs/experiments/rc_open_the_drawer.yaml` | 🗄️ Open the drawer task config |
| `configs/experiments/rc_put_cup_on_coaster.yaml` | ☕ Put cup on coaster task config |
| `configs/datasets/rc_*.yaml` | 🤖 Dataset configs (ARX5, etc.) |

**🚀 Run training:**
```bash
bash train_rc.sh
```

> **💡 Tip:** Modify the `vla_config_path` variable in the script to switch between different task configurations.

---

## 📦 Model Zoo & Datasets

### 🤖 Pretrained Models

| Model | Description | Checkpoint |
|:------|:------------|:----------:|
| **pretrain** | Pretrained model on large-scale VLA datasets | [Link](https://huggingface.co/spatialtemporal-ai/a1-pretrain) |
| **libero** | Fine-tuned on LIBERO simulation tasks | [Link](https://huggingface.co/spatialtemporal-ai/a1-libero) |
| **libero_exit** | LIBERO model with early exit mechanism | [Link](https://huggingface.co/spatialtemporal-ai/a1-libero-exit) |
| **vlabench** | Fine-tuned on VLABench simulation tasks | [Link](https://huggingface.co/spatialtemporal-ai/a1-vlabench) |
| **rc_put_cup_on_coaster** | Fine-tuned on RoboChallenge put cup task | [Link](https://huggingface.co/spatialtemporal-ai/a1-rc-put-cup-on-coaster) |
| **rc_open_the_drawer** | Fine-tuned on RoboChallenge open drawer task | [Link](https://huggingface.co/spatialtemporal-ai/a1-rc-open-the-drawer) |

### 📊 Training Datasets

| Dataset | Description | Download |
|:--------|:------------|:--------:|
| **Droid** | DROID dataset for robotic manipulation | [Link](https://huggingface.co/datasets/IPEC-COMMUNITY/droid_lerobot) |
| **RoboChallenge** | RoboChallenge competition data | [Link](https://huggingface.co/datasets/RoboChallenge/Table30) |
| **RoboCOIN** | RoboCOIN dataset | [Link](https://huggingface.co/RoboCOIN) |
| **RoboMIND** | RoboMIND benchmark dataset | [Link](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND) |
| **AgiBot** | AgiBot dataset | [Link](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) |
| **LIBERO** | LIBERO simulation tasks | [Link](https://huggingface.co/datasets/spatialtemporal-ai/libero_rlds) |
| **VlaBench** | VLABench simulation environment | [Link](https://huggingface.co/datasets/spatialtemporal-ai/vlabench) |

**📁 Storage Paths:**
- **Model weights**: Place downloaded model weights in the `model/` directory
- **Training data**: Place downloaded datasets in the `data/` directory

> **⚠️ Important Notes:**
> 1. **RoboMIND dataset preprocessing**: Before using RoboMIND dataset, you need to run the indexing script:
>    ```bash
>    bash scripts/robomind_build_index.sh
>    ```
> 2. **LeRobot dataset patch for pretraining**: Before pretraining, you must replace the LeRobot dataset file:
>    ```bash
>    cp a1/data/vla/lerobot_datasets_replace.py <CONDA_ENV_PATH>/lib/python3.10/site-packages/lerobot/datasets/lerobot_dataset.py
>    ```
>    Replace `<CONDA_ENV_PATH>` with your actual conda environment path (e.g., `/path/to/conda/envs/a1`)

> **📝 Note:** Please fill in the actual download links for models and datasets in the table above.

---

## 🙏 Acknowledgements

This project is built upon the [Molmo](https://github.com/allenai/molmo) project. We thank the Allen Institute for AI for their excellent open-source work.

---

## 📚 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@misc{zhang2026a1fullytransparentopensource,
      title={A1: A Fully Transparent Open-Source, Adaptive and Efficient Truncated Vision-Language-Action Model}, 
      author={Kaidong Zhang and Jian Zhang and Rongtao Xu and Yu Sun and Shuoshuo Xue and Youpeng Wen and Xiaoyu Guo and Minghao Guo and Weijia Liufu and Liu Zihou and Kangyi Ji and Yangsong Zhang and Jiarun Zhu and Jingzhi Liu and Zihang Li and Ruiyi Chen and Meng Cao and Jingming Zhang and Shen Zhao and Xiaojun Chang and Feng Zheng and Ivan Laptev and Xiaodan Liang},
      year={2026},
      eprint={2604.05672},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2604.05672}, 
}
```
