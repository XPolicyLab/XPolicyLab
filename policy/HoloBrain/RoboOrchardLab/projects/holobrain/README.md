<div align="center">
  <img src="https://github.com/HorizonRobotics/robot_lab/blob/master/holobrain/assets/holobrain_logo.png?raw=true" alt="HoloBrain Logo" width="400" style="vertical-align: middle; margin-right: 15px;">
  <h1 style="display: inline-block; margin: 10; font-size: 2em">A foundation model for general embodied manipulation</h1>
</div>

<div align="center" class="authors">
Xuewu Lin, Yun Du, Hongyu Xie, Yiwei Jin, Jiawei Li, Shijie Wu, Qingze Wang, Mengao Zhao, Ziang Li, Chaodong Huang, Mengdi Li, Hongzhe Bi, Lichao Huang, Zhizhong Su, Tianwei Lin
</div>

<div align="center" style="line-height: 3;">
  <a href="https://horizonrobotics.github.io/robot_lab/holobrain/" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://img.shields.io/badge/🏠HoloBrain-HomePage-blue" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/abs/2602.12062" target="_blank" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/📄Paper-arXiv-red" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/HorizonRobotics/RoboOrchardLab/tree/master/projects/holobrain/" target="_blank" style="margin: 2px;">
    <img alt="Code" src="https://img.shields.io/badge/💻Code-Github-black" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/collections/HorizonRobotics/holobrain" target="_blank" style="margin: 2px;">
    <img alt="Model" src="https://img.shields.io/badge/⚙️HoloBrain Model-HuggingFace-orange" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>


## :book: Framework
<div align="center">
  <img src="https://github.com/HorizonRobotics/robot_lab/blob/master/holobrain/assets/holobrain_framework.png?raw=true" width="90%" alt="HoloBrain" />
  <p style="font-size:1em; color:#555;">By incorporating explicit embodiment modeling (e.g., camera parameters and kinematic descriptions), our model effectively unifies training across heterogeneous robots. Together with a full-stack VLA infrastructure (RoboOrchard) and an effective test-driven data strategy, HoloBrain-0 delivers superior performance on both real world and simulation manipulation benchmarks.</p>
</div>

## :file_folder: Quick Start

Get up and running with HoloBrain using RoboTwin2.0 simulation data. This walkthrough covers data preparation, training, evaluation, and model export — everything you need to understand the basic pipeline.

> [!NOTE]
> **Working with a real robot?**
> This Quick Start uses simulation data. If you're working with physical hardware, check out these guides instead:
> * **[Real Robot Pipeline Guide](REALBOT_PIPELINE_GUIDE.md)** — from data recording and packaging all the way through to model training.
> * **[Real Robot Deployment Guide](REALBOT_DEPLOY_GUIDE.md)** — hardware setup, camera calibration, and running inference on the real robot.
>
> **Looking for other benchmarks?** HoloBrain also supports other simulation benchmarks:
> * **[Using GenieSim 3.0 Benchmark](benchmarks/geniesim3.md)** — Instructions for packaging GenieSim 3.0 data and running benchmark evaluations with the HoloBrain policy through a GenieSim3-compatible model server.

###  1. Installation
```bash
cd /path/to/robo_orchard_lab
make version
pip install ".[holobrain_0]"
```
note: pytorch3d==0.7.8 is recommended to be installed from [source](https://github.com/facebookresearch/pytorch3d), flash-attn is recommended to be installed from [whl package](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.1).

###  2. Prepare Data
#### Preparing [RoboTwin2.0](https://github.com/RoboTwin-Platform/RoboTwin) Training Data.
Follow the instructions in the RoboTwin code repository to download the required assets and generate data.
Then, use the following command to package the data into LMDB format for training.
```bash
# require data format from the robotwin2.0 master branch before commit e71140e9734e69686daa420a9be8b75a20ff4587
python3 -m robo_orchard_lab.dataset.robotwin.robotwin_packer \
    --input_path path/to/robotwin_data \
    --output_path "projects/holobrain/data/lmdb" \
    --task_names ${task_names} \
    --config_name demo_clean
```

Visualize data for checking.
```bash
cd projects/holobrain
CONFIG=configs/config_holobrain_qwen_common.py # or configs/config_holobrain_gd_common.py
python3 scripts/data_visualize.py --config ${CONFIG}  $@
```

### 3. Run Training
```bash
cd projects/holobrain
CONFIG=configs/config_holobrain_qwen_common.py # or configs/config_holobrain_gd_common.py

# train with single-gpu
python3 scripts/train.py --config ${CONFIG}

# train with multi-gpu multi-machine
# example: 2 machines × 8 gpus
accelerate launch  \
    --num_machines 2 \
    --num-processes 16  \
    --multi-gpu \
    --gpu-ids 0,1,2,3,4,5,6,7  \
    --machine_rank ${current_rank} \
    --main_process_ip ${main_process_ip} \
    --main_process_port 1227 \
    scripts/train.py \
    --workspace ./workspace \
    --config ${CONFIG}
```

### 4. Run Evaluation

#### Close loop evaluation on RoboTwin2.0 Env
Use the maintained evaluation entrypoint:

```bash
cd projects/holobrain
python3 scripts/robotwin_eval.py \
  --model_dir ${MODEL_DIR} \
  --task_names ["place_empty_cup","adjust_bottle","stack_blocks_three"] \
  --mode ray \
  --device cuda \
  --gpu_ids [0,1,2,3,4,5,6,7] \
  --workers_per_gpu 1
```

### 5. Export Model and Processors and Pipeline

Export bundles the trained checkpoint, processor configs, and pipeline definition into a single self-contained artifact that is ready for deployment.

```bash
cd projects/holobrain
CONFIG=configs/config_holobrain_qwen_common.py # or configs/config_holobrain_gd_common.py

python3 scripts/export.py --config ${CONFIG} --workspace ./model_export_path
```

### 6. Model Inference

The exported artifact can be used very conveniently. You can insert the code below into any location to perform model inference.

```python
from robo_orchard_lab.models.holobrain.pipeline import (
  HoloBrainInferencePipeline
)
from robo_orchard_lab.models.holobrain.processor import (
    MultiArmManipulationInput,
    MultiArmManipulationOutput,
)
# use robotwin2_0 as example
pipeline = HoloBrainInferencePipeline.load_pipeline(
    directory="hf://model/HorizonRobotics/HoloBrain_v0.0_Qwen/post_training_robotwin", # or your model dir
    inference_prefix="robotwin2_0",
    device="cuda",
    load_impl="native",
)
pipeline.model.eval()

input_data: MultiArmManipulationInput
output_data: MultiArmManipulationOutput = pipeline(input_data)
```

## :page_facing_up: Citation
```
@misc{lin2026holobrain0technicalreport,
      title={HoloBrain-0 Technical Report}, 
      author={Xuewu Lin and Tianwei Lin and Yun Du and Hongyu Xie and Yiwei Jin and Jiawei Li and Shijie Wu and Qingze Wang and Mengdi Li and Mengao Zhao and Ziang Li and Chaodong Huang and Hongzhe Bi and Lichao Huang and Zhizhong Su},
      year={2026},
      eprint={2602.12062},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.12062}, 
}
```
