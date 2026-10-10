# EgoVLA

This is the repo of training code and eval for our work:

### EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos

Ruihan Yang<sup>1*</sup>, Qinxi Yu<sup>2*</sup>, Yecheng Wu<sup>3,4</sup>, Rui Yan<sup>1</sup>, Borui Li<sup>1</sup>, An-Chieh Cheng<sup>1</sup>, Xueyan Zou<sup>1</sup>, Yunhao Fang<sup>1</sup>, Xuxin Cheng<sup>1</sup>, Ri-Zhao Qiu<sup>1</sup>, Hongxu Yin<sup>4</sup>, Sifei Liu<sup>4</sup>, Song Han<sup>3,4</sup>, Yao Lu<sup>4</sup>, Xiaolong Wang<sup>1</sup>

<sup>1</sup>UC San Diego / <sup>2</sup>UIUC / <sup>3</sup>MIT / <sup>4</sup>NVIDIA

[Project Page](https://rchalyang.github.io/EgoVLA) / [Arxiv](https://arxiv.org/abs/2507.12440) / [Simulation Benchmark](https://github.com/quincy-u/Ego_Humanoid_Manipulation_Benchmark)

![img](./media/EgoVLA-Teaser.jpg)

## Installation

### Setup VILA dependency

follow the VILA setup instruction

```bash
cd VILA
./environment_setup.sh vila
```

### Install EgoVLA related dependency

```bash
bash ./build_env.sh
```

Register at the MANO website and download the models.
* Download Mano Hand model [link](https://mano.is.tue.mpg.de/)
* Unzip the mano models and place it in the repo directory (EgoVLA/mano_v1_2)


```sh
git clone https://github.com/hassony2/manopth # This is for hand pose preprocessing
git clone https://github.com/facebookresearch/hot3d # This is for Hot3d data preprocessing
export PYTHONPATH=$PYTHONPATH:/path/to/your/manopth
export PYTHONPATH=$PYTHONPATH:/path/to/your/hot3d
```

### Setup Simulation

Overall instruction for setup IsaacLab: https://isaac-sim.github.io/IsaacLab/main/index.html

* Follow the instruction to install IsaacSim (4.2.0.2)
  ```bash
  pip install isaacsim==4.2.0.2 --extra-index-url https://pypi.nvidia.com

  # if this command failed please try the following command:
  # pip install isaacsim==4.2.0.2 isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 --extra-index-url https://pypi.nvidia.com
  ```
* Clone [Ego Humanoid Manipulation Benchmark](https://github.com/quincy-u/Ego_Humanoid_Manipulation_Benchmark), then install it with the command in the instruction


## 

## Data Preparation

For the following human dataset, we use egocentric RGB video & Hand/Head/Camera Pose & language labels.

* TACO

Download Raw data follow official instruction: https://taco2024.github.io/


And follow instruction (https://github.com/leolyliu/TACO-Instructions) to setup the virtrual environment to process the TACO data. (It's a bit complicated to merge all dependency)

```sh
# RAW DATA: data/TACO
# HF DATA: data/TACO_HF

conda activate taco
# Preprocess RAW -> HF:
sh human_plan/dataset_preprocessing/taco/hf_dataset/generate_dataset_hands_30hz.sh
sh human_plan/dataset_preprocessing/taco/hf_dataset/generate_dataset_image_30hz.sh

```

* HOT3D


Download Raw data follow official instruction: https://github.com/facebookresearch/hot3d

And follow instruction to setup the virtrual environment to process the HOT3D data. (It's a bit complicated to merge all dependency)

```sh
# RAW DATA: data/hot3d
# HF DATA: data/hot3d_hf

# Preprocess RAW -> HF:

conda activate hot3d
sh human_plan/dataset_preprocessing/hot3d/hf_dataset/generate_dataset_hands_job_set1.sh
sh human_plan/dataset_preprocessing/hot3d/hf_dataset/generate_dataset_hands_job_set2.sh
sh human_plan/dataset_preprocessing/hot3d/hf_dataset/generate_dataset_hands.sh
sh human_plan/dataset_preprocessing/hot3d/hf_dataset/generate_dataset_image.sh

```

* HOI4D

Download Raw data from official website: https://hoi4d.github.io/

Follow instruction (https://github.com/leolyliu/HOI4D-Instructions) to setup the virtrual environment to process the HOI4D data. (It's a bit complicated to merge all dependency)
```sh
# RAW DATA: data/HOI4D
# HF DATA: data/hoi4d_hf

conda activate hoi4d
sh human_plan/dataset_preprocessing/hoi4d/hf_dataset/generate_dataset_hands.sh
sh human_plan/dataset_preprocessing/hoi4d/hf_dataset/generate_dataset_image.sh
```

* HoloAssist

Download the data from [HoloAssist Official site](https://holoassist.github.io/#HoloAssist)

```sh
# RAW DATA: data/HoloAssist
# HF DATA: data/ha_dataset

# Preprocess RAW -> HF:
sh human_plan/dataset_preprocessing/holoassist/hf_dataset/generate_dataset_image.sh
sh human_plan/dataset_preprocessing/holoassist/hf_dataset/generate_dataset_hand_set1.sh
sh human_plan/dataset_preprocessing/holoassist/hf_dataset/generate_dataset_hand_set2.sh
sh human_plan/dataset_preprocessing/holoassist/hf_dataset/generate_dataset_hand_merge.sh

```

### Robot Data

#### Original Demonstration

Download from [HuggingFace](https://huggingface.co/datasets/quincyu/EgoVLA-Humanoid-Sim/tree/main)

```bash
huggingface-cli download EgoVLA/EgoVLA-Humanoid-Sim --repo-type dataset --local-dir data/EgoVLA_SIM
```

Data processing 
```bash
#Without Augmentation Version

bash human_plan/dataset_preprocessing/otv_isaaclab/hf_dataset_fixed_set/generate_dataset_image.sh
bash human_plan/dataset_preprocessing/otv_isaaclab/hf_dataset_fixed_set/generate_dataset_hands.sh
```

## Training

#### Pretrained Model:

* [Base VLM](https://huggingface.co/rchal97/egovla_base_vlm)
* [Pretrained Checkpoints on human video](https://huggingface.co/rchal97/ego_vla_human_video_pretrained)
* [EgoVLA](https://huggingface.co/rchal97/egovla)

```
huggingface-cli download rchal97/egovla_base_vlm --repo-type model --local-dir checkpoints

huggingface-cli download rchal97/ego_vla_human_video_pretrained --repo-type model --local-dir checkpoints

huggingface-cli download rchal97/egovla --repo-type model --local-dir checkpoints
```

#### Human Video Pretraining


```bash
bash training_scripts/human_video_pretraining/trans_v2_f1p30_split.sh
```

#### Robot Data Fine-tuning

Download [Pretrained Checkpoints on human video](https://huggingface.co/rchal97/ego_vla_human_video_pretrained)

Put the checkpoints to the correct directory

* **Pretained on Human Video**
  ```bash
    bash training_scripts/robot_finetuning/hoi4dhot3dholotaco_p30_h5_transv2.sh
  ```

  Second stage finetuning
  ```bash
    bash training_scripts/robot_finetuning/hoi4dhot3dholotaco_p30_h5_transv2_continual_lr.sh
  ```

* **Not Pretained on Human Video**

  nopretrained [base model (VILA)](https://huggingface.co/rchal97/egovla_base_vlm)

  ```bash
  bash training_scripts/robot_finetuning/nopretrain_p30_h5_transv2.sh
  ```

  Second stage finetuning
  ```bash
  bash training_scripts/robot_finetuning/nopretrain_p30_h5_transv2_continual_lr.sh
  ```

#### Training retargetting MLP for inference
*The following script will output 
hand_actuation_net.pth and hand_mano_retarget_net.pth used for EgoVLA inference*

*The checkpoint we used (hand_actuation_net.pth & hand_mano_retarget_net.pth) already included* 
```bash
python human_plan/utils/nn_retarget_formano.py
python human_plan/utils/nn_retarget_tomano.py
```

## Simulation on Ego Humanoid Manipulation Benchmark

Before training eval on our Ego Humanoid Manipulation Benchmark. Please follow the instruction on [Ego Humanoid Manipulation Benchmark](https://github.com/quincy-u/Ego_Humanoid_Manipulation_Benchmark)

### Evaluation Single Task on Single Visual Config:

```bash
mkdir video_output
# Evaluation Result will be stored in result_log.txt
# Evaluation Videos will be stored in video_output
# This command is evaluate the given model 
bash human_plan/ego_bench_eval/fullpretrain_p30_h5_transv2.sh Humanoid-Push-Box-v0 1 2 0.2 3 1 result_log.txt 0 0 0.8 video_output evaluation_tag
```

### Evaluation Cross Tasks and Visual Configs:


```bash
python human_plan/ego_bench_eval/batch_script_30hz.py
```

### Note:
The code hasn't been fully tested yet: There might be some hard-coded path issue. Let me know if there is any issues.

This software is part of the BAIR Commons HIC Repository as of calendar year 2025.

## FAQ:

1. Error: 
`"deepspeed/runtime/config_utils.py", line 116, in get_config_default field name). required,
f"'{field name?" is a required field and does not have a default value"
`

    Cause: VILA requires pydantic==v1, and IsaacSim 4.2.0.2 would install one pydantic==V2

    Fix: remove `anaconda3/envs/vila/lib/python3.10/site-packages/isaacsim/extscache/omni.kit.pip_archive-0.0.0+10a4b5c0.lx64.cp310/pip_prebundle/pydantic`  


## Bibtex
```
@misc{yang2025egovlalearningvisionlanguageactionmodels,
      title={EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos}, 
      author={Ruihan Yang and Qinxi Yu and Yecheng Wu and Rui Yan and Borui Li and An-Chieh Cheng and Xueyan Zou and Yunhao Fang and Xuxin Cheng and Ri-Zhao Qiu and Hongxu Yin and Sifei Liu and Song Han and Yao Lu and Xiaolong Wang},
      year={2025},
      eprint={2507.12440},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2507.12440}, 
}
```
