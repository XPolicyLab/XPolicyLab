# Mem_0 Usage

Mem_0 includes two components: Planning Module and Execution Module. The **environment installation**, **training procedure** and **inference procedure** are listed below.

## Environment Preparation

```bash
cd policy/Mem_0

# create conda environment
conda create -n mem0 python=3.10 -y
conda activate mem0

# Our Project is built on Pytorch2.6.0 + CUDA12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install torchcodec --index-url https://download.pytorch.org/whl/cu124

# install other requirements
pip install -r requirements.txt

# install FlashAttention2
pip install "flash-attn==2.6.1" --no-build-isolation

# install ffmpeg
conda install "ffmpeg" -c conda-forge
```

## Training

The training procedures of Planning Module and Execution Module are separate. 
- As for **M(1)-type** task, only training of Execution Module is needed. 
- As for **M(n)-type** task, training of both Execution Module and Planning Module are needed.

### Execution Module

Assume that the data of RMBench task has been downloaded into ```'{RMBench_workspace}/data'```.

#### 1. Data Preparation

execute following scripts to prepare lerobot type data. The data will be saved into ```lerobot_datasets/``` directory.

```python
# M1 task
python scripts/hdf5_to_lerobot/M1_dataset_to_lerobot.py

# Mn task
python scripts/hdf5_to_lerobot/Mn_dataset_to_lerobot.py

# Note:
# modify **TASK_NAMES** in the file to specify the dataset.
# modify **episode_num** in the file to define the processed episode number.
```

#### 2. Download VLM Checkpoints

In Execution Module, Qwen3-VL-2B is used as VLM backbone. In Planning Module, Qwen3-VL-8B is used as VLM backbone. Please download the checkpoint using follow instructions.

```python
cd checkpoints
python _download.py
```

#### 3. Modify the training config

Please modify the parameters defined in ```source/config/execution_module_train.yaml``` to your own configuration. Some important parameters are listed below:

- ```is_debug```: ```True``` or ```False```, set it to ```True``` to examine the execution of training procedure.
- ```trainer.checkpoint_dir```: define your checkpoint save path.
- ```trainer.wandb_run_name```: define your wandb run name.
- ```trainer.batch_size```: define batch size according to your GPU VRAM.
- ```trainer.train_steps```: define global training steps.
- ```vla_dataset.RMBench.repo_id```: define your training dataset path.

#### 4. Start training

In ```source/training/train_low_standalone.sh```, define your GPU index and nproc_per_node. Then run

```python
bash source/training/train_low_standalone.sh
```

### Planning Module

In the Planning Module, we fine-tune the vision–language model (Qwen3-VL-8B-Instruct) using **LoRA** via **LLaMA-Factory** to enable reasoning over key memories.

#### 1. Prepare the LLaMA-Factory Environment

```python
# open new conda env
conda create -n llama_factory python=3.11
conda activate llama_factory

git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt
# wandb login
pip install wandb
wandb login
```

#### 2. Prepare Fine-Tuning Data, Train, and Merge LoRA

You can either run the pipeline with one script or follow steps 2, 3, and 4 manually.

**Run with script (recommended)**

Edit the "User configuration" section at the top of `run_planning_pipeline.sh`, then from the Mem_0 directory:

```bash
cd policy/Mem_0

bash ./run_planning_pipeline.sh
```

Required variables: `LEROBOT_DATASET_PATH`, `LLAMAFACTORY_ROOT`, `BASE_OUTPUT_DIR`. Optional: `EXPORT_DIR`, `EPISODE_START_ID`, `EPISODE_END_ID`, and training/merge options. To run only specific steps: `STEPS="copy train merge" ./run_planning_pipeline.sh`

<details>
<summary><strong>Follow steps 2, 3, 4 in order (manual)</strong></summary>

**Step 2. Prepare Fine-Tuning Data**

Run the data preparation script (in the `mem0` conda env). Output goes to `llamafactory_data/`.

```bash
python scripts/llama_data_preparation/llamafactory_data_preparation.py \
  --lerobot_dataset_path /path/to/lerobot_datasets/xxx \
  --episode_start_id 0 \
  --episode_end_id 50
```

Copy the generated `.json` file and the images folder from `llamafactory_data/XXX` to `LlamaFactory/data`. Then add an entry to `LlamaFactory/data/dataset_info.json`:

```
  "dataset_name": {
    "file_name": "XXXX.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
```

**Step 3. Train**

In `LLaMA-Factory/examples/train_lora`, create `qwen3_vl_lora_sft.yaml`:

```yaml
### model
model_name_or_path: checkpoints/Qwen3-VL-8B-Instruct
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: cover_blocks_real_data
template: qwen3_vl_nothink
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 8

### output
output_dir: XXX/XXXX_sft_lora
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: wandb

### train
per_device_train_batch_size: 16
gradient_accumulation_steps: 1
learning_rate: 1.0e-4
num_train_epochs: 25
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
```

Then run:

```bash
llamafactory-cli train examples/train_lora/qwen3_vl_lora_sft.yaml
```

**Step 4. Merge LoRA**

In `LLaMA-Factory/examples/merge_lora`, create `qwen3_vl_lora_sft.yaml`:

```yaml
### model
model_name_or_path: checkpoints/Qwen3-VL-8B-Instruct
adapter_name_or_path: XXX/XXXX_sft_lora
template: qwen3_vl_nothink
trust_remote_code: true

### export
export_dir: /save/final/weights/path
export_size: 5
export_device: cpu
export_legacy_format: false
```

Then run:

```bash
llamafactory-cli export examples/merge_lora/qwen3_vl_lora_sft.yaml
```

</details>

#### 5. Load the model with vLLM

Finally we can get the fine-tuned model in the 'export_dir' you defined.

Here we utilize vLLM to load the model.

You should open a new conda environment to configure the vLLM.

```python
conda create -n vllm python=3.10
conda activate vllm

pip install vllm
```

Then using following code to load your model. By default, we load the model using 4 gpus.

```python
export CUDA_VISIBLE_DEVICES=0,1,2,3
vllm serve /save/final/weights/path \
--tensor-parallel-size 4 \
--mm-encoder-tp-mode data \
--async-scheduling \
--media-io-kwargs '{"video": {"num_frames": -1}}' \
--host 0.0.0.0 \
--port 8123

# you can change model_load path, tensor-parallel-size, and port.
# tensor-parallel-size should be aligned with GPU number.
```

This procedure will be used for inference, we set the fine-tuned model as server, and the client part will be introduced in the Inference part.

## Inference

First, place the trained weights in the `./checkpoints` folder.

### 1. Environment

Because Qwen requires compatibility with dependencies such as FlashAttention, the Mem_0 inference Python environment should be installed on top of the RMBench environment (some existing packages will be overwritten). In practice, simply run the RMBench environment setup commands first, and then run the Mem_0 environment setup commands.

i.e.

```bash
conda activate RMBench

bash script/_install.sh # for RMBench, if have done, skip.

# ==== Mem_0 requirements below ====

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install torchcodec --index-url https://download.pytorch.org/whl/cu124

# install other requirements
pip install -r requirements.txt

# install FlashAttention2
pip install "flash-attn==2.6.1" --no-build-isolation

# install ffmpeg
conda install "ffmpeg" -c conda-forge
```

### 2. Normalization

After modifying the `repo_id` related information in the `__main__` section of `dataloader/dataset_min_max.py`, run the script to generate the `norm_stats` for the corresponding dataset. The save path is `Mem_0/assets/<task_name>/norm_stats.json`.

We also support other normalization methods. You just need to use the corresponding `dataloader` and then modify `NORM_WAY` in `deploy_policy.py`.

### 3. Start Evaluation

Just simply run `eval.sh`! We provide an example in `eval.sh` with the main parameters from `deploy_policy.yml` that may need to be replaced. You can quickly start the test by adjusting the parameters in `eval.sh`.

Below are descriptions of some parameters in the `eval.sh` example:
- `global_task`: See `data/<task_name>/demo_clean/instructions/episode0.json`.
- `vllm_url`: Required for Mn tasks when using the planning module; this is the planning module endpoint.
- `action_horizon`: The prefix length of the predicted action chunk that is actually executed. For Mem_0, this value can be up to 30.

Below are descriptions of some parameters that are not overridden in the `eval.sh` example:
- `threshold`: For Mn tasks, this determines how many sub-task termination signals the classifier must output before switching to the next sub-task.

```
bash eval.sh
```

## GPU Resource Requirements

As for **Planning Module**, training is conducted on **8 NVIDIA A800 GPUs** and the duration of training for a single task is approximately **half an hour**.

The **Execution Module** of Mem_0 utilizes a single-task training strategy, where the model is trained from scratch for each specific task. Training is conducted on **8 NVIDIA A800** GPUs with a global batch size of 448 (56 batch size each) over 30K iterations. The duration of training for a single task is approximately **18 hours**.

During training, we set the batch size to the largest value that fully utilizes the **80 GB GPU memory of the A800**. Users can adjust the batch size according to the memory capacity of their available GPUs.

During inference, the Planning Module runs on a dedicated server and is invoked via a vLLM client–server interface. The **Simulation + Execution Module** of Mem_0 requires approximately **12–15 GB of GPU memory**. When the **Execution Module is** executed independently, the GPU memory consumption is reduced to approximately **7–10 GB**.
