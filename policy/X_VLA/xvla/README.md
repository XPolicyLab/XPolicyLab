# X-VLA

## Environment Preparation

### 1️⃣ Installation

```bash
# Clone the repository
cd policy/X-VLA

# Create and activate Conda environment
conda create -n XVLA python=3.10 -y
conda activate XVLA

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Download X-VLA-Base-Model

```bash
huggingface-cli download --repo-type model 2toINF/X-VLA-Pt --local-dir ./checkpoints/X-VLA-Pt
```

## Training

### 1️⃣ Data Preparation

The RMBench dataset downloaded from Hugging Face needs to have a **language_instruction** key added to the HDF5 files in order to be compatible with X-VLA training. To do this, modify the **data_dir** and **instructions_dir** fields in hdf5_add_language_instruction.py, and then run:

```bash
python hdf5_add_language_instruction.py
```
This will add the required target key to the dataset.


### 2️⃣ Modify 'meta.json' File

You can still list explicit files in **datalist**, but it now also supports automatic path expansion:

- Set **data_dir** to a dataset directory and all `.hdf5` / `.h5` files under it will be discovered automatically.
- Set **data_dirs** to multiple dataset directories if you want to merge several roots.
- Put a directory path or glob pattern such as `/path/to/data/**/*.hdf5` directly inside **datalist** and it will be expanded automatically.

Example:

```json
{
	"dataset_name": "aloha_joint",
	"language_instruction_key": "language_instruction",
	"observation_key": ["observation/head_camera/rgb"],
	"data_dir": "/path/to/your/data"
}
```

### 3️⃣ Start Training

Modify the **models**, **train_metas_path**, and **output_dir** fields in **train.sh** so that they match your actual paths. 

Specifically，
- **models** refers to the path of the previously downloaded X-VLA-Pt
- **train_metas_path** refers to the path of meta.json
- **output_dir** refers to the directory where checkpoints will be saved.

Then, run

```bash
bash train.sh
```

## Evaluation

**Attention:** The saved checkpoint directory may only contain 'config.json', 'model.safetensors', 'state.json'. Please copy all the other files in 'X-VLA-Pt' to the target checkpoint directory (don't overwrite 'config.json', 'model.safetensors' and 'state.json' files).

### 1️⃣ Start the X-VLA Server

Run the X-VLA model as an inference server (in a clean environment to avoid dependency conflicts):

```bash
conda activate X-VLA
python -m deploy --port 4567 --model_path X-VLA-RMBench # change to your own model_path
```

### 2️⃣ Run the Client Evaluation

Add the absolute path of your RMBench repository at line 12 of evaluation/RMBench/client.py:

```bash
robowin_root = Path("/home/admin02/RMBench") # <- Add your path
```

Fill the **ALL_TASKS** and **TASK_INSTRUCTIONS** in line 46 and 61.

Besides, modify the **task_name** in 'evaluation/RMBench/eval_RMBench.sh' to your target task name (the name should be in **ALL_TASKS** with its language instruction in **TASK_INSTRUCTIONS**).

Launch the RMBench evaluation client to connect to your X-VLA server:

```bash
# reopen a clean terminal

cd evaluation/RMBench

conda activate RMBench

bash eval_RMBench.sh
```