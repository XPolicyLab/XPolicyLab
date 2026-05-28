# LIBERO Evaluation Guide

This README provides instructions for evaluating GO-1 model fine-tuned on the [LIBERO benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO). For model fine-tuning details, please refer to our main [README](../../README.md).

LIBERO requires a different environment, so we employ the server-client inference for evaluation. The server handles policy inference while the client runs the simulation environment, with communication established through HTTP requests.

## Start GO-1 Server

Start the server first:

```bash
conda activate go1

python evaluate/deploy.py --model_path /path/to/your/checkpoint --data_stats_path /path/to/your/dataset_stats.json --port <SERVER_PORT>
```

## Start LIBERO Client

The client requires a separate terminal session. We strongly recommend using `tmux` or `screen` for this process, as evaluation can take several hours to complete.

1. Clone LIBERO repo:

```bash
cd evaluate/libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
```

2. Create a new conda environment:

```bash
conda create -n libero python=3.8.13 -y
conda activate libero
```

3. Install the required dependencies as [specified](https://github.com/Lifelong-Robot-Learning/LIBERO?tab=readme-ov-file#installtion):

```bash
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .

# Additional required packages
pip install json_numpy draccus
```

4. Run the evaluation script on client side:

```bash
cd ..
bash eval_libero.sh <TASK_SUITE> <SAVE_NAME> <SERVER_IP> <SERVER_PORT>
```

**Arguments:**
- `TASK_SUITE` - Name of the task suite to evaluate (*e.g.*, `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, default: `libero_10`)
- `SAVE_NAME` - Identifier for saving evaluation results
- `SERVER_IP` - IP address of the server (default: `127.0.0.1`)
- `SERVER_PORT` - Port number of the server (default: `9000`)
  

See [main.py](main.py) for more options and details.

## Results
We report the performance of GO-1 model and other baselines in the table below. Our model is fine-tuned jointly on four task suites for 50k steps.

|  Model   | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average  |
| :------: | :------------: | :-----------: | :---------: | :-------: | :------: |
| GR00T N1 |      94.4      |     97.6      |    93.0     |   90.6    |   93.9   |
| $\pi_0$  |    **96.8**    |   **98.8**    |    95.8     |   85.2    |   94.2   |
| GO-1 Air |      94.0      |     96.8      |  **96.2**   | **91.2**  |   94.6   |
|   GO-1   |      96.2      |     97.8      |    96.0     |   89.2    | **94.8** |

