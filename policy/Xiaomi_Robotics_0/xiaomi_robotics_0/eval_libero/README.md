## 🧪 LIBERO Benchmark Evaluation Guide

This document outlines the complete evaluation workflow for our model on the LIBERO benchmark: [https://github.com/Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO). 

We utilize LIBERO to assess the model's lifelong learning and generalization capabilities across four distinct subtasks: **Spatial**, **Object**, **Goal**, and **Long**. The guide below covers environment setup, server deployment, client execution, and result aggregation.

---

### 1️⃣ Environment Setup

#### Create and Activate Conda Environment
First, set up the LIBERO simulation environment following the [official instructions](https://github.com/Lifelong-Robot-Learning/LIBERO). We use Python 3.10 and a specific PyTorch version for compatibility.
Run the following commands to set up the environment:

```bash
# Create and activate a new conda environment
conda create -n libero python=3.10 -y
conda activate libero

# Clone the LIBERO repository
cd eval_libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

# Install basic requirements
pip install -r requirements.txt
# Install the LIBERO package in editable mode
pip install --config-settings editable_mode=compat -e ./LIBERO

sudo apt update && sudo apt install -y xvfb
cd ../
```

---

### 2️⃣ Start the Inference Server

We deploy the Xiaomi-Robotics-0 model by launching multiple parallel inference servers to maximize efficiency during the evaluation.

```bash
bash scripts/deploy.sh XiaomiRobotics/Xiaomi-Robotics-0-LIBERO 8 8
```

Configuration Details:
*   **Model:** Path to your Xiaomi-Robotics-0-LIBERO model checkpoint.
*   **Resources:** Launches 8 servers across 8 GPUs.
*Note: Port numbers for subsequent servers are incremented sequentially starting from the base port.*

---

### 3️⃣ Start the Evaluation Client

Launch the LIBERO evaluation clients. These clients will connect to the distributed servers to perform tasks across the Spatial, Object, Goal, and Long suites.

```bash
bash scripts/launch_libero.sh 8 libero_10 ./logs
bash scripts/launch_libero.sh 8 libero_goal ./logs
bash scripts/launch_libero.sh 8 libero_object ./logs
bash scripts/launch_libero.sh 8 libero_spatial ./logs
```

Configuration Details:
*   **Port numbers:** Launches 8 workers for evaluation.
*   **Task:** Evaluate the libero_10 task suite.
*   **Log dir:** Save the visualization into ./log/libero.

Each client will independently save its evaluation metrics (success rates, task completion logs) to the designated results directory.

---

### 4️⃣ Merge Results

Once the evaluation is complete for all subtasks, aggregate the distributed logs to calculate the final performance metrics.

```bash
python merge_results.py path/to/results
```
The final results are as follows:
| LIBERO-10 | LIBERO-Goal | LIBERO-Object | LIBERO-Spatial | **Average** |
| :---: | :---: | :---: | :---: | :---: |
| 97.2% | 98.8% | 100.0% | 98.8% | **98.7%** |

Note: Results may vary when testing on different GPU machines. Therefore, we provide our evaluation logs in the `eval_logs` folder, including detailed per-rank evaluation results and the final merged results, to facilitate comparison and debugging.
