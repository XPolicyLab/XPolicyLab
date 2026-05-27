## 🧪 SimplerEnv Benchmark Evaluation Guide

This document outlines the complete evaluation workflow for our model on the SimplerEnv benchmark. The guide below covers environment setup, server deployment, client execution, and result aggregation.

---

### 1️⃣ Environment Setup

#### 1.1 Create and Activate Conda Environment
First, set up the SimplerEnv simulation environment following the [instructions](https://github.com/allenzren/SimplerEnv#Installation). We use Python 3.10 and a specific PyTorch version for compatibility.

If you meet any issues with Vulkan, please following the official [troubleshooting in ManiSkill2](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#troubleshooting).

Run the following commands to set up the environment:

```bash
# Create and activate a new conda environment
conda create -n simplerenv python=3.10 -y
conda activate simplerenv

# Clone the SimplerEnv repository
cd eval_simplerenv
git clone https://github.com/allenzren/SimplerEnv.git --recurse-submodules

# Install numpy<2.0 (otherwise errors in IK might occur in pinocchio):
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install ManiSkill2 real-to-sim environments and their dependencies:
cd SimplerEnv/ManiSkill2_real2sim
pip install -e .

# Install SimplerEnv
cd ..
pip install -e .

pip install numpy==1.24.4 tyro transformers==4.57.1 pandas
```

---

### 2️⃣ Start the Inference Server

We deploy the Xiaomi-Robotics-0 model by launching multiple parallel inference servers to maximize efficiency during the evaluation.

```bash
bash scripts/deploy.sh XiaomiRobotics/Xiaomi-Robotics-0-SimplerEnv-Google-Robot 8 8
bash scripts/deploy.sh XiaomiRobotics/Xiaomi-Robotics-0-SimplerEnv-WidowX 8 8
```

Configuration Details:
*   **Model:** Path to your Xiaomi-Robotics-0-SimplerEnv-Google-Robot model checkpoint.
*   **Resources:** Launches 8 servers across 8 GPUs.
*Note: Port numbers for subsequent servers are incremented sequentially starting from the base port.*

---

### 3️⃣ Start the Evaluation Client

Launch the SimplerEnv evaluation clients. These clients will connect to the distributed servers to perform tasks across the Spatial, Object, Goal, and Long suites.

```bash
bash scripts/launch_simplerenv.sh 8 fractal ./logs
bash scripts/launch_simplerenv.sh 8 bridge ./logs
```

Configuration Details:
*   **Port numbers:** Launches 8 workers for evaluation.
*   **Task:** Evaluate the SimplerEnv Fractal/Bridge task suite.
*   **Log dir:** Save the visualization into ./logs/.

Each client will independently save its evaluation metrics (success rates, task completion logs) to the designated results directory, and the results will be automatically aggregrated by the first client after all evaluations are finished.

---

### 4️⃣ Results
The final results are as follows:

#### 4.1 WidowX Results (Bridge Dataset)

| Task Name | Success / Total | Success Rate |
| :--- | :---: | :---: |
| Put Carrot On Plate | 90 / 144 | **62.5%** |
| Put Eggplant In Basket | 120 / 144 | **83.3%** |
| Put Spoon On Table Cloth | 138 / 144 | **95.8%** |
| Stack Green Cube On Yellow | 108 / 144 | **75.0%** |

#### 4.2 Google Robot Results (Fractal Dataset)
| Evaluation Setting | Task Name | Success / Total | Success Rate |
| :--- | :--- | :---: | :---: |
| Visual Aggregation | Close Drawer | 145 / 189 | **76.7%** |
| | Move Near | 461 / 600 | **76.8%** |
| | Open Drawer | 109 / 189 | **57.7%** |
| | Pick Coke Can | 728 / 825 | **88.2%** |
| | Put In Drawer | 126 / 189 | **66.7%** |
| Visual Matching | Close Drawer | 95 / 108 | **88.0%** |
| | Move Near | 213 / 240 | **88.8%** |
| | Open Drawer | 77 / 108 | **71.3%** |
| | Pick Coke Can | 296 / 300 | **98.7%** |
| | Put In Drawer | 81 / 108 | **75.0%** |

Note: Results may vary when testing on different GPU machines. Therefore, we provide our evaluation logs in the `eval_logs` folder, including detailed per-rank evaluation results and the final merged results, to facilitate comparison and debugging.
