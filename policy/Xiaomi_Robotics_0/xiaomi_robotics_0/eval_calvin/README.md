## 🧪 CALVIN Benchmark Evaluation Guide

This document outlines the complete evaluation workflow for the CALVIN benchmark. We utilize CALVIN to assess the model's generalization capabilities in long-horizon, multi-stage, language-guided manipulation tasks. The guide below covers environment setup, server deployment, client execution, and result aggregation to ensure an efficient and reliable validation process.

---

### 1️⃣ Environment Setup

#### 1.1 Create and Activate Conda Environment
First, install the CALVIN simulation environment and its dependencies. For comprehensive details, refer to the official repositories:
- **calvin:** [https://github.com/mees/calvin](https://github.com/mees/calvin)
- **calvin_env:** [https://github.com/mees/calvin_env](https://github.com/mees/calvin_env)

Run the following commands to set up the environment:

```bash
# Create and activate a new conda environment
conda create -n calvin python=3.10 -y
conda activate calvin

# Clone the main repository
cd eval_calvin
git clone --recurse-submodules https://github.com/mees/calvin.git
cd ./calvin

# Install the specific version of calvin_env
rm -rf calvin_env
git clone --recurse-submodules https://github.com/mees/calvin_env.git
cd calvin_env
git checkout 797142c

# Install tacto and other dependencies
cd tacto
pip install -e .
cd ..
pip install -e .
cd ../calvin_models
pip install -e .

pip install tyro
pip install moviepy==1.0.3
pip install networkx==3.4.2
pip install numpy==1.23.0

sudo apt-get update && sudo apt-get install -y libxrender1 libxrender-dev
```

#### 1.2 Install Additional Dependencies
To resolve potential compatibility issues with `pyhash`, we recommend using the modified version provided by [flowers_vla](https://github.com/intuitive-robots/flower_vla_calvin/tree/main/pyhash-0.9.3).

Install the modified version using the following commands:

```bash
git clone https://github.com/intuitive-robots/flower_vla_calvin.git
cd flower_vla_calvin/pyhash-0.9.3
python setup.py build
python setup.py install
```

---

### 2️⃣ Start the Inference Server

We deploy the Xiaomi-Robotics-0 model by launching multiple parallel inference servers to maximize efficiency.

```bash
bash scripts/deploy.sh XiaomiRobotics/Xiaomi-Robotics-0-Calvin-ABC_D 8 8
bash scripts/deploy.sh XiaomiRobotics/Xiaomi-Robotics-0-Calvin-ABCD_D 8 8
```

Configuration Details:
*   **Model:** XiaomiRobotics/Xiaomi-Robotics-0-Calvin-ABC_D or XiaomiRobotics/Xiaomi-Robotics-0-Calvin-ABCD_D
*   **Resources:** Launches 8 servers across 8 GPUs.

Note: Port numbers for subsequent servers are incremented sequentially starting from the base port.

---

### 3️⃣ Start the Evaluation Client

Launch multiple CALVIN clients in parallel. These clients will connect to the distributed servers to perform the evaluation tasks.

```bash
bash scripts/launch_calvin.sh 8 ./logs/calvin_abc
```

Configuration Details:
*   **Ports numbers:** Launches 8 workers for evaluation.
*   **Log dir:** Save the visulization into ./log/calvin_abc.

Each client will independently save its evaluation metrics to the designated results directory.

---

### 4️⃣ Merge Results

Once the evaluation is complete, aggregate the distributed logs to calculate the final performance metrics.

```bash
python merge_results.py --eval_log_dir ./logs/calvin_abc
```
The final results are as follows:
| Setting | 1 | 2 | 3 | 4 | 5 | Avg. Len. ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ABCD → D** | 99.7% | 98.0% | 96.7% | 94.2% | 91.8% | **4.80** |
| **ABC → D** | 100.0% | 98.3% | 96.0% | 92.6% | 88.1% | **4.75** |

Note: Results may vary when testing on different GPU machines. Therefore, we provide our evaluation logs in the `eval_logs` folder, including detailed per-rank evaluation results and the final merged results, to facilitate comparison and debugging.