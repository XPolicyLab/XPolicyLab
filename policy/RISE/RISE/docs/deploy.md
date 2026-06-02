# Inference Deployment on Piper

Real-time inference deployment for OpenPi policies on Agilex Piper dual-arm robots.

---

## Prerequisites

### System Requirements

- **OS**: Ubuntu 20.04
- **ROS**: ROS Noetic

### Hardware Setup

**1. Piper Dual Arms**

Configure master + slave arms following the [Piper Arm Setup Guide](../deploy/Piper_ros_private-ros-noetic/README.md).

**2. RealSense Cameras**

Install ROS wrapper:
```bash
sudo apt-get install ros-noetic-realsense2-camera
```

Update camera serials in `deploy/multi_camera.launch`:
```xml
<arg name="serial_no_camera1" default="YOUR_CAMERA_1_SERIAL"/>
<arg name="serial_no_camera2" default="YOUR_CAMERA_2_SERIAL"/>
<arg name="serial_no_camera3" default="YOUR_CAMERA_3_SERIAL"/>
```

Deploy and test:
```bash
cp deploy/multi_camera.launch /opt/ros/noetic/share/realsense2_camera/launch/
roslaunch realsense2_camera multi_camera.launch
```

---

## Installation

**1. Create Environment**
```bash
conda create -n deploy python=3.10 -y
conda activate deploy
```

**2. Install Dependencies**
```bash
cd deploy
pip install -r requirements.txt
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```
> For other CUDA versions, see [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

**3. Install openpi-client**
```bash
cd deploy/packages/openpi-client
pip install -e .
```

---

## Running Inference

Start the [openpi policy server](https://github.com/Physical-Intelligence/openpi) on your GPU machine first.

**Launch in 3 terminals:**

```bash
# Terminal 1: Cameras
roslaunch realsense2_camera multi_camera.launch

# Terminal 2: Robot Arms
bash deploy/Piper_ros_private-ros-noetic/can_config.sh

roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true

# Terminal 3: Inference
conda activate deploy
python deploy/piper_deploy.py \
  --host 172.16.99.11 \
  --port 8000 \
  --ctrl_type joint \
  --use_temporal_smoothing \
  --chunk_size 50 \
  --lang_embeddings "Pick and sort bricks on the conveyor."
```

---

## Data Collection

Both scripts save data in HDF5 format with synchronized images, joint states, and actions.

**Output Structure:**
```
dataset_dir/
├── task_name/
│   ├── episode_0.hdf5
│   ├── episode_1.hdf5
│   └── video/
│       ├── cam_high/
│       ├── cam_left_wrist/
│       └── cam_right_wrist/
└── aloha_mobile/           # for inference data
    ├── aloha_mobile_success/
    └── aloha_mobile_fail/
```

### Teleoperation Collection

Manual control via master-slave setup.

```bash
# Terminal 1: Cameras
roslaunch realsense2_camera multi_camera.launch

# Terminal 2: Arms (master-slave mode)
bash deploy/Piper_ros_private-ros-noetic/can_config.sh

roslaunch piper start_ms_piper.launch mode:=0 auto_enable:=false

# Terminal 3: Collection
conda activate deploy
cd deploy/data_collection
python collect_data.py \
  --max_timesteps 5000 \
  --export_video \
  --dataset_dir ~/data/my_task \
  --episode_idx 0
```

**Controls:** Press `Space` to stop and save early.

### Inference Collection

Record autonomous policy execution. Requires running policy server.

```bash
# Terminal 1: Cameras
roslaunch realsense2_camera multi_camera.launch

# Terminal 2: Arms (inference mode)
bash deploy/Piper_ros_private-ros-noetic/can_config.sh

roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true

# Terminal 3: Collection
conda activate deploy
cd deploy/data_collection
python collect_inference_data.py \
  --use_temporal_smoothing \
  --ctrl_type joint \
  --export_video \
  --chunk_size 50 \
  --host 172.16.99.11 \
  --port 8000 \
  --dataset_dir ~/data/rl_task
```

**Workflow:**
1. Press `Enter` to start recording
2. Press `s` to stop robot
3. Label episode: `1` (success) or `0` (failure)
