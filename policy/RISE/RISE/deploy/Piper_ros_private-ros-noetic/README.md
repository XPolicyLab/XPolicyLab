# Piper Arm Setup Guide

Setup guide for dual Piper arms (master + slave, left + right) including dependencies, workspace build, CAN configuration, and verification.

> Adapted from [Agilex Piper SDK](https://github.com/agilexrobotics) and [χ₀ documentation](https://github.com/OpenDriveLab/kai0)

---

## 1. Install Dependencies
```bash
pip install python-can piper_sdk
sudo apt install ethtool can-utils
```

---

## 2. Build ROS Workspace

⚠️ **Deactivate conda first** - ROS Noetic requires system Python.
```bash
conda deactivate
cd deploy/Piper_ros_private-ros-noetic
source /opt/ros/noetic/setup.bash
catkin_make
```

Auto-source workspace in new terminals:
```bash
echo "source $(pwd)/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## 3. CAN Configuration

Dual-arm setup uses **2 USB-to-CAN modules** (one per arm pair).

**Step 1: Set CAN count**

Edit `can_config.sh`:
```bash
EXPECTED_CAN_COUNT=2
```

**Step 2: Get bus-info for each module**

Insert **left arm CAN module** only:
```bash
sudo ethtool -i can0 | grep bus
# Example output: bus-info: 1-13:1.0
```

Insert **right arm CAN module** (different USB port):
```bash
sudo ethtool -i can1 | grep bus
# Example output: bus-info: 1-12:1.0
```

**Step 3: Update configuration**

Edit `can_config.sh` with your recorded bus-info values:
```bash
USB_PORTS["1-13:1.0"]="can_left:1000000"   # replace with your left arm bus-info
USB_PORTS["1-12:1.0"]="can_right:1000000"  # replace with your right arm bus-info
```

**Step 4: Activate CAN**
```bash
sudo bash can_config.sh
ifconfig | grep can  # verify can_left and can_right are up
```

---

## 4. Launch & Verify

### Mode 0: Teleoperation / Data Collection

Slave follows master. Both arms publish joint states.
```bash
roslaunch piper start_ms_piper.launch mode:=0 auto_enable:=false
```

### Mode 1: Inference / External Control

Slave controlled via ROS topics. **Disconnect master aviation connector first.**
```bash
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true
```

### Verify Joint States
```bash
rostopic echo /puppet/joint_left   # left arm
rostopic echo /puppet/joint_right  # right arm
```

Continuous updates indicate successful configuration.