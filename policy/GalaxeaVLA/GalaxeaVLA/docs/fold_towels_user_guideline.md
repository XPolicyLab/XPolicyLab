# Out-of-the-Box is All You Need: User Guides for Fold Towels Demo

**Fold Towels Demo** shows how to combine a host computer and a robot (R1 Lite) to implement a fold-towel task.

What you should expect:

* A fully command-line–driven deployment and startup workflow.
* One-click startup scripts for both the robot and the host system.
* Stable, fast, and smooth execution of a fold-towel task.  

## Scene

Our scene setup is shown below:

<p align="center">
  <img src="assets/fold_towel_demo/desktop_setup.png" alt="desktop setup" width="400"/>
</p>

The desktop and the R1 Lite are viewed from the side as follows:

<p align="center">
  <img src="assets/fold_towel_demo/robot_setup.png" alt="robot_setup" width="400"/>
</p>

After calibrating the zero point, you can run the following command on the robot to adjust the R1 Lite to the appropriate position (assuming you have already logged into the robot via SSH):

```bash
ros2 topic pub /motion_target/target_joint_state_torso sensor_msgs/msg/JointState "header:
  stamp: {sec: 0, nanosec: 0}
  frame_id: ''
name: ['']
position: [-0.82, 1.5, 0.5]
velocity: [0]
effort: [0]"
```

**Note:** Before executing the above commands, make sure there are no obstacles near the robot to prevent it from suddenly rising or falling.

## One-Click Startup

### 1. Preparation

#### 1.1 Machine Preparation & Docker Image Build

After preparing the Docker startup environment, use the following command to pull the pre-built image we have provided:

```bash
docker pull edp-image-registry-cn-beijing.cr.volces.com/infer-public/galaxea_infer
```

Once completed, you can verify the image has been pulled locally by using the `docker image` command.

The first time you use it, run the following command to start the container:

```bash
docker run -itd \
--name galaxea \
--network host \
--gpus all \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix/X1:/tmp/.X11-unix/X11 \
edp-image-registry-cn-beijing.cr.volces.com/infer-public/galaxea_infer:latest
```

To enter the container:

```bash
docker exec -it galaxea bash
```

After entering the container, download the model from [TODO: Huggingface](xxx) to the  `~/.galaxea/models/` directory.

### 2. Startup Robot

Open a new command-line window and perform the following steps:

1. (Host) SSH into the robot:

   ```bash
   ssh r1lite@10.42.0.<ROBOT_PORT>
   ```

   * Replace `<ROBOT_PORT>` with the actual IP address of your robot.

2. (Host) Run the following commands on the robot:

   ```bash
   cd ~
   tmux kill-ser
   ros2 daemon stop
   ros2 daemon start
   cd ~/galaxea/install/startup_config/share/startup_config/script/
   ./robot_startup.sh boot ../sessions.d/ATCStandard/R1LITEBody.d/
   ```

### 3. Startup Fold System on Host

1. (Host-Docker) Run the one-click discovery script:

   ```bash
   glx discovery 10.42.0.<ROBOT_PORT>
   ```

   * Replace `<ROBOT_PORT>` with the actual IP address of your robot.

2. (Host-Docker) Run the one-click startup script:

   ```bash
   glx run <MODEL_DIR_NAME>
   ```

   * Replace `<MODEL_DIR_NAME>` with the actual name of the model directory.

   * This command will start the *server-client* mode for model inference, which usually requires a few seconds for the model to load.

## 😊 Enjoy it Now!

Feel free to raise an issue if you have any questions.
