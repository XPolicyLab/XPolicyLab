#!/usr/bin/env bash
set -euo pipefail

# ------------------ Iterative Input: EXEC_MODE ------------------
while true; do
  echo "Please enter the execution mode (0 or 1, press q to exit):"
  echo "0: Initial execution (docker run, and copy the checkpoint weight)"
  echo "1: Second execution (docker start only)"
  read -r EXEC_MODE

  if [[ "$EXEC_MODE" == "q" ]]; then
    echo "Exited."
    exit 0
  fi

  if [[ "$EXEC_MODE" == "0" || "$EXEC_MODE" == "1" ]]; then
    break
  else
    echo "Invalid input, please re-enter."
  fi
done

# ------------------ Iterative Input：USE_QWEN_FLAG ------------------
while true; do
  echo "Enable the Qwen model for VLM? (0 or 1, press q to exit):"
  echo "0: Disabled (uses the default Gemini; note that non-overseas users need to correctly configure the proxy on their host machine, e.g. turn on the Clash)"
  echo "1: Enable (no proxy required on the host machine)"
  read -r USE_QWEN_FLAG

  if [[ "$USE_QWEN_FLAG" == "q" ]]; then
    echo "Exited."
    exit 0
  fi

  if [[ "$USE_QWEN_FLAG" == "0" || "$USE_QWEN_FLAG" == "1" ]]; then
    break
  else
    echo "Invalid input, please re-enter."
  fi
done

# ------------------ Varibale Definition ------------------

if [[ "$USE_QWEN_FLAG" == "0" ]]; then
  vlm_use_qwen="0"
else
  vlm_use_qwen="1"
fi

if [[ "$EXEC_MODE" == "0" ]]; then
  DEFAULT_VLM_API_KEY="NaN"
  DEFAULT_VLM_API_KEY_QWEN="NaN"
  echo "Please enter the api key for VLM(Gemini): (Press Enter to use the default: ${DEFAULT_VLM_API_KEY}):"
  echo "** If you haven't fit in a valid Gemini API key in this step, you cannot modify it after the container is created."
  read -r USER_VLM_API_KEY

  if [[ -z "$USER_VLM_API_KEY" ]]; then
    VLM_API_KEY="$DEFAULT_VLM_API_KEY"
  else
    VLM_API_KEY="$USER_VLM_API_KEY"
  fi

  echo "Please enter the api key for VLM(Qwen): (Press Enter to use the default: ${DEFAULT_VLM_API_KEY_QWEN}):"
  echo "** If you haven't fit in a valid Qwen API key in this step, you cannot modify it after the container is created."
  read -r USER_VLM_API_KEY_QWEN

  if [[ -z "$USER_VLM_API_KEY_QWEN" ]]; then
    VLM_API_KEY_QWEN="$DEFAULT_VLM_API_KEY_QWEN"
  else
    VLM_API_KEY_QWEN="$USER_VLM_API_KEY_QWEN"
  fi

  echo "-------------[Initial Execution!] Used Configuration-------------"
  echo "VLM API Key Gemini：${USER_VLM_API_KEY}"
  echo "VLM API Key Qwen：${USER_VLM_API_KEY_QWEN}"
  echo "VLM Enable Qwen Model：${vlm_use_qwen}"
  echo "------------------------------------------"
else
  echo "-------------[Second Execution!] Used Configuration-------------"
  echo "VLM API Keys Passed" 
  echo "VLM Enable Qwen Model：${vlm_use_qwen}"
  echo "------------------------------------------"
fi




DEFAULT_RLROBOT_PORT="180"
echo "Please enter the robot port ID of R1Lite: 10.42.0.<PORT> (Press Enter to use the default: ${DEFAULT_RLROBOT_PORT}):"
read -r USER_RLROBOT_PORT

if [[ -z "$USER_RLROBOT_PORT" ]]; then
  rlrobot_port="$DEFAULT_RLROBOT_PORT"
else
  rlrobot_port="$USER_RLROBOT_PORT"
fi



DEFAULT_CONTAINER="g0plus_ros2_v1"

echo "[Last Option] Please enter the name of the Docker container (press Enter to use the default: ${DEFAULT_CONTAINER}):"
read -r USER_CONTAINER_INPUT

if [[ -z "$USER_CONTAINER_INPUT" ]]; then
  CONTAINER="$DEFAULT_CONTAINER"
else
  CONTAINER="$USER_CONTAINER_INPUT"
fi



# ------------------ Execute Commands According to EXEC_MODE ------------------

if [[ "$EXEC_MODE" == "0" ]]; then
  echo "Execute Mode 0: Initial execution, starting the container..."
  docker run -itd --name "$CONTAINER" \
  --network host --gpus all --shm-size 256G -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$HOME/g0plus_ros2/data":/data \
  -v /usr/local/cuda-12.8:/usr/local/cuda-12.8:ro \
  -e CUDA_HOME=/usr/local/cuda-12.8 \
  -e LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-} \
  -e VLM_API_KEY="${VLM_API_KEY}" \
  -e VLM_API_KEY_QWEN="${VLM_API_KEY_QWEN}" \
  -u ros g0plus:ros2_v1-trt /bin/bash 

  echo "Copying the weight file...Copying the configuration file..."

  docker exec -it "$CONTAINER" bash -lc "\
    cp -f /data/docker_g0plus_hs_start_v1.sh /home/ros/docker_g0plus_hs_start_v1.sh && \
    echo \"export FASTRTPS_DEFAULT_PROFILES_FILE=/home/ros/super_client_configuration_file.xml\" >> /home/ros/.bashrc && \
    sed \"s/RLROBOT_PORT/${rlrobot_port}/g\" \
    /home/ros/super_client_configuration_file.xml.tpl \
    > /home/ros/super_client_configuration_file.xml && \
    cd /home/ros && \
    chmod +x docker_g0plus_hs_start_v1.sh && \
    ./docker_g0plus_hs_start_v1.sh ${vlm_use_qwen}"

else
  echo "Execute Mode 1: Launches Docker containers with one click, without copying checkpoint."
  if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
      echo "Error: Container '$CONTAINER' is not existed." >&2
      exit 1
  fi

  is_running=$(docker inspect -f '{{.State.Running}}' "$CONTAINER")

  if [ "$is_running" = "true" ]; then
      echo "Container '$CONTAINER' is running, skip docker start."
  else
      echo "Container '$CONTAINER' has not been started, try to start..."
      docker start "$CONTAINER" || { echo "Error: Container is started fail."; exit 1; }
  fi
  docker exec -it "$CONTAINER" bash -lc "\
    cp -f /data/docker_g0plus_hs_start_v1.sh /home/ros/docker_g0plus_hs_start_v1.sh && \
    cd /home/ros && \
    chmod +x docker_g0plus_hs_start_v1.sh && \
    ./docker_g0plus_hs_start_v1.sh ${vlm_use_qwen}"
fi

echo " Complete execution! Entering the container and attaching the tmux session..."
docker exec -it "$CONTAINER" bash -lc "tmux a"
