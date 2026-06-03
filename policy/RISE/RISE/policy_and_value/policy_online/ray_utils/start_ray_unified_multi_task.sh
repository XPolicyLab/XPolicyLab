#!/bin/bash

echo nnodes=$nnodes
echo node_rank=$node_rank

# Parameter check
if [ -z "$node_rank" ]; then
    echo "Error: RANK environment variable not set!"
    exit 1
fi

# Configuration file path (modify according to actual needs)
SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname "$SCRIPT_PATH")
RAY_PORT=${RAY_PORT:-29500}  # Default port for Ray, can be modified if needed

# Head node startup logic
if [ "$node_rank" -eq 0 ]; then
    # Get local machine IP address (assumed to be intranet IP)
    IP_ADDRESS=$master_addr
    # Start Ray head node
    echo "Starting Ray head node on rank 0, IP: $IP_ADDRESS"
    ray start --head --memory=461708984320 --port=$RAY_PORT

    echo "[HEAD] Ray Cluster is up. Waiting a few seconds for workers..."
    sleep 10

    
else
    HEAD_ADDRESS=$master_addr
    ray start --memory=461708984320 --address="$HEAD_ADDRESS:$RAY_PORT" --block
fi