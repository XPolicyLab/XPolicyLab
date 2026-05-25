#!/bin/bash
# Launcher script for SLURM jobs
# Usage: ./scripts/launch.sh [script_path]
# Example: ./scripts/launch.sh scripts/slurm_single_node.sh

if [ $# -eq 0 ]; then
    echo "Usage: ./scripts/launch.sh [script_path]"
    echo "Examples:"
    echo "  ./scripts/launch.sh scripts/slurm_single_node.sh"
    echo "  ./scripts/launch.sh scripts/slurm_multi_node.sh"
    exit 1
fi

SCRIPT_PATH=$1

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script $SCRIPT_PATH not found"
    exit 1
fi

# Make script executable
chmod +x $SCRIPT_PATH

# Submit job
echo "Submitting job: $SCRIPT_PATH"
sbatch $SCRIPT_PATH

# Show job status
echo "Job submitted. Checking status..."
sleep 2
squeue -u $(whoami)