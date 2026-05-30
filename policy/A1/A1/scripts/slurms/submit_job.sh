#!/usr/bin/bash
# ------------------------ Config ------------------------ #
job_name="pretrain"        # Job name
partition="faculty"        # Partition name
nnodes=8                      # Number of nodes required
time="72:00:00"
mem="728G"
gpus_per_node=8               # Number of GPUs required per node
cpus_per_gpu=16               # Number of CPUs required per GPU
quotatype="xdqos"             # QOS type, e.g., `normal`, `gtqos`, `stqos`, etc.
output_dir="sbatch_output"    # Output directory
# ------------------------ Setup ------------------------ #
timestamp=$(date +%Y%m%d_%H%M%S)
output_dir=${output_dir}/${timestamp}
export TIMESTAMP=${timestamp}
export OUTPUT_DIR=${output_dir}

# export NCCL_DEBUG_SUBSYS=ALL
mkdir -p ${output_dir}
# ------------------------ Submit ------------------------ #
sbatch \
    --job-name=${job_name} \
    --partition=${partition} \
    --qos=${quotatype} \
    --ntasks-per-node=1 \
    --nodes=${nnodes} \
    --mem=${mem} \
    --gres=gpu:${gpus_per_node} \
    --cpus-per-task=$((cpus_per_gpu * gpus_per_node)) \
    --time=${time} \
    --exclusive \
    --output=${output_dir}_slurm-%j.log \
    scripts/slurms/job.sh