#!/bin/bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
POLICY_DIR="${XPL_ROOT}/policy/${policy_name}"
FLEXPI_DIR="${POLICY_DIR}/FlexPi"
yaml_file="${POLICY_DIR}/deploy.yml"

# Checkpoint is selected by ckpt_name, not task_name: task_name is only the
# simulator task forwarded to the env client, so one checkpoint can be
# evaluated across many tasks.
#
# Directory precedence (CKPT_ROOT = POLICY_DIR/checkpoints):
#   1. FLEXPI_CKPT_SETTING explicit override (CKPT_ROOT/<setting>)
#   2. ckpt_name as an absolute path
#   3. ckpt_name as a POLICY_DIR-relative path
#   4. the 5-tuple concat run dir under checkpoints/
#   5. checkpoints/<ckpt_name> verbatim
CKPT_ROOT="${FLEXPI_CKPT_ROOT:-${POLICY_DIR}/checkpoints}"
ckpt_setting="${FLEXPI_CKPT_SETTING:-${ckpt_name}}"
run_dir_name="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
if [[ -n "${FLEXPI_CKPT_SETTING:-}" ]]; then
    ckpt_dir="${CKPT_ROOT}/${ckpt_setting}"
elif [[ "${ckpt_name}" == /* ]]; then
    ckpt_dir="${ckpt_name}"
elif [[ "${ckpt_name}" == */* ]]; then
    ckpt_dir="${POLICY_DIR}/${ckpt_name}"
elif [[ -d "${CKPT_ROOT}/${run_dir_name}" ]]; then
    ckpt_dir="${CKPT_ROOT}/${run_dir_name}"
else
    ckpt_dir="${CKPT_ROOT}/${ckpt_setting}"
fi

# flex-pi checkpoints appear in two shapes, and upstream's _find_trained_config
# walks up to 5 parents for config.yaml, so both work once the weights file is
# named correctly:
#   run dir    <run_dir>/config.yaml + <run_dir>/checkpoints/weights/step_*.pt
#   packaged   <dir>/{config.yaml, dataset_stats.json, step_*.pt}
# A symlinked step_*.pt is picked up as well, but upstream walks up from the
# link's TARGET, so it is checked below against the config.yaml upstream reads.
pick_latest_step() {
    find -L "$1" -maxdepth 1 -type f -name 'step_*.pt' 2>/dev/null | sort -V | tail -n 1
}

# The config.yaml upstream's _find_trained_config loads for a weights file: the
# nearest one among the first 5 parents of its resolved path.
trained_config_of() {
    local dir
    dir="$(dirname "$(realpath "$1")")"
    for _ in 1 2 3 4 5; do
        if [[ -f "${dir}/config.yaml" ]]; then
            realpath "${dir}/config.yaml"
            return
        fi
        dir="$(dirname "${dir}")"
    done
}

allow_dummy_policy="${FLEXPI_ALLOW_DUMMY_POLICY:-false}"
checkpoint_path="${FLEXPI_CHECKPOINT_PATH:-}"
if [[ -z "${checkpoint_path}" && "${allow_dummy_policy}" != "true" ]]; then
    weights_dir="${ckpt_dir}/checkpoints/weights"
    if [[ -d "${weights_dir}" ]]; then
        checkpoint_path=$(pick_latest_step "${weights_dir}")
    fi
    if [[ -z "${checkpoint_path}" ]]; then
        checkpoint_path=$(pick_latest_step "${ckpt_dir}")
    fi
    if [[ -z "${checkpoint_path}" ]]; then
        echo -e "\033[31m[SERVER] No step_*.pt found under ${ckpt_dir} (checked ./ and ./checkpoints/weights/).\033[0m" >&2
        echo -e "\033[31m[SERVER] Point FLEXPI_CHECKPOINT_PATH at the .pt directly, or set FLEXPI_ALLOW_DUMMY_POLICY=true to test the wiring.\033[0m" >&2
        exit 1
    fi
    # Weights symlinked in from another directory make upstream load the config
    # beside the target (another run's, or none: the task preset), not this one.
    trained_config=$(trained_config_of "${checkpoint_path}")
    if [[ -f "${ckpt_dir}/config.yaml" && "${trained_config}" != "$(realpath "${ckpt_dir}/config.yaml")" ]]; then
        echo -e "\033[31m[SERVER] ${checkpoint_path} resolves to $(realpath "${checkpoint_path}"); upstream would load ${trained_config:-no config.yaml (the task preset)} instead of ${ckpt_dir}/config.yaml.\033[0m" >&2
        echo -e "\033[31m[SERVER] Hardlink or copy the weights into ${ckpt_dir} instead of symlinking them.\033[0m" >&2
        exit 1
    fi
fi

# dataset_stats.json sits next to config.yaml in both layouts; upstream raises
# if it is missing, so resolve it here where the message can be actionable.
dataset_stats_path="${FLEXPI_DATASET_STATS_PATH:-${ckpt_dir}/dataset_stats.json}"
if [[ "${allow_dummy_policy}" != "true" && ! -f "${dataset_stats_path}" ]]; then
    echo -e "\033[31m[SERVER] dataset_stats.json not found: ${dataset_stats_path}\033[0m" >&2
    echo -e "\033[31m[SERVER] Override with FLEXPI_DATASET_STATS_PATH.\033[0m" >&2
    exit 1
fi

echo -e "\033[33m[SERVER] policy=${policy_name}, task=${task_name}, ckpt=${ckpt_name}\033[0m"
echo -e "\033[33m[SERVER] checkpoint_path: ${checkpoint_path:-<dummy-policy>}\033[0m"
if [[ -n "${checkpoint_path}" ]]; then
    trained_config=$(trained_config_of "${checkpoint_path}")
    echo -e "\033[33m[SERVER] trained config: ${trained_config:-<none: upstream falls back to the task preset>}\033[0m"
fi
echo -e "\033[33m[SERVER] dataset_stats_path: ${dataset_stats_path}\033[0m"
echo -e "\033[33m[SERVER] policy_server_host=${policy_server_host} policy_server_port=${policy_server_port}\033[0m"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")
echo -e "\033[33m[SERVER] action_dim=${action_dim}\033[0m"

# Optional per-run deploy.yml overrides, as space-separated key=value tokens
# (e.g. "infer_joint_video=False use_depth=False"). They are appended after the
# launcher's own, so parallel runs can differ without editing the shared deploy.yml.
read -r -a extra_overrides <<< "${FLEXPI_EXTRA_OVERRIDES:-}"
if (( ${#extra_overrides[@]} )); then
    echo -e "\033[33m[SERVER] extra deploy.yml overrides: ${extra_overrides[*]}\033[0m"
fi

# Scope env vars to this server process only; exporting them in the
# orchestrator would leak them into the env client.
exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${BENCH_ROOT}:${FLEXPI_DIR}:${FLEXPI_DIR}/src:${PYTHONPATH:-}" \
    DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${FLEXPI_DIR}/checkpoints}" \
    python -u "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim="${action_dim}" \
            checkpoint_path="${checkpoint_path}" \
            dataset_stats_path="${dataset_stats_path}" \
            allow_dummy_policy="${allow_dummy_policy}" \
            ${extra_overrides[@]+"${extra_overrides[@]}"}
