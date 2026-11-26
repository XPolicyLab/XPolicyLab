#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?}
task_name=${2:?}
ckpt_name=${3:?}
env_cfg_type=${4:?}
action_type=${5:?}
seed=${6:?}
policy_gpu_id=${7:?}
policy_conda_env=${8:?}
policy_server_port=${9:?}
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
policy_name="$(basename "${SCRIPT_DIR}")"
source "${SCRIPT_DIR}/inference_launch_common.sh"
wsp_select_bench_root
wsp_select_deploy "${bench_name}"
wsp_select_python "${policy_conda_env}" "${WSP_PYTHON:-}"

worldscape_root="${WORLDSCAPE_POLICY_ROOT:-${SCRIPT_DIR}/worldscape-policy}"
if [[ ! -d "${worldscape_root}/src/worldscape_policy" ]]; then
    echo "[SERVER][ERROR] Invalid local runtime: ${worldscape_root}" >&2
    exit 2
fi
worldscape_root="$(cd "${worldscape_root}" && pwd)"
if [[ "${action_type}" != joint ]]; then
    echo "[SERVER][ERROR] ${policy_name} requires action_type=joint." >&2
    exit 2
fi

# Model defaults belong to the selected YAML. Environment overrides are applied
# only when explicitly set, so the dense RoboDojo cadence cannot become stride 3.
overrides=(
    "port=${policy_server_port}" "host=${policy_server_host}"
    "bench_name=${bench_name}" "task_name=${task_name}" "ckpt_name=${ckpt_name}"
    "env_cfg_type=${env_cfg_type}" "action_type=${action_type}" "seed=${seed}"
    "policy_name=${policy_name}" "worldscape_root=${worldscape_root}"
)
add_override() {
    local name=$1 key=$2
    if [[ -v "${name}" ]]; then overrides+=("${key}=${!name}"); fi
}
add_override WORLDSCAPE_CHECKPOINT checkpoint_path
add_override WORLDSCAPE_DEVICE device
add_override WSP_MODE mode
add_override ROBODOJO_ACTION_HORIZON action_horizon
add_override ROBODOJO_REPLAN_STEPS replan_steps
add_override ROBODOJO_OBSERVATION_INTERVAL observation_interval
add_override ROBODOJO_MEMORY_RESET_CHUNKS memory_reset_chunks
add_override ROBODOJO_DIFFUSION_VIEW_LAYOUT diffusion_view_layout
add_override ROBODOJO_T5_PROMPT_TEMPLATE t5_prompt_template
add_override WSP_LOG_INFERENCE log_inference
add_override WORLDSCAPE_ALLOW_DUMMY_POLICY allow_dummy_policy

echo "[SERVER] policy=${policy_name} task=${task_name} config=${yaml_file}"
echo "[SERVER] runtime=${worldscape_root} checkpoint=${WORLDSCAPE_CHECKPOINT:-${ckpt_name}}"
echo "[SERVER] endpoint=${policy_server_host}:${policy_server_port} python=${WSP_SELECTED_PYTHON}"
# Prefer this checkout over any XPolicyLab copy in the benchmark workspace.
wsp_launch_or_print env \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH="${XPL_ROOT}/..:${XPL_ROOT}:${worldscape_root}/src:${worldscape_root}:${BENCH_ROOT}:${PYTHONPATH:-}" \
    "${WSP_SELECTED_PYTHON}" -P -u "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${yaml_file}" --overrides "${overrides[@]}"
