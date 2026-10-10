#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -ne 11 ]]; then
    echo "Usage: bash run_hf_robodojo_env_client.sh <eval_env> <port> <bench> <task> <env_cfg> <policy> <additional_info> <bench_root> <seed> <gpu> <host>" >&2
    exit 2
fi

eval_env_ref=$1
policy_server_port=$2
bench_name=$3
task_name=$4
env_cfg_type=$5
policy_name=$6
additional_info=$7
bench_root=$8
seed=$9
env_gpu_id=${10}
policy_server_host=${11}

if [[ "${bench_name}" != "RoboDojo" ]]; then
    echo "[starVLA HF][CLIENT][ERROR] direct client only supports bench=RoboDojo, got ${bench_name}" >&2
    exit 2
fi

if [[ -x "${eval_env_ref}/bin/python" ]]; then
    eval_env_dir="$(cd "${eval_env_ref}" && pwd -L)"
    export PATH="${eval_env_dir}/bin:${PATH}"
    if [[ -d "${eval_env_dir}/conda-meta" ]]; then
        export CONDA_PREFIX="${eval_env_dir}"
    elif [[ -f "${eval_env_dir}/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "${eval_env_dir}/bin/activate"
    fi
    eval_python="${eval_env_dir}/bin/python"
    echo "[starVLA HF][CLIENT] simulator environment prefix: ${eval_env_dir}"
else
    if ! command -v conda >/dev/null 2>&1; then
        echo "[starVLA HF][CLIENT][ERROR] Conda is unavailable and eval_env is not a prefix path: ${eval_env_ref}" >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${eval_env_ref}"
    eval_python="$(command -v python)"
    echo "[starVLA HF][CLIENT] simulator Conda environment: ${eval_env_ref} (${eval_python})"
fi

if [[ "${STARVLA_ISAAC_RUNTIME_PREFLIGHTED:-0}" != "1" ]]; then
    "${eval_python}" "${SCRIPT_DIR}/check_isaac_runtime.py"
fi

main_py="${bench_root}/src/eval_client/main.py"
main_runner="${SCRIPT_DIR}/run_robodojo_main.py"
eval_cfg="${bench_root}/env_cfg/${env_cfg_type}.yml"
if [[ ! -f "${main_py}" || ! -f "${main_runner}" || ! -f "${eval_cfg}" ]]; then
    echo "[starVLA HF][CLIENT][ERROR] latest RoboDojo client/config not found under ${bench_root}" >&2
    exit 1
fi

read -r sim_cfg_name num_envs < <(
    "${eval_python}" -c '
import sys
import yaml

eval_cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
sim_name = eval_cfg["config"]["sim"]
sim_cfg = yaml.safe_load(open(sys.argv[2] % sim_name, encoding="utf-8"))
print(sim_name, int(sim_cfg.get("scene", {}).get("num_envs", 1)))
' "${eval_cfg}" "${bench_root}/env_cfg/sim/%s.yml"
)

# Keep one Isaac environment as the portable default instead of inheriting the
# generic sim YAML's parallel value. The archived PI-v3 reference used five;
# eval_hf_robodojo.sh documents the explicit environment override for that run.
runtime_num_envs="${STARVLA_ROBODOJO_NUM_ENVS:-1}"
if ! [[ "${runtime_num_envs}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[starVLA HF][CLIENT][ERROR] STARVLA_ROBODOJO_NUM_ENVS must be a positive integer: ${runtime_num_envs}" >&2
    exit 2
fi

kit_args="${STARVLA_ROBODOJO_KIT_ARGS:---/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1 --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera}"
policy_server_url="ws://${policy_server_host}:${policy_server_port}"
export PYTHONPATH="${bench_root}:${bench_root}/XPolicyLab:${PYTHONPATH:-}"
export ROBODOJO_RUN_ID="${ROBODOJO_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"

# AppLauncher receives the physical device explicitly. Keeping
# CUDA_VISIBLE_DEVICES set makes Vulkan and CUDA enumerate different ordinal
# spaces and can crash on a busy multi-GPU host.
unset CUDA_VISIBLE_DEVICES || true

echo "[starVLA HF][CLIENT] sim_cfg=${sim_cfg_name}, configured_num_envs=${num_envs}, runtime_num_envs=${runtime_num_envs}, device=cuda:${env_gpu_id}"
echo "[starVLA HF][CLIENT] Kit args: ${kit_args}"
echo "[starVLA HF][CLIENT] ROBODOJO_RUN_ID=${ROBODOJO_RUN_ID}, EVAL_NUM=${EVAL_NUM:-native}"

client_args=(
    -u "${main_runner}" "${main_py}"
    --task_name "${task_name}"
    --env_cfg_type "${env_cfg_type}"
    --num_envs "${runtime_num_envs}"
    --enable_cameras
    --kit_args "${kit_args}"
    --device "cuda:${env_gpu_id}"
    --device_id "${env_gpu_id}"
    --policy_name "${policy_name}"
    --port "${policy_server_port}"
    --protocol ws
    --policy_server_url "${policy_server_url}"
    --additional_info "${additional_info}"
    --seed "${seed}"
    --host "${policy_server_host}"
    --headless
)
if [[ "${STARVLA_ROBODOJO_CLIENT_DRY_RUN:-0}" == "1" ]]; then
    printf '[starVLA HF][CLIENT] dry-run: %q' "${eval_python}"
    printf ' %q' "${client_args[@]}"
    printf '\n'
    exit 0
fi

client_pid=""
cleanup() {
    if [[ -n "${client_pid}" ]]; then
        kill "${client_pid}" 2>/dev/null || true
        wait "${client_pid}" 2>/dev/null || true
        client_pid=""
    fi
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

cd "${bench_root}"
max_retries="${ROBODOJO_MAX_BASH_RETRIES:-10}"
attempt=0
while :; do
    set +e
    "${eval_python}" "${client_args[@]}" &
    client_pid=$!
    wait "${client_pid}"
    rc=$?
    client_pid=""
    set -e

    case "${rc}" in
        0)
            exit 0
            ;;
        99|134|139)
            attempt=$((attempt + 1))
            if [[ "${attempt}" -ge "${max_retries}" ]]; then
                echo "[starVLA HF][CLIENT][ERROR] giving up after ${attempt} restart(s), rc=${rc}" >&2
                exit "${rc}"
            fi
            echo "[starVLA HF][CLIENT] simulator exited rc=${rc}; restarting ${attempt}/${max_retries}" >&2
            sleep 5
            ;;
        *)
            exit "${rc}"
            ;;
    esac
done
