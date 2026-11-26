#!/usr/bin/env bash
# AGP eval client: RoboDojo's src/eval_client/main.py with --policy_name AGP; with EVAL_ENV_TYPE=debug, XPolicyLab's
# utils/debug_env_client.py.
#
#   bash setup_eval_env_client.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
#        <env_gpu_id> <eval_env> <additional_info> <policy_server_port> [policy_server_ip]
#
# eval_env: a conda environment such as RoboDojo's `RoboDojo`; "default" = agp_runtime/scripts/activate_eval.sh with
# $AGP_HOME/.venv-sim; or a shell script that sets up the simulator runtime (sourced).
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
env_gpu_id=$7
eval_env=${8:-default}
additional_info=$9
policy_server_port=${10}
policy_server_ip=${11:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
policy_name="$(basename "${SCRIPT_DIR}")"
export AGP_HOME="${AGP_HOME:-${HOME}/.agp-robodojo}"

# shellcheck source=/dev/null
source "${XPL_ROOT}/utils/resolve_eval_env_type.sh"
eval_env_mode="$(resolve_eval_env_type)" || exit 1
if [[ "${eval_env_mode}" == "real_world" ]]; then
    echo "[CLIENT][ERROR] EVAL_ENV_TYPE=${EVAL_ENV_TYPE}: AGP supports sim and debug only" >&2
    exit 1
fi

debug_client_python() {
    local env=$1 python=""
    if [[ "${env}" == "default" || "${env}" == "uv" ]]; then
        python="${AGP_HOME}/.venv-sim/bin/python"
    elif [[ -x "${env}/bin/python" ]]; then
        python="${env}/bin/python"
    elif [[ -f "${env}" && -x "${env}" ]]; then
        python="${env}"
    elif command -v conda >/dev/null 2>&1; then
        python="$(
            set +eu
            # shellcheck source=/dev/null
            source "$(conda info --base)/etc/profile.d/conda.sh" >/dev/null 2>&1 \
                && conda activate "${env}" >/dev/null 2>&1 && command -v python
        )" || python=""
    fi
    if [[ -z "${python}" ]]; then
        python="${AGP_HOME}/.venv-sim/bin/python"
        echo "[CLIENT] eval env '${env}' is no Python, virtualenv or activatable conda environment: the debug client" \
             "uses ${python}" >&2
    fi
    echo "${python}"
}

if [[ "${eval_env_mode}" == "debug" ]]; then
    debug_python="$(debug_client_python "${eval_env}")"
    if [[ ! -x "${debug_python}" ]]; then
        echo "[CLIENT][ERROR] debug client Python not found: ${debug_python}" >&2
        exit 1
    fi
    echo "[CLIENT] EVAL_ENV_TYPE=debug: XPolicyLab debug client, policy=${policy_name} task=${task_name}" \
         "server=${policy_server_ip}:${policy_server_port} python=${debug_python}"
    export PYTHONPATH="${XPL_ROOT}:${BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    export PYTHONDONTWRITEBYTECODE=1
    exec "${debug_python}" -B "${XPL_ROOT}/utils/debug_env_client.py" \
        --bench_name "${bench_name}" \
        --task_name "${task_name}" \
        --env_cfg_type "${env_cfg_type}" \
        --policy_name "${policy_name}" \
        --protocol ws \
        --host "${policy_server_ip}" \
        --port "${policy_server_port}" \
        --eval_batch false
fi

conda_env=""
if [[ "${eval_env}" != "default" && "${eval_env}" != "uv" && "${eval_env}" != */* && ! -f "${eval_env}" ]]; then
    if command -v conda >/dev/null 2>&1 && (
        set +eu
        source "$(conda info --base)/etc/profile.d/conda.sh" >/dev/null 2>&1 && conda activate "${eval_env}" >/dev/null 2>&1
    ); then
        conda_env="${eval_env}"
    else
        echo "[CLIENT] eval env '${eval_env}' is not an activatable conda environment; using the default runtime (\$AGP_HOME/.venv-sim)"
    fi
    eval_env="default"
fi
if [[ "${eval_env}" == "default" || "${eval_env}" == "uv" ]]; then
    eval_env="${SCRIPT_DIR}/agp_runtime/scripts/activate_eval.sh"
fi
if [[ ! -f "${eval_env}" ]]; then
    echo "[CLIENT][ERROR] simulator runtime script not found: ${eval_env}" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${eval_env}"
if [[ -n "${conda_env}" ]]; then
    set +eu
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${conda_env}"
    set -eu
    echo "[CLIENT] eval env: conda environment ${conda_env}"
fi
export ROBODOJO_ROOT="${BENCH_ROOT}"
case ":${PYTHONPATH:-}:" in
    *":${BENCH_ROOT}:"*) ;;
    *) echo "[CLIENT][ERROR] ${eval_env} does not put ${BENCH_ROOT} on PYTHONPATH" >&2; exit 1 ;;
esac

if [[ -z "${ROBODOJO_RUN_ID:-}" ]]; then
    export ROBODOJO_RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_agp-${task_name}-s${seed}-$$"
fi
if [[ -n "${AGP_RESULTS_DIR:-}" ]]; then
    workdir="${AGP_RESULTS_DIR}"
    mkdir -p "${workdir}"
elif [[ -n "${AGP_SCRIPTED:-}" ]]; then
    workdir="${AGP_HOME}/runs/official/${ROBODOJO_RUN_ID}"
    mkdir -p "${workdir}"
else
    workdir="${BENCH_ROOT}"
fi
mkdir -p "${AGP_HOME}/config" "${AGP_HOME}/cache" "${AGP_HOME}/logs"
cd "${workdir}"

export CUDA_VISIBLE_DEVICES="${env_gpu_id}"
kit_args="--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
kit_args+=" --/app/userConfigPath=${AGP_HOME}/config/kit --/app/cachePath=${AGP_HOME}/cache/kit"
kit_args+=" --/app/logPath=${AGP_HOME}/logs/kit"

echo "[CLIENT] policy=${policy_name} task=${task_name} seed=${seed} server=${policy_server_ip}:${policy_server_port}"
echo "[CLIENT] ROBODOJO_RUN_ID=${ROBODOJO_RUN_ID} eval_result=${workdir}/eval_result python=$(command -v python)"

max_retries="${ROBODOJO_MAX_BASH_RETRIES:-10}"
attempt=0
while : ; do
    set +e
    python -u "${BENCH_ROOT}/src/eval_client/main.py" \
        --task_name "${task_name}" \
        --env_cfg_type "${env_cfg_type}" \
        --num_envs 1 \
        --enable_cameras \
        --headless \
        --kit_args "${kit_args}" \
        --device_id "${env_gpu_id}" \
        --policy_name "${policy_name}" \
        --port "${policy_server_port}" \
        --host "${policy_server_ip}" \
        --protocol ws \
        --additional_info "${additional_info}" \
        --seed "${seed}"
    rc=$?
    set -e
    case "${rc}" in
        0)
            exit 0
            ;;
        99|134|139)
            attempt=$((attempt + 1))
            if [[ "${attempt}" -ge "${max_retries}" ]]; then
                echo "[CLIENT] giving up after ${attempt} restart(s) (rc=${rc}, run_id=${ROBODOJO_RUN_ID})" >&2
                exit "${rc}"
            fi
            echo "[CLIENT] main.py exited rc=${rc}, restarting (${attempt}/${max_retries})" >&2
            sleep 5
            ;;
        *)
            echo "[CLIENT] main.py exited rc=${rc}" >&2
            exit "${rc}"
            ;;
    esac
done
