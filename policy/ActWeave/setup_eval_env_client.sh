#!/bin/bash
set -euo pipefail

bench_name=${1:?}
task_name=${2:?}
ckpt_name=${3:?}
env_cfg_type=${4:?}
action_type=${5:?}
seed=${6:?}
env_gpu_id=${7:?}
eval_env=${8:?}
additional_info=${9:?}
server_port=${10:?}
server_ip=${11:-localhost}
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
xpl_root="$(cd "${script_dir}/../.." && pwd -P)"
bench_root="$(cd "${xpl_root}/.." && pwd)"

exec bash "${xpl_root}/utils/setup_env_client.sh" \
    "${xpl_root}/utils" "${script_dir}/deploy.yml" "${eval_env}" "${server_port}" \
    "${bench_name}" "${task_name}" "${env_cfg_type}" ActWeave "${additional_info}" \
    "${bench_root}" "${seed}" "${env_gpu_id}" "${server_ip}"
