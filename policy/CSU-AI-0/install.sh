#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROUTER_PYTHON="${CSU_ROUTER_PYTHON:-}"
if [[ -z "${ROUTER_PYTHON}" ]]; then
    ROUTER_PYTHON="$(command -v python3 || command -v python)"
fi
[[ -x "${ROUTER_PYTHON}" ]] || { echo "missing router python: ${ROUTER_PYTHON}" >&2; exit 1; }

if [[ "${CSU_SKIP_ROUTER_INSTALL:-0}" != "1" ]]; then
    echo "[CSU-AI-0] installing XPolicyLab websocket dependencies into ${ROUTER_PYTHON}"
    "${ROUTER_PYTHON}" -m pip install -e "${XPL_ROOT}"
else
    echo "[CSU-AI-0] CSU_SKIP_ROUTER_INSTALL=1; router dependency installation skipped"
fi

PYTHONPATH="$(cd "${XPL_ROOT}/.." && pwd):${XPL_ROOT}:${PYTHONPATH:-}" \
    "${ROUTER_PYTHON}" -c 'import yaml, websockets.asyncio; from XPolicyLab.model_template import ModelTemplate'
for policy in Xiaomi_Robotics_1 Spatial_Forcing Hy_Embodied_05_VLA; do
    [[ -f "${XPL_ROOT}/policy/${policy}/setup_eval_policy_server.sh" ]] || {
        echo "missing expert policy server: ${policy}" >&2
        exit 1
    }
done

if [[ "${CSU_SKIP_EXPERT_INSTALL:-0}" != "1" ]]; then
    for policy in Xiaomi_Robotics_1 Spatial_Forcing Hy_Embodied_05_VLA; do
        echo "[CSU-AI-0] installing isolated expert environment: ${policy}"
        bash "${XPL_ROOT}/policy/${policy}/install.sh"
    done
else
    echo "[CSU-AI-0] CSU_SKIP_EXPERT_INSTALL=1; dependency installation skipped"
fi

CSU_SKIP_RUNTIME_CHECKS=1 "${ROUTER_PYTHON}" "${SCRIPT_DIR}/route.py" --task-name stack_bowls --format json >/dev/null
CSU_SKIP_RUNTIME_CHECKS=1 "${ROUTER_PYTHON}" "${SCRIPT_DIR}/route.py" --task-name cover_blocks --format json >/dev/null
CSU_SKIP_RUNTIME_CHECKS=1 "${ROUTER_PYTHON}" "${SCRIPT_DIR}/route.py" --task-name fold_clothes_random --format json >/dev/null

if [[ -d "${SCRIPT_DIR}/checkpoints/CSU-AI-0-v1" ]]; then
    "${ROUTER_PYTHON}" "${SCRIPT_DIR}/verify_checkpoints.py"
else
    echo "[CSU-AI-0] checkpoint bundle is not installed yet. Run download_checkpoints.sh."
fi
echo "[CSU-AI-0] standard Model adapter and isolated expert launchers are ready"
