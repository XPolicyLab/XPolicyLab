# XPolicyLab deploy: policy server env=uv; run setup_eval_policy_server.sh with this env.
#!/usr/bin/env bash
# MolmoAct2 一键安装（对应 INSTALLATION.md）
#
# Usage:
#   bash install.sh          # lerobot 训练环境 + XPolicyLab（RoboDojo 默认，训练与 eval 共用）
#   bash install.sh train    # 同上
#   bash install.sh infer    # 仅上游 FastAPI server venv（非 XPolicyLab eval）
#   bash install.sh all      # lerobot + XPolicyLab + FastAPI server两套 venv
#
# 环境变量（可选）:
#   MOLMOACT2_REPO       默认 https://github.com/allenai/molmoact2.git
#   LEROBOT_REPO         默认 https://github.com/allenai/lerobot
#   LEROBOT_BRANCH       默认 molmoact2-policy
#   SKIP_XPOLICYLAB=1    跳过 XPolicyLab 安装

set -euo pipefail

MODE="${1:-train}"
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOLMOACT2_DIR="${POLICY_DIR}/molmoact2"
LEROBOT_DIR="${MOLMOACT2_DIR}/lerobot"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"

MOLMOACT2_REPO="${MOLMOACT2_REPO:-https://github.com/allenai/molmoact2.git}"
LEROBOT_REPO="${LEROBOT_REPO:-https://github.com/allenai/lerobot}"
LEROBOT_BRANCH="${LEROBOT_BRANCH:-molmoact2-policy}"

ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "错误: 未找到 uv，请先安装:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
  fi
  echo "uv: $(uv --version)"
}

init_molmoact2_source() {
  echo ""
  echo "=== 1. 初始化上游源码 ==="

  if [[ -d "${MOLMOACT2_DIR}/.git" ]]; then
    echo "molmoact2 已存在: ${MOLMOACT2_DIR}"
    git -C "${MOLMOACT2_DIR}" submodule update --init --recursive
  else
    echo "clone ${MOLMOACT2_REPO} -> ${MOLMOACT2_DIR}"
    git clone --recurse-submodules "${MOLMOACT2_REPO}" "${MOLMOACT2_DIR}"
  fi

  if [[ ! -f "${LEROBOT_DIR}/pyproject.toml" ]]; then
    if [[ -d "${LEROBOT_DIR}" ]] && [[ -n "$(ls -A "${LEROBOT_DIR}" 2>/dev/null)" ]]; then
      echo "错误: ${LEROBOT_DIR} 已存在但缺少 pyproject.toml，请删除后重试 install.sh" >&2
      exit 1
    fi
    echo "lerobot submodule 为空，手动 clone 分支 ${LEROBOT_BRANCH}"
    git clone -b "${LEROBOT_BRANCH}" "${LEROBOT_REPO}" "${LEROBOT_DIR}"
  elif [[ ! -d "${LEROBOT_DIR}/.git" ]]; then
    echo "lerobot 源码已存在（无 .git），跳过 clone: ${LEROBOT_DIR}"
  fi

  if [[ ! -f "${LEROBOT_DIR}/pyproject.toml" ]]; then
    echo "错误: LeRobot 源码不完整: ${LEROBOT_DIR}" >&2
    exit 1
  fi
}

install_infer_env() {
  echo ""
  echo "=== 2. 推理环境 (molmoact2/.venv) ==="
  cd "${MOLMOACT2_DIR}"
  UV_LINK_MODE=copy uv sync
  uv run python -c "import torch; print('cuda:', torch.cuda.is_available())"
  echo "推理环境就绪: ${MOLMOACT2_DIR}/.venv"
}

install_train_env() {
  echo ""
  echo "=== 3. 训练环境 (lerobot/.venv) ==="
  cd "${LEROBOT_DIR}"
  UV_LINK_MODE=copy uv pip install -e ".[molmoact2,training,scipy-dep]" --index-strategy unsafe-best-match
  uv run python -c "from lerobot.policies.factory import get_policy_class; print('molmoact2 policy:', get_policy_class('molmoact2'))"
  echo "训练环境就绪: ${LEROBOT_DIR}/.venv"
}

install_xpolicylab() {
  if [[ "${SKIP_XPOLICYLAB:-0}" == "1" ]]; then
    echo "跳过 XPolicyLab 安装 (SKIP_XPOLICYLAB=1)"
    return
  fi

  echo ""
  echo "=== 4. 安装 XPolicyLab 到 lerobot/.venv（训练与 eval 共用） ==="
  # shellcheck disable=SC1091
  source "${LEROBOT_DIR}/.venv/bin/activate"
  cd "${XPOLICYLAB_ROOT}"
  if ! python -m pip --version >/dev/null 2>&1; then
    python -m ensurepip --upgrade
  fi
  uv pip install -e .
  uv pip install h5py opencv-python
  python -c "import XPolicyLab; import cv2, h5py; print('XPolicyLab ok')"
}

verify_all() {
  echo ""
  echo "=== 安装完成 ==="
  echo "XPolicyLab venv: ${LEROBOT_DIR}/.venv  (train + eval)"
  echo "FastAPI venv:     ${MOLMOACT2_DIR}/.venv  (optional upstream server)"
  echo "训练入口:   bash ${POLICY_DIR}/train.sh ..."
}

main() {
  case "${MODE}" in
    train|infer|all) ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      echo "未知模式: ${MODE}（可选: train | infer | all）" >&2
      exit 1
      ;;
  esac

  ensure_uv
  init_molmoact2_source

  case "${MODE}" in
    infer)
      install_infer_env
      ;;
    train)
      install_train_env
      install_xpolicylab
      ;;
    all)
      install_infer_env
      install_train_env
      install_xpolicylab
      ;;
  esac

  verify_all
}

main "$@"
