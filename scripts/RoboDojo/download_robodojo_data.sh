#!/usr/bin/env bash
# 从 ModelScope / HuggingFace 下载 RoboDojo 数据集到 XPolicyLab/data/
#
# 用法:
#   bash scripts/RoboDojo/download_robodojo_data.sh <source> <type>
#
# 示例:
#   bash scripts/RoboDojo/download_robodojo_data.sh modelscope lerobot_v3.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"  # XPolicyLab 根目录
DATA_ROOT="${PROJECT_ROOT}/data"

SOURCE="${1:-}"      # 下载源: modelscope / huggingface
DATA_TYPE="${2:-}"   # 数据格式: lerobot_v3.0 / lerobot_v2.1 / hdf5 / hdf5_w_depth

usage() {
	cat <<'EOF'
Usage: bash scripts/RoboDojo/download_robodojo_data.sh <source> <type>

Sources:
  modelscope
  huggingface

Types:
  lerobot_v3.0
  lerobot_v2.1
  hdf5
  hdf5_w_depth

Example:
  bash scripts/RoboDojo/download_robodojo_data.sh modelscope lerobot_v3.0
EOF
}

if [[ -z "${SOURCE}" || -z "${DATA_TYPE}" ]]; then
	usage
	exit 1
fi

if ! command -v git >/dev/null 2>&1; then
	echo "git not found" >&2
	exit 1
fi

mkdir -p "${DATA_ROOT}"

# 只 clone 仓库中的指定子目录，避免下载整个数据集仓库
# repo_url:   远程 git 地址
# remote_dir: 仓库内要下载的文件夹名
# target_dir: 本地保存路径 (XPolicyLab/data/...)
clone_sparse_folder() {
	local repo_url="$1"
	local remote_dir="$2"
	local target_dir="$3"
	local tmp_dir

	if [[ -d "${target_dir}" ]]; then
		echo "==> Target already exists, skip: ${target_dir}"
		return 0
	fi

	tmp_dir="$(mktemp -d)"
	trap 'rm -rf "${tmp_dir}"' RETURN

	echo "==> Downloading ${remote_dir}"
	git clone --depth 1 --filter=blob:none --sparse "${repo_url}" "${tmp_dir}/repo"
	git -C "${tmp_dir}/repo" sparse-checkout set "${remote_dir}"

	if [[ ! -d "${tmp_dir}/repo/${remote_dir}" ]]; then
		echo "Remote folder not found: ${remote_dir}" >&2
		exit 1
	fi

	mv "${tmp_dir}/repo/${remote_dir}" "${target_dir}"
	echo "==> Saved to ${target_dir}"
}

case "${SOURCE}:${DATA_TYPE}" in
	modelscope:lerobot_v3.0)
		clone_sparse_folder \
			"https://oauth2:ms-98d73e79-a89f-4cfa-ac03-039f2d26c7b4@www.modelscope.cn/datasets/niantianshinidie/RoboDojo_release.git" \
			"RoboDojo_lerobot_v30_video" \
			"${DATA_ROOT}/RoboDojo_lerobot_v30_video"
		;;
	modelscope:lerobot_v2.1|modelscope:hdf5|modelscope:hdf5_w_depth|huggingface:*)
		echo "Not implemented yet: source=${SOURCE}, type=${DATA_TYPE}" >&2
		exit 1
		;;
	*)
		echo "Invalid source or type: source=${SOURCE}, type=${DATA_TYPE}" >&2
		usage
		exit 1
		;;
esac
