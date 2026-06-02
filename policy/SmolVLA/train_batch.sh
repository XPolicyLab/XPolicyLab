#!/usr/bin/env bash
# 批量 SmolVLA 训练：每个 task_name(=ckpt_name) 绑定一张 GPU，在独立 tmux 会话中跑 train.sh。
#
# 用法 1 — 环境变量（逗号分隔 task:gpu 或 task:gpu:seed）:
#   TASK_GPU_MAP="stack_bowls:0:0,push_T:1:42,build_tower:2" bash train_batch.sh
#
# 用法 2 — 命令行参数:
#   bash train_batch.sh stack_bowls:0:0 push_T:1:42 build_tower:2
#
# 用法 3 — 编辑下方 DEFAULT_TASK_GPU 数组后:
#   bash train_batch.sh
#
# 环境变量:
#   SMOVLA_CONDA_ENV        conda 环境名 (默认 smolvla；若你用 smo_vla 请 export)
#   SMOVLA_DATASET_NAME     默认 RoboDojo
#   SMOVLA_ENV_CFG_TYPE     默认 arx_x5
#   SMOVLA_EXPERT_DATA_NUM  默认 100
#   SMOVLA_ACTION_TYPE      默认 joint
#   SMOVLA_SEED             未写 seed 时的默认值 (默认 0)
#   SMOVLA_TMUX_PREFIX      tmux 会话名前缀 (默认 smolvla)
#   SMOVLA_TMUX_REPLACE     1=同名会话已存在则先 kill (默认 1)
#   SMOVLA_DRY_RUN          1=只打印命令不启动 tmux
#   SMOVLA_REPO_ID_PREFIX   LeRobot repo 前缀 (默认 RoboDojo_sim)
#   SMOVLA_REPO_ID_SUFFIX   LeRobot repo 后缀 (默认 v30)
#   SMOVLA_BASHRC           启动前 source 的 bashrc (默认 /mnt/nfs/niantian/.bashrc)
#   SMOVLA_HF_LEROBOT_HOME  LeRobot 数据根目录 (默认 /mnt/xspark-data/xspark_shared/lerobot)
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${POLICY_DIR}/train.sh"

CONDA_ENV="${SMOVLA_CONDA_ENV:-smolvla}"
DATASET_NAME="${SMOVLA_DATASET_NAME:-RoboDojo}"
ENV_CFG_TYPE="${SMOVLA_ENV_CFG_TYPE:-arx_x5}"
EXPERT_DATA_NUM="${SMOVLA_EXPERT_DATA_NUM:-100}"
ACTION_TYPE="${SMOVLA_ACTION_TYPE:-joint}"
SEED="${SMOVLA_SEED:-0}"
TMUX_PREFIX="${SMOVLA_TMUX_PREFIX:-smolvla}"
TMUX_REPLACE="${SMOVLA_TMUX_REPLACE:-1}"
DRY_RUN="${SMOVLA_DRY_RUN:-0}"

# 无 CLI / 无 TASK_GPU_MAP 时使用；按需修改
DEFAULT_TASK_GPU=(
	"arrange_largest_number:4:0"
	"arrange_largest_number:4:1"
	"arrange_largest_number:4:2"
	"build_tower:5:0"
	"build_tower:5:1"
	"build_tower:5:2"
	"classify_objects:6:0"
	"classify_objects:6:1"
	"classify_objects:6:2"
	"cover_blocks:7:0"
	"cover_blocks:7:1"
	"cover_blocks:7:2"
)

usage() {
	cat <<EOF
Usage:
  TASK_GPU_MAP="task_a:0:0,task_b:1:42" bash train_batch.sh
  bash train_batch.sh <ckpt_name>:<gpu_id>[:<seed>] ...

Each ckpt_name is passed to train.sh as the 2nd argument (task_name).
LeRobot dataset.repo_id: \${SMOVLA_REPO_ID_PREFIX}_<ckpt_name>_\${SMOVLA_REPO_ID_SUFFIX}
  e.g. build_tower -> RoboDojo_sim_build_tower_v30
tmux session name: \${SMOVLA_TMUX_PREFIX}_<ckpt_name>

Shared train args (override via env):
  dataset=${DATASET_NAME} env_cfg=${ENV_CFG_TYPE} expert_num=${EXPERT_DATA_NUM}
  action=${ACTION_TYPE} default_seed=${SEED} conda=${CONDA_ENV}
  per-task seed: ckpt_name:gpu_id:seed (seed 可省略)

Attach:  tmux attach -t ${TMUX_PREFIX}_<ckpt_name>_s<seed>
List:    tmux list-sessions | grep '^${TMUX_PREFIX}_'
EOF
}

if [[ ! -x "${TRAIN_SCRIPT}" ]]; then
	chmod +x "${TRAIN_SCRIPT}" 2>/dev/null || true
fi

if ! command -v tmux >/dev/null 2>&1; then
	echo "[SmolVLA batch] ERROR: tmux not found. Install: apt install tmux" >&2
	exit 1
fi

# shellcheck disable=SC1091
source "${POLICY_DIR}/conda_init.sh"
smolvla_source_bashrc
smolvla_resolve_conda_base || exit 1
echo "[SmolVLA batch] CONDA_BASE=${CONDA_BASE} BASHRC=${SMOVLA_BASHRC}"
echo "[SmolVLA batch] repo_id pattern: ${SMOVLA_REPO_ID_PREFIX:-RoboDojo_sim}_<task>_${SMOVLA_REPO_ID_SUFFIX:-v30}"

declare -A TASK_TO_GPU=()
declare -A TASK_TO_SEED=()
declare -a TASK_ORDER=()

normalize_pair() {
	# 允许 "task : 4 : 0" 这类空格写法
	local pair="$1"
	pair="${pair// /}"
	echo "${pair}"
}

add_task_entry() {
	local task="$1"
	local gpu="$2"
	local seed="$3"
	if [[ -z "${task}" || -z "${gpu}" ]]; then
		echo "[SmolVLA batch] ERROR: invalid entry '${task}:${gpu}:${seed}'" >&2
		exit 1
	fi
	if [[ ! "${gpu}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
		echo "[SmolVLA batch] ERROR: invalid gpu_id '${gpu}' for task '${task}'" >&2
		exit 1
	fi
	if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
		echo "[SmolVLA batch] ERROR: invalid seed '${seed}' for task '${task}'" >&2
		exit 1
	fi
	local key="${task}::${seed}"
	if [[ -n "${TASK_TO_GPU[${key}]+x}" ]]; then
		echo "[SmolVLA batch] WARN: duplicate ${task} seed=${seed}, updating gpu=${gpu}" >&2
	fi
	TASK_TO_GPU["${key}"]="${gpu}"
	TASK_TO_SEED["${key}"]="${seed}"
	if [[ " ${TASK_ORDER[*]} " != *" ${key} "* ]]; then
		TASK_ORDER+=("${key}")
	fi
}

parse_pair() {
	local pair="$1"
	local task gpu seed rest
	pair="$(normalize_pair "${pair}")"
	if [[ "${pair}" != *:* ]]; then
		echo "[SmolVLA batch] ERROR: expected ckpt_name:gpu_id[:seed], got '${pair}'" >&2
		exit 1
	fi
	task="${pair%%:*}"
	rest="${pair#*:}"
	gpu="${rest%%:*}"
	if [[ "${rest}" == *:* ]]; then
		seed="${rest#*:}"
	else
		seed="${SEED}"
	fi
	add_task_entry "${task}" "${gpu}" "${seed}"
}

if [[ $# -gt 0 ]]; then
	for pair in "$@"; do
		parse_pair "${pair}"
	done
elif [[ -n "${TASK_GPU_MAP:-}" ]]; then
	IFS=',' read -ra _pairs <<< "${TASK_GPU_MAP}"
	for pair in "${_pairs[@]}"; do
		pair="$(echo "${pair}" | xargs)"
		[[ -n "${pair}" ]] || continue
		parse_pair "${pair}"
	done
elif [[ ${#DEFAULT_TASK_GPU[@]} -gt 0 ]]; then
	for pair in "${DEFAULT_TASK_GPU[@]}"; do
		[[ "${pair}" =~ ^[[:space:]]*# ]] && continue
		pair="$(echo "${pair}" | xargs)"
		[[ -n "${pair}" ]] || continue
		parse_pair "${pair}"
	done
fi

if [[ ${#TASK_ORDER[@]} -eq 0 ]]; then
	usage >&2
	exit 1
fi

sanitize_session_name() {
	local task="$1"
	local seed="$2"
	local name="${task}"
	name="${name//\//-}"
	name="${name//./_}"
	echo "${TMUX_PREFIX}_${name}_s${seed}"
}

launch_one() {
	local task_key="$1"
	local ckpt_name="${task_key%%::*}"
	local seed="${TASK_TO_SEED[${task_key}]}"
	local gpu_id="${TASK_TO_GPU[${task_key}]}"
	local session
	session="$(sanitize_session_name "${ckpt_name}" "${seed}")"

	local run_dir="${POLICY_DIR}/.tmux_runs"
	local runner="${run_dir}/${session}.sh"
	mkdir -p "${run_dir}"

	local repo_id
	repo_id="$(smolvla_repo_id_for_task "${ckpt_name}")"

	cat >"${runner}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export SMOVLA_BASHRC="${SMOVLA_BASHRC:-/mnt/nfs/niantian/.bashrc}"
# shellcheck disable=SC1091
source "${POLICY_DIR}/conda_init.sh"
smolvla_setup_runtime "${CONDA_ENV}"
cd "${POLICY_DIR}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export HF_LEROBOT_HOME="\${HF_LEROBOT_HOME:-${SMOVLA_HF_LEROBOT_HOME:-/mnt/xspark-data/xspark_shared/lerobot}}"
export SMOVLA_REPO_ID="${repo_id}"
echo "[tmux ${session}] GPU=${gpu_id} task=${ckpt_name} seed=${seed} repo_id=${repo_id}"
echo "[tmux ${session}] HF_LEROBOT_HOME=\${HF_LEROBOT_HOME} conda=${CONDA_ENV}"
exec bash "${TRAIN_SCRIPT}" \\
	"${DATASET_NAME}" "${ckpt_name}" "${ENV_CFG_TYPE}" "${EXPERT_DATA_NUM}" \\
	"${ACTION_TYPE}" "${seed}" "${gpu_id}"
EOF
	chmod +x "${runner}"

	if [[ "${TMUX_REPLACE}" == "1" ]] && tmux has-session -t "${session}" 2>/dev/null; then
		echo "[SmolVLA batch] Replacing existing tmux session: ${session}"
		tmux kill-session -t "${session}"
	fi

	if tmux has-session -t "${session}" 2>/dev/null; then
		echo "[SmolVLA batch] SKIP ${ckpt_name}: tmux session exists (${session}), set SMOVLA_TMUX_REPLACE=1 to replace"
		return 0
	fi

	echo "[SmolVLA batch] START ${ckpt_name} -> GPU ${gpu_id} seed=${seed} repo_id=${repo_id}"
	echo "              tmux: ${session}  runner: ${runner}"

	if [[ "${DRY_RUN}" == "1" ]]; then
		return 0
	fi

	tmux new-session -d -s "${session}" -n train "bash ${runner}"
}

echo "[SmolVLA batch] dataset=${DATASET_NAME} env=${ENV_CFG_TYPE} expert=${EXPERT_DATA_NUM} action=${ACTION_TYPE} default_seed=${SEED}"
echo "[SmolVLA batch] conda=${CONDA_ENV} tasks=${#TASK_ORDER[@]}"
echo

_launched_sessions=()
for task_key in "${TASK_ORDER[@]}"; do
	launch_one "${task_key}"
	_launched_sessions+=("$(sanitize_session_name "${task_key%%::*}" "${TASK_TO_SEED[${task_key}]}")")
done

echo
echo "[SmolVLA batch] Launched ${#TASK_ORDER[@]} session(s)."
for _s in "${_launched_sessions[@]}"; do
	echo "  Attach:  tmux attach -t ${_s}"
done
echo "  List:    tmux list-sessions | grep '^${TMUX_PREFIX}_'"
