# Shared bootstrap for train.sh / train_batch.sh (bashrc, conda, repo_id).

SMOVLA_BASHRC="${SMOVLA_BASHRC:-/mnt/nfs/niantian/.bashrc}"
# 与 .bashrc 中 HF_LEROBOT_HOME 保持一致；source 失败时作兜底
SMOVLA_HF_LEROBOT_HOME="${SMOVLA_HF_LEROBOT_HOME:-/mnt/xspark-data/xspark_shared/lerobot}"

smolvla_source_bashrc() {
	if [[ ! -f "${SMOVLA_BASHRC}" ]]; then
		echo "[SmolVLA] WARN: bashrc not found: ${SMOVLA_BASHRC}" >&2
		export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${SMOVLA_HF_LEROBOT_HOME}}"
		return 0
	fi

	# 用户 .bashrc 开头有: [ -z "$PS1" ] && return
	# 非交互 / tmux runner 下 PS1 为空会直接 return，HF_LEROBOT_HOME 等都不会生效
	set +u
	local _saved_ps1="${PS1-}"
	PS1="${PS1:-smolvla-noninteractive}"
	# shellcheck disable=SC1090
	source "${SMOVLA_BASHRC}"
	if [[ -n "${_saved_ps1}" ]]; then
		PS1="${_saved_ps1}"
	else
		unset PS1
	fi
	set -u

	export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${SMOVLA_HF_LEROBOT_HOME}}"
	echo "[SmolVLA] sourced ${SMOVLA_BASHRC} -> HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
}

smolvla_repo_id_for_task() {
	local task_name="$1"
	local prefix="${SMOVLA_REPO_ID_PREFIX:-RoboDojo_sim}"
	local suffix="${SMOVLA_REPO_ID_SUFFIX:-v30}"
	echo "${prefix}_${task_name}_${suffix}"
}

smolvla_resolve_conda_base() {
	if [[ -n "${CONDA_BASE:-}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
		return 0
	fi

	if command -v conda >/dev/null 2>&1; then
		CONDA_BASE="$(conda info --base 2>/dev/null || true)"
		if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
			return 0
		fi
	fi

	local candidate
	for candidate in \
		"${CONDA_ROOT:-}" \
		"/mnt/nfs/miniconda3" \
		"/data/miniconda3" \
		"/root/miniforge3" \
		"/root/miniconda3" \
		"${HOME}/miniforge3" \
		"${HOME}/miniconda3"; do
		[[ -n "${candidate}" ]] || continue
		if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
			CONDA_BASE="${candidate}"
			return 0
		fi
	done

	echo "[SmolVLA] ERROR: conda not found. Set CONDA_BASE or install miniconda." >&2
	echo "[SmolVLA] Tried: /mnt/nfs/miniconda3, /data/miniconda3, ~/miniforge3, ..." >&2
	return 1
}

smolvla_activate_conda() {
	local env_name="${1:-${SMOVLA_CONDA_ENV:-smolvla}}"
	smolvla_resolve_conda_base || return 1
	# shellcheck disable=SC1091
	source "${CONDA_BASE}/etc/profile.d/conda.sh"
	conda activate "${env_name}"
}

smolvla_setup_runtime() {
	local env_name="${1:-${SMOVLA_CONDA_ENV:-smolvla}}"
	smolvla_source_bashrc
	smolvla_activate_conda "${env_name}"
}
