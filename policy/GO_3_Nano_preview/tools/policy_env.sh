# Internal policy-process setup; never source in eval.sh or the eval client.
go3_policy_environment() {
  export PYTHONPATH="${XPL_ROOT}/..:${POLICY_DIR}:${POLICY_DIR}/go3_nano_preview${PYTHONPATH:+:${PYTHONPATH}}"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
  export STAT_PATH="${POLICY_DIR}/assets/stats/robotwin.json"
}

go3_require_policy_python() {
  if [[ -z "$POLICY_PYTHON" || ! -x "$POLICY_PYTHON" ]]; then
    echo "Set POLICY_ENV in $POLICY_DIR/configs/runtime.conf to an environment with an executable bin/python." >&2
    return 1
  fi
}
