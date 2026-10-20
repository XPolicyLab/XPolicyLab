#!/usr/bin/env bash
# Shared shell helpers; no environment is activated merely by sourcing this file.
crwm_policy_dir() { (cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); }
crwm_xpl_dir() { (cd "$(crwm_policy_dir)/../.." && pwd); }

crwm_python() {
    local stage="${1:?inference or posttrain}" override=""
    case "$stage" in
        inference) override="${CRWM_INFERENCE_PYTHON:-}" ;;
        posttrain) override="${CRWM_TRAIN_PYTHON:-}" ;;
        *) echo "Unknown CR-WM stage: $stage" >&2; return 2 ;;
    esac
    local interpreter="${override:-$(crwm_policy_dir)/source_${stage}/.venv/bin/python}"
    if [[ ! -x "$interpreter" ]]; then
        echo "Missing $stage interpreter: $interpreter; run install.sh $stage --python /path/to/python" >&2
        return 2
    fi
    printf '%s\n' "$interpreter"
}

crwm_checkpoint_root() {
    local bench="${1:?}" ckpt="${2:?}" robot="${3:?}" action="${4:?}" seed="${5:?}"
    local must_exist="${6:-true}" policy_dir xpl_dir interpreter
    policy_dir="$(crwm_policy_dir)"
    xpl_dir="$(crwm_xpl_dir)"
    interpreter="${CRWM_PATH_PYTHON:-python3}"
    # This helper imports only the upstream stdlib checkpoint resolver, never a model.
    "$interpreter" - "$policy_dir" "$xpl_dir" "$bench" "$ckpt" "$robot" "$action" "$seed" "$must_exist" <<'PY'
import sys
from pathlib import Path
policy, xpl, bench, ckpt, robot, action, seed, must_exist = sys.argv[1:]
sys.path.insert(0, str(Path(xpl).parent))
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
cfg = dict(bench_name=bench, ckpt_name=ckpt, env_cfg_type=robot, action_type=action, seed=seed)
print(resolve_checkpoint_root(cfg, Path(policy) / "checkpoints", policy_dir=policy,
                              must_exist=must_exist.lower() == "true"))
PY
}
