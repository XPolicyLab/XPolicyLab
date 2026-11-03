#!/usr/bin/env bash
# Resolve an explicit eval model or the highest complete checkpoint-N in a run.

wsp_has_complete_model_weights() {
    local checkpoint_dir="$1"
    if [[ -f "$checkpoint_dir/model.safetensors" ]]; then
        [[ ! -e "$checkpoint_dir/model.safetensors.index.json" ]]
        return
    fi
    [[ -f "$checkpoint_dir/model.safetensors.index.json" ]] || return 1
    "${WSP_PYTHON:-python3}" - "$checkpoint_dir" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
try:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    names = set(weight_map.values())
    valid = (
        isinstance(weight_map, dict)
        and bool(weight_map)
        and all(
            isinstance(key, str)
            and key
            and isinstance(name, str)
            and pathlib.PurePath(name).name == name
            and name.endswith(".safetensors")
            and name != "model.safetensors"
            for key, name in weight_map.items()
        )
        and all((root / name).is_file() for name in names)
    )
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

wsp_resolve_eval_checkpoint() {
    local experiment_dir="$1"
    local explicit_path="${2:-}"
    if [[ -n "$explicit_path" ]]; then
        printf '%s\n' "$explicit_path"
        return 0
    fi

    local candidate name step
    local latest_path=""
    local latest_step=-1
    shopt -s nullglob
    for candidate in "$experiment_dir"/checkpoint-*; do
        [[ -d "$candidate" ]] || continue
        [[ -f "$candidate/.complete" ]] || continue
        wsp_has_complete_model_weights "$candidate" || continue
        [[ -f "$candidate/checkpoint_manifest.json" ]] || continue
        name="${candidate##*/}"
        step="${name#checkpoint-}"
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        if (( step > latest_step )); then
            latest_step=$step
            latest_path="$candidate"
        fi
    done
    shopt -u nullglob

    if [[ -z "$latest_path" ]]; then
        echo "ERROR: No evaluation-ready checkpoint-N under $experiment_dir" >&2
        return 1
    fi
    printf '%s\n' "$latest_path"
}
