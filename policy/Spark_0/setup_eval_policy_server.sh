#!/bin/bash
set -euo pipefail

if [[ $# -lt 9 || $# -gt 10 ]]; then
    echo "Usage: bash setup_eval_policy_server.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <policy_conda_env> <policy_server_port> [policy_server_host]"
    exit 1
fi

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
# Web / caller may pin STARVLA_ROOT. Otherwise default to this adapter copy,
# then (after the checkpoint is resolved) prefer the training checkout that
# owns the weights.
_explicit_starvla_root="${STARVLA_ROOT:-}"
STARVLA_ROOT="${STARVLA_ROOT:-${SCRIPT_DIR}/source_starvla}"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

# Configuration parsing must use the requested environment too: the base
# Python on remote workers may not have PyYAML. Conda activation hooks can
# legitimately read unset variables, so relax nounset only during activation.
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"
set -u
echo "[SERVER] Python: $(command -v python) (Conda: ${CONDA_PREFIX})"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")
# Run-dir resolution: the default run dir is the 5-tuple
# <bench>-<ckpt>-<env>-<action>-<seed> under results/Checkpoints/ or checkpoints/.
# ckpt_name may also be a path (absolute, or relative containing '/', resolved
# under this policy dir), and checkpoints/<ckpt_name> is kept as a verbatim
# fallback. SPARK_0_CKPT_PATH stays the highest-priority explicit override.
run_id="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
result_run_dir="${SCRIPT_DIR}/results/Checkpoints/${ckpt_name}"
local_run_dir="${SCRIPT_DIR}/checkpoints/${ckpt_name}"

if [[ "${ckpt_name}" == /* ]]; then
    candidate_run_dirs=("${ckpt_name}")
elif [[ "${ckpt_name}" == */* ]]; then
    candidate_run_dirs=("${SCRIPT_DIR}/${ckpt_name}")
else
    # Training run dirs embed the training env_cfg_type (tianji_marvin_wuji
    # by default; older runs use the short tag "wuji"), so also glob any tag
    # for the same <bench>-<ckpt>-*-<action>-<seed> tuple. This lets eval.sh
    # take the short training CKPT_NAME (which scripts/finetune.sh defaults
    # to the model config stem) directly. The published pretrain directory
    # is checkpoints/pretrain_spark_0 and matches that short name.
    shopt -s nullglob
    tag_glob_dirs=(
        "${SCRIPT_DIR}/results/Checkpoints/${bench_name}-${ckpt_name}-"*"-${action_type}-${seed}"
        "${SCRIPT_DIR}/checkpoints/${bench_name}-${ckpt_name}-"*"-${action_type}-${seed}"
    )
    shopt -u nullglob
    candidate_run_dirs=(
        "${SCRIPT_DIR}/results/Checkpoints/${run_id}"
        "${SCRIPT_DIR}/checkpoints/${run_id}"
        ${tag_glob_dirs[@]+"${tag_glob_dirs[@]}"}
        "${result_run_dir}"
        "${local_run_dir}"
    )
fi

resolve_Spark_0_checkpoint() {
    local run_dir=$1
    local candidates=()

    if [[ ! -d "${run_dir}" ]]; then
        return 1
    fi

    shopt -s nullglob
    candidates=(
        "${run_dir}"/final_model/*.pt
        "${run_dir}"/final_model/*.safetensors
        "${run_dir}"/checkpoints/*.pt
        "${run_dir}"/checkpoints/*.safetensors
    )
    shopt -u nullglob

    if (( ${#candidates[@]} > 0 )); then
        printf '%s\n' "${candidates[@]}" | sort -V | tail -n 1
        return 0
    fi

    return 1
}

checkpoint_path="${SPARK_0_CKPT_PATH:-}"
if [[ -z "${checkpoint_path}" ]]; then
    for run_dir in "${candidate_run_dirs[@]}"; do
        if checkpoint_path=$(resolve_Spark_0_checkpoint "${run_dir}"); then
            break
        fi
        checkpoint_path=""
    done
fi
if [[ -z "${checkpoint_path}" ]]; then
    checkpoint_path="${local_run_dir}/checkpoints/<checkpoint>.pt"
fi
if [[ ! -f "${checkpoint_path}" ]]; then
    echo "[SERVER][ERROR] checkpoint file does not exist: ${checkpoint_path}" >&2
    echo "[SERVER][ERROR] set SPARK_0_CKPT_PATH=/path/to/pytorch_model.pt to override checkpoint lookup" >&2
    echo "[SERVER][ERROR] expected a .pt or .safetensors file under one of:" >&2
    echo "[SERVER][ERROR]   ${result_run_dir}/final_model/" >&2
    echo "[SERVER][ERROR]   ${result_run_dir}/checkpoints/" >&2
    echo "[SERVER][ERROR]   ${local_run_dir}/final_model/" >&2
    echo "[SERVER][ERROR]   ${local_run_dir}/checkpoints/" >&2
    exit 1
fi
checkpoint_path="$(realpath "${checkpoint_path}")"
echo "[SERVER] resolved Spark_0 checkpoint: ${checkpoint_path}"
if [[ -z "${_explicit_starvla_root}" ]]; then
    owner_cursor="$(dirname "${checkpoint_path}")"
    while [[ "${owner_cursor}" != "/" ]]; do
        if [[ -f "${owner_cursor}/eval.sh" && -d "${owner_cursor}/source_starvla" ]]; then
            STARVLA_ROOT="${owner_cursor}/source_starvla"
            echo "[SERVER] using checkpoint-owner Spark_0 checkout: ${STARVLA_ROOT}"
            break
        fi
        owner_cursor="$(dirname "${owner_cursor}")"
    done
fi

# ---------------------------------------------------------------------------
# Train/test consistency: replay the complete TRAINING config saved next to
# the checkpoint. The Spark_0 policy server
# rebuilds the training-time normalization (q99 over converted values) through
# the dataloader registry, which resolves `datasets.vla_data.data_mix` ->
# robot_type -> embodiment DataConfig. Runtime mixes (e.g. xpolicylab_runtime)
# only exist when these env vars are set, so derive them from the checkpoint:
#   - data_mix / embodiment (robot_type) from the run's full config;
#   - the embedded `embodiment_spec` is materialized into a generated
#     embodiment dir so eval uses the exact training-time schema even if
#     model_configs/embodiments/*.yaml changed since training;
#   - a saved multi-dataset `data_mixture` is forwarded as mixture JSON.
# Pre-set SPARK_0_XPOLICY_* / SPARK_0_EMBODIMENT_DIR values take precedence.
# ---------------------------------------------------------------------------
ckpt_registry=$(python - "${checkpoint_path}" "${SCRIPT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

checkpoint_path, script_dir = Path(sys.argv[1]), Path(sys.argv[2])
run_dir = checkpoint_path.parents[1]

# config.full.yaml is the authoritative complete merged config. config.yaml is
# the trainer's accessed-only snapshot and remains a fallback for legacy runs
# that do not have the full config.
configs = []
for name in ("config.full.yaml", "config.yaml"):
    path = run_dir / name
    if path.is_file():
        with path.open("r", encoding="utf-8") as fp:
            loaded = yaml.safe_load(fp)
        if isinstance(loaded, dict):
            configs.append(loaded)

OUTPUT_LINES = 8

if not configs:
    # Legacy checkpoint without a saved training config: leave everything
    # empty; the server then relies on statically registered mixtures.
    print("\n" * (OUTPUT_LINES - 1))
    raise SystemExit(0)


def first(*path_keys):
    for cfg in configs:
        value = cfg
        for key in path_keys:
            value = value.get(key) if isinstance(value, dict) else None
            if value is None:
                break
        if value not in (None, "", [], {}):
            return value
    return None


data_mix = first("datasets", "vla_data", "data_mix") or "xpolicylab_runtime"
# New XPolicyLab schema checkpoints record robot_type here, not as a
# top-level `embodiment` leaf. Falling back to the ARX 14-d builtin would
# split a 54-d SparkArena state with left_joints/gripper keys.
schema = first("datasets", "vla_data", "xpolicylab_schema") or {}
schema_robot = schema.get("robot_type") if isinstance(schema, dict) else None
# Legacy fallback for checkpoints saved before `embodiment` was recorded.
# robodojo_arx_x5_h50_q99 ships as a BUILT-IN spec in the data registry (no
# YAML file), which is exactly what those checkpoints were trained against.
embodiment = schema_robot or first("embodiment") or "robodojo_arx_x5_h50_q99"

embodiment_dir = script_dir / "model_configs" / "embodiments"
spec = first("embodiment_spec")
if isinstance(spec, dict) and spec:
    embodiment_dir = script_dir / ".generated" / "eval_embodiments" / run_dir.name
    embodiment_dir.mkdir(parents=True, exist_ok=True)
    with (embodiment_dir / f"{embodiment}.yaml").open("w", encoding="utf-8") as fp:
        yaml.safe_dump(spec, fp, sort_keys=False)

mixture = first("data_mixture") or []
mixture_json = json.dumps(mixture, separators=(",", ":")) if mixture else ""

# Training-time state/action pose formats for pose keys (datasets.vla_data.
# {state,action}_pose_format). Prefer those run-level fields, then the frozen
# embodiment_spec for older configs that only saved the latter.
pose_repr = (
    spec.get("pose_representation")
    if isinstance(spec, dict) and isinstance(spec.get("pose_representation"), dict)
    else {}
)
state_pose_format = (
    first("datasets", "vla_data", "state_pose_format")
    or pose_repr.get("state")
    or ""
)
action_pose_format = (
    first("datasets", "vla_data", "action_pose_format")
    or pose_repr.get("action")
    or ""
)
# Normalization of the converted pose actions (q99 / hold_centered /
# hold_centered_perrow). Legacy pose checkpoints have no key because q99 was
# the only/default path; make that fallback explicit so an ambient env var
# cannot silently reinterpret an old checkpoint as hold-centered.
pose_normalization = (
    first("datasets", "vla_data", "pose_normalization")
    or pose_repr.get("normalization")
    or ("q99" if action_pose_format else "")
    or ""
)

# Top-level schema knobs the embodiment YAML no longer pins (chunk length,
# per-modality normalization, observation indices, active cameras). Same
# reason as the pose formats above: the materialized spec already carries
# them, but forwarding them keeps the registry correct when
# SPARK_0_EMBODIMENT_DIR is pre-set to the repo's model_configs/embodiments/. Legacy
# checkpoints have no keys -> empty payload, and the materialized spec's
# own values apply. video_keys prefer the frozen embodiment_spec (train.sh
# already filtered the catalog there) over a raw train-config list.
spec_overrides = {}
for field, path_keys in (
    ("action_horizon", ("framework", "action_model", "action_horizon")),
    ("state_normalization", ("datasets", "vla_data", "state_normalization")),
    ("action_normalization", ("datasets", "vla_data", "action_normalization")),
    ("observation_indices", ("datasets", "vla_data", "observation_indices")),
):
    value = first(*path_keys)
    if value is not None:
        spec_overrides[field] = value
video_keys = (
    spec.get("video_keys")
    if isinstance(spec, dict)
    else None
)
if not video_keys:
    video_keys = first("datasets", "vla_data", "video_keys")
if video_keys is not None:
    spec_overrides["video_keys"] = list(video_keys)

print(data_mix)
print(embodiment)
print(embodiment_dir)
print(mixture_json)
print(state_pose_format)
print(action_pose_format)
print(pose_normalization)
print(json.dumps(spec_overrides, separators=(",", ":")) if spec_overrides else "")
PY
)
ckpt_data_mix=$(sed -n '1p' <<< "${ckpt_registry}")
ckpt_embodiment=$(sed -n '2p' <<< "${ckpt_registry}")
ckpt_embodiment_dir=$(sed -n '3p' <<< "${ckpt_registry}")
ckpt_mixture_json=$(sed -n '4p' <<< "${ckpt_registry}")
ckpt_state_pose_format=$(sed -n '5p' <<< "${ckpt_registry}")
ckpt_action_pose_format=$(sed -n '6p' <<< "${ckpt_registry}")
ckpt_pose_normalization=$(sed -n '7p' <<< "${ckpt_registry}")
ckpt_spec_overrides=$(sed -n '8p' <<< "${ckpt_registry}")

Spark_0_data_mix="${SPARK_0_XPOLICY_DATA_MIX:-${ckpt_data_mix}}"
Spark_0_robot_type="${SPARK_0_XPOLICY_ROBOT_TYPE:-${ckpt_embodiment}}"
Spark_0_embodiment_dir="${SPARK_0_EMBODIMENT_DIR:-${ckpt_embodiment_dir}}"
Spark_0_mixture_json="${SPARK_0_XPOLICY_MIXTURE_JSON:-${ckpt_mixture_json}}"
# Pose formats/normalization are training contracts, not safe eval knobs.
# Preserve the old env entry points for launches without a saved config, but
# when the checkpoint records a value an env override must be identical.
for contract in \
    "SPARK_0_XPOLICY_STATE_POSE_FORMAT:${ckpt_state_pose_format}:state_pose_format" \
    "SPARK_0_XPOLICY_ACTION_POSE_FORMAT:${ckpt_action_pose_format}:action_pose_format" \
    "SPARK_0_XPOLICY_POSE_NORMALIZATION:${ckpt_pose_normalization}:pose_normalization"
do
    IFS=: read -r env_name ckpt_value label <<< "${contract}"
    env_value="${!env_name:-}"
    if [[ -n "${env_value}" && -n "${ckpt_value}" && "${env_value}" != "${ckpt_value}" ]]; then
        echo "[SERVER][ERROR] ${env_name}=${env_value} conflicts with checkpoint ${label}=${ckpt_value}." >&2
        echo "[SERVER][ERROR] pose representation and normalization must match training; remove the override." >&2
        exit 1
    fi
done
Spark_0_state_pose_format="${SPARK_0_XPOLICY_STATE_POSE_FORMAT:-${ckpt_state_pose_format}}"
Spark_0_action_pose_format="${SPARK_0_XPOLICY_ACTION_POSE_FORMAT:-${ckpt_action_pose_format}}"
Spark_0_pose_normalization="${SPARK_0_XPOLICY_POSE_NORMALIZATION:-${ckpt_pose_normalization}}"
Spark_0_spec_overrides="${SPARK_0_XPOLICY_SPEC_OVERRIDES:-${ckpt_spec_overrides}}"
# Any non-empty dataset name registers the runtime mix entry; only the
# robot_type field is consumed for eval-time normalization.
Spark_0_dataset_name="${SPARK_0_XPOLICY_DATASET_NAME:-xpolicy_eval}"
if [[ -z "${ckpt_data_mix}" && -z "${SPARK_0_XPOLICY_DATA_MIX:-}" ]]; then
    # No training config found: do not register a runtime mixture.
    Spark_0_dataset_name="${SPARK_0_XPOLICY_DATASET_NAME:-}"
fi

Spark_0_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
Spark_0_server_host="127.0.0.1"
Spark_0_include_state="${SPARK_0_INCLUDE_STATE:-auto}"
# `auto` lets the wrapper pick the checkpoint's only stats key (self-trained
# runs store them under the embodiment tag, e.g. new_embodiment). Set
# SPARK_0_UNNORM_KEY explicitly only for multi-embodiment checkpoints.
Spark_0_unnorm_key="${SPARK_0_UNNORM_KEY:-auto}"
# execute_horizon is resolved with the model config below (same path as RTC).
# Precedence: SPARK_0_EXECUTE_HORIZON > model config execute_horizon > auto
# (model.py falls back to the training action_horizon — one infer per chunk).
extra_overrides=()
# `auto` replays the checkpoint's datasets.vla_data.obs_image_size (see
# tools.runtime_config.resolve_image_size), so the eval resize always matches the
# training resolution. Set SPARK_0_IMAGE_SIZE=[W,H] only to diverge on purpose.
Spark_0_image_size="${SPARK_0_IMAGE_SIZE:-auto}"

cleanup() {
    if [[ -n "${SPARK_0_SERVER_PID:-}" ]]; then
        echo "[SERVER] kill Spark_0 websocket server ${SPARK_0_SERVER_PID}"
        kill "${SPARK_0_SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Deliberate divergence from the checkpoint's frozen training config, for
# knobs that only exist at inference. Two entry points, SAME allowlist and
# validation, precedence env > model config rtc > checkpoint:
#   1. SPARK_0_CONFIG_OVERRIDE accepts allowlisted OmegaConf dotlist items
#      (space-separated):
#        SPARK_0_CONFIG_OVERRIDE=framework.action_model.num_inference_timesteps=10
#      (more flow-matching Euler steps shrink the shrinkage-toward-the-prior
#      term that drives wrist windup — the cheapest arm that needs no retrain).
#      Real-time chunking (RTC) knobs — inference-time only, no retrain:
#        SPARK_0_CONFIG_OVERRIDE="framework.action_model.rtc.enable=true"
#      (optionally plus rtc.inference_delay / rtc.execution_horizon /
#       rtc.prefix_attention_schedule / rtc.max_guidance_weight; rtc.mode
#       accepts only pigdm — simulated_delay needs a training-time loss this
#       repo lacks)
#   2. The model config's top-level `rtc` pointer (resolved below) selects a
#      shared config under model_configs/rtc/ whose server-side keys are
#      translated into the same --config_override arguments. RTC is
#      training-free, so its values come from model_configs/ at startup —
#      never from the checkpoint's saved training config.
# Everything else is replayed from the checkpoint; use this only when the
# divergence IS the experiment, and record it with the eval result.
config_override_args=()
applied_config_overrides=()
declare -A seen_override_keys=()

# Allowlist + value validation shared by both override entry points.
# $1=dotlist key, $2=value, $3=source label (for error messages).
validate_config_override() {
    local override_key=$1 override_value=$2 source=$3
    case "${override_key}" in
        framework.action_model.num_inference_timesteps)
            if [[ ! "${override_value}" =~ ^[1-9][0-9]*$ ]]; then
                echo "[SERVER][ERROR] framework.action_model.num_inference_timesteps must be a positive integer, got: ${override_value} (${source})" >&2
                exit 1
            fi
            ;;
        framework.action_model.rtc.enable)
            if [[ "${override_value}" != "true" && "${override_value}" != "false" ]]; then
                echo "[SERVER][ERROR] framework.action_model.rtc.enable must be true|false, got: ${override_value} (${source})" >&2
                exit 1
            fi
            ;;
        framework.action_model.rtc.inference_delay|framework.action_model.rtc.execution_horizon)
            if [[ ! "${override_value}" =~ ^[1-9][0-9]*$ ]]; then
                echo "[SERVER][ERROR] ${override_key} must be a positive integer, got: ${override_value} (${source})" >&2
                exit 1
            fi
            ;;
        framework.action_model.rtc.prefix_attention_schedule)
            if [[ ! "${override_value}" =~ ^(exp|linear|ones|zeros)$ ]]; then
                echo "[SERVER][ERROR] ${override_key} must be exp|linear|ones|zeros, got: ${override_value} (${source})" >&2
                exit 1
            fi
            ;;
        framework.action_model.rtc.max_guidance_weight)
            if [[ ! "${override_value}" =~ ^([1-9][0-9]*(\.[0-9]+)?|0\.[0-9]*[1-9][0-9]*)$ ]]; then
                echo "[SERVER][ERROR] ${override_key} must be a positive number, got: ${override_value} (${source})" >&2
                echo "[SERVER][ERROR] 0 disables the PiGDM guidance and a negative value inverts it (both silently); the server refuses to start." >&2
                exit 1
            fi
            ;;
        framework.action_model.rtc.mode)
            if [[ "${override_value}" != "pigdm" ]]; then
                echo "[SERVER][ERROR] ${override_key} must be pigdm, got: ${override_value} (${source})" >&2
                echo "[SERVER][ERROR] simulated_delay requires a checkpoint trained with the simulated-delay loss, which this repo does not implement; the server refuses to start." >&2
                exit 1
            fi
            ;;
        *)
            echo "[SERVER][ERROR] config overrides may only set inference-only keys" >&2
            echo "[SERVER][ERROR] (framework.action_model.num_inference_timesteps, framework.action_model.rtc.*); got: ${override_key} (${source})" >&2
            echo "[SERVER][ERROR] training-contract keys (action_horizon, dimensions, normalization, architecture) must be replayed from the checkpoint." >&2
            exit 1
            ;;
    esac
}

# $1=KEY=VALUE dotlist item, $2=source label. First writer wins
# (SPARK_0_CONFIG_OVERRIDE is processed before the model config's rtc
# pointer, so env > YAML).
add_config_override() {
    local override=$1 source=$2
    local override_key="${override%%=*}"
    if [[ -n "${seen_override_keys[${override_key}]:-}" ]]; then
        if [[ "${seen_override_keys[${override_key}]}" == "${source}" ]]; then
            echo "[SERVER][ERROR] ${source} sets ${override_key} twice; remove the duplicate." >&2
            exit 1
        fi
        echo "[SERVER][WARN] ${source} also sets ${override_key}; keeping the earlier value (precedence: SPARK_0_CONFIG_OVERRIDE > model config rtc)." >&2
        return 0
    fi
    seen_override_keys["${override_key}"]="${source}"
    validate_config_override "${override_key}" "${override#*=}" "${source}"
    config_override_args+=(--config_override "${override}")
    applied_config_overrides+=("${override} (${source})")
}

if [[ -n "${SPARK_0_CONFIG_OVERRIDE:-}" ]]; then
    for override in ${SPARK_0_CONFIG_OVERRIDE}; do
        if [[ "${override}" != *=* ]]; then
            echo "[SERVER][ERROR] SPARK_0_CONFIG_OVERRIDE entries must be KEY=VALUE dotlist items, got: ${override}" >&2
            exit 1
        fi
        add_config_override "${override}" "SPARK_0_CONFIG_OVERRIDE"
    done
fi

# Deploy-time knobs from the MODEL CONFIG (see the entry-points comment above).
# The model config carries top-level `rtc` (null = sync; a stem selects
# model_configs/rtc/<stem>.yaml) and `execute_horizon` (null = sync uses the
# training action_horizon; a positive int re-plans every N rows on the sync
# path; ignored while RTC owns the async cadence). Pointer resolution order:
#   1. SPARK_0_MODEL_CONFIG (a stem under model_configs/ or a path);
#   2. model_configs/<ckpt_name>.yaml (the finetune.sh default naming, where
#      ckpt_name IS the config stem);
#   3. model_configs/<the checkpoint's recorded `model_config`>.yaml — train.sh
#      writes the stem it launched from into the saved config, so a run-dir
#      style ckpt_name still lands on the LIVE model config (editing `rtc`
#      after training takes effect without a retrain);
#   4. the top-level `rtc` pointer frozen into the checkpoint's saved config
#      (legacy runs trained before the stem was recorded).
# Only the POINTER can come from the checkpoint; the VALUES always come from
# the live model_configs/rtc/ file.
# Output protocol: line 1 is the mode (off | auto | on:<rtc config path>);
# later lines prefixed "S " are server dotlist overrides, "C " are client
# overrides forwarded to the policy client.
rtc_resolution=$(python - "${SCRIPT_DIR}" "${ckpt_name}" "${checkpoint_path}" "${SPARK_0_MODEL_CONFIG:-}" <<'PY'
import sys
from pathlib import Path

import yaml

script_dir, ckpt_name, checkpoint_path, model_config_env = sys.argv[1:5]
config_dir = Path(script_dir) / "model_configs"

SERVER_KEYS = (
    "enable",
    "inference_delay",
    "execution_horizon",
    "prefix_attention_schedule",
    "max_guidance_weight",
    "mode",
)
CLIENT_KEYS = ("fire_at_remaining", "delay_window", "debug", "max_failures")
_NULLS = (None, "", "null", "None", "auto")


def _stem(text):
    text = str(text)
    return text[:-5] if text.endswith(".yaml") else text


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def _checkpoint_value(key):
    """Top-level key from the checkpoint's saved training config."""
    run_dir = Path(checkpoint_path).parents[1]
    for name in ("config.full.yaml", "config.yaml"):
        path = run_dir / name
        if path.is_file():
            value = _load_yaml(path).get(key)
            if value not in _NULLS:
                return value
    return None


def _model_config_path():
    """(path, why) of the model config whose `rtc` pointer applies."""
    if model_config_env:
        candidate = Path(model_config_env).expanduser()
        if candidate.is_file():
            return candidate, "SPARK_0_MODEL_CONFIG"
        candidate = config_dir / f"{_stem(model_config_env)}.yaml"
        if candidate.is_file():
            return candidate, "SPARK_0_MODEL_CONFIG"
        print(
            f"[SERVER][WARN] SPARK_0_MODEL_CONFIG={model_config_env} not found; "
            "falling back to ckpt_name / the checkpoint's recorded model_config.",
            file=sys.stderr,
        )
    candidate = config_dir / f"{_stem(ckpt_name)}.yaml"
    if candidate.is_file():
        return candidate, "ckpt_name"
    recorded = _checkpoint_value("model_config")
    if recorded:
        candidate = config_dir / f"{_stem(recorded)}.yaml"
        if candidate.is_file():
            return candidate, "the checkpoint's recorded model_config"
        print(
            f"[SERVER][WARN] the checkpoint was trained from model_config="
            f"{recorded}, which no longer exists under {config_dir}; falling "
            "back to its frozen rtc pointer.",
            file=sys.stderr,
        )
    return None, ""


cfg_path, cfg_why = _model_config_path()
model_cfg = _load_yaml(cfg_path) if cfg_path is not None else {}
if cfg_path is not None:
    rtc_ref = model_cfg.get("rtc")
    source = f"model config {cfg_path.name} (via {cfg_why})"
else:
    rtc_ref = _checkpoint_value("rtc")
    source = "the checkpoint's frozen rtc pointer"
print(f"[SERVER] rtc pointer source: {source}", file=sys.stderr)


def _emit_execute_horizon():
    """Sync-path cadence from the LIVE model config (null = action_horizon).

    Emitted as a client override. While RTC is on, model.py may ignore it in
    favour of the rtc config's fire_at_remaining / execution_horizon — that is
    intentional (RTC owns the async cadence).
    """
    if cfg_path is None:
        return
    value = model_cfg.get("execute_horizon")
    if value in _NULLS:
        return
    try:
        n = int(value)
    except (TypeError, ValueError):
        sys.exit(
            f"[SERVER][ERROR] {cfg_path.name} execute_horizon must be a "
            f"positive integer or null, got: {value!r}."
        )
    if n < 1:
        sys.exit(
            f"[SERVER][ERROR] {cfg_path.name} execute_horizon must be a "
            f"positive integer or null, got: {n}."
        )
    print(f"C execute_horizon={n}")


if rtc_ref in _NULLS:
    # A found model config means an explicit decision: pin the client to the
    # synchronous path regardless of how the server was started. No model
    # config at all (legacy checkpoint) stays on the old auto behaviour.
    if cfg_path is not None:
        print("off")
        print("C rtc=false")
        print("C rtc_enable=false")
    else:
        print("auto")
    _emit_execute_horizon()
    sys.exit(0)

rtc_path = Path(str(rtc_ref)).expanduser()
if not rtc_path.is_file():
    rtc_path = config_dir / "rtc" / f"{_stem(rtc_ref)}.yaml"
if not rtc_path.is_file():
    sys.exit(
        f"[SERVER][ERROR] {source} sets rtc: {rtc_ref}, but no RTC config "
        f"found at {rtc_path}."
    )
rtc_cfg = _load_yaml(rtc_path)
unknown = sorted(set(rtc_cfg) - set(SERVER_KEYS) - set(CLIENT_KEYS))
if unknown:
    sys.exit(
        f"[SERVER][ERROR] {rtc_path} has unknown keys {unknown}; "
        f"server keys: {list(SERVER_KEYS)}, client keys: {list(CLIENT_KEYS)}."
    )

enable = rtc_cfg.get("enable", True)
if isinstance(enable, str):
    enable = enable.strip().lower() in ("true", "1", "yes", "on")
enable_str = "true" if enable else "false"

print(f"on:{rtc_path}")
print(f"S framework.action_model.rtc.enable={enable_str}")
for key in SERVER_KEYS:
    if key == "enable":
        continue
    value = rtc_cfg.get(key)
    if value in _NULLS:
        continue
    print(f"S framework.action_model.rtc.{key}={value}")
print(f"C rtc={enable_str}")
print(f"C rtc_enable={enable_str}")
for key in CLIENT_KEYS:
    value = rtc_cfg.get(key)
    if value in _NULLS:
        continue
    if isinstance(value, bool):
        value = "true" if value else "false"
    print(f"C rtc_{key}={value}")
_emit_execute_horizon()
PY
)
rtc_mode="$(head -n 1 <<< "${rtc_resolution}")"
case "${rtc_mode}" in
    on:*)
        echo "[SERVER] rtc=${rtc_mode#on:} (training-free, from the model config's rtc pointer)"
        rtc_source="rtc config $(basename "${rtc_mode#on:}")"
        ;;
    off)
        echo "[SERVER] rtc=off (model config rtc: null)"
        rtc_source=""
        ;;
    *)
        rtc_source=""
        ;;
esac
while IFS= read -r line; do
    case "${line}" in
        "S "*) add_config_override "${line#S }" "${rtc_source}" ;;
        "C "*) extra_overrides+=("${line#C }") ;;
    esac
done <<< "${rtc_resolution}"

# execute_horizon: SPARK_0_EXECUTE_HORIZON > model config > auto (action_horizon).
# Appended AFTER the model-config C lines so the env wins on collision.
Spark_0_execute_horizon="${SPARK_0_EXECUTE_HORIZON:-}"
model_execute_horizon=""
for ov in ${extra_overrides[@]+"${extra_overrides[@]}"}; do
    if [[ "${ov}" == execute_horizon=* ]]; then
        model_execute_horizon="${ov#execute_horizon=}"
    fi
done
if [[ -n "${Spark_0_execute_horizon}" ]]; then
    if [[ ! "${Spark_0_execute_horizon}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[SERVER][ERROR] SPARK_0_EXECUTE_HORIZON must be a positive integer, got: ${Spark_0_execute_horizon}" >&2
        exit 1
    fi
    extra_overrides+=("execute_horizon=${Spark_0_execute_horizon}")
    execute_horizon_log="${Spark_0_execute_horizon} (SPARK_0_EXECUTE_HORIZON)"
elif [[ -n "${model_execute_horizon}" ]]; then
    execute_horizon_log="${model_execute_horizon} (model config)"
else
    execute_horizon_log="auto (training action_horizon; model config null/absent)"
fi

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}, Spark_0_port=${Spark_0_server_port}"
echo "[SERVER] Spark_0 overrides: include_state=${Spark_0_include_state}, unnorm_key=${Spark_0_unnorm_key}, execute_horizon=${execute_horizon_log}, image_size=${Spark_0_image_size}"
echo "[SERVER] training-config replay: data_mix=${Spark_0_data_mix:-<none>}, robot_type=${Spark_0_robot_type:-<none>}, embodiment_dir=${Spark_0_embodiment_dir:-<none>}, mixture=${Spark_0_mixture_json:-<none>}, state_pose_format=${Spark_0_state_pose_format:-<spec>}, action_pose_format=${Spark_0_action_pose_format:-<spec>}, pose_normalization=${Spark_0_pose_normalization:-<spec>}, spec_overrides=${Spark_0_spec_overrides:-<spec>}"
if (( ${#applied_config_overrides[@]} > 0 )); then
    echo "[SERVER][WARN] DIVERGING from the checkpoint config: ${applied_config_overrides[*]}"
fi

(
    cd "${STARVLA_ROOT}"
    PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}" \
    SPARK_0_XPOLICY_DATASET_NAME="${Spark_0_dataset_name}" \
    SPARK_0_XPOLICY_DATA_MIX="${Spark_0_data_mix}" \
    SPARK_0_XPOLICY_ROBOT_TYPE="${Spark_0_robot_type}" \
    SPARK_0_XPOLICY_MIXTURE_JSON="${Spark_0_mixture_json}" \
    SPARK_0_XPOLICY_STATE_POSE_FORMAT="${Spark_0_state_pose_format}" \
    SPARK_0_XPOLICY_ACTION_POSE_FORMAT="${Spark_0_action_pose_format}" \
    SPARK_0_XPOLICY_POSE_NORMALIZATION="${Spark_0_pose_normalization}" \
    SPARK_0_XPOLICY_SPEC_OVERRIDES="${Spark_0_spec_overrides}" \
    SPARK_0_EMBODIMENT_DIR="${Spark_0_embodiment_dir}" \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${STARVLA_ROOT}/deployment/model_server/server_policy.py" \
        --ckpt_path "${checkpoint_path}" \
        --port "${Spark_0_server_port}" \
        ${config_override_args[@]+"${config_override_args[@]}"} \
        --use_bf16
) &
SPARK_0_SERVER_PID=$!

sleep 6

PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}" \
PYTHONWARNINGS=ignore::UserWarning \
CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
python "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
        port="${policy_server_port}" \
        host="${policy_server_host}" \
        bench_name="${bench_name}" \
        task_name="${task_name}" \
        ckpt_name="${ckpt_name}" \
        checkpoint_path="${checkpoint_path}" \
        env_cfg_type="${env_cfg_type}" \
        seed="${seed}" \
        policy_name="${policy_name}" \
        action_type="${action_type}" \
        action_dim="${action_dim}" \
        include_state="${Spark_0_include_state}" \
        unnorm_key="${Spark_0_unnorm_key}" \
        ${extra_overrides[@]+"${extra_overrides[@]}"} \
        image_size="${Spark_0_image_size}" \
        starvla_root="${STARVLA_ROOT}" \
        Spark_0_server_host="${Spark_0_server_host}" \
        Spark_0_server_port="${Spark_0_server_port}"
