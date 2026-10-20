#!/bin/bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
    echo "Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> <train_config> [extra_args...]"
    echo "Example: bash train.sh SparkArena my_run tianji_marvin_wuji ee 0 0,1,2,3 sft_spark_0"
    echo "train_config is a YAML stem under model_configs/ or a path (also settable via SPARK_0_TRAIN_CONFIG)."
    echo "The training YAML's 'embodiment' key selects the schema under model_configs/embodiments/."
    exit 1
fi

bench_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
action_type=${4}
seed=${5}
gpu_id=${6}
shift 6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="${SCRIPT_DIR}/source_starvla"

# 7th positional: model config name (a YAML stem under model_configs/) or a
# path to a YAML file. There is no default recipe — SPARK_0_TRAIN_CONFIG is
# the only way to omit it.
train_config="${SPARK_0_TRAIN_CONFIG:-}"
if [[ $# -ge 1 && ${1} != -* ]]; then
    train_config=${1}
    shift
fi

list_model_configs() {
    for f in "${SCRIPT_DIR}/model_configs/"*.yaml; do
        [[ -e "${f}" ]] || continue
        echo "[Spark_0][ERROR]   $(basename "${f%.yaml}")" >&2
    done
}

if [[ -z "${train_config}" ]]; then
    echo "[Spark_0][ERROR] train_config is required (7th positional or SPARK_0_TRAIN_CONFIG)." >&2
    echo "[Spark_0][ERROR] available configs:" >&2
    list_model_configs
    exit 1
fi
if [[ -f "${train_config}" ]]; then
    base_config_yaml="$(realpath "${train_config}")"
else
    base_config_yaml="${SCRIPT_DIR}/model_configs/${train_config%.yaml}.yaml"
fi
if [[ ! -f "${base_config_yaml}" ]]; then
    echo "[Spark_0][ERROR] model config not found: ${base_config_yaml}" >&2
    echo "[Spark_0][ERROR] available configs:" >&2
    list_model_configs
    exit 1
fi
config_stem="$(basename "${base_config_yaml%.yaml}")"

data_dir_name="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
run_id="${data_dir_name}-${seed}"

# Multi-machine launch: set these on every node, with a unique SPARK_0_MACHINE_RANK per node.
# SPARK_0_ACCELERATE_CONFIG can point to e.g. source_starvla/starVLA/config/deepseeds/zero2.yaml
# to enable DeepSpeed; CLI flags below still override values inside the config file.
# Gradient accumulation comes from the training YAML
# (trainer.gradient_accumulation_steps); see the resolution block further down.
num_machines="${SPARK_0_NUM_MACHINES:-1}"
machine_rank="${SPARK_0_MACHINE_RANK:-0}"
main_process_ip="${SPARK_0_MAIN_PROCESS_IP:-127.0.0.1}"
main_process_port="${SPARK_0_MAIN_PROCESS_PORT:-29500}"
if [[ "${num_machines}" -gt 1 ]]; then
    case "${main_process_ip}" in
        127.0.0.1|localhost|::1)
            echo "[Spark_0][ERROR] SPARK_0_NUM_MACHINES=${num_machines} but SPARK_0_MAIN_PROCESS_IP is ${main_process_ip}." >&2
            echo "                 Set SPARK_0_MAIN_PROCESS_IP to the rank-0 node's reachable address;" >&2
            echo "                 loopback is not a multi-node rendezvous." >&2
            exit 1
            ;;
    esac
fi
accelerate_config="${SPARK_0_ACCELERATE_CONFIG:-}"
if [[ -n "${accelerate_config}" ]]; then
    # Resolve before `cd "${STARVLA_ROOT}"` below. Accept paths relative to
    # either the caller's cwd (e.g. source_starvla/starVLA/...) or the Spark_0
    # source root (e.g. starVLA/config/...).
    if [[ -f "${accelerate_config}" ]]; then
        accelerate_config="$(realpath "${accelerate_config}")"
    elif [[ -f "${STARVLA_ROOT}/${accelerate_config}" ]]; then
        accelerate_config="$(realpath "${STARVLA_ROOT}/${accelerate_config}")"
    else
        echo "[Spark_0][ERROR] accelerate config not found: ${accelerate_config}" >&2
        exit 1
    fi
    # `accelerate launch --config_file` wants an ACCELERATE config (it requires
    # a top-level `distributed_type`), not a raw DeepSpeed JSON. Both live in
    # starVLA/config/deepseeds/, and pointing at the JSON fails deep inside the
    # launcher with "A `distributed_type` must be specified in the config
    # file" — name the wrapper instead.
    if ! python - "${accelerate_config}" <<'PY'
import sys

import yaml

with open(sys.argv[1]) as handle:
    cfg = yaml.safe_load(handle)
if not isinstance(cfg, dict) or "distributed_type" not in cfg:
    sys.exit(1)
PY
    then
        echo "[Spark_0][ERROR] ${accelerate_config} is not an accelerate config" >&2
        echo "                 (no top-level 'distributed_type'). It looks like a raw" >&2
        echo "                 DeepSpeed JSON; accelerate can only consume those via a" >&2
        echo "                 wrapper's deepspeed_config.deepspeed_config_file." >&2
        echo "                 ZeRO-2: starVLA/config/deepseeds/zero2.yaml" >&2
        echo "                 ZeRO-3: starVLA/config/deepseeds/accelerate_zero3.yaml" >&2
        exit 1
    fi
fi

gpus_per_node=$(awk -F',' '{print NF}' <<< "${gpu_id}")
# world size is derived from THIS node's GPU list, so every node must launch
# with the same count. A node that passes a shorter SPARK_0_GPU_ID computes a
# smaller num_processes, the ranks never agree on the world size, and the job
# hangs in the rendezvous with no error. Let the operator pin it explicitly.
if [[ -n "${SPARK_0_GPUS_PER_NODE:-}" ]]; then
    if [[ "${SPARK_0_GPUS_PER_NODE}" != "${gpus_per_node}" ]]; then
        echo "[Spark_0][ERROR] SPARK_0_GPUS_PER_NODE=${SPARK_0_GPUS_PER_NODE} but this node's" >&2
        echo "                 SPARK_0_GPU_ID='${gpu_id}' has ${gpus_per_node} GPU(s)." >&2
        echo "                 Every node must contribute the same number of processes." >&2
        exit 1
    fi
elif [[ "${num_machines}" -gt 1 ]]; then
    echo "[Spark_0][WARN] multi-node launch with ${gpus_per_node} GPU(s) from SPARK_0_GPU_ID='${gpu_id}';" >&2
    echo "                world size assumes every node matches. Set SPARK_0_GPUS_PER_NODE to assert it." >&2
fi
num_processes=$((gpus_per_node * num_machines))
data_root_dir="${SPARK_0_DATA_ROOT:-${SCRIPT_DIR}/data}"
ckpt_root_dir="${SPARK_0_CKPT_ROOT:-${SCRIPT_DIR}/checkpoints}"
data_mix="${SPARK_0_DATA_MIX:-xpolicylab_runtime}"
dataset_name="${SPARK_0_XPOLICY_DATASET_NAME:-${data_dir_name}}"
config_yaml="${SCRIPT_DIR}/.generated/${config_stem}_${run_id}.yaml"
dataset_path="${data_root_dir}/${dataset_name}"

# A training config may declare its own multi-dataset co-training mixture
# (top-level `data_mixture` list); it then replaces the single positional
# dataset, so the single-dataset existence check is skipped.
has_data_mixture=$(python -c "import sys, yaml; cfg = yaml.safe_load(open(sys.argv[1])); print(1 if cfg.get('data_mixture') else 0)" "${base_config_yaml}")

if [[ ! -f "${dataset_path}/meta/modality.json" && -f "${dataset_path}/${env_cfg_type}/meta/modality.json" ]]; then
    dataset_name="${dataset_name}/${env_cfg_type}"
    dataset_path="${data_root_dir}/${dataset_name}"
fi

if [[ "${has_data_mixture}" != "1" && ! -f "${dataset_path}/meta/modality.json" ]]; then
    echo "[Spark_0][ERROR] LeRobot dataset not found or incomplete: ${dataset_path}" >&2
    echo "[Spark_0][ERROR] expected ${dataset_path}/meta/modality.json" >&2
    echo "[Spark_0][ERROR] Run process_data.sh first, or set SPARK_0_DATA_ROOT/SPARK_0_XPOLICY_DATASET_NAME." >&2
    exit 1
fi

embodiment_dir="${SPARK_0_EMBODIMENT_DIR:-${SCRIPT_DIR}/model_configs/embodiments}"

mkdir -p "$(dirname "${config_yaml}")"
gen_output=$(python - "${base_config_yaml}" "${config_yaml}" "${data_root_dir}" "${data_mix}" "${run_id}" "${seed}" "${embodiment_dir}" "${ckpt_root_dir}" "${SCRIPT_DIR}" <<'PY'
import json
import math
import os
import sys
import yaml

src, dst, data_root_dir, data_mix, run_id, seed, embodiment_dir, ckpt_root_dir, adapter_dir = sys.argv[1:10]
sys.path.insert(0, adapter_dir)
from tools import runtime_config  # noqa: E402

with open(src, "r", encoding="utf-8") as fp:
    cfg = yaml.safe_load(fp)

# Recipe YAML must not own run_id: the filename stem is the recipe identity
# (`model_config`), and the launch 5-tuple is the checkpoint / wandb name.
# A leftover YAML key is discarded so a copied template cannot silently
# rename the run.
cfg.pop("run_id", None)
cfg["model_config"] = os.path.splitext(os.path.basename(src))[0]
cfg["run_id"] = run_id
cfg["seed"] = int(seed)
# Deployment-time, training-free knobs (currently the top-level `rtc`
# pointer) are read from the LIVE model_configs/<stem>.yaml at eval, so
# the checkpoint only has to name it — see the RTC resolution order in
# setup_eval_policy_server.sh.
# Resolve output/data roots to absolute paths so the run does not depend on
# the training cwd (defaults: <adapter>/checkpoints and <adapter>/data).
cfg["run_root_dir"] = ckpt_root_dir
cfg.setdefault("datasets", {}).setdefault("vla_data", {})["data_root_dir"] = data_root_dir
cfg["datasets"]["vla_data"]["data_mix"] = data_mix

# Required: EE frame flag (co-train shared; no per-mixture-entry override).
vla_data = cfg["datasets"]["vla_data"]
if "ee_in_camera_frame" not in vla_data:
    raise ValueError(
        "datasets.vla_data.ee_in_camera_frame is required "
        "(true = chunk-t0 cam_high frame; false = world frame)."
    )
vla_data["ee_in_camera_frame"] = bool(vla_data["ee_in_camera_frame"])
# Everything a run decides about how the data is encoded is shared across the
# whole co-train mixture, so a per-entry value would silently apply to only
# part of the batch.
MIXTURE_FORBIDDEN_KEYS = (
    "ee_in_camera_frame",
    "state_pose_format",
    "action_pose_format",
    "pose_normalization",
    "state_normalization",
    "action_normalization",
    "observation_indices",
    "video_keys",
    "obs_image_size",
)
for entry in cfg.get("data_mixture") or []:
    if not isinstance(entry, dict):
        continue
    for key in MIXTURE_FORBIDDEN_KEYS:
        if key in entry:
            raise ValueError(
                f"data_mixture entries must not set {key}; "
                f"use datasets.vla_data.{key} (shared across the mixture)."
            )

# Embodiment schema selection: the training YAML's `embodiment` key picks a
# spec under embodiments/ (SPARK_0_XPOLICY_ROBOT_TYPE overrides it). The
# resolved name plus the full spec are recorded in the generated config, which
# the trainer saves next to the checkpoint (config.yaml/config.full.yaml), so
# every run is self-describing. action_dim/state_dim are synced FROM the spec
# (the robot's key layout); the run-level knobs below are synced INTO it.
embodiment = os.environ.get("SPARK_0_XPOLICY_ROBOT_TYPE") or cfg.get("embodiment")
if not embodiment:
    raise ValueError(
        "top-level `embodiment` is required: it names the schema under "
        f"{embodiment_dir} (or set SPARK_0_XPOLICY_ROBOT_TYPE). Guessing a "
        "layout here would silently train on the wrong key order."
    )
cfg["embodiment"] = embodiment
spec_path = os.path.join(embodiment_dir, f"{embodiment}.yaml")

# RTC is a TRAINING-FREE, deployment-time feature: the model config's
# top-level `rtc` key only POINTS at a shared config under
# model_configs/rtc/ (resolved by setup_eval_policy_server.sh at eval, never
# baked into the framework config). Eval re-reads the pointer from the LIVE
# model config named by cfg["model_config"] above, so flipping `rtc` after
# training takes effect without a retrain. Validate it at train launch anyway
# so a typo fails here instead of at eval.
rtc_ref = cfg.get("rtc")
if rtc_ref not in (None, "", "null", "None"):
    rtc_path = str(rtc_ref)
    if not os.path.isfile(rtc_path):
        rtc_path = os.path.join(
            adapter_dir, "model_configs", "rtc", f"{rtc_ref}.yaml"
        )
    if not os.path.isfile(rtc_path):
        raise ValueError(
            f"rtc: {rtc_ref!r} does not resolve to an RTC config "
            f"({rtc_path}); use null or a stem under model_configs/rtc/."
        )

# Tactile-fusion encoder is TRAINING-TIME: top-level `tactile_fusion: <stem>`
# loads model_configs/tactile_fusion/<stem>.yaml and deep-merges into
# framework.tactile (recipe-inline keys win), then turns the path on
# (encoder + DiT fusion) unless the recipe already set enable. This stem
# is the Action-DiT fusion path only; contact-region visual alignment uses
# `tactile_align:`. Unlike RTC, the resolved block is baked into the
# generated config / checkpoint so deploy replays the same encoder.
# null / absent leaves framework.tactile alone (QwenPI_v3 defaults off).
def _deep_merge(base, override):
    """Dict deep-merge; `override` wins on conflicts. Non-dicts replace."""
    if not isinstance(base, dict):
        return override
    out = dict(base)
    if not isinstance(override, dict):
        return override
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out

# Legacy alias: accept top-level `tactile:` but prefer `tactile_fusion:`.
tactile_ref = runtime_config.tactile_stem_ref(
    cfg.get("tactile_fusion"), key="tactile_fusion"
)
if tactile_ref is None and cfg.get("tactile") not in (None, "", "null", "None"):
    tactile_ref = runtime_config.tactile_stem_ref(
        cfg.get("tactile"), key="tactile_fusion"
    )
    print(
        "[Spark_0][WARN] top-level `tactile:` is deprecated; rename to "
        "`tactile_fusion:` (model_configs/tactile_fusion/).",
        file=sys.stderr,
    )
if tactile_ref is not None:
    tactile_path = str(tactile_ref)
    if not os.path.isfile(tactile_path):
        tactile_path = os.path.join(
            adapter_dir, "model_configs", "tactile_fusion", f"{tactile_ref}.yaml"
        )
    if not os.path.isfile(tactile_path):
        raise ValueError(
            f"tactile_fusion: {tactile_ref!r} does not resolve to a config "
            f"({tactile_path}); use null or a stem under "
            "model_configs/tactile_fusion/."
        )
    with open(tactile_path, "r", encoding="utf-8") as fp:
        tactile_cfg = yaml.safe_load(fp) or {}
    if not isinstance(tactile_cfg, dict):
        raise ValueError(f"tactile_fusion config {tactile_path} must be a mapping")
    fw = cfg.setdefault("framework", {})
    fw["tactile"] = _deep_merge(tactile_cfg, fw.get("tactile") or {})

# Contact-region → Qwen visual-token alignment (train-only, §2.5). Same
# stem-merge pattern as tactile_fusion; orthogonal to DA3 spatial_alignment.
# A selected stem turns the projector on; null / absent leaves it off.
tactile_align_ref = runtime_config.tactile_stem_ref(
    cfg.get("tactile_align"), key="tactile_align"
)
if tactile_align_ref is not None:
    ta_path = str(tactile_align_ref)
    if not os.path.isfile(ta_path):
        ta_path = os.path.join(
            adapter_dir, "model_configs", "tactile_align", f"{tactile_align_ref}.yaml"
        )
    if not os.path.isfile(ta_path):
        raise ValueError(
            f"tactile_align: {tactile_align_ref!r} does not resolve to a config "
            f"({ta_path}); use null or a stem under model_configs/tactile_align/."
        )
    with open(ta_path, "r", encoding="utf-8") as fp:
        ta_cfg = yaml.safe_load(fp) or {}
    if not isinstance(ta_cfg, dict):
        raise ValueError(f"tactile_align config {ta_path} must be a mapping")
    fw = cfg.setdefault("framework", {})
    fw["tactile_align"] = _deep_merge(ta_cfg, fw.get("tactile_align") or {})

runtime_config.apply_tactile_stem_enables(
    cfg.setdefault("framework", {}),
    fusion_selected=tactile_ref is not None,
    align_selected=tactile_align_ref is not None,
)
# Pressure log1p / scale live on the fusion stem; copy into vla_data
# so the dataloader and the frozen checkpoint see the same knobs.
# Recipe-inline tactile_pressure_* still win.
runtime_config.apply_tactile_stem_pressure(
    vla_data, (cfg.get("framework") or {}).get("tactile") or {}
)

# Default-off contract: packing is opt-in in the dataloader (unset = false).
# Write the flags so the frozen checkpoint yaml matches what training packs.
# Explicit pack_tactile / tactile_pack_heatmap in the recipe win.
fw = cfg.get("framework") or {}
tac_on = bool((fw.get("tactile") or {}).get("enable"))
align_on = bool((fw.get("tactile_align") or {}).get("enable"))
if vla_data.get("pack_tactile") in (None, "", "null", "None"):
    vla_data["pack_tactile"] = bool(tac_on or align_on)
if vla_data.get("tactile_pack_heatmap") in (None, "", "null", "None"):
    vla_data["tactile_pack_heatmap"] = bool(align_on)
# Packed windows must equal the encoder history (deploy rebuilds from
# streams.*.history_len). Unset tactile_history → copy every enabled
# encoder stream; a bare int applies to region only. Δp adds a region
# channel so channel_dim must be 2 in the frozen yaml.
tac = fw.get("tactile") or {}
streams = tac.get("streams") or {}
region = streams.get("region_pressure") or {}
taxel = streams.get("taxel_pressure_3d") or {}


def _role_on(stream, default_role):
    role = str((stream or {}).get("role") or default_role).lower()
    flag = tac.get("fast_enable" if role == "fast" else "slow_enable", True)
    if flag in (None, "", "auto", "null", "None"):
        flag = True
    if isinstance(flag, str) and flag.strip().lower() in ("false", "0", "no", "off"):
        flag = False
    return bool(flag)


th = vla_data.get("tactile_history")
if th in (None, "", "null", "None"):
    th = None
mapping = dict(th) if isinstance(th, dict) else {}
if tac_on and region.get("enable") and _role_on(region, "fast"):
    # Default matches TactileEncoder (region history_len defaults to 5);
    # datasets.py::_sync_tactile_history_with_encoder uses the same default.
    enc_hist = int(region.get("history_len") or 5)
    got = th.get("region_pressure") if isinstance(th, dict) else th
    if got is not None and int(got) != enc_hist:
        raise ValueError(
            "datasets.vla_data.tactile_history "
            f"({got}) != framework.tactile.streams.region_pressure.history_len "
            f"({enc_hist})"
        )
    mapping["region_pressure"] = enc_hist
    delta = vla_data.get("tactile_append_delta")
    delta_on = (
        bool(delta.get("region_pressure")) if isinstance(delta, dict) else bool(delta)
    )
    if delta_on:
        region["channel_dim"] = 2
if tac_on and taxel.get("enable") and _role_on(taxel, "slow"):
    enc_hist = int(taxel.get("history_len") or 1)
    got = th.get("taxel_pressure_3d") if isinstance(th, dict) else None
    if got is not None and int(got) != enc_hist:
        raise ValueError(
            "datasets.vla_data.tactile_history.taxel_pressure_3d "
            f"({got}) != framework.tactile.streams."
            f"taxel_pressure_3d.history_len ({enc_hist})"
        )
    mapping["taxel_pressure_3d"] = enc_hist
if mapping:
    vla_data["tactile_history"] = mapping

# Variant A future-contact: slice the first H frames of the exported future
# region window. Unset tactile_future_horizon follows the stem, then the
# head's own default (glove frames, not policy steps — do NOT tie this to
# action_horizon). Fusion-only runs leave the key unset so the loader never
# decodes the future column.
fc = (tac.get("tactile_expert") or {}).get("future_contact") or {}
if tac_on and bool(fc.get("enable")):
    hz = vla_data.get("tactile_future_horizon")
    if hz in (None, "", "null", "None"):
        hz = fc.get("horizon")
    if hz in (None, "", "null", "None"):
        # Mirrors future_contact.DEFAULT_FUTURE_HORIZON. Not imported: this
        # generator must stay dependency-free (it runs before the training
        # env is exercised); keep the two values equal.
        hz = 10
    hz = int(hz)
    if hz < 1:
        raise ValueError(f"tactile_future_horizon must be >= 1, got {hz}")
    vla_data["tactile_future_horizon"] = hz


# Sync-path replan cadence — also deploy-time only. null/absent = use the
# training action_horizon at eval; a positive int re-plans every N rows.
# While RTC is on the rtc config owns the cadence instead.
execute_horizon = cfg.get("execute_horizon")
if execute_horizon not in (None, "", "null", "None", "auto"):
    try:
        execute_horizon_n = int(execute_horizon)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"execute_horizon must be a positive integer or null, "
            f"got: {execute_horizon!r}."
        ) from exc
    if execute_horizon_n < 1:
        raise ValueError(
            f"execute_horizon must be a positive integer or null, "
            f"got: {execute_horizon_n}."
        )

# State/action pose formats are TRAIN-CONFIG switches (datasets.vla_data.
# state_pose_format / action_pose_format), not embodiment properties: required
# whenever the run touches an embodiment with a pose_representation block,
# forbidden otherwise. Resolved values are written into pose_representation.
# {state,action} of the embedded embodiment_spec (checkpoints stay
# self-describing and eval replays them) and exported to the dataloader
# registry via SPARK_0_XPOLICY_{STATE,ACTION}_POSE_FORMAT (shared across a
# co-train mixture). source_starvla still reads that name; train.sh copies
# the resolved value across at launch.
STATE_POSE_FORMATS = ("raw", "abs_xyz_rot6d", "cam_abs_xyz_rot6d")
ACTION_POSE_FORMATS = ("raw", "abs_xyz_rot6d", "cam_abs_xyz_rot6d", "delta_xyz_rot6d")
CAMERA_FRAME_POSE_FORMATS = ("cam_abs_xyz_rot6d",)
# Normalization of the CONVERTED pose ACTION values. Optional; the default
# reproduces the legacy affine map exactly. The `hold_centered*` modes only
# exist for formats with a constant hold (see HOLD_CENTERED_ACTION_FORMATS);
# `hold_centered_perrow` additionally needs the per-horizon-row quantiles,
# which the converted-pose statistics only produce for chunked action targets.
POSE_NORMALIZATIONS = ("q99", "hold_centered", "hold_centered_perrow")
HOLD_CENTERED_NORMALIZATIONS = ("hold_centered", "hold_centered_perrow")
HOLD_CENTERED_ACTION_FORMATS = ("delta_xyz_rot6d",)
state_pose_format = vla_data.get("state_pose_format")
action_pose_format = vla_data.get("action_pose_format")
pose_normalization = str(vla_data.get("pose_normalization") or "q99")
if pose_normalization not in POSE_NORMALIZATIONS:
    raise ValueError(
        f"datasets.vla_data.pose_normalization={pose_normalization!r}; "
        f"expected one of {POSE_NORMALIZATIONS}."
    )

def _spec_has_pose_repr(name):
    """(spec file found, declares pose keys) for one embodiment name."""
    path = os.path.join(embodiment_dir, f"{name}.yaml")
    if not os.path.isfile(path):
        return False, False
    with open(path, "r", encoding="utf-8") as fp:
        entry_spec = yaml.safe_load(fp) or {}
    return True, bool((entry_spec.get("pose_representation") or {}).get("keys"))

mixture_embodiments = {
    str(entry.get("embodiment") or embodiment)
    for entry in cfg.get("data_mixture") or []
    if isinstance(entry, dict)
}
_pose_repr_probe = [_spec_has_pose_repr(name) for name in ({embodiment} | mixture_embodiments)]
any_spec_found = any(found for found, _ in _pose_repr_probe)
any_pose_repr = any(has_keys for _, has_keys in _pose_repr_probe)

def _resolve_pose_format(key, value, allowed):
    """Validate one datasets.vla_data.*_pose_format against pose embodiments."""
    if value:
        value = str(value)
        if value not in allowed:
            raise ValueError(
                f"datasets.vla_data.{key}={value!r}; expected one of {allowed}."
            )
        if not any_pose_repr:
            if any_spec_found:
                raise ValueError(
                    f"datasets.vla_data.{key} is set but no embodiment "
                    "in this run declares a pose_representation block."
                )
            # No embodiment YAML resolved under embodiment_dir (built-in
            # registry defaults, or a spec supplied elsewhere): forward
            # the value instead of blocking the launch.
            print(f"[Spark_0][WARN] no embodiment spec found under {embodiment_dir}; "
                  f"forwarding {key}={value}.", file=sys.stderr)
        return value
    if any_pose_repr:
        raise ValueError(
            f"datasets.vla_data.{key} is required for embodiments "
            f"with a pose_representation block; choose one of {allowed}."
        )
    return value

state_pose_format = _resolve_pose_format(
    "state_pose_format", state_pose_format, STATE_POSE_FORMATS
)
action_pose_format = _resolve_pose_format(
    "action_pose_format", action_pose_format, ACTION_POSE_FORMATS
)
# cam_abs_xyz_rot6d is absolute xyz+rot6d in the chunk-t0 camera frame: the
# format name itself requires ee_in_camera_frame so train/eval cannot drift.
for fmt, key in (
    (state_pose_format, "state_pose_format"),
    (action_pose_format, "action_pose_format"),
):
    if fmt in CAMERA_FRAME_POSE_FORMATS and not vla_data["ee_in_camera_frame"]:
        raise ValueError(
            f"datasets.vla_data.{key}={fmt!r} requires "
            "datasets.vla_data.ee_in_camera_frame=true "
            "(camera-frame absolute spatial pose)."
        )
if action_pose_format:
    ref_noise_prob = float((vla_data.get("ref_noise") or {}).get("prob") or 0.0)
    # PARKED (2026-07-29): wrist windup was a normalization/representation
    # bug (hold ≠ q99 zero), not missing corrective demos. ref_noise also
    # only perturbs the last history row, so history>1 sees a physically
    # impossible state jump that deploy never produces. Keep the code path;
    # re-enable only after hold_centered (and friends) are measured.
    if ref_noise_prob > 0.0:
        raise ValueError(
            "datasets.vla_data.ref_noise is parked (prob must be 0); "
            "use pose_normalization='hold_centered_perrow' for the anti-windup "
            f"A/B. Got prob={ref_noise_prob}."
        )
    if action_pose_format != "delta_xyz_rot6d" and ref_noise_prob > 0.0:
        raise ValueError(
            "datasets.vla_data.ref_noise.prob > 0 only applies to "
            f"action_pose_format='delta_xyz_rot6d' (got {action_pose_format!r})."
        )
    if (
        pose_normalization in HOLD_CENTERED_NORMALIZATIONS
        and action_pose_format not in HOLD_CENTERED_ACTION_FORMATS
    ):
        raise ValueError(
            f"datasets.vla_data.pose_normalization={pose_normalization!r} needs an "
            f"action format with a constant hold value {HOLD_CENTERED_ACTION_FORMATS} "
            f"(got {action_pose_format!r}); absolute formats hold at the current "
            "state, which is not a constant."
        )

# action_mode (the base dataset's linear delta/rel pass) is applied to EVERY
# action key — including pose keys that EEPoseReprLeRobotDataset has already
# re-encoded (abs/delta rot6d). The combination silently double-converts the
# pose actions (delta-of-rot6d-delta), so pose embodiments must use abs; the
# pose delta lives in action_pose_format instead.
run_action_mode = str(vla_data.get("action_mode") or "abs").lower()
if run_action_mode in ("absolute", "raw"):
    run_action_mode = "abs"
if any_pose_repr and run_action_mode != "abs":
    raise ValueError(
        f"datasets.vla_data.action_mode={run_action_mode!r} is incompatible with "
        "embodiments declaring a pose_representation block: the EE-pose dataset "
        "converts pose keys itself, and the base action_mode pass would apply a "
        "linear delta/rel on top of the converted values (a silent double "
        "conversion). Use action_mode: abs — chunk-wise pose deltas are handled "
        "by datasets.vla_data.action_pose_format."
    )

# Run-level embodiment overrides. The embodiment YAML describes what the robot
# IS (cameras, keys, per-key dims, which keys carry EE poses); the chunk
# length, per-modality normalization and observation indices are decisions a
# TRAINING RUN makes, so they live in the train config. Resolved values are
# frozen into the embedded embodiment_spec (checkpoints stay self-describing
# and eval replays them) and passed to the dataloader as
# SPARK_0_XPOLICY_SPEC_OVERRIDES. source_starvla still reads that name;
# train.sh copies the resolved value across at launch.
# The `hold_centered*` modes are deliberately absent: they need a per-key hold
# vector and are reached only through datasets.vla_data.pose_normalization.
NORMALIZATION_MODES = ("q99", "mean_std", "min_max", "binary")

action_model = cfg.setdefault("framework", {}).setdefault("action_model", {})
if action_model.get("action_horizon") is None:
    raise ValueError(
        "framework.action_model.action_horizon is required: it is the action "
        "chunk length (previously the embodiment YAML's action_horizon)."
    )
action_horizon = int(action_model["action_horizon"])
if action_horizon < 1:
    raise ValueError(
        f"framework.action_model.action_horizon must be >= 1, got {action_horizon}."
    )
action_model["action_horizon"] = action_horizon


def _resolve_normalization(key):
    """One mode for every key of the modality, or a per-key short-name map."""
    value = vla_data.get(key) or "q99"
    for mode in (value.values() if isinstance(value, dict) else [value]):
        if mode not in NORMALIZATION_MODES:
            raise ValueError(
                f"datasets.vla_data.{key} uses mode {mode!r}; expected one of "
                f"{NORMALIZATION_MODES} (pose actions have their own switch, "
                "datasets.vla_data.pose_normalization)."
            )
    return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else str(value)


state_normalization = _resolve_normalization("state_normalization")
action_normalization = _resolve_normalization("action_normalization")

observation_indices = vla_data.get("observation_indices")
if observation_indices is None:
    observation_indices = [0]
if not isinstance(observation_indices, (list, tuple)) or not observation_indices:
    raise ValueError(
        "datasets.vla_data.observation_indices must be a non-empty list of frame "
        f"offsets relative to the current step, got {observation_indices!r}."
    )
observation_indices = [int(index) for index in observation_indices]

vla_data["state_normalization"] = state_normalization
vla_data["action_normalization"] = action_normalization
vla_data["observation_indices"] = observation_indices
spec_overrides = {
    "action_horizon": action_horizon,
    "state_normalization": state_normalization,
    "action_normalization": action_normalization,
    "observation_indices": observation_indices,
}

if os.path.isfile(spec_path):
    with open(spec_path, "r", encoding="utf-8") as fp:
        spec = yaml.safe_load(fp)
    spec.update(spec_overrides)

    # Active cameras are a TRAIN-CONFIG switch. The embodiment YAML lists the
    # catalog; datasets.vla_data.video_keys (explicit) or the named cameras in
    # obs_image_size (coupled enable+resize) select the subset used this run.
    # Resolved full keys are frozen into embodiment_spec + SPEC_OVERRIDES so
    # the dataloader / deploy see the same list as training.
    catalog_video_keys = list(spec.get("video_keys") or [])
    active_video_keys = runtime_config.resolve_active_video_keys(
        catalog_video_keys,
        video_keys=vla_data.get("video_keys"),
        obs_image_size=vla_data.get("obs_image_size") or vla_data.get("image_size"),
    )
    # Fail fast on a size mapping that cannot cover the active cameras
    # (same expansion the dataloader / predict_action will do).
    runtime_config.expand_obs_image_sizes(
        vla_data.get("obs_image_size") or vla_data.get("image_size"),
        active_video_keys,
    )
    spec["video_keys"] = active_video_keys
    vla_data["video_keys"] = active_video_keys
    spec_overrides["video_keys"] = active_video_keys

    # Keys under pose_representation are stored raw as 7-d xyz_wxyz but the
    # model sees the per-modality format's dim (raw=7, abs_xyz_rot6d=9,
    # delta_xyz_rot6d=9; see data_registry/ee_delta_dataset.py), so model
    # dims must count the converted representation.
    pose_repr = spec.get("pose_representation") or {}
    pose_keys = set(pose_repr.get("keys", []))
    if pose_keys and (state_pose_format or action_pose_format):
        # Freeze the per-run switches into the spec embedded in the checkpoint.
        if state_pose_format:
            pose_repr["state"] = state_pose_format
        if action_pose_format:
            pose_repr["action"] = action_pose_format
        pose_repr["normalization"] = pose_normalization
        spec["pose_representation"] = pose_repr
    pose_format_dims = {
        "raw": 7,
        "abs_xyz_rot6d": 9,
        "cam_abs_xyz_rot6d": 9,
        "delta_xyz_rot6d": 9,
    }

    def _model_key_dims(key_dims, pose_format):
        pose_dim = pose_format_dims[pose_format]
        return {
            key: (pose_dim if key in pose_keys else int(dim))
            for key, dim in key_dims.items()
        }

    action_key_dims = _model_key_dims(
        spec["action_keys"], pose_repr.get("action", "delta_xyz_rot6d")
    )
    action_model["action_dim"] = sum(action_key_dims.values())
    action_model["state_dim"] = sum(
        _model_key_dims(spec["state_keys"], pose_repr.get("state", "abs_xyz_rot6d")).values()
    )

    # framework.action_model.action_loss_weight.keys is a per-action-key map
    # (short names, missing keys default to 1.0). The action head only sees a
    # flat action vector, so expand it here into per-dimension weights while
    # the key -> model-dim layout is known. Absent/empty -> nothing written and
    # the head keeps the plain .mean().
    loss_weight = action_model.get("action_loss_weight") or {}
    key_weights = loss_weight.get("keys") or {}
    # horizon_power exists to compensate the row-dependent target amplitude a
    # POOLED scale leaves behind. Per-row normalization already removes it from
    # the targets, so stacking both over-weights the near rows twice. Legal as
    # an explicit arm, hence a warning rather than an error.
    if (
        pose_normalization == "hold_centered_perrow"
        and float(loss_weight.get("horizon_power") or 0.0) != 0.0
    ):
        print(
            "[Spark_0][WARN] framework.action_model.action_loss_weight.horizon_power="
            f"{loss_weight['horizon_power']} together with pose_normalization="
            "'hold_centered_perrow': the per-row scale already equalizes the "
            "target amplitude across the chunk, so this weights the near rows a "
            "second time. Set horizon_power: 0.0 unless that is the arm.",
            file=sys.stderr,
        )
    if key_weights:
        unknown = sorted(set(key_weights) - set(action_key_dims))
        if unknown:
            raise ValueError(
                f"framework.action_model.action_loss_weight.keys has unknown action "
                f"keys {unknown}; expected a subset of {sorted(action_key_dims)}."
            )
        dims = []
        for key, dim in action_key_dims.items():
            weight = float(key_weights.get(key, 1.0))
            if weight <= 0:
                raise ValueError(
                    f"framework.action_model.action_loss_weight.keys[{key}]={weight}; "
                    "weights must be > 0."
                )
            dims.extend([weight] * dim)
        loss_weight["dims"] = dims
        action_model["action_loss_weight"] = loss_weight

    # Strict co-train compatibility. A mixture batch mixes samples from every
    # dataset directly into ONE flat state/action vector and one image list,
    # so every mixture embodiment must produce the same model-side layout as
    # the main one: identical ordered state/action keys and per-key dims, the
    # same pose keys, and a camera catalog that covers the active video_keys.
    # Checked at launch — the alternatives are a cryptic merge-metadata assert
    # minutes in, or a ragged batch at the first mixed step.
    for mixture_name in sorted(mixture_embodiments - {embodiment}):
        mixture_spec_path = os.path.join(embodiment_dir, f"{mixture_name}.yaml")
        if not os.path.isfile(mixture_spec_path):
            raise ValueError(
                f"data_mixture embodiment {mixture_name!r} not found under "
                f"{embodiment_dir}."
            )
        with open(mixture_spec_path, "r", encoding="utf-8") as fp:
            mixture_spec = yaml.safe_load(fp) or {}
        for block in ("state_keys", "action_keys"):
            main_layout = [(str(k), int(v)) for k, v in (spec.get(block) or {}).items()]
            mixture_layout = [
                (str(k), int(v)) for k, v in (mixture_spec.get(block) or {}).items()
            ]
            if mixture_layout != main_layout:
                raise ValueError(
                    f"data_mixture embodiment {mixture_name!r} has a different "
                    f"{block} layout {mixture_layout} than the main embodiment "
                    f"{embodiment!r} {main_layout}. Mixture samples share one flat "
                    f"{block.split('_')[0]} vector, so the ordered keys and per-key "
                    "dims must be identical."
                )
        main_pose_keys = sorted((spec.get("pose_representation") or {}).get("keys") or [])
        mixture_pose_keys = sorted(
            (mixture_spec.get("pose_representation") or {}).get("keys") or []
        )
        if mixture_pose_keys != main_pose_keys:
            raise ValueError(
                f"data_mixture embodiment {mixture_name!r} declares pose keys "
                f"{mixture_pose_keys} but the main embodiment {embodiment!r} "
                f"declares {main_pose_keys}; co-trained embodiments must share "
                "one pose representation."
            )
        mixture_catalog = {
            str(k).split(".")[-1] for k in (mixture_spec.get("video_keys") or [])
        }
        # flexible_views: embodiment catalogs need only cover required_views;
        # individual datasets may omit optional wrists via modality.json.
        # Without it, every mixture schema must still cover the full active set.
        # String "false" must stay off (bash/python bool("false") is true).
        _flex_raw = vla_data.get("flexible_views", False)
        if isinstance(_flex_raw, str):
            flexible_views = _flex_raw.strip().lower() in ("true", "1", "yes", "on")
        else:
            flexible_views = bool(_flex_raw)
        if flexible_views:
            required_raw = vla_data.get("required_views")
            if required_raw is None:
                check_cameras = (
                    [active_video_keys[0]] if active_video_keys else []
                )
            elif isinstance(required_raw, str):
                check_cameras = [required_raw]
            else:
                check_cameras = list(required_raw)
        else:
            check_cameras = list(active_video_keys)
        missing_cameras = [
            key for key in check_cameras
            if str(key).split(".")[-1] not in mixture_catalog
        ]
        if missing_cameras:
            raise ValueError(
                f"data_mixture embodiment {mixture_name!r} camera catalog "
                f"{sorted(mixture_catalog)} does not cover "
                f"{'required_views' if flexible_views else 'active cameras'} "
                f"{check_cameras} (missing {missing_cameras})"
                f"{'; active run slots are ' + str(active_video_keys) if flexible_views else ''}; "
                f"{'with flexible_views, optional cameras may be absent per dataset' if flexible_views else 'every mixture dataset must provide the same views'}."
            )

    cfg["embodiment_spec"] = spec
else:
    if ((cfg.get("framework") or {}).get("action_model") or {}).get(
        "action_loss_weight", {}
    ).get("keys"):
        raise ValueError(
            "framework.action_model.action_loss_weight.keys needs an embodiment spec "
            f"to resolve the key -> dimension layout, but none was found at {spec_path}."
        )
    print(f"[Spark_0][WARN] embodiment spec not found at {spec_path}; "
          "relying on built-in registry defaults, action_model left as-is.", file=sys.stderr)

# Optional multi-dataset co-training mixture declared in the training YAML:
#   data_mixture:
#     - {dataset: <dir under data_root_dir>, weight: 1.0, embodiment: <schema>}
# Entries are normalized (weight defaults to 1.0, embodiment defaults to this
# config's embodiment) and handed to the dataloader registry via
# SPARK_0_XPOLICY_MIXTURE_JSON, registered under datasets.vla_data.data_mix.
# source_starvla still reads that name; train.sh copies the resolved value
# across at launch.
mixture_entries = []
for entry in cfg.get("data_mixture") or []:
    if not isinstance(entry, dict) or "dataset" not in entry:
        raise ValueError(f"data_mixture entries need a 'dataset' key, got: {entry!r}")
    weight = float(entry.get("weight", 1.0))
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(
            f"data_mixture entry {entry['dataset']!r} has weight {entry.get('weight')!r}; "
            "weights must be finite and > 0 (omit the entry to exclude a dataset)."
        )
    mixture_entries.append({
        "dataset": str(entry["dataset"]),
        "weight": weight,
        "embodiment": str(entry.get("embodiment") or embodiment),
    })
if mixture_entries:
    cfg["data_mixture"] = mixture_entries

# Optional environment overrides so per-user/per-cluster settings never
# require editing the checked-in base yaml.
if os.environ.get("SPARK_0_WANDB_ENTITY"):
    cfg["wandb_entity"] = os.environ["SPARK_0_WANDB_ENTITY"]
if os.environ.get("SPARK_0_WANDB_PROJECT"):
    cfg["wandb_project"] = os.environ["SPARK_0_WANDB_PROJECT"]
if os.environ.get("SPARK_0_BASE_VLM"):
    cfg.setdefault("framework", {}).setdefault("qwenvl", {})["base_vlm"] = os.environ["SPARK_0_BASE_VLM"]

# Continue a previous run (trainer.resume_from, default null): a steps_<N>
# checkpoint file, a run directory, or a run's checkpoints/ directory. Training
# runs with cwd=source_starvla, so a relative path is resolved here (against the
# caller's cwd first, then the adapter dir) and checked before the multi-minute
# model load.
resume_from = os.environ.get("SPARK_0_RESUME_FROM") or (cfg.get("trainer") or {}).get("resume_from")
if resume_from:
    resume_from = os.path.expanduser(str(resume_from))
    if not os.path.isabs(resume_from):
        for base in (os.getcwd(), adapter_dir):
            candidate = os.path.join(base, resume_from)
            if os.path.exists(candidate):
                resume_from = candidate
                break
    resume_from = os.path.abspath(resume_from)
    if not os.path.exists(resume_from):
        raise FileNotFoundError(f"trainer.resume_from not found: {resume_from}")
    cfg.setdefault("trainer", {})["resume_from"] = resume_from

# Atomic write: on a shared filesystem every node of a multi-node launch
# generates this same file concurrently; a plain open(dst, "w") lets one
# node read a half-written config. The content is deterministic across
# nodes, so last-rename-wins is safe.
tmp_dst = f"{dst}.tmp.{os.getpid()}"
with open(tmp_dst, "w", encoding="utf-8") as fp:
    yaml.safe_dump(cfg, fp, sort_keys=False)
os.replace(tmp_dst, dst)

# Captured by train.sh, then exported to source_starvla as SPARK_0_*:
# line 1 -> XPOLICY_ROBOT_TYPE, line 2 -> XPOLICY_MIXTURE_JSON (empty when
# no mixture is declared), line 3 -> the resolved trainer.resume_from (empty
# when starting from scratch), line 4 -> XPOLICY_STATE_POSE_FORMAT, line 5 ->
# XPOLICY_ACTION_POSE_FORMAT (empty for non-pose embodiments), line 6 ->
# XPOLICY_POSE_NORMALIZATION, line 7 -> XPOLICY_SPEC_OVERRIDES.
print(embodiment)
print(json.dumps(mixture_entries, separators=(",", ":")) if mixture_entries else "")
print(resume_from or "")
print(state_pose_format or "")
print(action_pose_format or "")
print(pose_normalization if (state_pose_format or action_pose_format) else "")
print(json.dumps(spec_overrides, separators=(",", ":")))
PY
)
embodiment=$(sed -n '1p' <<< "${gen_output}")
mixture_json=$(sed -n '2p' <<< "${gen_output}")
resume_from=$(sed -n '3p' <<< "${gen_output}")
state_pose_format=$(sed -n '4p' <<< "${gen_output}")
action_pose_format=$(sed -n '5p' <<< "${gen_output}")
pose_normalization=$(sed -n '6p' <<< "${gen_output}")
spec_overrides=$(sed -n '7p' <<< "${gen_output}")

# Fail fast when a declared mixture references missing datasets.
if [[ -n "${mixture_json}" ]]; then
    python - "${data_root_dir}" "${mixture_json}" <<'PY'
import json
import os
import sys

data_root_dir, mixture_json = sys.argv[1:3]
missing = [
    entry["dataset"]
    for entry in json.loads(mixture_json)
    if not os.path.isfile(os.path.join(data_root_dir, entry["dataset"], "meta", "modality.json"))
]
if missing:
    print(f"[Spark_0][ERROR] data_mixture datasets missing under {data_root_dir}: {missing}", file=sys.stderr)
    print("[Spark_0][ERROR] expected <dataset>/meta/modality.json for each entry.", file=sys.stderr)
    sys.exit(1)
PY
fi

# Fail fast when base_vlm is a local path that does not exist (relative paths
# resolve against source_starvla because training runs with cwd there).
base_vlm=$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['framework']['qwenvl']['base_vlm'])" "${config_yaml}")
if [[ "${base_vlm}" == /* || "${base_vlm}" == ./* ]]; then
    resolved_base_vlm="${base_vlm}"
    [[ "${base_vlm}" == ./* ]] && resolved_base_vlm="${STARVLA_ROOT}/${base_vlm#./}"
    if [[ ! -e "${resolved_base_vlm}" ]]; then
        echo "[Spark_0][ERROR] base_vlm path not found: ${resolved_base_vlm}" >&2
        echo "[Spark_0][ERROR] fix framework.qwenvl.base_vlm in ${base_config_yaml} or set SPARK_0_BASE_VLM." >&2
        exit 1
    fi
fi

# Gradient accumulation: single source of truth is the training YAML
# (trainer.gradient_accumulation_steps, default 1). Precedence:
#   ACCELERATE_GRADIENT_ACCUMULATION_STEPS env (per-user override, no YAML edit)
#   > trainer.gradient_accumulation_steps CLI override (any dotlist form)
#   > YAML value.
# The resolved value is exported to accelerate/DeepSpeed AND passed back to the
# trainer as an explicit override, so the run's config snapshot always records
# the effective value and train_starvla.py can fail fast on any mismatch.
grad_accum=$(SPARK_0_GA_ENV="${ACCELERATE_GRADIENT_ACCUMULATION_STEPS:-}" python - "${config_yaml}" "$@" <<'PY'
import os
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1])) or {}
ga = (cfg.get("trainer") or {}).get("gradient_accumulation_steps") or 1

# CLI dotlist override, in any accepted form:
#   trainer.gradient_accumulation_steps=4
#   --trainer.gradient_accumulation_steps=4
#   --trainer.gradient_accumulation_steps 4
KEY = "trainer.gradient_accumulation_steps"
args = sys.argv[2:]
for i, arg in enumerate(args):
    stripped = arg.lstrip("-")
    if stripped.startswith(KEY + "="):
        ga = stripped.split("=", 1)[1]
    elif stripped == KEY and i + 1 < len(args):
        ga = args[i + 1]

if os.environ.get("SPARK_0_GA_ENV"):
    ga = os.environ["SPARK_0_GA_ENV"]

try:
    ga = int(ga)
except (TypeError, ValueError):
    print(f"[Spark_0][ERROR] invalid gradient_accumulation_steps: {ga!r}", file=sys.stderr)
    sys.exit(1)
if ga < 1:
    print(f"[Spark_0][ERROR] gradient_accumulation_steps must be >= 1, got {ga}", file=sys.stderr)
    sys.exit(1)
print(ga)
PY
)

echo "[Spark_0] train_config=${config_stem} (${base_config_yaml})"
echo "[Spark_0] config_yaml=${config_yaml}"
echo "[Spark_0] base_vlm=${base_vlm}"
echo "[Spark_0] run_id=${run_id}"
echo "[Spark_0] seed=${seed}"
echo "[Spark_0] data_root_dir=${data_root_dir}"
echo "[Spark_0] ckpt_root_dir=${ckpt_root_dir}"
if [[ -n "${mixture_json}" ]]; then
    echo "[Spark_0] data_mix=${data_mix} (multi-dataset co-train): ${mixture_json}"
else
    echo "[Spark_0] data_mix=${data_mix}, dataset=${dataset_name}, dataset_path=${dataset_path}"
fi
echo "[Spark_0] embodiment=${embodiment} (embodiment_dir=${embodiment_dir})"
echo "[Spark_0] embodiment run overrides: ${spec_overrides}"
if [[ -n "${state_pose_format}" ]]; then
    echo "[Spark_0] state_pose_format=${state_pose_format} (datasets.vla_data.state_pose_format)"
fi
if [[ -n "${action_pose_format}" ]]; then
    echo "[Spark_0] action_pose_format=${action_pose_format} (datasets.vla_data.action_pose_format)"
fi
if [[ -n "${pose_normalization}" ]]; then
    echo "[Spark_0] pose_normalization=${pose_normalization} (datasets.vla_data.pose_normalization)"
fi
if [[ -n "${resume_from}" ]]; then
    echo "[Spark_0] resume_from=${resume_from} (continues the step counter + LR schedule; optimizer moments are not restored)"
fi
echo "[Spark_0] train_entry=starVLA/training/train_starvla.py"
echo "[Spark_0] num_processes=${num_processes} (gpus_per_node=${gpus_per_node} x num_machines=${num_machines}), mixed_precision=bf16"
echo "[Spark_0] gradient_accumulation_steps=${grad_accum} (effective batch = per_device_bs x ${num_processes} x ${grad_accum})"

launch_args=()
if [[ -n "${accelerate_config}" ]]; then
    launch_args+=(--config_file "${accelerate_config}")
    echo "[Spark_0] accelerate_config=${accelerate_config}"
fi
launch_args+=(
    --num_processes "${num_processes}"
    --num_machines "${num_machines}"
    --mixed_precision bf16
    --dynamo_backend no
)
if [[ "${num_machines}" -gt 1 ]]; then
    launch_args+=(
        --machine_rank "${machine_rank}"
        --main_process_ip "${main_process_ip}"
        --main_process_port "${main_process_port}"
    )
    echo "[Spark_0] machine_rank=${machine_rank}, main_process=${main_process_ip}:${main_process_port}"
fi

# source_starvla reads SPARK_0_* directly. Stats knobs are inherited.
# The XPOLICY_* and embodiment values are set on the command itself from
# the resolved shell variables, and those two launches must stay identical —
# the content-addressed stats cache names depend on them.
# Offline statistics precompute (SPARK_0_PRECOMPUTE_STATS_ONLY=1, normally via
# scripts/precompute_stats.sh): same resolved config + env as training, but
# only the dataset statistics caches are built, then exit.
if [[ "${SPARK_0_PRECOMPUTE_STATS_ONLY:-0}" == "1" ]]; then
    echo "[Spark_0] SPARK_0_PRECOMPUTE_STATS_ONLY=1: building dataset statistics caches, no training"
    cd "${STARVLA_ROOT}"
    PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}" \
    SPARK_0_XPOLICY_DATASET_NAME="${dataset_name}" \
    SPARK_0_XPOLICY_DATA_MIX="${data_mix}" \
    SPARK_0_XPOLICY_MIXTURE_JSON="${mixture_json}" \
    SPARK_0_XPOLICY_ROBOT_TYPE="${embodiment}" \
    SPARK_0_XPOLICY_STATE_POSE_FORMAT="${state_pose_format}" \
    SPARK_0_XPOLICY_ACTION_POSE_FORMAT="${action_pose_format}" \
    SPARK_0_XPOLICY_POSE_NORMALIZATION="${pose_normalization}" \
    SPARK_0_XPOLICY_SPEC_OVERRIDES="${spec_overrides}" \
    SPARK_0_EMBODIMENT_DIR="${embodiment_dir}" \
    SPARK_0_REQUIRE_PRECOMPUTED_STATS=0 \
    python starVLA/training/precompute_stats.py --config_yaml "${config_yaml}" "$@"
    exit 0
fi

# Fail fast on THIS node when DA3 is on but the teacher is missing / the
# package is not importable. The teacher is lazy-loaded on the first
# training forward; a missing copy otherwise errors (or hangs multi-node)
# minutes in. finetune.sh also runs this on the source recipe; this check
# uses the generated yaml so enable + teacher_model_path match training.
# SKIP_DA3_CHECK=1 bypasses (e.g. an HF-hub teacher id with no local copy).
bash "${SCRIPT_DIR}/scripts/check_da3_teacher.sh" "${config_yaml}"

cd "${STARVLA_ROOT}"
PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}" \
SPARK_0_XPOLICY_DATASET_NAME="${dataset_name}" \
SPARK_0_XPOLICY_DATA_MIX="${data_mix}" \
SPARK_0_XPOLICY_MIXTURE_JSON="${mixture_json}" \
SPARK_0_XPOLICY_ROBOT_TYPE="${embodiment}" \
SPARK_0_XPOLICY_STATE_POSE_FORMAT="${state_pose_format}" \
SPARK_0_XPOLICY_ACTION_POSE_FORMAT="${action_pose_format}" \
SPARK_0_XPOLICY_POSE_NORMALIZATION="${pose_normalization}" \
SPARK_0_XPOLICY_SPEC_OVERRIDES="${spec_overrides}" \
SPARK_0_EMBODIMENT_DIR="${embodiment_dir}" \
SPARK_0_REQUIRE_PRECOMPUTED_STATS="${SPARK_0_REQUIRE_PRECOMPUTED_STATS:-1}" \
WANDB_MODE="${WANDB_MODE:-offline}" \
NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}" \
NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}" \
ACCELERATE_GRADIENT_ACCUMULATION_STEPS="${grad_accum}" \
CUDA_VISIBLE_DEVICES="${gpu_id}" accelerate launch \
    "${launch_args[@]}" \
    starVLA/training/train_starvla.py \
    --config_yaml "${config_yaml}" \
    --run_id "${run_id}" \
    --seed "${seed}" \
    "$@" \
    --trainer.gradient_accumulation_steps "${grad_accum}"
