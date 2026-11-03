#!/bin/bash
set -euo pipefail

# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
# Output: checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>-<stage>
#
# Training runs in two stages, because the second one needs the first to exist.
#
#   WALLX_STAGE=memory   (default)  the policy and the keyframe selector
#   WALLX_STAGE=refiner             the residual refiner, over a frozen stage 1
#
# Stage 1 trains a policy that reads a memory of recorded keyframes, together
# with the selector that decides which frames to record -- the selector is what
# lets the policy rely on a memory it writes itself at inference.
#
# Stage 2 adds a second copy of the action branch that corrects the policy's own
# samples. A flow-matching policy is near a fixed point of its own denoiser, so
# it cannot correct itself; the refiner learns the one mapping demonstrations
# never show it, from its own output back to the demonstration.
#
# Stage 1 starts from the base model behind our published results, published as
# x-square-robot/wall-oss-1.0-lite-robodojo-base. It ships the normalization
# statistics it was trained with, which is where a run reads them from.
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
WALLX_REPO="${WALLX_REPO:-${BENCH_ROOT}/wall-x}"
STAGE="${WALLX_STAGE:-memory}"

ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
run_dir="${SCRIPT_DIR}/checkpoints/${ckpt_setting}-${STAGE}"
nproc="${WALLX_NPROC:-$(awk -F, '{print NF}' <<<"${gpu_id}")}"

if [[ ! -f "${WALLX_REPO}/wall_x/trainer/fsdp_trainer/train_fsdp.py" ]]; then
    echo "[WallX] no wall-x checkout at ${WALLX_REPO}" >&2
    echo "[WallX] clone https://github.com/X-Square-Robot/wall-x beside XPolicyLab," >&2
    echo "[WallX] or point WALLX_REPO at it; install.sh does this for you." >&2
    exit 1
fi

# The serving checkout and the training checkout are different branches, and a
# serving-only one fails much later, inside the trainer, with an unregistered
# dataset type. Say it here instead.
if [[ ! -d "${WALLX_REPO}/wall_x/data/backends/lerobot/robodojo" ]]; then
    echo "[WallX] ${WALLX_REPO} has no RoboDojo data backend: it is the serving" >&2
    echo "[WallX] checkout. Training lives on the branch starrick/oss-training:" >&2
    echo "[WallX]   git -C ${WALLX_REPO} fetch origin starrick/oss-training" >&2
    echo "[WallX]   git -C ${WALLX_REPO} checkout starrick/oss-training" >&2
    exit 1
fi

# The stage-1 run of the same setting is what stage 2 refines. Training writes
# one directory per saved step under that run, so resolve the newest actual
# checkpoint rather than the run directory, which exists from the first second.
stage1_root="${WALLX_STAGE1_CKPT:-${SCRIPT_DIR}/checkpoints/${ckpt_setting}-memory}"
margins="${WALLX_MARGINS:-${SCRIPT_DIR}/margins.json}"

resolve_stage1() {
    if [[ -f "${stage1_root}/model.safetensors" ]]; then
        echo "${stage1_root}"
        return 0
    fi
    find "${stage1_root}" -mindepth 2 -maxdepth 2 -name model.safetensors \
        -printf '%T@ %h\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-
}

case "${STAGE}" in
    memory)
        config="${WALLX_CONFIG:-${WALLX_REPO}/workspace/example/robodojo/stage1_memory.yml}"
        ;;
    refiner)
        config="${WALLX_CONFIG:-${WALLX_REPO}/workspace/example/robodojo/stage2_refiner.yml}"
        stage1_dir="$(resolve_stage1 || true)"
        if [[ -z "${stage1_dir}" ]]; then
            echo "[WallX] stage 2 refines a policy, and no saved checkpoint is under:" >&2
            echo "[WallX]   ${stage1_root}" >&2
            echo "[WallX] run this script without WALLX_STAGE first and let it save," >&2
            echo "[WallX] or set WALLX_STAGE1_CKPT to a stage-1 checkpoint." >&2
            exit 1
        fi
        echo "[WallX] refining ${stage1_dir}"
        ;;
    *)
        echo "[WallX] WALLX_STAGE must be 'memory' or 'refiner', got '${STAGE}'" >&2
        exit 1
        ;;
esac

if [[ ! -f "${config}" ]]; then
    echo "[WallX] no config at ${config}; set WALLX_CONFIG" >&2
    exit 1
fi

mkdir -p "${run_dir}"
run_config="${run_dir}/train.yml"

# The trainer reads the seed, the output path and the model to start from out of
# the config rather than the command line, so this invocation's values are
# written into a copy of it. The copy sits next to the checkpoint, which is also
# the honest record of what produced it.
python - "$config" "$run_config" "$STAGE" "$seed" "$run_dir" "${stage1_dir:-}" "$margins" <<'PY'
import sys
import yaml

src, dst, stage, seed, run_dir, stage1, margins = sys.argv[1:8]
with open(src, encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config["hyperparams"]["seed"] = int(seed)
config["checkpoint"]["save_path"] = run_dir
config["logging"]["log_name"] = run_dir.rsplit("/", 1)[-1]
if stage == "refiner":
    config["model"]["pretrained_path"] = stage1
    config["data"].setdefault("refiner", {})["margins"] = margins

with open(dst, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
print(f"[WallX] config for this run: {dst}")
PY

# The refiner is only scored on candidates the margins accept. Those margins
# describe the policy being refined, measured from it, so they are not shipped
# as constants -- and the measurement needs the run config written just above,
# which is the one pointing at this stage-1 checkpoint.
if [[ "${STAGE}" == "refiner" && ! -f "${margins}" ]]; then
    echo "[WallX] no calibrated margins at ${margins}; measure them first:" >&2
    echo "[WallX]   cd ${WALLX_REPO} && python scripts/calibrate_refiner_margins.py \\" >&2
    echo "[WallX]     --train-config ${run_config} --out ${margins}" >&2
    exit 1
fi

echo "[WallX] stage ${STAGE}, seed ${seed}, gpus ${gpu_id} (${nproc} processes)"
echo "[WallX] from ${config}"
echo "[WallX] into ${run_dir}"

cd "${WALLX_REPO}"
exec env CUDA_VISIBLE_DEVICES="${gpu_id}" \
    torchrun --nproc_per_node="${nproc}" \
    wall_x/trainer/fsdp_trainer/train_fsdp.py \
    --config "${run_config}"
