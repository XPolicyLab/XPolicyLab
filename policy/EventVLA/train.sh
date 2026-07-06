#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: bash train.sh <data_mix> <memory_ablation_mode> <keyframe_memory_policy> [data_root_dir] [train_args...]"
    echo "Example: bash train.sh robodojo pure_image_keyframe_memory teacher"
    echo "Example with training override: bash train.sh robodojo pure_image_keyframe_memory teacher --trainer.max_train_steps 20000"
    echo ""
    echo "The printed RUN_ID (= run directory name) is exactly the ckpt_name expected by eval.sh:"
    echo "  bash train.sh robodojo pure_image_keyframe_memory teacher"
    echo "  bash eval.sh <bench> <task> <RUN_ID> <env_cfg_type> <action_type> <seed> ..."
    echo ""
    echo "Override the run directory name (e.g. to match an existing eval ckpt_name) with RUN_ID:"
    echo "  RUN_ID=RoboDojo-eventvla-arx_x5-3500-joint-0 bash train.sh robodojo pure_image_keyframe_memory teacher"
    exit 1
fi

data_mix=${1}
memory_ablation_mode=${2}
keyframe_memory_policy=${3}
shift 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVENTVLA_ROOT="${SCRIPT_DIR}/source_eventvla"

# Default to the single-node entry that drives the in-repo eventvla/training/train_eventvla.py.
# The multi-node batch script (run_eventvla_train_batch.sh) launches starVLA/training/train_starvla.py,
# which is NOT vendored in this repo, so it must not be the default. Override with EVENTVLA_TRAIN_SCRIPT.
DEFAULT_TRAIN_SCRIPT="${EVENTVLA_ROOT}/examples/RoboTwin-Mem/train_files/run_eventvla_train.sh"
TRAIN_SCRIPT="${EVENTVLA_TRAIN_SCRIPT:-${DEFAULT_TRAIN_SCRIPT}}"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "[EventVLA][error] train script not found: ${TRAIN_SCRIPT}" >&2
    echo "[EventVLA][hint] set EVENTVLA_TRAIN_SCRIPT to an existing training entry." >&2
    exit 1
fi

# Map the keyframe_memory_policy argument onto the memory-source knobs consumed by
# run_eventvla_train.sh (kept semantically usable while the run script stays single-node).
case "${keyframe_memory_policy}" in
    teacher|with_teacher|teacher_to_predict)
        resolved_keyframe_memory_policy=teacher
        keyframe_train_memory_source=teacher_to_predict
        keyframe_train_memory_schedule=teacher_to_predict
        keyframe_schedule_teacher_prob_start=1.0
        keyframe_schedule_teacher_prob_end=0.0
        ;;
    predict|no_teacher|without_teacher|student)
        resolved_keyframe_memory_policy=predict
        keyframe_train_memory_source=predict
        keyframe_train_memory_schedule=predict
        keyframe_schedule_teacher_prob_start=0.0
        keyframe_schedule_teacher_prob_end=0.0
        ;;
    *)
        echo "[EventVLA][error] unsupported keyframe_memory_policy=${keyframe_memory_policy}" >&2
        echo "[EventVLA][error] supported policies: teacher, predict" >&2
        exit 1
        ;;
esac

# Deterministic run directory name that doubles as the eval ckpt_name.
# train writes results/Checkpoints/<RUN_ID>/... and eval.sh reads the same <RUN_ID>.
run_date=$(date +%Y%m%d)
default_run_id="${run_date}_${data_mix}_${memory_ablation_mode}_${resolved_keyframe_memory_policy}_eventvla"
RUN_ID="${RUN_ID:-${default_run_id}}"

# Keep training output under policy/EventVLA/results/Checkpoints/ so that
# setup_eval_policy_server.sh (which searches ${SCRIPT_DIR}/results/Checkpoints/<ckpt_name>) can find it.
RUN_ROOT_DIR="${EVENTVLA_RUN_ROOT_DIR:-${SCRIPT_DIR}/results/Checkpoints}"
mkdir -p "${RUN_ROOT_DIR}"

echo "[EventVLA] train_script=${TRAIN_SCRIPT}"
echo "[EventVLA] data_mix=${data_mix}"
echo "[EventVLA] memory_ablation_mode=${memory_ablation_mode}"
echo "[EventVLA] keyframe_memory_policy=${resolved_keyframe_memory_policy}"
echo "[EventVLA] run_root_dir=${RUN_ROOT_DIR}"
echo "[EventVLA] RUN_ID (= eval ckpt_name)=${RUN_ID}"
if [[ $# -gt 0 ]]; then
    echo "[EventVLA] train_args=$*"
fi

# The vendored run scripts use paths relative to the source_eventvla root.
cd "${EVENTVLA_ROOT}"

# All values below are consumed by run_eventvla_train.sh via env-var overrides
# (see its param block); the run directory / profile stay aligned with eval.
RUN_ID="${RUN_ID}" \
RUN_ROOT_DIR="${RUN_ROOT_DIR}" \
EVENTVLA_DATA_MIX="${data_mix}" \
EVENTVLA_MEMORY_ABLATION_MODE="${memory_ablation_mode}" \
KEYFRAME_TRAIN_MEMORY_SOURCE="${keyframe_train_memory_source}" \
KEYFRAME_TRAIN_MEMORY_SCHEDULE="${keyframe_train_memory_schedule}" \
KEYFRAME_SCHEDULE_TEACHER_PROB_START="${keyframe_schedule_teacher_prob_start}" \
KEYFRAME_SCHEDULE_TEACHER_PROB_END="${keyframe_schedule_teacher_prob_end}" \
bash "${TRAIN_SCRIPT}" "$@"

echo "[EventVLA] training finished."
echo "[EventVLA] eval ckpt_name=${RUN_ID}"
echo "[EventVLA] checkpoints under: ${RUN_ROOT_DIR}/${RUN_ID}/{final_model,checkpoints}"
echo "[EventVLA] to evaluate: bash eval.sh <bench> <task> ${RUN_ID} <env_cfg_type> <action_type> <seed> <policy_gpu> <env_gpu> <policy_conda_env> <eval_env_conda_env>"
