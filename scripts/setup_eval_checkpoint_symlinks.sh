#!/bin/bash
# Link final_ckpt weights into policy/*/checkpoints/ using README 6-tuple or model-specific names.
set -euo pipefail

XPOLICYLAB_ROOT="${XPOLICYLAB_ROOT:-/mnt/nfs/niantian/robodojo_test/XPolicyLab}"
FINAL_CKPT="${FINAL_CKPT:-/mnt/xspark-data/final_ckpt}"
POLICY_ROOT="${XPOLICYLAB_ROOT}/policy"

link_ckpt() {
    local policy_dir="$1"
    local link_name="$2"
    local src="$3"
    local ckpt_dir="${policy_dir}/checkpoints"
    local target="${ckpt_dir}/${link_name}"
    mkdir -p "$(dirname "${target}")"
    if [[ -e "${target}" || -L "${target}" ]]; then
        rm -rf "${target}"
    fi
    if [[ ! -e "${src}" ]]; then
        echo "[SKIP] missing source: ${src}" >&2
        return 1
    fi
    ln -sfn "${src}" "${target}"
    echo "[OK] ${target} -> ${src}"
}

COTRAIN_6="RoboDojo-cotrain-arx_x5-3500-joint-0"
OPENVLA_FINETUNE="${FINAL_CKPT}/OpenVLA_OFT/seed0/openvla-7b+aloha_sim_arx+b8+lr-0.0005+lora-r32+dropout-0.0--8gpu--cuda01234567--3img--proprio--film--100000_chkpt"

# --- A group ---
link_ckpt "${POLICY_ROOT}/A1" "${COTRAIN_6}" "${FINAL_CKPT}/A1/${COTRAIN_6}"
link_ckpt "${POLICY_ROOT}/Abot_M0" "RoboDojo-sim-arx_x5-100-joint-0" "${FINAL_CKPT}/Abot_M0/RoboDojo-sim-arx_x5-100-joint-0"
link_ckpt "${POLICY_ROOT}/GO1" "${COTRAIN_6}" "${FINAL_CKPT}/GO1/${COTRAIN_6}"
link_ckpt "${POLICY_ROOT}/MolmoACT2" "${COTRAIN_6}" "${FINAL_CKPT}/MolmoACT2/${COTRAIN_6}"
link_ckpt "${POLICY_ROOT}/OpenVLA_OFT" "${COTRAIN_6}" "${OPENVLA_FINETUNE}"
link_ckpt "${POLICY_ROOT}/OpenVLA_OFT" "shared/openvla-7b" \
    "/mnt/xspark-data/xspark_shared/model_weights/openvla-7b"
# Motus: model.py loads <dir>/mp_rank_00_model_states.pt, which lives under pytorch_model/.
link_ckpt "${POLICY_ROOT}/Motus" "${COTRAIN_6}" "${FINAL_CKPT}/Motus/checkpoint_step_80000/pytorch_model"

# RISE: whole checkpoints tree (if not already linked)
RISE_CKPT="${POLICY_ROOT}/RISE/checkpoints"
if [[ ! -L "${RISE_CKPT}" ]]; then
    mkdir -p "${POLICY_ROOT}/RISE"
    ln -sfn "/mnt/xspark-data/zijian/XPolicyLab/policy/RISE/checkpoints" "${RISE_CKPT}"
    echo "[OK] ${RISE_CKPT} -> zijian RISE checkpoints"
fi

# --- B group (model-specific ckpt dir names + 6-tuple aliases) ---
link_ckpt "${POLICY_ROOT}/LingBot_VA" "${COTRAIN_6}" \
    "${FINAL_CKPT}/Lingbot_VA/robodojo_sim_arx_x5_v21/checkpoints"
link_ckpt "${POLICY_ROOT}/LingBot_VA" "robodojo_sim_arx_x5_v21" \
    "${FINAL_CKPT}/Lingbot_VA/robodojo_sim_arx_x5_v21/checkpoints"

link_ckpt "${POLICY_ROOT}/InternVLA_A1" "RoboDojo_sim_seed_0" "${FINAL_CKPT}/InternVLA_A1/RoboDojo_sim_seed_0"
link_ckpt "${POLICY_ROOT}/InternVLA_A1" "${COTRAIN_6}" "${FINAL_CKPT}/InternVLA_A1/RoboDojo_sim_seed_0"
# InternVLA_A1 base VLM / tokenizer (model.py defaults QWEN3_2B_PATH / COSMOS_PATH to checkpoints/shared/).
link_ckpt "${POLICY_ROOT}/InternVLA_A1" "shared/Qwen3-VL-2B-Instruct" \
    "/mnt/xspark-data/xspark_shared/model_weights/Qwen3-VL-2B-Instruct"
link_ckpt "${POLICY_ROOT}/InternVLA_A1" "shared/Cosmos-Tokenizer-CI8x8" \
    "/mnt/xspark-data/xspark_shared/model_weights/Cosmos-Tokenizer-CI8x8"

link_ckpt "${POLICY_ROOT}/X_VLA" "XVLA_sim_arx-x5" "${FINAL_CKPT}/X_VLA/XVLA_sim_arx-x5"
link_ckpt "${POLICY_ROOT}/X_VLA" "${COTRAIN_6}" "${FINAL_CKPT}/X_VLA/XVLA_sim_arx-x5"
link_ckpt "${POLICY_ROOT}/X_VLA" "shared/X-VLA-Pt" \
    "/mnt/xspark-data/xspark_shared/model_weights/X-VLA-Pt"

link_ckpt "${POLICY_ROOT}/SmolVLA" "RoboDojo_sim_arx-x5_seed_0" "${FINAL_CKPT}/SmoVLA/RoboDojo_sim_arx-x5_seed_0"
link_ckpt "${POLICY_ROOT}/SmolVLA" "${COTRAIN_6}" "${FINAL_CKPT}/SmoVLA/RoboDojo_sim_arx-x5_seed_0"

link_ckpt "${POLICY_ROOT}/RDT_1B" "RoboDojo_sim_seed_0" "${FINAL_CKPT}/RDT_1B/RoboDojo_sim_seed_0"
link_ckpt "${POLICY_ROOT}/RDT_1B" "${COTRAIN_6}" "${FINAL_CKPT}/RDT_1B/RoboDojo_sim_seed_0"

link_ckpt "${POLICY_ROOT}/Spirit_v15" "RoboDojo_sim_arx-x5_seed_0" "${FINAL_CKPT}/Spirit_v1.5/RoboDojo_sim_arx-x5_seed_0"
link_ckpt "${POLICY_ROOT}/Spirit_v15" "${COTRAIN_6}" "${FINAL_CKPT}/Spirit_v1.5/RoboDojo_sim_arx-x5_seed_0"
link_ckpt "${POLICY_ROOT}/Spirit_v15" "shared/Spirit-v1.5" \
    "/mnt/xspark-data/xspark_shared/model_weights/Spirit-v1.5"
link_ckpt "${POLICY_ROOT}/Spirit_v15" "shared/Qwen3-VL-4B-Instruct" \
    "/mnt/xspark-data/xspark_shared/model_weights/Qwen3-VL-4B-Instruct"

link_ckpt "${POLICY_ROOT}/Pi_05" "Pi_05_sim_arx-x5_seed_1" "${FINAL_CKPT}/Pi_05/Pi_05_sim_arx-x5_seed_1"

# GigaWorldPolicy: model.py expects transformer_ema/config.json directly under the linked
# checkpoint dir; the real weights live one level deeper under checkpoint_epoch_4_step_100000_old/.
GIGAWORLD_CKPT_INNER="${FINAL_CKPT}/GigaWorldPolicy/RoboDojo_sim_arx_seed_0/checkpoint_epoch_4_step_100000_old"
link_ckpt "${POLICY_ROOT}/GigaWorldPolicy" "RoboDojo_sim_arx_seed_0" "${GIGAWORLD_CKPT_INNER}"
link_ckpt "${POLICY_ROOT}/GigaWorldPolicy" "${COTRAIN_6}" "${GIGAWORLD_CKPT_INNER}"

# --- C group (deploy-testing additions: Dexbotic / GR00T / Xiaomi / Pi_0 / Pi_0_Fast) ---
# Dexbotic_DM0: ee cotrain, model.py resolves checkpoints/<6-tuple>/checkpoint-*.
link_ckpt "${POLICY_ROOT}/Dexbotic_DM0" "RoboDojo-cotrain-arx_x5-3500-ee-0" \
    "${FINAL_CKPT}/DM_0/RoboDojo-cotrain-arx_x5-3500-ee-0"

# GR00T_N17: joint cotrain, model.py resolves checkpoints/<6-tuple>/.../checkpoint-*.
link_ckpt "${POLICY_ROOT}/GR00T_N17" "${COTRAIN_6}" \
    "${FINAL_CKPT}/GR00T_N17/${COTRAIN_6}"
link_ckpt "${POLICY_ROOT}/GR00T_N17" "shared/Cosmos-Reason2-2B" \
    "/mnt/xspark-data/xspark_shared/model_weights/Cosmos-Reason2-2B"

# Xiaomi_Robotics_0: ee cotrain (100); weights under <6-tuple>/last.ckpt/checkpoint/.
link_ckpt "${POLICY_ROOT}/Xiaomi_Robotics_0" "RoboDojo-cotrain-arx_x5-100-ee-0" \
    "${FINAL_CKPT}/Xiaomi_Robotics_0/RoboDojo-cotrain-arx_x5-100-ee-0"

# Pi_0 / Pi_0_Fast: reuse their own openpi-style checkpoint dirs (synced from Pi_05 flow).
link_ckpt "${POLICY_ROOT}/Pi_0" "RoboDojo_sim_arx_seed_0" \
    "${FINAL_CKPT}/Pi_0/RoboDojo_sim_arx_seed_0"
link_ckpt "${POLICY_ROOT}/Pi_0_Fast" "RoboDojo_sim_arx_seed_0" \
    "${FINAL_CKPT}/Pi_0/RoboDojo_sim_arx_seed_0"

echo "[DONE] checkpoint symlinks under ${POLICY_ROOT}"
