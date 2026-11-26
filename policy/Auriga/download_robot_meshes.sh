#!/bin/bash
# ARX X5 meshes for policy-side robot geometry (robot_model manifest, gripper collision bounds,
# robot-mesh overlays). Unmodified RoboDojo assets from the Hugging Face dataset
# RoboDojo-Benchmark/RoboDojo at a pinned revision; every file is checked against
# vendor/robodojo_planner/Assets/Robots/x5/meshes.sha256 (the dataset's LFS object ids). No token is needed.
# They sit next to the vendored X5A.urdf, which the planner, the calibration FK and Auriga share.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REVISION="${ROBODOJO_ASSETS_REVISION:-63fcc045f2e974902eb7800d29de48004cf86016}"
BASE_URL="${HF_ENDPOINT:-https://huggingface.co}/datasets/RoboDojo-Benchmark/RoboDojo/resolve/${REVISION}/Assets/Robots/x5/meshes"
X5_DIR="${SCRIPT_DIR}/vendor/robodojo_planner/Assets/Robots/x5"
DEST="${X5_DIR}/meshes"
mkdir -p "${DEST}"
cd "${DEST}"
while read -r sha name; do
    [[ -z "${name}" ]] && continue
    if [[ -f "${name}" ]] && echo "${sha}  ${name}" | sha256sum -c --status; then
        continue
    fi
    curl -fsSL --retry 5 -o "${name}.part" "${BASE_URL}/${name}"
    mv "${name}.part" "${name}"
done < "${X5_DIR}/meshes.sha256"
sha256sum -c --quiet "${X5_DIR}/meshes.sha256"
echo "[download_robot_meshes] ARX X5 meshes verified in ${DEST}"
