#!/bin/bash
# cuRobo v0.7.8 for RoboTwin, built in place and used only by RoboTwin processes.
#
# RoboTwin's planners (and the expert that selects evaluation seeds) are written
# for cuRobo v0.7.8; the environment's cuRobo is RoboDojo's newer checkout with a
# different package layout, and both import as `curobo`. So v0.7.8 is not
# installed: its CUDA extensions are built in its source tree and RoboTwin workers
# put `$HARNESS_ROBOTWIN_CUROBO/src` first on PYTHONPATH.
#
# torch is built with CUDA 12.8; the host toolkit is 13.0, which torch refuses for
# extensions. A CUDA 12.8 build toolkit (nvcc, cudart, cccl) is assembled from
# NVIDIA's redist archives (sha256-checked). Nothing here changes the environment.
#
#   bash environment/robotwin_curobo.sh    (setup.sh --only robotwin runs it)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUROBO="${HARNESS_ROBOTWIN_CUROBO:-$ROOT/third_party/curobo-0.7.8}"
TOOLKIT="${HARNESS_CUDA_BUILD_TOOLKIT:-$ROOT/third_party/cuda-12.8}"
REDIST="${HARNESS_CUDA_REDIST:-https://developer.download.nvidia.com/compute/cuda/redist}"

if ls "$CUROBO"/src/curobo/curobolib/*.so >/dev/null 2>&1; then
  echo "cuRobo v0.7.8 already built in $CUROBO"
  exit 0
fi
[ -d "$CUROBO" ] || git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$CUROBO"

if [ ! -x "$TOOLKIT/bin/nvcc" ]; then
  mkdir -p "$TOOLKIT" "$TOOLKIT.downloads"
  while read -r path sha; do
    file="$TOOLKIT.downloads/$(basename "$path")"
    [ -f "$file" ] || curl -sS -L -o "$file" "$REDIST/$path"
    echo "$sha  $file" | sha256sum -c -
    tar -xJf "$file" -C "$TOOLKIT.downloads"
    cp -a "${file%.tar.xz}"/. "$TOOLKIT"/
  done <<LIST
cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz 9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3
cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz 8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68
cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz 0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b
LIST
fi

cd "$CUROBO"
CUDA_HOME="$TOOLKIT" PATH="$TOOLKIT/bin:$PATH" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}" \
  python setup.py build_ext --inplace
ls "$CUROBO"/src/curobo/curobolib/*.so
