#!/usr/bin/env bash
set -euo pipefail
policy_python=${1:?Usage: install.sh /path/to/policy/python}
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
xpl_root="$(cd "${script_dir}/../.." && pwd)"
packages=(-e "${xpl_root}" "${script_dir}/runtime" 'numpy>=1.26,<2' 'Pillow>=10,<13' 'pyyaml>=6,<7' 'shapely>=2,<3' 'rich>=13,<15' 'prompt-toolkit>=3,<4' 'jsonschema>=4.23,<5' 'scikit-learn>=1.6,<2' 'transforms3d>=0.4,<0.5' 'trimesh>=4,<5' 'warp-lang==1.17.0' 'pin==2.7.0' 'yourdfpy==0.0.60' 'numpy-quaternion==2024.0.13' 'iopath==0.1.10' 'ftfy==6.1.1' 'timm==1.0.17' 'pycocotools')
if "$policy_python" -m pip --version >/dev/null 2>&1; then
    exec "$policy_python" -m pip install "${packages[@]}"
fi
exec uv pip install --python "$policy_python" "${packages[@]}"
