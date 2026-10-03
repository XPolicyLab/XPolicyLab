#!/usr/bin/env bash
set -euo pipefail
policy_python=${1:?Usage: install.sh /path/to/policy/python}
exec "$policy_python" -m pip install 'numpy>=1.26,<3' 'Pillow>=10,<13' 'pyyaml>=6,<7' 'shapely>=2,<3' 'rich>=13,<15' 'prompt-toolkit>=3,<4' 'jsonschema>=4.23,<5'
