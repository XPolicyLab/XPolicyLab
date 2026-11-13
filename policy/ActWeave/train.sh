#!/bin/bash
set -euo pipefail
printf '%s\n' 'ActWeave uses frozen external models; weight training is not supported.' >&2
exit 2
