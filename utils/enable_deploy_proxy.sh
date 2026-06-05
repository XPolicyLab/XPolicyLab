#!/bin/bash
# Enable HTTP(S) proxy for HuggingFace / external downloads on deploy hosts.
# Matches interactive alias: proxyup in ~/.bashrc
set -euo pipefail

PROXY_HOST="${DEPLOY_PROXY_HOST:-192.168.16.76}"
PROXY_PORT="${DEPLOY_PROXY_PORT:-18000}"
PROXY_URL="http://${PROXY_HOST}:${PROXY_PORT}"

export http_proxy="${PROXY_URL}"
export https_proxy="${PROXY_URL}"
export HTTP_PROXY="${PROXY_URL}"
export HTTPS_PROXY="${PROXY_URL}"

echo "[PROXY] http_proxy=${http_proxy}"
