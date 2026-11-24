#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${XPL_ROOT}/utils/resolve_eval_env_type.sh"

case "$(resolve_eval_env_type)" in
    debug)
        # Use the standard XPolicyLab debug client, including its encoded mode.
        exec bash "${XPL_ROOT}/utils/setup_env_client.sh" "${XPL_ROOT}/utils" \
            "${SCRIPT_DIR}/deploy.yml" "${8}" "${10}" "${1}" "${2}" "${4}" \
            ReStart "${9}" "$(dirname "${XPL_ROOT}")" "${6}" "${7}" "${11:-127.0.0.1}"
        ;;
    sim)
        : "${UNIWAM_ROBOTWIN_ROOT:?Set the score-run RoboTwin checkout}"
        : "${UNIWAM_ROBOTWIN_TASK_CONFIG:?Select demo_clean or demo_randomized}"
        case "${UNIWAM_ROBOTWIN_TASK_CONFIG}" in
            demo_clean|demo_randomized) ;;
            *) echo "Choose the official RoboTwin clean or randomized condition." >&2; exit 2 ;;
        esac
        if [[ "${UNIWAM_TEST_NUM:-100}" != 100 || "${UNIWAM_EVAL_VIDEO_LOG:-true}" != true ]]; then
            echo "Use the original 100 episodes and enabled video logging." >&2
            exit 2
        fi
        export UNIWAM_TEST_NUM=100
        # The pinned RoboTwin loop writes under its checkpoint setting, which
        # the upstream UniWAM bridge names "xpolicylab".
        export UNIWAM_RESULT_PATH_TAG=xpolicylab
        exec bash "${SCRIPT_DIR}/../UniWAM/setup_eval_env_client.sh" "$@"
        ;;
    *)
        echo "ReStart supports EVAL_ENV_TYPE=debug and EVAL_ENV_TYPE=sim." >&2
        exit 2
        ;;
esac
