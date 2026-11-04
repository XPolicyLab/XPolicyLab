# Shared by all train/eval launchers.
# DataLoader multiprocessing writes tiny pymp-* listener dirs under TMPDIR
# (locks/sockets for fd handoff, not videos or checkpoints).
# Prefer FLEXWAM_TMPDIR; otherwise $HOME/tmp so overlay /tmp is unused.
_pick_tmpdir() {
  if [ -n "${FLEXWAM_TMPDIR:-}" ]; then
    printf '%s\n' "${FLEXWAM_TMPDIR}"
    return
  fi
  printf '%s\n' "${HOME}/tmp"
}
export TMPDIR="$(_pick_tmpdir)"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
mkdir -p "${TMPDIR}"
