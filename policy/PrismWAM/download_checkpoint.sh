#!/usr/bin/env bash
# Download a released model folder into checkpoints/<name>/.
#   bash download_checkpoint.sh prismwam/robotwin|prismwam/robodojo|prismwam_star/robotwin
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
name="${1:-prismwam/robotwin}"
case "$name" in
    prismwam/robotwin) expected=c6ddd4ed984f69544ccd4ea1c9eadf171a709b1019cf51576c5b9f9af1581143 ;;
    prismwam/robodojo) expected=ddbbc70515a03d876740110e3bb6112a50bbd8724dceeabc085dd28ab04cdbe0 ;;
    prismwam_star/robotwin) expected=bd9c95e04d760f06db040a4bc58b3752defdabd754ccc59e73fe3e25463d69bf ;;
    *) echo 'Usage: download_checkpoint.sh prismwam/robotwin|prismwam/robodojo|prismwam_star/robotwin' >&2; exit 2 ;;
esac
model="${name%%/*}"
folder="${name#*/}"
# Each model has its own repository; both store one folder per benchmark.
case "$model" in
    prismwam) repo=THUSI-Lab/PrismWAM ;;
    prismwam_star) repo=THUSI-Lab/PrismWAM-Star ;;
esac
run_command hf download "$repo" --include "$folder/*" --local-dir "$POLICY_DIR/checkpoints/$model"
is_dry_run && exit 0
actual=$(sha256sum "$POLICY_DIR/checkpoints/$name/policy.pt" | awk '{print $1}')
[[ "$actual" == "$expected" ]] || { echo "policy.pt SHA256 mismatch: $actual" >&2; exit 1; }
echo "checkpoints/$name is ready."
