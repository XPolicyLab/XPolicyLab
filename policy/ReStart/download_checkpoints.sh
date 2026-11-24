#!/usr/bin/env bash
set -euo pipefail

destination=${1:?Usage: bash download_checkpoints.sh DESTINATION}
hf download chenpyyy/UniWAM-base mp_rank_00_model_states.pt \
    --revision 1b0fa99836f8311475baaffae0c35f1c86facb86 \
    --local-dir "$destination/UniWAM-base/pytorch_model"
hf download chenpyyy/UniWAM-robotwin-clean config.json \
    --revision f805f8148b779aa833bed2f83c148bdd22d9a978 \
    --local-dir "$destination/UniWAM-base"
hf download jingang/TabICL tabicl-regressor-v2-20260212.ckpt \
    --revision 4dcd344ece2c00be9e831fdd35bed57b5ad83e19 \
    --local-dir "$destination/TabICLv2"

python3 - "$destination" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
weight = "UniWAM-base/pytorch_model/mp_rank_00_model_states.pt"
expected = {
    weight: "7017c83c20493d7ab1ca257c483113112b58e14e929503ed464f27931a2c0bb0",
    "UniWAM-base/config.json": "66b99f21ddb7945fe595e67bf51fc9466636903d043819338e0cc914caaca4d5",
    "TabICLv2/tabicl-regressor-v2-20260212.ckpt": "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a",
}
for name, fingerprint in expected.items():
    digest = hashlib.sha256()
    with (root / name).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != fingerprint:
        raise SystemExit(f"SHA256 mismatch: {name}")
    print(f"Verified: {name}")

receipt = root / "UniWAM-base/embodpfn-asset-receipt.json"
receipt.write_text(json.dumps({
    "verified_weight_sha256": expected[weight],
    "cloud_path": str(root / weight),
}, indent=2) + "\n")
print(f"Checkpoint receipt: {receipt}")
PY
