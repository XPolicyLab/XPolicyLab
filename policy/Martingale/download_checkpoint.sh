#!/bin/bash
# Copyright © 2026 Martingale. All Rights Reserved.
set -euo pipefail

dest=${1:-"${XDG_CACHE_HOME:-$HOME/.cache}/martingale"}
revision=3c879f39826c281e95690f02c7821c4de09afae7
sha256=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
mkdir -p -- "$dest"
hf download facebook/sam3 sam3.pt --revision "$revision" --local-dir "$dest"
(cd "$dest" && echo "$sha256  sam3.pt" | sha256sum --check)
