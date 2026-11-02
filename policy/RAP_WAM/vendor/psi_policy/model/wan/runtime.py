from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _candidate_code_roots() -> list[Path]:
    candidates: list[Path] = [
        _repo_root() / "third_party" / "wan_official",
    ]
    for env_name in ("WAN_OFFICIAL_CODE_ROOT", "WAN_CODE_ROOT"):
        raw_value = os.environ.get(env_name)
        if raw_value:
            candidates.append(Path(raw_value).expanduser().resolve())
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_text = str(candidate)
        if candidate_text in seen:
            continue
        seen.add(candidate_text)
        deduped.append(candidate)
    return deduped


def _ensure_wan_code_available() -> None:
    for root in _candidate_code_roots():
        if not root.exists():
            continue
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        try:
            import wan  # noqa: F401
            return
        except Exception:
            continue

    try:
        import wan  # noqa: F401
        return
    except Exception:
        pass

    tried = ", ".join(str(path) for path in _candidate_code_roots())
    raise ModuleNotFoundError(
        "Could not import official `wan` runtime modules. "
        "Expected a vendored copy under third_party/wan_official or "
        "set WAN_OFFICIAL_CODE_ROOT / WAN_CODE_ROOT to the directory containing the `wan` package. "
        f"Tried: {tried}"
    )


_ensure_wan_code_available()

from wan.modules.model import WanModel  # type: ignore  # noqa: E402
from wan.modules.t5 import T5EncoderModel  # type: ignore  # noqa: E402
from wan.modules.vae2_2 import Wan2_2_VAE  # type: ignore  # noqa: E402

__all__ = ["WanModel", "Wan2_2_VAE", "T5EncoderModel"]
