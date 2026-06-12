from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPER_CONTROLLER = REPO_ROOT / "src/robot/controller/Piper_controller.py"
DUAL_PIPER_ORBBEC = REPO_ROOT / "src/robot/robot/dual_piper_orbbec.py"


def test_piper_controller_exposes_cleanup_for_dual_piper_orbbec() -> None:
    assert "def cleanup(self):" in PIPER_CONTROLLER.read_text(encoding="utf-8")
    assert "controller.cleanup()" in DUAL_PIPER_ORBBEC.read_text(encoding="utf-8")
