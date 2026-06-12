from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ORBBEC_MULTI_SENSOR = REPO_ROOT / "src/robot/sensor/orbbec_multi_sensor.py"


def test_shutdown_disables_multi_device_sync_before_dropping_context() -> None:
    source = ORBBEC_MULTI_SENSOR.read_text(encoding="utf-8")
    assert "enable_multi_device_sync(0)" in source
    assert "failed to disable multi-device sync during shutdown" in source


@pytest.mark.usefixtures("orbbec_session_module")
def test_shutdown_calls_disable_multi_device_sync(orbbec_session_module) -> None:
    OrbbecMultiSensorSession = orbbec_session_module.OrbbecMultiSensorSession
    session = OrbbecMultiSensorSession("test-session")
    context = MagicMock()
    session.context = context
    session._configured = True
    session.devices = {"CP2N163000AA": object()}

    session._shutdown()

    context.enable_multi_device_sync.assert_called_once_with(0)
    assert session.context is None
    assert session._configured is False
    assert not session.devices


@pytest.mark.usefixtures("orbbec_session_module")
def test_acquire_rolls_back_ref_count_when_configure_fails(orbbec_session_module) -> None:
    OrbbecMultiSensorSession = orbbec_session_module.OrbbecMultiSensorSession
    robot_config = {
        "CAMERA_SERIALS": {"head": "CP2N163000AA"},
        "CAMERA_SYNC": {"enabled": True, "use_async_frames": True},
    }

    with patch.object(
        OrbbecMultiSensorSession,
        "configure",
        side_effect=RuntimeError("configure failed"),
    ):
        with pytest.raises(RuntimeError, match="configure failed"):
            OrbbecMultiSensorSession.acquire(robot_config)

    assert OrbbecMultiSensorSession._sessions == {}


@pytest.fixture
def orbbec_session_module():
    import sys

    src_root = str(REPO_ROOT / "src")
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    from robot.sensor import orbbec_multi_sensor

    orbbec_multi_sensor.OrbbecMultiSensorSession._sessions.clear()
    yield orbbec_multi_sensor
    orbbec_multi_sensor.OrbbecMultiSensorSession._sessions.clear()
