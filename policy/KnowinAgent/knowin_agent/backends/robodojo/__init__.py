"""RoboDojo (Isaac Sim, dual ARX-X5) backend. Simulator modules are imported only when booting."""

from knowin_agent.backends.robodojo.calibration import CalibrationError, OpticalCalibration, calibrate_cameras
from knowin_agent.backends.robodojo.env import RoboDojoBackend
from knowin_agent.backends.robodojo.kinematics import ArxKinematics
from knowin_agent.backends.robodojo.layouts import available_scenes, decode_scene, encode_seed, resolve_root
from knowin_agent.backends.robodojo.native import LayoutUnstable
from knowin_agent.backends.robodojo.profile import arx_x5_profile
from knowin_agent.backends.robodojo.scoring import read_score
from knowin_agent.backends.robodojo.video import FrameRecorder, write_videos

__all__ = [
    "ArxKinematics",
    "CalibrationError",
    "FrameRecorder",
    "LayoutUnstable",
    "OpticalCalibration",
    "RoboDojoBackend",
    "arx_x5_profile",
    "available_scenes",
    "calibrate_cameras",
    "decode_scene",
    "encode_seed",
    "read_score",
    "resolve_root",
    "write_videos",
]
