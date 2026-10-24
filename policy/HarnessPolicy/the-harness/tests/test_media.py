import base64
import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from harness_policy.core.contracts import Transition
from harness_policy.core.media import concat_episode_videos, decode_video_data_url, materialize_videos, redact_video_payloads
from harness_policy.core.trace import read_events
from benches.robodojo.video import encode_mp4

FAKE_MP4 = bytes.fromhex("000000186674797069736f6d0000000069736f6d61766331")
VIDEO_URL = "data:video/mp4;base64," + base64.b64encode(FAKE_MP4).decode("ascii")


class MediaTests(unittest.TestCase):
    def test_video_url_must_be_an_mp4(self):
        with self.assertRaisesRegex(ValueError, "MP4"):
            decode_video_data_url("data:image/png;base64,AAAA")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "camera"):
                materialize_videos(Path(directory), 0, {"a/b": VIDEO_URL})

    def test_control_frames_join_into_one_playable_clip(self):
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg is not installed")
        red = bytes([220, 40, 40]) * (16 * 16)
        blue = bytes([40, 80, 220]) * (16 * 16)
        first = encode_mp4([red, red], 16, 16, 25)
        second = encode_mp4([blue, blue], 16, 16, 25)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            materialize_videos(folder, 0, {"cam_head": "data:video/mp4;base64," + base64.b64encode(first).decode()})
            materialize_videos(folder, 1, {"cam_head": "data:video/mp4;base64," + base64.b64encode(second).decode()})
            joined = concat_episode_videos(folder)
            self.assertEqual(joined["cam_head"], "videos/cam_head.mp4")
            probe = subprocess.run(
                [shutil.which("ffprobe"), "-v", "error", "-count_frames", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
                 str(folder / "videos" / "cam_head.mp4")],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(probe.returncode, 0, probe.stderr)
            self.assertEqual(int(probe.stdout.strip()), 4)
        self.assertTrue(redact_video_payloads({"clip": VIDEO_URL})["clip"].startswith("data:video/mp4;base64,<"))


class RuntimeOutcomeVideoTests(unittest.TestCase):
    def test_failed_encode_is_reported_not_raised(self):
        from benches.robodojo.codec import CAMERAS
        from benches.robodojo.runtime import RoboDojoRuntimeEnvironment

        class Worker:
            resets = 0

            def _encode_clip(self):
                raise RuntimeError("ffmpeg is required to encode a RoboDojo motion clip")

            def _reset_clip(self):
                self.resets += 1

        worker = Worker()
        result = RoboDojoRuntimeEnvironment._outcome_videos(worker)
        self.assertEqual(result, {"video_error": "RuntimeError: ffmpeg is required to encode a RoboDojo motion clip"})
        self.assertEqual(worker.resets, 1)


class MotionClipTests(unittest.TestCase):
    def setUp(self):
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg is not installed")

    @staticmethod
    def frame(value, width=16, height=16):
        import numpy as np
        return np.full((height, width, 3), value, dtype=np.uint8)

    @staticmethod
    def count_frames(payload):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.mp4"
            path.write_bytes(payload)
            probe = subprocess.run(
                [shutil.which("ffprobe"), "-v", "error", "-count_frames", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                capture_output=True, text=True, check=False)
        return int(probe.stdout.strip())

    def test_frames_are_streamed_not_kept(self):
        from benches.robodojo.video import MotionClip

        clip = MotionClip()
        self.assertFalse(clip)
        for index in range(40):
            clip.add("cam_head", self.frame(index * 5), 25)
            clip.add("cam_left_wrist", self.frame(200), 25)
        self.assertTrue(clip)
        # Only the encoder handles are held, no frame data.
        self.assertEqual(set(clip._encoders), {"cam_head", "cam_left_wrist"})
        directory = Path(clip._directory)
        videos = clip.finish()
        self.assertEqual(set(videos), {"cam_head", "cam_left_wrist"})
        for payload in videos.values():
            self.assertEqual(payload[4:8], b"ftyp")
            self.assertEqual(self.count_frames(payload), 40)
        self.assertFalse(directory.exists())  # temporary files removed
        self.assertFalse(clip)
        self.assertEqual(clip.finish(), {})  # reusable once finished

    def test_discard_stops_encoders_and_removes_files(self):
        from benches.robodojo.video import MotionClip

        clip = MotionClip()
        clip.add("cam_head", self.frame(10), 25)
        process = clip._encoders["cam_head"][0]
        directory = Path(clip._directory)
        clip.discard()
        self.assertIsNotNone(process.poll())
        self.assertFalse(directory.exists())
        self.assertFalse(clip)

    def test_bad_frames_fail_at_finish_and_clean_up(self):
        from benches.robodojo.video import MotionClip

        clip = MotionClip()
        clip.add("cam_head", self.frame(10, width=15), 25)  # odd width
        with self.assertRaisesRegex(RuntimeError, "even numbers"):
            clip.finish()
        clip.add("cam_head", self.frame(10), 25)
        clip.add("cam_head", self.frame(10, width=32), 25)  # size changed mid-clip
        directory = Path(clip._directory)
        with self.assertRaisesRegex(RuntimeError, "size changed"):
            clip.finish()
        self.assertFalse(directory.exists())


class MotionCaptureRateTests(unittest.TestCase):
    def test_one_frame_per_control_step(self):
        from benches.robodojo.adapter import RoboDojoEnvironment

        class ObsManager:
            collect_interval = 10.0  # physics steps per control step

        class Env:
            obs_manager = ObsManager()

            def __init__(self):
                self.physics_steps = 0

            def step(self, env_idx_list, decimation=1):
                self.physics_steps += 1

        class Worker:
            _env = Env()
            _reset_clip = RoboDojoEnvironment._reset_clip
            _install_motion_video = RoboDojoEnvironment._install_motion_video

            def __init__(self):
                self.captured = []

            def _capture_motion_frame(self):
                self.captured.append(self._env.physics_steps)

        worker = Worker()
        worker._install_motion_video()
        for _ in range(25):
            worker._env.step([0])
        # Captured at the end of control steps 1 and 2 (physics steps 10 and 20), not every step.
        self.assertEqual(worker.captured, [10, 20])
        worker._reset_clip()  # a new clip starts counting again
        for _ in range(10):
            worker._env.step([0])
        self.assertEqual(worker.captured, [10, 20, 35])
