"""Geometry regressions and opt-in comparison with the upstream FFmpeg filter."""
import os
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from .phoenix_rgb import CudaRGBPreprocessor, resize_crop_geometry


class RGBTests(unittest.TestCase):
    def test_sensor_geometry_and_nv12_offsets(self):
        self.assertEqual(resize_crop_geometry(640, 480), (342, 256, 42, 0))
        self.assertEqual(resize_crop_geometry(480, 640), (256, 342, 0, 42))
        self.assertEqual(resize_crop_geometry(1280, 720), (456, 256, 100, 0))
        self.assertEqual(resize_crop_geometry(256, 256), (256, 256, 0, 0))
        for width, height in [(0, 480), (641, 480), (640, 479)]:
            with self.assertRaises(ValueError):
                resize_crop_geometry(width, height)

    def test_prepared_frames_unchanged_without_ffmpeg(self):
        frames = [np.random.default_rng(i).integers(0, 256, (256, 256, 3), dtype=np.uint8)
                  for i in range(3)]
        preprocessor = CudaRGBPreprocessor(executable='/does/not/exist')
        frames[0].flags.writeable = False
        with patch('subprocess.Popen', side_effect=AssertionError('Unexpected FFmpeg launch')):
            for actual, expected in zip(preprocessor(frames), frames):
                np.testing.assert_array_equal(actual, expected)
        preprocessor.close()

    def test_missing_ffmpeg_never_uses_approximate_fallback(self):
        preprocessor = CudaRGBPreprocessor(executable='/does/not/exist')
        with self.assertRaisesRegex(RuntimeError, 'FFmpeg with scale_cuda'):
            preprocessor([np.zeros((480, 640, 3), np.uint8)])
        preprocessor.close()

    def test_ffmpeg_failure_closes_pipe(self):
        preprocessor = CudaRGBPreprocessor(executable='false', timeout=2)
        with self.assertRaisesRegex(RuntimeError, 'No PIL/CPU fallback'):
            preprocessor([np.zeros((480, 640, 3), np.uint8)])
        self.assertEqual(preprocessor.pipes, {})
        preprocessor.close()

    @unittest.skipUnless(os.environ.get('PHOENIX_TEST_CUDA_RGB') == '1',
                         'Set PHOENIX_TEST_CUDA_RGB=1 on a CUDA/FFmpeg host')
    def test_exact_upstream_filter_and_persistent_frame_order(self):
        executable = os.environ.get('PHOENIX_FFMPEG', 'ffmpeg')
        preprocessor = CudaRGBPreprocessor(executable=executable)
        try:
            # Independent reference retains upstream default interpolation and
            # centered crop expression; FFmpeg itself applies NV12 rounding.
            for width, height, scaled_w, scaled_h in [(640,480,342,256),
                                                       (480,640,256,342),
                                                       (1280,720,456,256)]:
                for iteration in range(2):
                    frames = [np.random.default_rng(100*iteration+i).integers(
                        0, 256, (height,width,3), dtype=np.uint8) for i in range(3)]
                    frames[0][:, :width//4] = [255, 0, 0]
                    frames[1][:, width//2:] = [0, 255, 0]
                    frames[2][:height//2] = [0, 0, 255]
                    raw = np.stack(frames).tobytes()
                    reference_filter = (
                        f'format=nv12,hwupload_cuda=device=0,scale_cuda={scaled_w}:{scaled_h},'
                        'hwdownload,format=nv12,crop=256:256:(iw-256)/2:(ih-256)/2,format=rgb24')
                    command = [executable, '-hide_banner', '-loglevel', 'error',
                               '-threads', '1', '-filter_threads', '1',
                               '-f', 'rawvideo', '-pixel_format', 'rgb24',
                               '-video_size', f'{width}x{height}', '-i', 'pipe:0',
                               '-vf', reference_filter, '-frames:v', '3', '-threads', '1',
                               '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
                    reference = subprocess.run(command, input=raw, capture_output=True,
                                               check=True, timeout=30).stdout
                    actual = np.stack(preprocessor(frames))
                    self.assertEqual(actual.tobytes(), reference)
            mixed = [frames[0], np.full((256,256,3), 123, np.uint8), frames[2]]
            result = preprocessor(mixed)
            np.testing.assert_array_equal(result[0], actual[0])
            np.testing.assert_array_equal(result[1], mixed[1])
            np.testing.assert_array_equal(result[2], actual[2])
        finally:
            preprocessor.close()


if __name__ == '__main__':
    unittest.main()
