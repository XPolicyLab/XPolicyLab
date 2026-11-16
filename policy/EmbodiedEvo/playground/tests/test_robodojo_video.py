import subprocess
import types
import unittest
from unittest.mock import Mock

from simulators.robodojo_video import configure_video_threads


class VideoThreadTests(unittest.TestCase):
    def test_scoped_limits_preserve_video_parameters_and_pipe(self):
        original = Mock(PIPE=subprocess.PIPE)
        module = types.SimpleNamespace(subprocess=original)
        command = ['ffmpeg', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                   '-video_size', '640x480', '-framerate', '30', '-i', '-',
                   '-pix_fmt', 'yuv420p', '-vcodec', 'libx264', '-crf', '23', 'out.mp4']
        configure_video_threads(module, 2)
        configure_video_threads(module, 2)
        module.subprocess.Popen(command, stdin=module.subprocess.PIPE)
        actual = original.Popen.call_args.args[0]
        self.assertEqual(actual[5:-3], command[1:-1])
        self.assertEqual(actual[:5], ['ffmpeg', '-threads', '2', '-filter_threads', '2'])
        self.assertEqual(actual[-3:], ['-threads', '2', 'out.mp4'])
        self.assertEqual(original.Popen.call_args.kwargs, {'stdin': subprocess.PIPE})
        self.assertEqual(command[1], '-f')
        self.assertIsInstance(subprocess.Popen, type)


if __name__ == '__main__':
    unittest.main()
