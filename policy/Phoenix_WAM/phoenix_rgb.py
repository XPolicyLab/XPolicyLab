"""Training-export CUDA resize and NV12-aligned crop for decoded RGB arrays.

FFmpeg transports raw pixels here; XPolicyLab still owns image-bit decoding.
Keep a process alive per input geometry to avoid a CUDA context startup per chunk.
"""
import atexit
from concurrent.futures import ThreadPoolExecutor
import os
import select
import shutil
import subprocess
import tempfile
import threading
import time
import weakref

import numpy as np


INPUT_GEOMETRY = 'ffmpeg_scale_cuda_bicubic_even_size_nv12_crop_256_v1'
_LIVE = weakref.WeakSet()


def resize_crop_geometry(width, height):
    """Match robodojo_v30_resize_256.py and crop's default 4:2:0 alignment."""
    if min(width, height) <= 0 or width % 2 or height % 2:
        raise ValueError('NV12 preprocessing requires positive even image dimensions')
    if width >= height:
        scaled_h, scaled_w = 256, int(round(width * 256 / height))
        scaled_w += scaled_w % 2
    else:
        scaled_w, scaled_h = 256, int(round(height * 256 / width))
        scaled_h += scaled_h % 2
    # FFmpeg crop rounds the centered offset down to the NV12 chroma grid.
    x = ((scaled_w - 256) // 2) & ~1
    y = ((scaled_h - 256) // 2) & ~1
    return scaled_w, scaled_h, x, y


def cuda_filter(width, height, device):
    w, h, x, y = resize_crop_geometry(width, height)
    return (f'format=nv12,hwupload_cuda=device={int(device)},'
            f'scale_cuda={w}:{h}:interp_algo=bicubic,hwdownload,format=nv12,'
            f'crop=256:256:{x}:{y}:exact=1,format=rgb24')


class _RGBPipe:
    def __init__(self, width, height, device, executable, timeout):
        self.timeout = timeout
        self.errors = tempfile.TemporaryFile()
        command = [executable, '-hide_banner', '-loglevel', 'error',
                   '-threads', '1', '-filter_threads', '1',
                   '-probesize', '32', '-analyzeduration', '0',
                   '-f', 'rawvideo', '-pixel_format', 'rgb24',
                   '-video_size', f'{width}x{height}', '-i', 'pipe:0',
                   '-vf', cuda_filter(width, height, device),
                   '-threads', '1', '-vsync', '0', '-flush_packets', '1',
                   '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=self.errors, bufsize=0)
        except BaseException:
            self.errors.close()
            raise
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix='phoenix-rgb')

    def _write(self, data):
        view = memoryview(data)
        while view:
            count = self.process.stdin.write(view)
            if not count:
                raise BrokenPipeError('FFmpeg stopped consuming RGB frames')
            view = view[count:]

    def transform(self, frames):
        deadline = time.monotonic() + self.timeout
        pending = self.writer.submit(self._write, np.stack(frames).tobytes())
        expected = len(frames) * 256 * 256 * 3
        output = bytearray()
        try:
            while len(output) < expected:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                    raise TimeoutError('FFmpeg CUDA image preprocessing timed out')
                chunk = os.read(self.process.stdout.fileno(), expected - len(output))
                if not chunk:
                    raise RuntimeError('FFmpeg ended before returning all RGB frames')
                output.extend(chunk)
            pending.result(timeout=max(.001, deadline - time.monotonic()))
        except BaseException as exc:
            self.close_process()
            self.errors.seek(0)
            detail = self.errors.read().decode('utf-8', errors='replace')[-3000:]
            raise RuntimeError('CUDA RGB preprocessing failed; install FFmpeg with scale_cuda '
                               f'and hwupload_cuda. No PIL/CPU fallback is used. {detail}') from exc
        # This is raw RGB pipe output, never encoded JPEG/PNG image bits.
        return np.ndarray((len(frames), 256, 256, 3), dtype=np.uint8, buffer=output).copy()

    def close_process(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            if stream and not stream.closed:
                stream.close()
        self.writer.shutdown(wait=True)

    def close(self):
        self.close_process()
        self.errors.close()


class CudaRGBPreprocessor:
    def __init__(self, device=0, *, executable=None, timeout=30):
        if int(device) < 0 or timeout <= 0:
            raise ValueError('Expected a nonnegative CUDA device and positive timeout')
        self.device = int(device)
        self.executable = executable or os.environ.get('PHOENIX_FFMPEG', 'ffmpeg')
        self.timeout = timeout
        self.pipes = {}
        self.lock = threading.Lock()
        _LIVE.add(self)

    def __call__(self, frames):
        groups, output = {}, [None] * len(frames)
        for index, rgb in enumerate(frames):
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError('Expected uint8 HWC RGB arrays')
            h, w = rgb.shape[:2]
            resize_crop_geometry(w, h)
            if (w, h) == (256, 256):
                output[index] = rgb  # Already-prepared training frames remain unchanged.
            else:
                groups.setdefault((w, h), []).append(index)
        with self.lock:
            for (w, h), indices in groups.items():
                key = (w, h)
                if key not in self.pipes:
                    executable = shutil.which(self.executable)
                    if executable is None:
                        raise RuntimeError('FFmpeg with scale_cuda is required; set PHOENIX_FFMPEG')
                    # Bound CUDA contexts if a client changes sensor resolutions repeatedly.
                    if len(self.pipes) >= 3:
                        self.pipes.pop(next(iter(self.pipes))).close()
                    self.pipes[key] = _RGBPipe(w, h, self.device, executable, self.timeout)
                try:
                    converted = self.pipes[key].transform([frames[i] for i in indices])
                except BaseException:
                    self.pipes.pop(key).close()
                    raise
                for i, rgb in zip(indices, converted):
                    output[i] = rgb
        return output

    def close(self):
        with self.lock:
            for pipe in self.pipes.values():
                pipe.close()
            self.pipes.clear()

    def __del__(self):
        if hasattr(self, 'lock'):
            self.close()


@atexit.register
def _close_pipes():
    for instance in list(_LIVE):
        instance.close()
