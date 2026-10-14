"""Optional video worker limits without editing native RoboDojo sources."""

from pathlib import Path


class _VideoSubprocess:
    def __init__(self, original, threads):
        self.original = original
        self.threads = threads

    def __getattr__(self, name):
        return getattr(self.original, name)

    def Popen(self, command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and Path(command[0]).name == 'ffmpeg':
            # Input decoding, pixel-format conversion, and output encoding have
            # independent pools. Preserve every native format/quality option.
            count = str(self.threads)
            command = [command[0], '-threads', count, '-filter_threads', count,
                       *command[1:-1], '-threads', count, command[-1]]
        return self.original.Popen(command, *args, **kwargs)


def configure_video_threads(module, threads):
    """Scope the override to utils.save_file; never patch global subprocess."""
    if threads <= 0:
        raise ValueError('Video thread count must be positive')
    original = module.subprocess
    if isinstance(original, _VideoSubprocess):
        original = original.original
    module.subprocess = _VideoSubprocess(original, threads)
