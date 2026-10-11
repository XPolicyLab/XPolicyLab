"""Host-wide four-slot lease; independent of the existing two-slots-per-GPU leases."""
from pathlib import Path

class Admission:
    def __init__(self, root):
        self.root=Path(root);self.handle=None
    def __enter__(self):
        import fcntl
        self.root.mkdir(parents=True,exist_ok=True)
        for i in range(4):
            handle=(self.root/f'adaptive-global-{i}.lock').open('a+')
            try:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.handle=handle;return self
            except BlockingIOError:handle.close()
        raise RuntimeError('Adaptive concurrency limit is four; no fifth launch')
    def __exit__(self,*args):
        if self.handle:self.handle.close()
