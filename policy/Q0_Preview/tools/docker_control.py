"""Bound concurrent Docker cold starts; tolerate slow daemon cleanup without hiding errors."""
import fcntl
from functools import wraps
from pathlib import Path
import subprocess
import time


def acquire_startup_slot(timeout=300., slots=4):
    folder=Path('/tmp/q0-preview-docker-start');folder.mkdir(exist_ok=True)
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        for i in range(slots):
            lock=(folder/f'{i}.lock').open('a')
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                return lock
            except BlockingIOError:lock.close()
        time.sleep(.2)
    raise TimeoutError('Docker cold-start admission timed out')


def startup_limited(fn):
    @wraps(fn)
    def wrapped(*args,**kwargs):
        deadline=kwargs.get('deadline',args[1] if len(args)>1 else None)
        timeout=min(300.,max(0.,deadline-time.time())) if deadline is not None else 300.
        with acquire_startup_slot(timeout=timeout):return fn(*args,**kwargs)
    return wrapped


def remove_container(name):
    # Docker may acknowledge removal while cleanup is still in progress.
    # Wait until this invocation's named container has actually disappeared.
    until=time.monotonic()+240
    last=''
    while time.monotonic()<until:
        try:
            subprocess.run(['docker','rm','-f',name],capture_output=True,text=True,timeout=30)
            inspected=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=15)
        except subprocess.TimeoutExpired:
            time.sleep(1);continue
        if inspected.returncode!=0 and 'no such' in inspected.stderr.lower():return
        last=inspected.stderr
        time.sleep(1)
    raise RuntimeError('Docker container cleanup incomplete: '+name+' '+last[-200:])
