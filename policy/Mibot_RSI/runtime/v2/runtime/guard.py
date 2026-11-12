"""In-process file, process and network guard for model-written Python.

A Python audit hook refuses reads outside the readable roots, writes outside the
writable roots, directory listings outside the readable roots, new processes
other than allowed executables (and URLs in their arguments), network access
other than the episode's robot service, ctypes, and walking the garbage
collector's object lists. It prevents accidental or casual access to evaluation
results, task code, other runs and the network; it is not an OS sandbox. Code that
sets out to defeat it can: the guard object stays reachable from the session's
functions, Python does not audit replacing functions or changing attributes, and
_posixsubprocess.fork_exec starts processes without an audit event.

On Linux kernels with Landlock the worker also runs under a Landlock sandbox (landlock.py) that the kernel enforces
for native code too: files, TCP connections and executables, but not MPTCP, UDP or Unix sockets, or file metadata.
On a kernel without Landlock this guard is the only layer.
"""
from __future__ import annotations

import contextlib
import json
import os
import site
import sys
import sysconfig
import time

WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
PROCESS_EVENTS = ('subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn', 'os.spawn',
                  'os.startfile', 'os.fork', 'os.forkpty', 'pty.spawn')
LOCAL_HOSTS = frozenset({'127.0.0.1', 'localhost', '::1'})
NETWORK_EVENTS = ('socket.connect', 'socket.sendto', 'socket.sendmsg')
NAME_EVENTS = ('socket.getaddrinfo', 'socket.gethostbyname', 'socket.gethostbyaddr', 'socket.getnameinfo')
INTROSPECTION_EVENTS = ('gc.get_objects', 'gc.get_referrers', 'gc.get_referents')
MEDIA_TOOLS = frozenset({'ffmpeg', 'ffprobe'})
WRITE_EVENTS = {'os.remove': (0,), 'os.rmdir': (0,), 'os.mkdir': (0,), 'os.chmod': (0,),
                'os.chown': (0,), 'os.truncate': (0,), 'os.utime': (0,), 'shutil.rmtree': (0,),
                'os.rename': (0, 1), 'os.replace': (0, 1), 'os.link': (0, 1), 'os.symlink': (0, 1)}


def system_read_roots():
    """Interpreter, library and OS locations that imports and system calls need."""
    # Only this process's own /proc entries: /proc/<pid>/environ of other processes
    # (such as the harness parent) must stay unreadable.
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix,
             '/usr', '/lib', '/lib64', '/etc', '/sys', '/dev',
             f'/proc/{os.getpid()}', '/proc/self', '/proc/thread-self', '/proc/cpuinfo', '/proc/meminfo',
             '/proc/stat', '/proc/loadavg', '/proc/version', '/proc/sys', '/proc/mounts', '/proc/filesystems',
             '/System', '/Library', '/opt/homebrew', '/private/etc', '/private/var/db'}
    for name in ('stdlib', 'platstdlib', 'purelib', 'platlib'):
        path = sysconfig.get_paths().get(name)
        if path:
            roots.add(path)
    try:
        roots.update(site.getsitepackages())
    except AttributeError:
        pass
    user_site = site.getusersitepackages() if hasattr(site, 'getusersitepackages') else None
    if user_site:
        roots.add(user_site)
    return roots


class Guard:
    def __init__(self, read_roots, write_roots, executables=(), log_path=None, endpoints=(), helper_roots=()):
        self.write_roots = [self._norm(root) for root in write_roots]
        self.read_roots = [self._norm(root) for root in list(read_roots) + list(system_read_roots())
                           ] + self.write_roots
        self.executables = frozenset(os.path.basename(str(name)) for name in executables)
        # The only network peers: (host, port) of the episode's robot service on this machine.
        self.endpoints = frozenset((str(host), int(port)) for host, port in endpoints)
        # The only roots allow() may open for a while (the knowledge directory of the session's helpers).
        self.helper_roots = frozenset(self._norm(root) for root in helper_roots)
        self.log = open(log_path, 'a') if log_path else None
        self.blocked = 0

    @staticmethod
    def _norm(path):
        return os.path.realpath(os.fsdecode(path))

    @staticmethod
    def _under(path, roots):
        return any(path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in roots)

    def _deny(self, event, target, reason='is outside the allowed paths for this session'):
        self.blocked += 1
        if self.log is not None:
            self.log.write(json.dumps({'time': time.time(), 'event': event, 'target': str(target)}) + '\n')
            self.log.flush()
        raise PermissionError(f'v2 guard: {event} {target} {reason}')

    @contextlib.contextmanager
    def allow(self, *roots):
        """Temporarily allow writes under roots, for harness helpers such as knowledge writes; only roots
        named at install (helper_roots) can be opened."""
        extra = [self._norm(root) for root in roots]
        if any(root not in self.helper_roots for root in extra):
            raise PermissionError('v2 guard: only the session\'s helper directories can be opened for writing')
        self.write_roots.extend(extra)
        self.read_roots.extend(extra)
        try:
            yield
        finally:
            for root in extra:
                self.write_roots.remove(root)
                self.read_roots.remove(root)

    def check_read(self, event, path):
        if isinstance(path, int) or path is None:
            return
        resolved = self._norm(path)
        if not self._under(resolved, self.read_roots):
            self._deny(event, resolved)

    def check_write(self, event, path):
        if isinstance(path, int) or path is None:
            return
        resolved = self._norm(path)
        if not self._under(resolved, self.write_roots):
            self._deny(event, resolved)

    def check_process(self, event, args):
        executable = None
        if event == 'subprocess.Popen' and args:
            executable = args[0]
            if executable is None and len(args) > 1:
                command = args[1]
                if isinstance(command, (list, tuple)) and command:
                    executable = command[0]
                elif isinstance(command, (str, bytes)):
                    executable = os.fsdecode(command).split()[0] if command.strip() else None
        elif event in ('os.exec', 'os.posix_spawn') and args:
            executable = args[0]
        elif event == 'os.spawn' and len(args) > 1:
            executable = args[1]
        name = os.path.basename(os.fsdecode(executable)) if executable is not None else None
        if name is None or name not in self.executables:
            self._deny(event, name or 'process')
        # ffmpeg and ffprobe open URLs, playlists and concat lists on the model's behalf: they run only when
        # told to open nothing but local files and pipes (the whitelist also binds nested inputs).
        if name in MEDIA_TOOLS:
            position = 2 if event == 'os.spawn' else 1
            argv = args[position] if len(args) > position else ()
            argv = os.fsdecode(argv).split() if isinstance(argv, (str, bytes)) else list(argv or ())
            words = [os.fsdecode(word) for word in argv if isinstance(word, (str, bytes, os.PathLike))]
            allowed = [words[i + 1] for i, word in enumerate(words[:-1]) if word == '-protocol_whitelist']
            if not allowed or any(set(value.split(',')) - {'file', 'pipe'} for value in allowed):
                self._deny(event, name, 'runs only with -protocol_whitelist file,pipe in this session')

    @staticmethod
    def _library_of_python(path):
        """A shared library shipped inside this interpreter's installation (site-packages), e.g. torch's."""
        if not isinstance(path, (str, bytes, os.PathLike)):
            return False
        resolved = os.path.realpath(os.fsdecode(path))
        prefixes = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)}
        return os.path.isabs(resolved) and any(resolved.startswith(p.rstrip(os.sep) + os.sep) for p in prefixes)

    def check_address(self, event, address):
        """Network peers: only the robot service; a connected socket's sends (no address) follow its connect."""
        if address is None and event != 'socket.connect':
            return
        if isinstance(address, tuple) and len(address) >= 2:
            host, port = address[0], address[1]
            host = host.decode(errors='replace') if isinstance(host, bytes) else str(host)
            if host in LOCAL_HOSTS and port in {p for _, p in self.endpoints}:
                return
            address = f'{host}:{port}'
        self._deny(event, address, 'is not allowed in this session (no network access)')

    def check_signal(self, event, args):
        """Signals only within the worker's own process group (itself and the processes it started):
        model code must not stop the harness or the simulator."""
        target = args[0] if args else None
        group = os.getpgrp()
        try:
            if event == 'os.killpg' and target == group:
                return
            if event == 'os.kill' and (target in (0, os.getpid()) or os.getpgid(target) == group):
                return
        except (OSError, TypeError):
            pass
        self._deny(event, f'pid {target}')

    def hook(self, event, args):
        if event == 'open':
            path, mode, flags = (list(args) + [None, None, None])[:3]
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                mode is None and isinstance(flags, int) and flags & WRITE_FLAGS)
            (self.check_write if writing else self.check_read)(event, path)
        elif event in ('os.listdir', 'os.scandir'):
            self.check_read(event, args[0] if args and args[0] is not None else '.')
        elif event == 'glob.glob' and args:
            pattern = os.fsdecode(args[0])
            base = pattern.split('*')[0].split('?')[0].split('[')[0]
            self.check_read(event, os.path.dirname(base) or '.')
        elif event in WRITE_EVENTS:
            for index in WRITE_EVENTS[event]:
                if len(args) > index:
                    self.check_write(event, args[index])
        elif event in PROCESS_EVENTS:
            self.check_process(event, args)
        elif event in ('os.kill', 'os.killpg'):
            self.check_signal(event, args)
        elif event in NETWORK_EVENTS:
            self.check_address(event, args[1] if len(args) > 1 else None)
        elif event in NAME_EVENTS:
            host = args[0] if args else None
            if event == 'socket.getnameinfo' and isinstance(host, tuple) and host:
                host = host[0]
            host = host.decode(errors='replace') if isinstance(host, bytes) else host
            if host not in LOCAL_HOSTS:
                self._deny(event, host, 'is not allowed in this session (no network access)')
        elif event == 'socket.bind':
            self._deny(event, args[1] if len(args) > 1 else 'socket', 'is not allowed in this session')
        elif event == 'ctypes.dlopen' and self._library_of_python(args[0] if args else None):
            return
        elif event.startswith('ctypes.'):
            self._deny(event, args[0] if event == 'ctypes.dlopen' and args else 'ctypes',
                       'is not allowed in this session')
        elif event in INTROSPECTION_EVENTS:
            self._deny(event, 'gc', 'is not allowed in this session')


def install(read_roots, write_roots, executables=(), log_path=None, endpoints=(), helper_roots=()):
    """Install the guard for the rest of this process; audit hooks cannot be removed."""
    guard = Guard(read_roots, write_roots, executables, log_path, endpoints, helper_roots)
    sys.addaudithook(guard.hook)
    return guard
