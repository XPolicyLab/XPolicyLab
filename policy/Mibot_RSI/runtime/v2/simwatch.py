"""Simulator watchdog of the submission's robot service.

In RoboDojo's evaluation process (v2/submission.py) the agent's robot commands run on the simulator's main thread. When
the simulator dies inside a command (e.g. GPU memory runs out, PhysX logs "Scene state is
corrupted. Simulation cannot continue!" and Isaac stops the simulation) that thread never returns, so the robot
service's loop, its wall limit and RoboDojo's own main loop never run again. RoboDojo recovers from this with its PhysX
log monitor (src/eval_client/physx_warning_monitor.py: rc 99, then scripts/eval_policy.sh restarts the process with the
same ROBODOJO_RUN_ID and the interrupted layout runs again), but main.py starts that monitor only for tasks whose config
has an Articulation section. This watchdog, a thread of the evaluation process that runs while an episode is served,
covers the other tasks:

  A. the simulator's log: RoboDojo's fatal lines ("fail to launch kernel", "PhysX has reported too many errors,
     simulation has been stopped.", "... Simulation cannot continue! Error code 700!") and PhysX's other
     "Simulation cannot continue!" lines (Scene state is corrupted; an internal CUDA error with any code), read from
     the regular file behind fd 1 or 2 (the launcher's log) or else Kit's own log file. Only a Kit log line counts
     ("<time> [N ms] [Error] [channel] ..." from the line's start), never a line Python printed, and not one
     of another Kit process sharing the file (its N ms does not date back to this process's start). Read once
     more as the episode ends, so that a line logged in the last second is not missed. Not read where RoboDojo's
     monitor runs for the task: it handles these itself. "Invalid PhysX transform" (RoboDojo abandons such a
     layout when its monitor runs) is only counted: RoboDojo does not act on it for these tasks either, and a
     layout that breaks the same way every time would be rerun until the restarts run out.
  B. a stalled command: a robot command is executing and neither RoboDojo's step counter nor the physics substep
     counter has moved for STALL_S, far longer than any healthy gap, counted from the resume after a paused
     process; not while RoboDojo's monitor has seen a fatal line (its own watchdog leaves the process within 45 s).
  C. the episode wall limit, which the robot service's loop checks only between commands, WALL_GRACE_S after it.

A finding goes to `on_failure(kind, reason)` (AttachedService._watchdog_failure), which records it, stops the session
and leaves the process with rc 99. The watchdog never touches the simulator: it reads files and two counters.
"""
from __future__ import annotations

import calendar
import os
import re
import stat
import sys
import threading
import time

# In recorded evaluation episodes the longest command took 70 s and the longest gap between two steps inside a
# command was 40 s (one camera capture); the longest tool call that moved nothing (dry runs, observations) took 62 s.
# The session's worker gives up on a command after 180 s.
STALL_S = 600.0
WALL_GRACE_S = 120.0  # longer than any command, so that the robot service's own check comes first when it can
POLL_S = 1.0
PAUSE_S = 30.0  # a longer gap between two looks: the whole process was paused (SIGSTOP); a stall counts from then
READ_CHUNK = 1 << 20
READ_MAX = 1 << 24  # per look; a longer backlog (a warning storm) is read in the next looks
PARTIAL_MAX = 1 << 16
# Kit's "[N ms]" counter runs from its process's start: in recorded logs a line's origin (its time minus N ms) was
# within about 1 s of the process start, and within one process the origins spread by at most 1.0 s (whole-second
# stamps).
# The error is about 2 s (whole-second stamps, /proc/stat btime in whole seconds); -10 s leaves a wide margin, and
# +60 s also covers a Kit whose counter would start at its own start-up rather than the process's. A stamped line
# whose origin falls outside was written by another Kit process sharing the file.
ORIGIN_WINDOW_S = (-10.0, 60.0)

FATAL = re.compile(r'fail to launch kernel|PhysX has reported too many errors, simulation has been stopped\.'
                   r'|Simulation cannot continue!')
NEEDLES = (b'fail to launch kernel', b'too many errors', b'Simulation cannot continue!', b'Invalid PhysX transform')
TAG = r'(?P<tag>\[(?:Error|Fatal|Warning)\] \[[^\]\s]+\] )'
# A Kit log line, from its start: '2026-10-22T08:42:11Z [13,516,581ms] [Error] [omni.physx.plugin] ...'. It may
# follow RoboDojo's progress line, which python -u writes in two parts ("env0 step: ..." and its "\r").
PROGRESS = r'(?:env\d+ step: \x1b\[92m\d+ / \d+\x1b\[0m)?'
KIT_LINE = re.compile(PROGRESS + r'(?P<time>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.\d+)?Z \[(?P<ms>[\d,]+)ms\] ' + TAG)
BROKEN_ENV = re.compile(r'Invalid PhysX transform detected for /World/envs/env_(\d+)/')
KIT_LOG = re.compile(r'/logs/Kit/.+/kit_[^/]*\.log$')
LINE_END = re.compile(rb'\r\n|\r|\n')
CONTROL = re.compile(r'[\x00-\x1f\x7f]')
EVIDENCE_CHARS = 300


def printable(line):
    """A log line as evidence for a record: control characters (terminal colours) removed, at most 300 characters."""
    return CONTROL.sub('', line)[:EVIDENCE_CHARS]


def process_start(proc='/proc', ticks=None):
    """This process's start time (seconds since the epoch), or that of a process of this boot that started `ticks`
    clock ticks after it; None where /proc does not tell."""
    try:
        if ticks is None:
            text = open(f'{proc}/self/stat').read()
            ticks = text[text.rfind(')') + 2:].split()[19]
        boot = next(int(line.split()[1]) for line in open(f'{proc}/stat') if line.startswith('btime '))
        return boot + int(ticks) / os.sysconf('SC_CLK_TCK')
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def kit_log_path(fd_dir='/proc/self/fd'):
    """The Kit log file this process has open (logs/Kit/<app>/<version>/kit_<time>.log), or None."""
    for source in log_sources(fd_dir, kit_only=True):
        return source['target']
    return None


def fatal_line(path, started, since=None):
    """The first fatal Kit line of a dead evaluation process's own Kit log, written at or after `since` (an ISO time,
    the episode's start) by the process that started at `started`; printable, or None."""
    try:
        data = open(path, 'rb').read()
    except OSError:
        return None
    since = (since or '')[:19]
    for part in LINE_END.split(data):
        if not any(needle in part for needle in NEEDLES):
            continue
        line = part.decode('utf-8', 'replace')
        kit = KIT_LINE.match(line)
        if kit is None or kit.group('time') < since:
            continue
        found = scan_line(line, started)
        if found and found[0] == 'fatal':
            return printable(line)
    return None


def scan_line(line, started=None):
    """('fatal', reason), ('broken_env', env index), ('foreign', None) or None for one log line (str). Only a line
    written as Kit writes its log lines counts, from its start, not one Python printed, which may quote one (as this
    watchdog's own message does); with `started` (this process's start time), a line of another Kit process sharing
    the file is 'foreign'."""
    kit = KIT_LINE.match(line)
    if kit is None:
        return None
    fatal = FATAL.search(line, kit.end())
    broken = None if fatal else BROKEN_ENV.search(line, kit.end())
    if not (fatal or broken):
        return None
    if started is not None:
        origin = (calendar.timegm(time.strptime(kit.group('time'), '%Y-%m-%dT%H:%M:%S'))
                  - int(kit.group('ms').replace(',', '')) / 1000)
        if not ORIGIN_WINDOW_S[0] <= origin - started <= ORIGIN_WINDOW_S[1]:
            return 'foreign', None
    if fatal:
        return 'fatal', line[kit.start('tag'):].split(', FILE ')[0].strip()[:300]
    return 'broken_env', int(broken.group(1))


def log_sources(fd_dir='/proc/self/fd', kit_only=False):
    """[{label, path, target, key}] of the simulator's log to read: the regular files behind fd 1 and fd 2 (one entry
    per file; e.g. a launcher's log that Kit's console output is redirected to), else Kit's own log file
    (an open file logs/Kit/<app>/<version>/kit_<time>.log, which Kit keeps open for its whole run), else none (a
    terminal or a pipe: then only B and C watch). `path` opens the file this process has open, even if renamed."""
    found, seen = [], set()

    def add(label, path):
        try:
            info = os.stat(path)
            target = os.readlink(path)
        except OSError:
            return
        key = (info.st_dev, info.st_ino)
        if stat.S_ISREG(info.st_mode) and key not in seen:
            seen.add(key)
            found.append({'label': label, 'path': path, 'target': target, 'key': key})
    for fd in () if kit_only else (1, 2):
        add(f'fd{fd}', f'{fd_dir}/{fd}')
    if found:
        return found
    try:
        names = sorted(os.listdir(fd_dir), key=lambda name: int(name) if name.isdigit() else -1)
    except OSError:
        return found
    for name in names:
        try:
            target = os.readlink(f'{fd_dir}/{name}')
        except OSError:
            continue
        if KIT_LOG.search(target):
            add('kit_log', f'{fd_dir}/{name}')
    return found


class LogTail:
    """New whole lines of one log file that may hold a watched pattern, from `offset` (where an earlier episode of this
    process stopped; default: the end of the file now)."""

    def __init__(self, source, offset=None):
        self.source = source
        self.stream = open(source['path'], 'rb')
        size = os.fstat(self.stream.fileno()).st_size
        self.offset = size if offset is None else (offset if offset <= size else 0)  # larger: truncated since
        self.partial = b''

    @property
    def resume_offset(self):
        """Where the next episode reads on: the start of the line not yet complete."""
        return self.offset - len(self.partial)

    def lines(self):
        size = os.fstat(self.stream.fileno()).st_size
        if size < self.offset:  # truncated: read it again from the start
            self.offset, self.partial = 0, b''
        out, budget = [], READ_MAX
        while self.offset < size and budget > 0:
            self.stream.seek(self.offset)
            data = self.stream.read(min(READ_CHUNK, size - self.offset, budget))
            if not data:
                break
            self.offset += len(data)
            budget -= len(data)
            parts = LINE_END.split(self.partial + data)
            self.partial = parts.pop()
            if len(self.partial) > PARTIAL_MAX:  # no line is this long: keep its end only
                self.partial = self.partial[-PARTIAL_MAX:]
            out.extend(part.decode('utf-8', 'replace') for part in parts
                       if any(needle in part for needle in NEEDLES))
        return out

    def close(self):
        try:
            self.stream.close()
        except OSError:
            pass


class SimWatch:
    """The watchdog of one served episode. The robot service (main thread) calls start(), command(name) when a robot
    command starts, command(None) when it ends, final_look() and then stop() before it records the episode's end;
    both block for good once the watchdog has begun to act (on_failure leaves the process).

    `progress()` returns a value that changes whenever the simulator advances (the step and substep counters);
    `official` is (enabled, monitor) of RoboDojo's PhysX log monitor for this task; `offsets` maps a log file's
    (device, inode) to where this process stopped reading it, so that lines logged between two episodes are read."""

    def __init__(self, progress, on_failure, *, official=(False, None), sources=None, offsets=None,
                 wall_limit_s=None, stall_s=None, wall_grace_s=None, poll_s=None, clock=time.monotonic,
                 started_at=None):
        self.progress, self.on_failure, self.clock = progress, on_failure, clock
        self.official_enabled, self.official = bool(official[0]), official[1]
        self.stall_s = STALL_S if stall_s is None else stall_s
        self.wall_grace_s = WALL_GRACE_S if wall_grace_s is None else wall_grace_s
        self.poll_s = POLL_S if poll_s is None else poll_s
        self.wall_limit_s = wall_limit_s
        self.offsets = {} if offsets is None else offsets
        self.process_started = process_start() if started_at is None else started_at
        self.tails = []
        if not self.official_enabled:
            for source in (log_sources() if sources is None else sources):
                try:
                    self.tails.append(LogTail(source, self.offsets.get(source['key'])))
                except OSError:
                    continue
        self.broken_envs = set()
        self.foreign_lines = 0
        self.fired = None
        self.finding = None  # which check fired ('log', 'stall' or 'wall') and its evidence, for the records
        self.started = None
        self._command = self._since = self._last = self._last_look = None
        self._epoch = 0
        self._lock = threading.Lock()
        self._logs = threading.Lock()
        self._final = threading.Lock()
        self._stopped = threading.Event()
        self._thread = None
        self._errors = 0

    def describe(self):
        """For the episode record: what watches this episode."""
        return {'log_sources': [{'label': t.source['label'], 'file': t.source['target']} for t in self.tails],
                'official_monitor': self.official_enabled, 'stall_s': self.stall_s,
                'wall_grace_s': self.wall_grace_s}

    def start(self):
        self.started = self.clock()
        self._thread = threading.Thread(target=self._loop, name='Mibot-RSI-simwatch', daemon=True)
        self._thread.start()
        return self

    def command(self, name):
        """A robot command (or other simulator work) begins on the main thread, or (None) has ended."""
        value = self._progress() if name is not None else None
        with self._lock:
            self._epoch += 1
            self._command, self._since, self._last = name, self.clock(), value

    def in_command(self):
        with self._lock:
            return self._command is not None

    def final_look(self):
        """The main thread, as the episode ends: the log lines written since the last look (a fatal line in the last
        second, or during the end check, would otherwise let the episode be scored). Acts on a fatal line as the
        thread does."""
        try:
            found = self.check_logs()
        except Exception:
            found = None
        if found:
            self.fire(*found)

    def stop(self):
        """Called by the main thread before it records the episode's end. Returns the finding if the watchdog acted
        (only where on_failure returns, i.e. in tests)."""
        with self._final:
            self._stopped.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10)
        with self._logs:
            for tail in self.tails:
                self.offsets[tail.source['key']] = tail.resume_offset
                tail.close()
            self.tails = []
        return self.fired

    # -- the watchdog thread

    def _progress(self):
        try:
            return self.progress()
        except Exception:
            return None

    def _official_fatal(self):
        try:
            return bool(self.official is not None and self.official.is_fatal())
        except Exception:
            return False

    def check_logs(self):
        """A fatal line among the new log lines: ('simulator_failure', reason), or None."""
        with self._logs:
            for tail in self.tails:
                try:
                    lines = tail.lines()
                except OSError:
                    continue
                for line in lines:
                    found = scan_line(line, self.process_started)
                    if found is None:
                        continue
                    if found[0] == 'fatal':
                        self.finding = {'check': 'log', 'evidence': printable(line)}
                        return 'simulator_failure', 'simulator log: ' + found[1]
                    if found[0] == 'foreign':
                        self.foreign_lines += 1
                    else:
                        self.broken_envs.add(found[1])
        return None

    def check(self):
        """One look: (kind, reason) of a finding, or None."""
        found = self.check_logs()
        if found:
            return found
        now = self.clock()
        paused = self._last_look is not None and now - self._last_look > max(PAUSE_S, 10 * self.poll_s)
        self._last_look = now
        with self._lock:
            if paused and self._command is not None:  # the process was stopped: nothing could move meanwhile
                self._since = now
            command, since, last, epoch = self._command, self._since, self._last, self._epoch
        if command is not None:
            value = self._progress()
            if value != last:
                with self._lock:
                    if self._epoch == epoch:
                        self._since, self._last = now, value
            elif now - since >= self.stall_s and not self._official_fatal():
                self.finding = {'check': 'stall', 'evidence': f'progress (steps, physics substeps) stayed {value}'}
                return 'simulator_failure', (f'the simulator made no progress for {now - since:.0f} s while running '
                                             f'robot command {command}')
        if (self.wall_limit_s is not None and self.started is not None
                and now - self.started >= self.wall_limit_s + self.wall_grace_s):
            self.finding = {'check': 'wall', 'evidence': None}
            return 'wall_limit', f'episode wall limit of {self.wall_limit_s} s reached'
        return None

    def _loop(self):
        while not self._stopped.wait(self.poll_s):
            try:
                found = self.check()
            except Exception as exc:  # the watchdog must keep watching
                self._errors += 1
                if self._errors == 1:
                    print(f'[Mibot-RSI] simulator watchdog: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
                continue
            if found:
                self.fire(*found)
                return

    def fire(self, kind, reason):
        """Act on a finding unless the main thread has already ended the episode (stop())."""
        with self._final:
            if self._stopped.is_set():
                return False
            self.fired = (kind, reason)
            try:
                self.on_failure(kind, reason)  # leaves the process in an evaluation
            finally:
                self._stopped.set()
        return True
