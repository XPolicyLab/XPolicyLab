"""Evaluator-side process and explicit Kit-fault monitoring; never score cases."""
import json
import os
from pathlib import Path
import re
import time


class OfficialRuntimeFault(RuntimeError):
    def __init__(self, kind, evidence, returncode=None):
        self.kind, self.evidence, self.returncode = kind, evidence, returncode
        super().__init__(kind + ': ' + evidence)


class RuntimeMonitor:
    def __init__(self, client, server, result_path, health_path, *, clock=time.monotonic,
                 hang_grace=30., auxiliary_logs=(), policy_episode_path=None):
        self.client, self.server = client, server
        self.result_path = Path(result_path) if result_path else None
        self.health_path = Path(health_path) if health_path else None
        self.clock, self.hang_grace = clock, hang_grace
        self.last_progress = clock()
        self.graded, self.actions = 0, None
        self.episode_started = False
        self.initial_reset_seen = False
        self.policy_episode_path = Path(policy_episode_path) if policy_episode_path else None
        self.pending_fault = None
        self.provided_logs = {Path(p) for p in auxiliary_logs}
        self.logs = {p: 0 for p in self.provided_logs}
        self.suffix = ''
        self.last_write = float('-inf')
        self._official_progress()

    def _official_progress(self):
        if self.result_path is None or not self.result_path.is_file():
            return
        try:
            from .official_resume import official_details
            details = official_details(json.loads(self.result_path.read_text()))
        except (OSError, ValueError, KeyError, TypeError):
            return  # SDK can be between writes. Partial JSON is never a score.
        if len(details) > self.graded:
            self.graded = len(details)
            self.episode_started = True
            self.last_progress = self.clock()
            self.pending_fault = None
            # This action token belonged to the newly scored episode. A Kit
            # UI warning during the next reset is not its terminal stall.
            self.actions = None
            self.suffix = ''

    def feed(self, text):
        combined = self.suffix + text
        # The SDK's action tqdm emits N/H [elapsed...]. File size, timestamps,
        # log chatter and process existence do not count as execution progress.
        plain = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', combined)
        if 'Resetting all environments' in plain:
            self.initial_reset_seen = True
        if re.search(r'Running Episode\s+\d+', plain):
            self.episode_started = True
        boundaries = list(re.finditer(r'Resetting all environments|Running Episode\s+\d+', plain))
        reset_end = boundaries[-1].end() if boundaries else 0
        if boundaries:
            # A reset starts a new SDK episode, not another official score.
            # Its first plan must not inherit the former horizon counter.
            self.actions = None
        matches = list(re.finditer(r'(?:env\d+ step:\s*|(?=\d+\s*/\s*\d+\s+\[))(\d+)\s*/\s*(\d+)', plain))
        matches = [match for match in matches if match.start() >= reset_end]
        advanced = False
        if matches:
            self.episode_started = True
            token = tuple(map(int, matches[-1].groups()))
            if token != self.actions:
                self.actions = token
                self.last_progress = self.clock()
                self.pending_fault = None
                combined = plain[matches[-1].end():]  # Earlier fault markers precede real recovery progress.
                advanced = True
        if re.search(r'(?:GPU Narrowphase failed|Cuda context manager error|'
                     r'CUDA error: an illegal memory access)', combined, re.I):
            if self.pending_fault is None or self.pending_fault[2] != 'client_cuda_context_failed':
                self.pending_fault = (self.clock(), combined[-1600:], 'client_cuda_context_failed', 5.)
        if re.search(r'Hang detected|Kit appears to be hanging', combined, re.I):
            if self.pending_fault is None:
                self.pending_fault = (self.clock(), combined[-1600:], 'client_kit_hang', self.hang_grace)
        # Consume phase markers once. Retain split fault markers, while a reset
        # without a subsequent action cannot clear an explicit CUDA failure.
        self.suffix = (plain[reset_end:] if boundaries and not advanced else combined)[-256:]

    def _kit_logs(self):
        # Read only Kit logs already opened by this owned SDK process. Do not
        # search shared storage or use a different process's stale Kit log.
        directory = Path('/proc') / str(self.client.pid) / 'fd'
        opened = set()
        try:
            for fd in directory.iterdir():
                try:
                    target = Path(os.readlink(fd))
                    if target.name == 'kit.log' and target.is_absolute():
                        opened.add(target)
                        self.logs.setdefault(target, 0)
                except OSError:
                    continue
        except OSError:
            pass
        removed = set(self.logs) - opened - self.provided_logs
        if removed:
            for path in removed:
                del self.logs[path]
            self.pending_fault = None  # Native execv closed the former Kit context.
        for path, offset in list(self.logs.items()):
            try:
                with path.open('rb') as stream:
                    stream.seek(offset)
                    # Bound work per poll; only explicit markers are considered.
                    chunk = stream.read(1024 * 1024)
                    self.logs[path] = stream.tell()
                if chunk:
                    self.feed(chunk.decode(errors='replace'))
            except OSError:
                continue

    def check(self):
        try:
            if self.client.poll() is not None:
                return  # Let the pinned SDK's native exit/restart contract win.
            if self.server is not None:
                code = self.server.poll()
                if code is not None:
                    raise OfficialRuntimeFault('policy_server_exited',
                                              'Owned policy server exited during the episode', code)
            self._official_progress()
            self._kit_logs()
            if self.pending_fault is not None:
                since, evidence, kind, grace = self.pending_fault
                terminal = self.actions and self.actions[0] == self.actions[1] and self.actions[1] >= 100
                if kind == 'client_kit_hang' and not terminal:
                    # Rule runners create this directory before perception or
                    # planning. Its absence, the SDK's initial-reset marker and
                    # no episode/action/score evidence distinguish a native
                    # bootstrap hang from legitimate slow policy computation.
                    bootstrap = False
                    if self.initial_reset_seen and not self.episode_started and self.policy_episode_path is not None:
                        try:
                            bootstrap = not self.policy_episode_path.exists()
                        except OSError:
                            pass  # Unknown filesystem state cannot authorize cleanup.
                    if bootstrap and self.clock() - max(since, self.last_progress) >= max(grace, 300.):
                        raise OfficialRuntimeFault('client_kit_bootstrap_hang', evidence)
                    # Kit also reports an unresponsive UI while the client waits
                    # for a legitimate slow policy plan. Only its terminal,
                    # unscored or proven pre-episode stall is confirmed; other
                    # hangs remain observable suspicions, never a general cap.
                    self.write()
                    return
                if self.clock() - max(since, self.last_progress) >= grace:
                    raise OfficialRuntimeFault(kind, evidence)
            self.write()
        except OfficialRuntimeFault as error:
            self.write(error)
            raise

    def write(self, error=None):
        if self.health_path is None or error is None and self.clock() - self.last_write < 5:
            return
        value = dict(schema='official-runtime-health-v1', at=time.time(),
                     status='fault' if error else ('hang_suspected' if self.pending_fault else
                         'terminal_unscored' if self.actions and self.actions[0] == self.actions[1] and self.actions[1] >= 100 else 'running'),
                     client_pid=self.client.pid, server_pid=self.server.pid if self.server else None,
                     actions=self.actions, official_graded=self.graded,
                     official_episode_started=self.episode_started,
                     initial_reset_seen=self.initial_reset_seen,
                     rule_bootstrap_guard_enabled=self.policy_episode_path is not None,
                     progress_idle_seconds=self.clock()-self.last_progress,
                     kit_logs=[str(p) for p in self.logs],
                     fault=error.kind if error else None, evidence=error.evidence if error else None,
                     server_returncode=error.returncode if error else None,
                     scores_fabricated=False, total_episode_wall_limit=False)
        temporary = self.health_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(self.health_path)
        self.last_write = self.clock()
