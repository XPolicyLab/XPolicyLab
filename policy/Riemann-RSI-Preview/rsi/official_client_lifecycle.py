"""Match the pinned SDK shell's bounded client recovery, never regrade cases."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import sys
import threading
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = 'scripts/eval_policy.sh'


def official_restart_contract(sdk):
    lock = json.loads((ROOT / 'configs/official-runtime-inputs.json').read_text())
    path = Path(sdk) / SCRIPT
    expected = lock['official_files'][SCRIPT]
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('The official client restart script changed')
    # These are the semantics of this exact, separately protected SDK file.
    # Tests execute its actual Bash loop as an independent oracle.
    return dict(official_commit=lock['official_commit'], script=SCRIPT,
                script_sha256=expected, returncodes=[99, 134, 139],
                default_attempt_cap=10, delay_seconds=5)


class OfficialClientRestarts:
    def __init__(self, sdk, output, environment, run_id, result_path=None, *, expected=None, expected_count=None):
        self.sdk = sdk
        self.path = Path(output) / 'official-client-restarts.json'
        self.run_id = run_id
        self.limit = int(environment.get('ROBODOJO_MAX_BASH_RETRIES', '10'))
        self.events = []
        self.contract = None
        self.delay_seconds = 5
        self.result_path = Path(result_path) if result_path is not None else None
        self.prefix = None
        self.expected = expected
        self.expected_count = expected_count
        self.client_attempt = 0

    def _fatal_shutdown(self, client_log):
        """Kit may terminate close() before the SDK can exec/exit(99).

        Admit only the pinned SDK's fatal marker plus its matching native
        checkpoint. An arbitrary zero exit, policy exception or timeout is not
        recoverable through this path.
        """
        if client_log is None:
            return False
        marker = '[FATAL] PhysX kernel failure detected: '
        with Path(client_log).open(errors='replace') as stream:
            fatal = any(line.startswith(marker) and
                        '; persisted manifest. In-process restart attempt ' in line for line in stream)
        if not fatal:
            return False
        lock = json.loads((ROOT / 'configs/official-runtime-inputs.json').read_text())
        main = Path(self.sdk) / 'src/eval_client/main.py'
        if hashlib.sha256(main.read_bytes()).hexdigest() != lock['official_files']['src/eval_client/main.py']:
            raise ValueError('The official fatal recovery source changed')
        if self.expected is None or self.result_path is None or type(self.expected_count) is not int:
            raise ValueError('Fatal shutdown needs the original public run identity and count')
        from .official_resume import document, official_details
        result = document(self.result_path) if self.result_path.exists() else dict(details={}, eval_time=0)
        details = official_details(result)
        if len(details) == self.expected_count:
            # A single transcript spans the SDK's own os.execv restarts. An
            # earlier fatal marker must not reject its later normal completion
            # (which intentionally removes the native checkpoint).
            return False
        manifest = document(self.result_path.parent.parent / ('_resume_' + self.run_id + '.json'))
        if any(manifest.get(key) != value for key, value in self.expected.items()):
            raise ValueError('Fatal shutdown SDK checkpoint identity differs')
        if (manifest.get('details') != details or len(details) >= self.expected_count
                or manifest.get('completed_layout_ids') != sorted(d['layout_id'] for d in details.values())
                or manifest.get('success_nums') != sum(d['success'] for d in details.values())
                or manifest.get('fail_nums') != sum(not d['success'] for d in details.values())
                or not math.isclose(manifest.get('total_score', -1),
                                    sum(d['score'] for d in details.values()), abs_tol=1e-8)):
            raise ValueError('Fatal shutdown needs a matching incomplete official checkpoint')
        return True

    def should_restart(self, returncode, *, client_log=None):
        from .official_resume import document, official_details, verify_extension
        if self.prefix is not None:
            verify_extension(self.prefix, document(self.result_path))
        # subprocess reports POSIX signals as negative numbers; Bash uses
        # 128+signal, which is what the official shell's case statement sees.
        shutdown_fatal = returncode == 0 and self._fatal_shutdown(client_log)
        effective = 99 if shutdown_fatal else (128 - returncode if returncode < 0 else returncode)
        if effective not in (99, 134, 139):
            if self.events:
                self._write('finished' if returncode == 0 else 'client_error', returncode)
            return False
        if self.contract is None:
            self.contract = official_restart_contract(self.sdk)
        prefix = manifest = None
        if self.result_path is not None and self.result_path.exists():
            prefix = document(self.result_path)
            details = official_details(prefix)
            if details:
                # Never let a restart without the SDK's checkpoint replay an
                # already graded success or failure. The actor sees no scores.
                path = self.result_path.parent.parent / ('_resume_' + self.run_id + '.json')
                manifest = document(path)
                if manifest.get('details') != details:
                    raise ValueError('SDK restart checkpoint differs from scored official cases')
                self.prefix = prefix
        attempt = len(self.events) + 1
        retry = attempt < self.limit
        self.events.append(dict(at=datetime.now(timezone.utc).isoformat(),
                                fatal_attempt=attempt, returncode=returncode,
                                shell_returncode=effective, will_restart=retry,
                                fatal_during_kit_close=shutdown_fatal, client_log=str(client_log) if client_log else None,
                                official_prefix=prefix, sdk_checkpoint=manifest))
        self._write('restarting' if retry else 'attempt_cap_reached', returncode)
        print(json.dumps(dict(event='official_client_fatal_exit', run_id=self.run_id,
                              fatal_attempt=attempt, attempt_cap=self.limit,
                              returncode=returncode, will_restart=retry)), flush=True)
        return retry

    def _write(self, status, returncode):
        receipt = dict(schema='official-client-restarts-v1', run_id=self.run_id,
                       status=status, last_returncode=returncode, attempt_cap=self.limit,
                       official_contract=self.contract, events=self.events,
                       sdk_owns_resume_layouts_and_scores=True,
                       policy_server_restarted=False)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(receipt, indent=2) + '\n')
        temporary.replace(self.path)


@contextmanager
def client_transcript(output, restarts):
    """Durable per-client output, mirrored to the existing campaign log.

    Read a file rather than a pipe: inherited descendant stdout cannot keep
    this coordinator blocked after its owned SDK client has exited.
    """
    attempt = 1 if restarts is None else restarts.client_attempt + 1
    if restarts is not None:
        restarts.client_attempt = attempt
    path = Path(output) / ('official-client-attempt-' + str(attempt) + '.log')
    stopped = threading.Event()

    def mirror():
        with path.open('rb') as reader:
            while True:
                chunk = reader.read(65536)
                if chunk:
                    try:
                        target = getattr(sys.stdout, 'buffer', None)
                        if target is not None:
                            target.write(chunk); target.flush()
                        else:
                            sys.stdout.write(chunk.decode(errors='replace')); sys.stdout.flush()
                    except (OSError, ValueError):
                        pass  # The durable transcript remains the source of evidence.
                elif stopped.is_set():
                    return
                else:
                    stopped.wait(.1)

    with path.open('xb') as stream:
        thread = threading.Thread(target=mirror, name='official-client-transcript', daemon=True)
        thread.start()
        try:
            yield stream, path
        finally:
            stopped.set()
            thread.join(timeout=5)


def wait_client(client, transcript, timeout=None, *, server=None, result_path=None, health_path=None,
                policy_episode_path=None):
    """Monitor owned peers and explicit runtime faults without a default wall cap."""
    from .official_runtime_monitor import RuntimeMonitor, OfficialRuntimeFault
    monitor = RuntimeMonitor(client, server, result_path, health_path,
                             policy_episode_path=policy_episode_path)
    deadline = None if timeout is None else time.monotonic() + timeout
    marker = b'Failed to create any GPU devices, including an attempt with compatibility mode.'
    with Path(transcript).open('rb') as log:
        tail = b''
        while True:
            while chunk := log.read(65536):
                if marker in tail + chunk:
                    raise RuntimeError('Official renderer bootstrap failed before usable observations; see ' + str(transcript))
                tail = (tail + chunk)[-len(marker):]
                try:
                    monitor.feed(chunk.decode(errors='replace'))
                except OfficialRuntimeFault as error:
                    monitor.write(error)
                    raise
            monitor.check()
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise subprocess.TimeoutExpired(client.args, timeout)
            try:
                code = client.wait(timeout=1 if remaining is None else min(1, remaining))
            except subprocess.TimeoutExpired:
                continue
            # Include final buffered lines even when startup exits immediately.
            if marker in tail + log.read():
                raise RuntimeError('Official renderer bootstrap failed; see ' + str(transcript))
            return code
