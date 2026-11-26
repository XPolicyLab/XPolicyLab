"""Robot SDK for v2 sessions: the v1 SDK plus a time limit that never cuts a motion short.

When a python_exec time limit arrives while a command is executing, the worker
sets `deadline_pending` instead of raising. The command finishes normally, its
report is recorded, and `DeadlineReached` is raised at the command boundary so
the rest of the code does not run. Nothing is cancelled or resubmitted.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time
import uuid

_V1 = str(Path(__file__).resolve().parents[2] / 'embodied_agent')
if _V1 not in sys.path:
    sys.path.insert(0, _V1)

from robot_sdk import Arm, Gripper, Robot as V1Robot  # noqa: E402

__all__ = ['Arm', 'DeadlineReached', 'Gripper', 'Robot']

INITIAL_MARK_NOTE = ("'initial' is recorded automatically at the start of the episode and cannot be re-marked; "
                     "nothing was changed (move_to_mark('initial') returns to it).")


class DeadlineReached(BaseException):
    """The python_exec time limit passed; the last command completed before stopping.

    Derived from BaseException so that `except Exception` in model code does not hide it.
    """


def public_error(error):
    """A service-side traceback reduced to its last line (exception type and message), so the
    simulator and adapter source it quotes stays out of the model's view."""
    if isinstance(error, str) and 'Traceback (most recent call last)' in error:
        lines = [line for line in error.strip().splitlines() if line.strip()]
        return lines[-1] if lines else error
    return error


class Robot(V1Robot):
    # Longest wait for one command. The worker raises it to its grace cap (a command may run past the
    # python_exec limit while the simulator keeps stepping); harness-side replay and diagnosis keep v1's 600 s.
    wait_limit_s = 600

    def __init__(self, connection=None):
        super().__init__(connection)
        self.deadline_pending = False   # set by the worker when the limit arrives during a command
        self.deadline_passed = False    # once set, no further command is submitted in this call

    def status(self, command_id):
        record = super().status(command_id)
        if isinstance(record, dict):
            if 'error' in record:
                record = {**record, 'error': public_error(record['error'])}
            if isinstance(record.get('result'), dict) and 'error' in record['result']:
                record = {**record, 'result': {**record['result'], 'error': public_error(record['result']['error'])}}
        return record

    def wait(self, command_id, timeout=None):
        return super().wait(command_id, self.wait_limit_s if timeout is None else timeout)

    def mark_pose(self, name):
        if name == 'initial':
            # The service refuses to overwrite the start pose; answering here keeps the rest of the code running.
            print(INITIAL_MARK_NOTE)
            return {'bookmark': 'initial', 'status': 'unchanged', 'note': INITIAL_MARK_NOTE}
        return super().mark_pose(name)

    def submit(self, operation, command_id=None, **params):
        if self.deadline_pending or self.deadline_passed:
            # Past the time limit only the command already running may finish (also for code that polls
            # robot.status itself instead of using call()).
            self.deadline_pending, self.deadline_passed = False, True
            raise DeadlineReached('python_exec time limit reached; no further robot command was submitted.')
        # Short IDs (8 hex digits) fit in every action line; they only need to be
        # unique within one robot service, which refuses a reused ID without moving.
        return super().submit(operation, command_id=command_id or uuid.uuid4().hex[:8], **params)

    def call(self, operation, **params):
        # As V1Robot.call, plus the command's wall time before the result observer sees it.
        started = time.monotonic()
        handle = self.submit(operation, **params)
        result = self.wait(handle['id'])
        if isinstance(result, dict) and result.get('command_id'):
            result['wall_s'] = round(time.monotonic() - started, 1)  # real time; the simulation is slower
        if self.result_observer is not None:
            self.result_observer(operation, params, result)
        if self.deadline_pending:
            self.deadline_pending = False
            self.deadline_passed = True
            raise DeadlineReached('python_exec time limit reached. The last robot command completed and is '
                                  'listed above; the rest of the code was not run and nothing was cancelled.')
        return result
