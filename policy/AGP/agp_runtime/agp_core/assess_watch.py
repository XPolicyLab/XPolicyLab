"""OURS_END_ON_ASSESS: end the codex policy once the runtime has accepted an assessment.

The episode's last useful act is an accepted assess; the closing message after it only costs a model turn
(15 of 15 recorded episodes wrote one, median 4.1 s). The official evaluator is independent of the policy process.
"""
import json
import subprocess
import time


def accepted_assessment(session):
    """True once the runtime has written an accepted assessment (control_runtime/assessment_*.json, ok true)."""
    for path in (session / "control_runtime").glob("assessment_*.json"):
        try:
            if json.loads(path.read_text()).get("ok") is True:
                return True
        except (OSError, ValueError):
            continue
    return False


def stop_process(process):
    process.terminate()
    try:
        return process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait()


def end_on_assess(process, session, timeout_s, poll_s=0.1, record=None):
    """Wait for codex like process.wait(timeout_s) but end it once an assessment is accepted (return code 0).

    record (a dict) receives terminated_returncode, the raw code of the stopped process. A timeout returns 124.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return process.wait(timeout=poll_s)
        except subprocess.TimeoutExpired:
            pass
        if accepted_assessment(session):
            code = stop_process(process)
            if record is not None:
                record["terminated_returncode"] = code
            return 0
        if time.monotonic() > deadline:
            stop_process(process)
            return 124
