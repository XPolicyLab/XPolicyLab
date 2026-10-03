"""Bounded JSON mailbox bridge; callbacks must honor the supplied deadline.

Only the trusted host dispatches capabilities. Child messages/results are
untrusted data and never constitute environment success evidence.
"""

import json
import os
import time

from .processes import ManagedProcess
from .storage import atomic_json


def run(
    job,
    output,
    mailbox,
    responses,
    handlers,
    *,
    max_calls,
    message_bytes,
    cancelled=None,
):
    output.mkdir(parents=True, exist_ok=False)
    receipt = output / "process.json"
    process = ManagedProcess(
        job.id, job.command, cwd=job.cwd, log_path=output / "process.log"
    )
    record = dict(state="started", calls=[])
    atomic_json(receipt, record)
    deadline = time.monotonic() + job.timeout_s
    final = None
    try:
        process.start()
        record["pid"] = process.pid
        atomic_json(receipt, record)
        while True:
            if cancelled is not None and cancelled.is_set():
                raise RuntimeError("Isolated policy cancelled")
            returncode = process.poll()
            if time.monotonic() >= deadline:
                raise TimeoutError("Isolated policy deadline reached")
            try:
                with mailbox.open("rb") as stream:
                    raw = stream.read(message_bytes + 1)
                if len(raw) > message_bytes:
                    raise ValueError("Policy message exceeds limit")
                message = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                message = None  # Writer may be in the middle of a bounded write.
            if message is not None:
                if not isinstance(message, dict):
                    raise ValueError("Policy message must be an object")
                if message.get("kind") == "result":
                    final = message
                elif message.get("kind") == "call":
                    if final is not None:
                        raise ValueError("Policy called a primitive after returning")
                    index = message.get("id")
                    if type(index) is not int or index < 0:
                        raise ValueError("Invalid policy call identity")
                    if index < len(record["calls"]):
                        if message != record["calls"][index]["request"]:
                            raise ValueError("Policy changed a previous call")
                    else:
                        if index != len(record["calls"]) or index >= max_calls:
                            raise ValueError("Policy call budget or sequence violated")
                        if (
                            set(message) != {"kind", "id", "method", "args", "kwargs"}
                            or message["method"] not in handlers
                            or not isinstance(message["args"], list)
                            or not isinstance(message["kwargs"], dict)
                        ):
                            raise ValueError("Policy capability or arguments rejected")
                        event = dict(request=message, state="started")
                        record["calls"].append(event)
                        atomic_json(receipt, record)
                        # Explicit allowlist only; never getattr on a robot/environment.
                        result = handlers[message["method"]](
                            message["args"], message["kwargs"], deadline=deadline
                        )
                        response = dict(id=index, value=result)
                        encoded = json.dumps(response, allow_nan=False).encode()
                        if len(encoded) > message_bytes:
                            raise ValueError("Primitive response exceeds limit")
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Primitive returned after deadline")
                        event.update(state="completed", response=response)
                        atomic_json(receipt, record)
                        prepared = output / "response.json"
                        atomic_json(prepared, response)
                        prepared.chmod(0o444)
                        os.replace(prepared, responses)
                else:
                    raise ValueError("Unknown policy message")
            if returncode is not None:
                if returncode != 0 or final is None:
                    raise RuntimeError("Isolated policy failed or omitted result")
                record.update(state="completed", returncode=returncode)
                return final["value"]
            time.sleep(0.01)
    except BaseException as error:
        record.update(state="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        try:
            process.stop()
        finally:
            atomic_json(receipt, record)
