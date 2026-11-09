"""Record the actual CLI arguments separately from resolved evaluation settings."""

import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

SCHEMA = "orimapi.invocation/v1"
REDACTED = "<redacted>"
PUBLIC_HEADERS = {"editor-version", "copilot-integration-id"}


def public_environment(environment):
    values = {}
    for name, allowed in {
        "OMNI_KIT_ACCEPT_EULA": {"yes", "no", "true", "false", "1", "0"},
        "XLA_PYTHON_CLIENT_PREALLOCATE": {"true", "false"},
        "LIVESTREAM": {"0", "1", "2"},
    }.items():
        value = environment.get(name)
        if isinstance(value, str) and value.lower() in allowed:
            values[name] = value
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if isinstance(visible, str) and re.fullmatch(r"[0-9, -]*", visible):
        values["CUDA_VISIBLE_DEVICES"] = visible
    return values


def public_argv(argv):
    """Keep order/spelling; mask credential-bearing arguments before persistence."""
    result = list(argv)
    index = 0
    while index < len(result):
        flag, separator, inline = result[index].partition("=")
        # argparse accepts unambiguous long-option abbreviations too.
        kind = next(
            (
                name
                for name in ("--llm-auth-command", "--llm-header", "--llm-endpoint")
                if flag.startswith("--") and name.startswith(flag)
            ),
            flag,
        )
        start = index if separator else index + 1
        if start >= len(result):
            break
        value = inline if separator else result[start]
        replacement = value
        if kind == "--llm-auth-command":
            end = index + 1
            while end < len(result) and not result[end].startswith("--"):
                end += 1
            command = ([inline] if separator else []) + result[index + 1 : end]
            # Plain executable/script paths are useful; arbitrary credential scripts
            # and their arguments may embed tokens and must not be displayed.
            paths_only = (
                len(command) == 2
                and re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(command[0]).name)
                and command[1].endswith(".py")
                and all(re.fullmatch(r"[\w./~ -]+", part) for part in command)
            )
            if not paths_only:
                if separator:
                    result[index] = flag + "=" + REDACTED
                for position in range(index + 1, end):
                    result[position] = REDACTED
            index = end
            continue
        if kind == "--llm-header":
            name, _, _ = value.partition("=")
            if name.lower() not in PUBLIC_HEADERS:
                replacement = REDACTED
        elif kind == "--llm-endpoint":
            try:
                url = urlsplit(value)
                if "@" in url.netloc or url.query or url.fragment:
                    replacement = urlunsplit(
                        (
                            url.scheme,
                            url.netloc.rsplit("@", 1)[-1],
                            url.path,
                            REDACTED if url.query else "",
                            REDACTED if url.fragment else "",
                        )
                    )
            except ValueError:
                replacement = REDACTED
        elif flag.startswith("-") and (
            flag == "-c"
            or any(
                word in flag.lower()
                for word in ("token", "secret", "password", "api-key")
            )
        ):
            if flag != "--tokenizer-path":
                replacement = REDACTED
        else:
            index += 1
            continue
        result[start] = flag + "=" + replacement if separator else replacement
        index += 1 if separator else 2
    return result


def public_invocation(record):
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError("Invalid invocation schema")
    argv = record.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or len(argv) > 4096
        or any(not isinstance(arg, str) or "\0" in arg for arg in argv)
        or record.get("source") not in {"process_argv", "api_argv"}
        or not isinstance(record.get("cwd"), str)
        or not isinstance(record.get("environment"), dict)
        or not isinstance(record.get("captured_at"), str)
    ):
        raise ValueError("Invalid invocation metadata")
    arguments = public_argv(argv)
    return {
        "schema": SCHEMA,
        "source": record["source"],
        "argv": arguments,
        "cwd": record["cwd"],
        "environment": public_environment(record["environment"]),
        "captured_at": record["captured_at"],
        "redacted": record.get("redacted") is True or arguments != argv,
    }


def capture_invocation(argv=None):
    process_argv = argv is None
    return public_invocation(
        {
            "schema": SCHEMA,
            "source": "process_argv" if process_argv else "api_argv",
            "argv": (
                list(sys.orig_argv)
                if process_argv
                else [sys.executable, "-m", "orimapi", *argv]
            ),
            "cwd": str(Path.cwd()),
            "environment": public_environment(os.environ),
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def record_invocation(trace, invocation, *, resume=False):
    if invocation is None:
        return
    record = public_invocation(invocation)
    if not resume:
        trace.write_json("invocation.json", record)
    trace.write("invocation_recorded", resume=bool(resume), invocation=record)
