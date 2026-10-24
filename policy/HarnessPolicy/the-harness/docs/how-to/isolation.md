# Isolate programs

Model-written programs run in a kernel process that sees only what a policy may: the
Python installation and the harness source, read-only; a fresh workspace; the services
named in its options. The task's files, checkers, object annotations and the
simulator's state are out of reach.

## What the kernel gets

With `runtime.isolation = true` (the default):

| | |
|---|---|
| Namespaces | `unshare --user --map-root-user --mount --pid --fork --mount-proc --kill-child`: a user, mount and PID namespace of its own |
| Files | the home directory and each top-most directory owned by the user (e.g. `/data/<user>` for a checkout in it) are hidden under empty tmpfs; the Python prefix, `src/`, `skills/`, `benches/` and the XPolicyLab client package are mounted back read-only; `/tmp` is a fresh tmpfs and the working directory `/tmp/workspace` |
| Environment | `PATH`, locale, `PYTHONPATH`, proxy variables, and the API keys its model roles name; `HOME` empty, `TMPDIR=/tmp` |
| Network | connections only to the services in its options (worker endpoint, model URLs, service URLs) and the proxies; name resolution only for those hosts |
| Processes | no `subprocess`, `os.system`, `exec*`, `fork`, `posix_spawn`, `pty.spawn`; no `ctypes` loading |

The network and process checks are an audit hook inside the kernel; it cannot be
removed from inside, but it guards against reaching data and services, not against a
deliberate escape from the interpreter.

## Options

| `runtime.isolation` | |
|---|---|
| `true` | the sandbox and the guard |
| `"auto"` | the sandbox where user namespaces work, else the guard alone (with a message) |
| `{hide = ["/data/secrets"]}` | the sandbox, also hiding these directories |
| `false` | neither: for debugging only |

`"auto"` is for machines you do not control: containers often forbid user namespaces.
The agent uses it (`configs/default.toml`), since an evaluator's machine need not allow them.

## Checking it

```bash
unshare --user --map-root-user --mount --pid --fork --mount-proc true && echo namespaces work
python -m unittest discover -s tests -p test_isolation.py
```

The tests check that user files are hidden, the source is read-only and the workspace
writable, only allowed services are reachable, processes can neither be started nor
seen, and the guard stays where the sandbox cannot run.
