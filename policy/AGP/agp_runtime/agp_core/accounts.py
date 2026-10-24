#!/usr/bin/env python3
"""Codex account pool: one ChatGPT login per slot, up to episodes_per_login (default 3) episodes per login at a time.

The RoboDojo committee runs the benchmark with model access we provide: several dedicated ChatGPT accounts, each
logged in once per slot with the pinned codex, whose login files are placed under AGP_HOME. agp_core/run_agent.py leases a
slot for every episode; without a pool it uses the local user's own login (~/.codex/auth.json).

    /usr/bin/python3 -B agp_core/accounts.py add LABEL STAGING_DIR [--replace]
                                                  take a fresh login (STAGING_DIR/auth.json) into slots/LABEL
    /usr/bin/python3 -B agp_core/accounts.py check [LABEL ...]      every slot (or the named ones): file modes, login kind,
                                                  plan, account / session hash, last_refresh, access expiry, lease,
                                                  marks; exit 1 on a problem
    /usr/bin/python3 -B agp_core/accounts.py status                 accounts (slots, leases, used percent, reset, cooling)
    /usr/bin/python3 -B agp_core/accounts.py refresh LABEL [--codex PATH]
                                                  renew the slot's login now with the pinned codex (`codex app-server`,
                                                  getAuthStatus refreshToken=true, includeToken=false) and write it back
    /usr/bin/python3 -B agp_core/accounts.py ratelimits LABEL [--codex PATH]
                                                  read the account's usage window (account/rateLimits/read; no model
                                                  turn) into state.json
    /usr/bin/python3 -B agp_core/accounts.py release-stale          settle the leases of dead processes (write a renewed
                                                  login back, delete the copy) and old login copies without a lease,
                                                  then remove the episode homes agp_core/run_agent.py kept (KEPT_MARK)
                                                  once nothing holds them (also done at every lease)
    /usr/bin/python3 -B agp_core/accounts.py enable LABEL           clear the slot's disabled mark and its account's cooling

Options: --home AGP_HOME (default: agp_core/paths.py, i.e. the AGP_HOME variable, else the tree / ~/.agp-robodojo).
--codex: default deploy.yml agp.codex_binary of the policy folder.

Creating a slot (the account owner, once per slot; device code login must be enabled in the ChatGPT security settings):
    S=$AGP_HOME/private/accounts/staging/a1s1; mkdir -p -m 700 "$S"
    CODEX_HOME="$S" ~/.codex/packages/standalone/releases/0.160.1-x86_64-unknown-linux-musl/bin/codex \
        -c 'cli_auth_credentials_store="file"' login --device-auth
    /usr/bin/python3 -B agp_core/accounts.py add a1s1 "$S"
Never run `codex login` or `codex logout` inside a slot folder (both revoke the login there first), never copy one
login file into two slots or onto two machines (one login = one serialized stream; codex rotates the refresh token, the
other copy then holds a used one), never restore an old copy over a slot.

Layout (AGP_HOME/private/accounts, 0700; `private` is runtime state for agp_core/freeze.py, never frozen):
    pool.lock                fcntl lock of every change below (lease, renewal, write-back, state, add)
    state.json               0600, no secret: per account hash the last usage reading and cooling, per slot the marks
    pool.json                optional settings: cap_percent, est_percent_per_episode, episodes_per_login
    slots/<label>/auth.json  0600 (folder 0700): one codex login
    leases/<id>.json         0600: who holds a login (pid + start time), the copy, the sha256 of the login at lease
                             time, whether the lease is exclusive
    staging/<label>/         where a login is made before `add`
    tmp/                     temporary CODEX_HOMEs of refresh / ratelimits and of the renewal before a lease

Sharing a login: up to episodes_per_login episodes (pool.json, default EPISODES_PER_LOGIN = 3) hold copies of one
login at the same time, so the pool runs at most (logins x episodes_per_login) episodes; an account still has at most
MAX_SLOTS_PER_ACCOUNT slots. codex renews a login only 5 minutes before its access token expires (or after a 401), and
a renewal rotates the refresh token, which the other copies of the login then hold used. So a login is shared only while
its access token stays valid longer than the longest episode (deploy.yml agp.episode_timeout_s of the policy folder, or
the launcher's own codex limit when longer) + SHARE_MARGIN_S (1 h): no copy of it then reaches a renewal. Closer to its
expiry a login is leased exclusively (one episode, which may renew it as before); a login no lease holds is renewed
first (as `refresh`: the pinned codex app-server, under the pool lock, written back when renewed), after which it can be
shared again. A renewal that fails leaves the lease exclusive.

Lease (agp_core/run_agent.py, under the lock): settle the leases of dead processes first; then among the slots that are not
disabled, whose account is not cooling and that can take one more episode (no exclusive lease on it, fewer than
episodes_per_login leases, and shared leases only while it can be shared), the one whose account has the lowest
effective use = last used_percent (0 once the reading's window has reset) + leases of that account x
est_percent_per_episode, below cap_percent; ties: fewer leases of the account, then fewer leases of the login, then the
label. The login is copied into the episode's CODEX_HOME. No usable slot: NoAccount, naming every account's use,
reset and cooling (an infrastructure failure, re-run by the launcher; nothing waits). End (agp_core/run_agent.py
`finally`, under the lock): a copy codex changed is written back atomically only when it is a valid login of the same
account and session as the one leased, newer (last_refresh) than the slot file, and the slot file still holds that
session (a newer or another login there stays); then the copy and the lease are deleted, the last usage reading of the
episode's rollout updates the account, and a codex failure marks: usage limit -> the account cools until its reset,
refresh/401 failure -> the slot is disabled (needs a new login: `add --replace`; not when the slot meanwhile holds a
newer renewal of the same login, written back from another copy), 429/overloaded -> the account cools for 10 minutes.
Without a pool (no slot): the local user's ~/.codex/auth.json is copied (no sharing limit, no renewal); at the end the
copy replaces it under the same write-back rule (so a renewal is never lost and a newer login never overwritten).

An "account" here is the quota identity: one ChatGPT user within its workspace (account_hash = sha256 prefix of
account_id:chatgpt_user_id). A Business workspace gives all its users one account_id, but each user has a usage window
of their own (observed: two users of one workspace read 2% and 0% with different resets), so slots group by user;
workspace_hash (account_id alone) is recorded too.
Records never hold a credential value: labels, sha256 prefixes (account, user, session_id claim, file contents), key
names, last_refresh, access expiry time, plan, usage numbers. Stdlib only.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import subprocess
import sys
import time

FORMAT = "agp-accounts/1"
MAX_SLOTS_PER_ACCOUNT = 3
EPISODES_PER_LOGIN = 3           # default of pool.json episodes_per_login: episodes sharing one login at a time
SHARE_MARGIN_S = 3600.0          # a login is shared only while its access token outlives the longest episode by this
DEFAULT_EPISODE_S = 14400.0      # the longest episode when deploy.yml agp.episode_timeout_s cannot be read
RENEW_TIMEOUT_S = 90.0           # the renewal before a lease (codex app-server), under the pool lock
CAP_PERCENT = 95.0               # no new episode on an account whose effective use reaches this
# our measured use: 6-6.5% of a promax weekly window per 54 episodes (~0.12% per episode), rounded up
EST_PERCENT_PER_EPISODE = 0.15
RATE_LIMIT_COOLING_S = 600       # 429 / overloaded
USAGE_LIMIT_COOLING_S = 3600     # usage limit without a known reset time
LABEL = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
AUTH_NS = "https://api.openai.com/auth"
# codex failure text (error events, stderr) -> mark
USAGE_LIMIT = re.compile(r"usage.?limit|out of credits|spend cap|quota exceeded", re.I)
AUTH_FAILURE = re.compile(r"could not be refreshed|sign in again|invalidated oauth|refresh_token_(?:reused|expired|"
                          r"invalidated)|\b401\b|unauthori[sz]ed|not logged in", re.I)
RATE_LIMITED = re.compile(r"\b429\b|too many requests|rate.?limit|overloaded|\b503\b", re.I)


class NoAccount(Exception):
    """No slot can take an episode now (all leased, disabled, cooling or at the cap)."""


class PoolError(Exception):
    """A refused pool operation (CLI)."""


# ----------------------------------------------------------------------------- small helpers
def pool_root(home):
    return Path(home) / "private" / "accounts"


def home_login():
    """The local user's own codex login (HOME of the policy's environment), used when there is no pool."""
    return Path.home() / ".codex" / "auth.json"


def _now():
    return time.time()


def iso(epoch):
    if not isinstance(epoch, (int, float)) or isinstance(epoch, bool):
        return None
    return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def epoch_of(text):
    """Seconds of an ISO time (codex writes nanoseconds and Z), None when unreadable."""
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text)
    if not isinstance(text, str):
        return None
    match = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:?\d{2})?$", text.strip())
    if not match:
        return None
    base, fraction, zone = match.groups()
    try:
        moment = dt.datetime.fromisoformat(base + "." + (fraction or "0")[:6].ljust(6, "0"))
    except ValueError:
        return None
    if zone in (None, "Z"):
        moment = moment.replace(tzinfo=dt.timezone.utc)
    else:
        sign = 1 if zone[0] == "+" else -1
        digits = zone[1:].replace(":", "")
        moment = moment.replace(tzinfo=dt.timezone(sign * dt.timedelta(hours=int(digits[:2]), minutes=int(digits[2:]))))
    return moment.timestamp()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def short_hash(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12] if value else None


def identity_hash(account, user):
    """The quota identity of a login: its user within its workspace (account_id alone when no user id is present)."""
    if not account:
        return None
    return short_hash(f"{account}:{user}" if user else account)


def _claims(jwt):
    """The payload claims of a JWT (only ever used to derive hashes, times and the plan)."""
    if not isinstance(jwt, str) or jwt.count(".") < 2:
        return {}
    segment = jwt.split(".")[1]
    try:
        claims = json.loads(base64.urlsafe_b64decode((segment + "=" * (-len(segment) % 4)).encode("ascii")))
    except (ValueError, TypeError):
        return {}
    return claims if isinstance(claims, dict) else {}


def describe_bytes(data):
    """What a login file is, without any credential value: {ok, error, key_names, auth_mode, has_refresh, plan,
    account_hash, session_hash, last_refresh, access_expires, file_sha256}."""
    info = {"ok": False, "error": None, "file_sha256": sha256_bytes(data)}
    try:
        doc = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        info["error"] = "not JSON"
        return info
    if not isinstance(doc, dict):
        info["error"] = "not a JSON object"
        return info
    tokens = doc.get("tokens") if isinstance(doc.get("tokens"), dict) else {}
    access = _claims(tokens.get("access_token"))
    ident = _claims(tokens.get("id_token"))
    auth = ident.get(AUTH_NS) if isinstance(ident.get(AUTH_NS), dict) else (
        access.get(AUTH_NS) if isinstance(access.get(AUTH_NS), dict) else {})
    account = tokens.get("account_id") or auth.get("chatgpt_account_id")
    user = auth.get("chatgpt_user_id") or ident.get("sub")
    info.update(key_names=sorted(doc), token_key_names=sorted(tokens), auth_mode=doc.get("auth_mode"),
                has_api_key=bool(doc.get("OPENAI_API_KEY")), has_refresh=bool(tokens.get("refresh_token")),
                plan=auth.get("chatgpt_plan_type"), account_hash=identity_hash(account, user),
                workspace_hash=short_hash(account),
                session_hash=short_hash(access.get("session_id")), last_refresh=doc.get("last_refresh"),
                access_expires=iso(access.get("exp")))
    problems = []
    if doc.get("auth_mode") not in (None, "chatgpt"):
        problems.append(f"auth_mode {doc.get('auth_mode')!r} (a ChatGPT login is needed)")
    if info["has_api_key"]:
        problems.append("holds an API key")
    if not info["has_refresh"] or not tokens.get("access_token"):
        problems.append("no ChatGPT login (tokens missing)")
    if not account:
        problems.append("no account id")
    info["ok"] = not problems
    info["error"] = "; ".join(problems) or None
    return info


def describe(path):
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        return {"ok": False, "error": f"unreadable ({type(exc).__name__})", "file_sha256": None}
    return describe_bytes(data)


def _write_atomic(path, data, mode=0o600):
    """Write a file by a temporary file in the same folder and os.replace (readers never see a partial file)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _write_json(path, doc):
    _write_atomic(path, (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode("utf-8"))


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _mkdir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o700)
    return path


@contextlib.contextmanager
def locked(root):
    """The pool lock (fcntl, exclusive): every lease, write-back, state change and add holds it for milliseconds."""
    _mkdir(root)
    fd = os.open(Path(root) / "pool.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _pid_start(pid):
    """Start time of a live process (/proc/<pid>/stat field 22), None when it does not run (zombies included)."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None
    if not fields or fields[0] in ("Z", "X"):
        return None
    return fields[19] if len(fields) > 19 else None


def alive(pid, start=None):
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    found = _pid_start(pid)
    if found is None:
        if Path("/proc/self/stat").exists():
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    return start is None or found == start


def slot_file(root, label):
    return Path(root) / "slots" / label / "auth.json"


def slot_labels(root):
    folder = Path(root) / "slots"
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir()
                  if p.is_dir() and LABEL.match(p.name) and (p / "auth.json").is_file())


def load_state(root):
    state = _read_json(Path(root) / "state.json", {})
    if not isinstance(state, dict):
        state = {}
    state.setdefault("format", FORMAT)
    for key in ("accounts", "slots"):
        if not isinstance(state.get(key), dict):
            state[key] = {}
    return state


def save_state(root, state):
    state["updated"] = iso(_now())
    _write_json(Path(root) / "state.json", state)


def settings(root):
    """pool.json over the defaults: cap_percent, est_percent_per_episode (numbers), episodes_per_login (an integer >= 1;
    1 = one episode per login, never shared)."""
    extra = _read_json(Path(root) / "pool.json", {})
    extra = extra if isinstance(extra, dict) else {}
    out = {"cap_percent": CAP_PERCENT, "est_percent_per_episode": EST_PERCENT_PER_EPISODE}
    for key in out:
        if isinstance(extra.get(key), (int, float)) and not isinstance(extra.get(key), bool):
            out[key] = float(extra[key])
    per = extra.get("episodes_per_login")
    out["episodes_per_login"] = per if isinstance(per, int) and not isinstance(per, bool) and per >= 1 \
        else EPISODES_PER_LOGIN
    return out


def episode_limit_s(paths_module=None):
    """The longest episode: deploy.yml agp.episode_timeout_s of the policy folder (agp_core/paths.py POLICY_DIR), else
    DEFAULT_EPISODE_S (never raises)."""
    try:
        paths_module = paths_module or _load_paths()
        text = (Path(paths_module.POLICY_DIR) / "deploy.yml").read_text()
    except Exception:
        return DEFAULT_EPISODE_S
    try:
        import yaml
        value = ((yaml.safe_load(text) or {}).get("agp") or {}).get("episode_timeout_s")
    except ImportError:                          # the agent venv has no PyYAML
        match = re.search(r"^\s+episode_timeout_s:\s*([0-9]+(?:\.[0-9]*)?)\s*(?:#.*)?$", text, re.M)
        value = float(match.group(1)) if match else None
    except Exception:
        value = None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return DEFAULT_EPISODE_S


def share_horizon_s(episode_s=None, paths_module=None):
    """How long a login's access token must stay valid for the login to be shared: the longest episode (the larger of
    deploy.yml agp.episode_timeout_s and `episode_s`) + SHARE_MARGIN_S."""
    longest = episode_limit_s(paths_module)
    if isinstance(episode_s, (int, float)) and not isinstance(episode_s, bool):
        longest = max(longest, float(episode_s))
    return longest + SHARE_MARGIN_S


def shareable(info, now, horizon):
    """Whether a login (describe()) can be shared now: its access token expires more than `horizon` seconds from now."""
    expires = epoch_of((info or {}).get("access_expires"))
    return bool((info or {}).get("ok")) and expires is not None and expires - now > horizon


def leases(root):
    """{lease id: lease record} of the lease files."""
    folder = Path(root) / "leases"
    out = {}
    if folder.is_dir():
        for path in sorted(folder.glob("*.json")):
            record = _read_json(path)
            if isinstance(record, dict):
                out[path.stem] = record
    return out


# ----------------------------------------------------------------------------- usage readings
def _reading(windows, *, plan=None, limit_id=None, reached=None, observed_at=None):
    usable = [w for w in windows if isinstance(w, dict) and isinstance(w.get("used_percent"), (int, float))
              and not isinstance(w.get("used_percent"), bool)]
    if not usable:
        return None
    window = max(usable, key=lambda w: w["used_percent"])          # the binding window
    return {"used_percent": float(window["used_percent"]), "resets_at": window.get("resets_at"),
            "window_minutes": window.get("window_minutes"), "plan_type": plan, "limit_id": limit_id,
            "reached": reached, "observed_at": observed_at}


def reading_from_snapshot(snapshot, observed_at=None):
    """A usage reading from codex's rate_limits (rollout token_count, snake_case) or from the app-server's
    rateLimits (camelCase)."""
    if not isinstance(snapshot, dict):
        return None
    windows = []
    for name in ("primary", "secondary"):
        window = snapshot.get(name)
        if isinstance(window, dict):
            windows.append({"used_percent": window.get("used_percent", window.get("usedPercent")),
                            "resets_at": window.get("resets_at", window.get("resetsAt")),
                            "window_minutes": window.get("window_minutes", window.get("windowDurationMins"))})
    return _reading(windows, plan=snapshot.get("plan_type", snapshot.get("planType")),
                    limit_id=snapshot.get("limit_id", snapshot.get("limitId")),
                    reached=snapshot.get("rate_limit_reached_type", snapshot.get("rateLimitReachedType")),
                    observed_at=observed_at)


def reading_from_rollout(path):
    """The last usage reading of a codex rollout (token_count events), None when it has none."""
    last = None
    try:
        with open(path, errors="replace") as handle:
            for line in handle:
                if '"token_count"' not in line or '"rate_limits"' not in line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                payload = event.get("payload") if isinstance(event, dict) else None
                if isinstance(payload, dict) and payload.get("type") == "token_count":
                    found = reading_from_snapshot(payload.get("rate_limits"), event.get("timestamp"))
                    if found is not None:
                        last = found
    except OSError:
        return None
    return last


def _effective_used(account, now):
    """The account's last used_percent, 0 once that reading's window has reset (or when it was never read)."""
    used = account.get("used_percent")
    if not isinstance(used, (int, float)):
        return 0.0
    resets = account.get("resets_at")
    if isinstance(resets, (int, float)) and resets <= now:
        return 0.0
    return float(used)


def record_reading(state, account_hash, reading):
    """Keep the newest reading of an account (episodes end out of order)."""
    if not account_hash or not reading:
        return False
    account = state["accounts"].setdefault(account_hash, {})
    new = epoch_of(reading.get("observed_at")) or _now()
    old = epoch_of(account.get("observed_at"))
    if old is not None and new < old:
        return False
    account.update({k: reading.get(k) for k in ("used_percent", "resets_at", "window_minutes", "plan_type",
                                                "limit_id", "reached")})
    account["observed_at"] = reading.get("observed_at") or iso(new)
    return True


# ----------------------------------------------------------------------------- codex failures
def codex_error_text(session, tail_chars=20000, tail_events=4000):
    """codex's own error output of an episode: its stderr and the error / turn.failed events (never tool results or
    model messages)."""
    session = Path(session)
    texts = []
    try:
        texts.append((session / "agent_stderr.log").read_text(errors="replace")[-tail_chars:])
    except OSError:
        pass
    try:
        lines = (session / "agent_events.jsonl").read_text(errors="replace").splitlines()[-tail_events:]
    except OSError:
        lines = []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        item = event.get("item")
        if event.get("type") in ("error", "turn.failed") or (isinstance(item, dict) and item.get("type") == "error"):
            texts.append(json.dumps(event)[:4000])
    return "\n".join(texts)


def classify_failure(text):
    """usage_limit | auth | rate_limited | None for codex's failure text."""
    if not text:
        return None
    if USAGE_LIMIT.search(text):
        return "usage_limit"
    if AUTH_FAILURE.search(text):
        return "auth"
    if RATE_LIMITED.search(text):
        return "rate_limited"
    return None


# ----------------------------------------------------------------------------- write-back
def newer(last_refresh, than):
    """Whether last_refresh is later than `than` (an unreadable `than` counts as older; an unreadable last_refresh is
    never newer)."""
    mine, other = epoch_of(last_refresh), epoch_of(than)
    return mine is not None and (other is None or mine > other)


def _settle(lease):
    """Write a renewed login back and delete the copy (the caller holds the lock and removes the lease file). Several
    copies of one login may be out at once (sharing), so a changed copy replaces the target only when it is a valid
    login of the same account and session as the one leased, the target still holds that session (or is gone), and the
    copy is newer (last_refresh) than the target now is. Returns {write_back, last_refresh_before,
    last_refresh_after}; write_back is one of unchanged, written_back, copy_missing, skipped_invalid_copy,
    skipped_other_account, skipped_other_session, skipped_target_changed (another login was put there meanwhile),
    skipped_not_newer (the target is as new or newer, e.g. another copy's renewal was written back first). An OSError
    of the write propagates (the copy and the lease stay for a later release-stale)."""
    copy, target = Path(lease["copy"]), Path(lease["target"])
    out = {"write_back": None, "last_refresh_before": lease.get("last_refresh"), "last_refresh_after": None}
    try:
        data = copy.read_bytes()
    except FileNotFoundError:
        out["write_back"] = "copy_missing"
        return out
    if sha256_bytes(data) == lease.get("sha256_at_lease"):
        out["write_back"] = "unchanged"
    else:
        found = describe_bytes(data)
        out["last_refresh_after"] = found.get("last_refresh")
        try:
            current = target.read_bytes()
        except FileNotFoundError:
            current = None
        there = describe_bytes(current) if current is not None else None
        same_login = (there is not None and there["ok"] and there.get("account_hash") == lease.get("account_hash")
                      and there.get("session_hash") == lease.get("session_hash"))
        if not found["ok"]:
            out["write_back"] = "skipped_invalid_copy"
        elif found.get("account_hash") != lease.get("account_hash"):
            out["write_back"] = "skipped_other_account"
        elif found.get("session_hash") != lease.get("session_hash"):
            out["write_back"] = "skipped_other_session"
        elif current is not None and sha256_bytes(current) != lease.get("sha256_at_lease") and not same_login:
            out["write_back"] = "skipped_target_changed"      # another login was put there meanwhile: it stays
        elif not newer(found.get("last_refresh"),
                       there.get("last_refresh") if current is not None else lease.get("last_refresh")):
            out["write_back"] = "skipped_not_newer"           # the slot file is as new or newer: it stays
        else:
            _write_atomic(target, data)
            out["write_back"] = "written_back"
    copy.unlink()
    return out


def _lease_path(root, lease_id):
    return Path(root) / "leases" / f"{lease_id}.json"


def _lease_id(root, base):
    """A lease id not in use: base, else base.1, base.2, ... (one process may hold several leases of one login)."""
    candidate, number = base, 0
    while _lease_path(root, candidate).exists():
        number += 1
        candidate = f"{base}.{number}"
    return candidate


def recover_stale(root, state):
    """Settle the leases whose process is gone (SIGKILL, crash): [{id, label, write_back}] (lock held)."""
    out = []
    for lease_id, lease in leases(root).items():
        if alive(lease.get("pid"), lease.get("pid_start")):
            continue
        entry = {"id": lease_id, "label": lease.get("label"), "source": lease.get("source")}
        try:
            entry.update(_settle(lease))
        except OSError as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            out.append(entry)
            continue
        if lease.get("source") == "pool" and entry.get("write_back") == "skipped_invalid_copy":
            _disable(state, lease.get("label"), "copy_invalid")
        with contextlib.suppress(FileNotFoundError):
            _lease_path(root, lease_id).unlink()
        out.append(entry)
    return out


def _disable(state, label, reason):
    if label:
        state["slots"].setdefault(label, {})["disabled"] = {"reason": reason, "at": iso(_now())}


# ----------------------------------------------------------------------------- lease / finish (agp_core/run_agent.py)
def public(lease):
    """The part of a lease that goes into launch.json (no path of a credential copy, no file hash)."""
    keys = ("source", "label", "account_hash", "session_hash", "plan", "last_refresh", "access_expires", "exclusive",
            "used_percent_at_select", "resets_at", "effective_percent_at_select", "account_leases", "login_leases",
            "renewal", "recovered_stale")
    return {k: lease.get(k) for k in keys if k in lease}


def _copy_in(source, copy_to):
    data = Path(source).read_bytes()
    _mkdir(Path(copy_to).parent)
    _write_atomic(copy_to, data)
    return data


def _new_lease(root, lease_id, *, source, label, target, copy_to, data, found, owner, pid, exclusive=False):
    lease = {"id": lease_id, "source": source, "label": label, "target": str(target), "copy": str(copy_to),
             "sha256_at_lease": sha256_bytes(data), "account_hash": found.get("account_hash"),
             "session_hash": found.get("session_hash"), "plan": found.get("plan"),
             "last_refresh": found.get("last_refresh"), "access_expires": found.get("access_expires"),
             "exclusive": bool(exclusive), "pid": pid, "pid_start": _pid_start(pid), "owner": owner,
             "started": iso(_now())}
    _mkdir(Path(root) / "leases")
    _write_json(_lease_path(root, lease_id), lease)
    return lease


def _holders(root, alive_only=False):
    """{label: [lease records]} of the pool leases (alive_only: those whose process still runs)."""
    out = {}
    for record in leases(root).values():
        if record.get("source") == "pool" and record.get("label"):
            if alive_only and not alive(record.get("pid"), record.get("pid_start")):
                continue
            out.setdefault(record["label"], []).append(record)
    return out


def _held_mark(records, per_login):
    if any(r.get("exclusive") for r in records):
        return "leased exclusively"
    return f"leased by {len(records)} of {per_login}"


def _pool_report(root, state, holders, now, conf, horizon):
    """Why no slot is usable: one line per account and per slot."""
    by_account = {}
    for label in slot_labels(root):
        found = describe(slot_file(root, label))
        by_account.setdefault(found.get("account_hash") or "unreadable", []).append((label, found))
    parts = []
    for account_hash, entries in sorted(by_account.items()):
        account = state["accounts"].get(account_hash, {})
        slots = []
        for label, found in entries:
            records = holders.get(label, [])
            mark = (f"disabled ({(state['slots'].get(label) or {}).get('disabled', {}).get('reason')})"
                    if (state["slots"].get(label) or {}).get("disabled") else
                    f"unreadable ({found.get('error')})" if not found.get("ok") else
                    (_held_mark(records, conf["episodes_per_login"])
                     + ("" if shareable(found, now, horizon) else ", near its access expiry")) if records else "free")
            slots.append(f"{label} {mark}")
        used = _effective_used(account, now)
        cooling = account.get("cooling_until")
        parts.append(f"account {account_hash}: used {used:.0f}% (cap {conf['cap_percent']:.0f}%), resets "
                     f"{iso(account.get('resets_at')) or 'unknown'}"
                     + (f", cooling until {iso(cooling)} ({account.get('cooling_reason')})"
                        if isinstance(cooling, (int, float)) and cooling > now else "")
                     + "; slots " + ", ".join(slots))
    return "; ".join(parts) or "no slot"


def _renew_locked(root, label, codex, *, pid, timeout_s=RENEW_TIMEOUT_S):
    """The renewal before a lease (the pool lock held, no lease on the slot): the login is copied into a temporary
    CODEX_HOME under a lease of its own (exclusive, so a crash in the middle is settled by recover_stale like an
    episode), renewed by the pinned codex as `refresh` does (codex app-server getAuthStatus refreshToken=true) and
    written back by _settle. Returns {result: renewed | not_renewed, write_back, last_refresh_before,
    last_refresh_after, error} (never raises; no secret)."""
    out = {"result": "not_renewed", "error": None}
    codex_home = Path(root) / "tmp" / f"{label}-renew-{pid}"
    lease_id = None
    try:
        _mkdir(codex_home)
        data = _copy_in(slot_file(root, label), codex_home / "auth.json")
        lease_id = _lease_id(root, f"{label}.renew.{pid}")
        record = _new_lease(root, lease_id, source="pool", label=label, target=slot_file(root, label),
                            copy_to=codex_home / "auth.json", data=data, found=describe_bytes(data),
                            owner="accounts.py renewal before a lease", pid=pid, exclusive=True)
        try:
            _refresh_action(codex or default_codex(_load_paths()), codex_home, timeout_s=timeout_s)
        except (PoolError, OSError) as exc:
            out["error"] = redact(f"{type(exc).__name__}: {exc}")
        out.update(_settle(record))
        _lease_path(root, lease_id).unlink()
        lease_id = None
        if out.get("write_back") == "written_back":
            out["result"] = "renewed"
    except Exception as exc:                  # never break the lease; a left lease is settled by recover_stale
        out["error"] = redact(f"{type(exc).__name__}: {exc}")
    finally:
        if lease_id is None:
            shutil.rmtree(codex_home, ignore_errors=True)
    return out


def lease(home, copy_to, *, owner, pid=None, label=None, maintenance=False, episode_s=None, codex=None):
    """Lease a login for one codex run and copy it to copy_to (0600). Returns the lease (keep it for finish(); put
    public(lease) into records). With a pool: the best slot that can take one more episode (see the docstring: shared
    while its access token outlives share_horizon_s(episode_s), else exclusive; a free login closer to its expiry is
    renewed first with `codex`, default deploy.yml agp.codex_binary); NoAccount when none is usable. `label` +
    `maintenance` (CLI refresh / ratelimits): that slot whatever its marks and use, exclusively, unless it is leased.
    Without a pool: the local user's ~/.codex/auth.json ({"source": "none"} when there is none: codex then fails without
    a login). Under the lock it first settles the leases of dead processes (recover_stale) and then removes the episode
    homes that are kept only for that (release_kept_homes)."""
    root = pool_root(home)
    pid = os.getpid() if pid is None else pid
    labels = slot_labels(root)
    if not labels:
        master = home_login()
        if not master.is_file():
            return {"source": "none"}
        with locked(root):
            state = load_state(root)
            if recover_stale(root, state):
                save_state(root, state)
            release_kept_homes(home)
            data = _copy_in(master, copy_to)
            return _new_lease(root, _lease_id(root, f"home-{pid}"), source="home", label=None, target=master,
                              copy_to=copy_to, data=data, found=describe_bytes(data), owner=owner, pid=pid)
    horizon = share_horizon_s(episode_s)
    with locked(root):
        state = load_state(root)
        recovered = recover_stale(root, state)
        release_kept_homes(home)
        now = _now()
        conf = settings(root)
        per_login = conf["episodes_per_login"]
        holders = _holders(root)                     # after recover_stale: live processes only
        found = {name: describe(slot_file(root, name)) for name in labels}
        per_account = {}
        for name, records in holders.items():
            if name in found and found[name].get("account_hash"):
                account_hash = found[name]["account_hash"]
                per_account[account_hash] = per_account.get(account_hash, 0) + len(records)
        candidates = []
        for name in ([label] if label else labels):
            info = found.get(name) or {}
            if not info.get("ok"):
                continue
            records = holders.get(name, [])
            share = shareable(info, now, horizon)
            account = state["accounts"].get(info["account_hash"], {})
            cooling = account.get("cooling_until")
            active = per_account.get(info["account_hash"], 0)
            used = _effective_used(account, now)
            effective = used + active * conf["est_percent_per_episode"]
            if maintenance:
                if records:
                    continue
            elif ((state["slots"].get(name) or {}).get("disabled")
                  or (isinstance(cooling, (int, float)) and cooling > now) or effective >= conf["cap_percent"]
                  or len(records) >= per_login
                  or (records and (not share or any(r.get("exclusive") for r in records)))):
                continue
            candidates.append((effective, active, len(records), name, used, account, share))
        if not candidates:
            if recovered:
                save_state(root, state)
            raise NoAccount(_pool_report(root, state, holders, now, conf, horizon))
        effective, active, sharing, name, used, account, share = min(candidates, key=lambda c: (c[0], c[1], c[2], c[3]))
        renewal = None
        if not maintenance and not share and per_login > 1:
            # no lease holds it (above): renew it first, so that it can be shared again
            renewal = _renew_locked(root, name, codex, pid=pid)
            share = shareable(describe(slot_file(root, name)), _now(), horizon)
            renewal["shareable_after"] = share
        data = _copy_in(slot_file(root, name), copy_to)
        record = _new_lease(root, _lease_id(root, f"{name}.{pid}"), source="pool", label=name,
                            target=slot_file(root, name), copy_to=copy_to, data=data, found=describe_bytes(data),
                            owner=owner, pid=pid, exclusive=maintenance or per_login == 1 or not share)
        slot_state = state["slots"].setdefault(name, {})
        slot_state.update(last_lease=record["started"], account_hash=record["account_hash"])
        if renewal is not None:
            slot_state["last_renewal"] = {"at": iso(_now()), "result": renewal.get("result"),
                                          "write_back": renewal.get("write_back"), "error": renewal.get("error")}
        save_state(root, state)
    record.update(used_percent_at_select=used, resets_at=iso(account.get("resets_at")),
                  effective_percent_at_select=round(effective, 3), account_leases=active + 1,
                  login_leases=sharing + 1, recovered_stale=[r.get("id") for r in recovered])
    if renewal is not None:
        record["renewal"] = renewal
    return record


def _renewed_elsewhere(lease_record):
    """Whether the slot file now holds a newer renewal of the leased login (same account and session), written back
    from another copy: a refresh failure of this copy then comes from that renewal, not from a broken login."""
    there = describe(lease_record.get("target") or "")
    return (bool(there.get("ok")) and there.get("file_sha256") != lease_record.get("sha256_at_lease")
            and there.get("account_hash") == lease_record.get("account_hash")
            and there.get("session_hash") == lease_record.get("session_hash")
            and newer(there.get("last_refresh"), lease_record.get("last_refresh")))


def finish(home, lease_record, *, rollout=None, failure_text=None, codex_failed=False, reading=None):
    """End of a codex run (lock held inside): write a renewed login back, delete the copy and the lease, record the
    usage reading (the rollout's last one, or `reading`), mark a failure. Never raises; returns the account_end
    record (no secret)."""
    out = {"source": (lease_record or {}).get("source")}
    if not lease_record or lease_record.get("source") not in ("pool", "home"):
        return out
    try:
        root = pool_root(home)
        with locked(root):
            state = load_state(root)
            try:
                out.update(_settle(lease_record))
            except OSError as exc:            # the copy and the lease stay: release-stale settles them later
                out["error"] = f"write-back failed: {type(exc).__name__}: {exc}"
                save_state(root, state)
                return out
            with contextlib.suppress(FileNotFoundError):
                _lease_path(root, lease_record["id"]).unlink()
            label, account_hash = lease_record.get("label"), lease_record.get("account_hash")
            if reading is None and rollout is not None:
                reading = reading_from_rollout(rollout)
            if reading is not None:
                record_reading(state, account_hash, reading)
                out.update(used_percent_end=reading.get("used_percent"), resets_at=iso(reading.get("resets_at")))
            kind = classify_failure(failure_text) if codex_failed else None
            out["failure"] = kind
            now = _now()
            if out.get("write_back") == "skipped_invalid_copy" and lease_record.get("source") == "pool":
                _disable(state, label, "copy_invalid")
                out["disabled"] = "copy_invalid"
            if kind == "usage_limit" and account_hash:
                resets = (reading or {}).get("resets_at")
                until = resets if isinstance(resets, (int, float)) and resets > now else now + USAGE_LIMIT_COOLING_S
                state["accounts"].setdefault(account_hash, {}).update(cooling_until=until, cooling_reason=kind)
                out["cooling_until"] = iso(until)
            elif kind == "rate_limited" and account_hash:
                until = now + RATE_LIMIT_COOLING_S
                state["accounts"].setdefault(account_hash, {}).update(cooling_until=until, cooling_reason=kind)
                out["cooling_until"] = iso(until)
            elif kind == "auth" and lease_record.get("source") == "pool":
                if _renewed_elsewhere(lease_record):
                    out["not_disabled"] = "the slot holds a newer renewal of the same login (another copy's)"
                else:
                    _disable(state, label, "auth")
                    out["disabled"] = "auth"
            if label:
                state["slots"].setdefault(label, {}).update(last_end=iso(now), last_write_back=out.get("write_back"))
            save_state(root, state)
    except Exception as exc:                  # never break the launcher's cleanup
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def pool_summary(home):
    """Not frozen (agp_core/freeze.py): which slots exist, by label and account hash; no secret."""
    root = pool_root(home)
    labels = slot_labels(root)
    accounts = {}
    for name in labels:
        accounts.setdefault(describe(slot_file(root, name)).get("account_hash") or "unreadable", []).append(name)
    return {"pool": bool(labels), "slots": len(labels), "labels": labels, "accounts": accounts,
            "home_login_present": home_login().is_file()}


def capacity(home, now=None, episode_s=None):
    """How many episodes could lease a login now, by the rule of lease() (read only, no lock; no credential value in the
    result): {pool, slots, episodes_per_login, leased (live leases of the pool), usable, usable_labels, per_slot: {label:
    episodes}, not_usable: {label: why}}. A login can take episodes_per_login episodes less its live leases, none while a lease holds it exclusively or
    while it is leased and too close to its access expiry to be shared; a free login close to its expiry counts in full
    (the next lease renews it first; should that renewal fail, the login takes one episode). Per account, episodes are
    counted only while the account's effective use stays below the cap. A lease whose process is gone counts as free
    (the next lease settles it first). agp_core/run_official.py caps its parallel codex episodes to `usable`."""
    root = pool_root(home)
    labels = slot_labels(root)
    conf = settings(root)
    per_login = conf["episodes_per_login"]
    out = {"pool": bool(labels), "slots": len(labels), "episodes_per_login": per_login, "leased": 0, "usable": 0,
           "usable_labels": [], "per_slot": {}, "not_usable": {}}
    if not labels:
        return out
    state = load_state(root)
    now = _now() if now is None else now
    horizon = share_horizon_s(episode_s)
    holders = _holders(root, alive_only=True)
    out["leased"] = sum(len(records) for records in holders.values())
    found = {name: describe(slot_file(root, name)) for name in labels}
    active = {}
    for name, records in holders.items():
        account_hash = (found.get(name) or {}).get("account_hash")
        if account_hash:
            active[account_hash] = active.get(account_hash, 0) + len(records)
    free = {}
    for name in labels:
        info = found[name]
        records = holders.get(name, [])
        account = state["accounts"].get(info.get("account_hash"), {}) if info.get("ok") else {}
        cooling = account.get("cooling_until")
        why = (f"unreadable ({info.get('error')})" if not info.get("ok") else
               "disabled" if (state["slots"].get(name) or {}).get("disabled") else
               "account cooling" if isinstance(cooling, (int, float)) and cooling > now else
               "leased exclusively" if any(r.get("exclusive") for r in records) else
               f"leased by {len(records)}, near its access expiry (not shared)"
               if records and not shareable(info, now, horizon) else
               f"leased by {len(records)} of {per_login}" if len(records) >= per_login else None)
        if why:
            out["not_usable"][name] = why
        else:
            free.setdefault(info["account_hash"], []).append((name, per_login - len(records)))
    for account_hash, entries in free.items():
        used = _effective_used(state["accounts"].get(account_hash, {}), now)
        taken = active.get(account_hash, 0)
        for name, room in sorted(entries):
            count = 0
            while count < room and used + taken * conf["est_percent_per_episode"] < conf["cap_percent"]:
                count += 1
                taken += 1
            if count:
                out["per_slot"][name] = count
            else:
                out["not_usable"][name] = "account at its usage cap"
    out["usable_labels"] = sorted(out["per_slot"])
    out["usable"] = sum(out["per_slot"].values())
    return out


# ----------------------------------------------------------------------------- episode homes (agp_core/run_agent.py)
# Each codex episode runs with a CODEX_HOME of its own (AGP_HOME/private/codex_episodes/<run>-<session>-<pid>). At the
# end agp_core/run_agent.py copies codex's rollout and its logs database into the session and deletes the home; it keeps
# the home (and writes KEPT_MARK into it) when its login copy could not be settled or a copy into the session failed.
# release_kept_homes() removes such a home later (release-stale, and at every lease).
CODEX_LOGS = "codex_logs"                 # session subfolder for codex's logs database
KEPT_MARK = "codex_home_kept.json"        # a home agp_core/run_agent.py kept: no secret, only the session and the reason
ROLLOUT = "policy_rollout.jsonl"
# Credential-like content in a copy of codex's logs database: such a copy is deleted, not kept. Only
# counted, never printed or recorded. Each pattern starts with a literal, so the scan stays linear on binary data; an
# e-mail address is found from its '@' (EMAIL_DOMAIN) and checked backwards for a local part.
CREDENTIAL_LIKE = (re.compile(rb"eyJ[A-Za-z0-9_-]{16,}"),                    # a JWT (base64url of '{"')
                   re.compile(rb"refresh_token"),
                   re.compile(rb"Bearer "),
                   re.compile(rb"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{16,}"))   # an API key
EMAIL_DOMAIN = re.compile(rb"@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
EMAIL_LOCAL_CHAR = re.compile(rb"[A-Za-z0-9._%+-]")


def credential_like_hits(paths):
    """Number of credential-like matches (CREDENTIAL_LIKE, e-mail addresses) in the files; the matches themselves are
    never returned."""
    hits = 0
    for path in paths:
        data = Path(path).read_bytes()
        hits += sum(sum(1 for _ in pattern.finditer(data)) for pattern in CREDENTIAL_LIKE)
        hits += sum(1 for m in EMAIL_DOMAIN.finditer(data)
                    if m.start() > 0 and EMAIL_LOCAL_CHAR.fullmatch(data[m.start() - 1:m.start()]))
    return hits


def copy_codex_logs(codex_home, session):
    """Copy codex's logs database into session/codex_logs/ once codex has exited: every logs_*.sqlite together with its
    -wal and -shm files, byte for byte as one set (the database file first); nothing is opened, so the copy holds what
    codex wrote, including the frames still in the WAL. Returns the copied names (an OSError propagates)."""
    copied = []
    for database in sorted(Path(codex_home).glob("logs_*.sqlite")):
        for path in (database, database.with_name(database.name + "-wal"), database.with_name(database.name + "-shm")):
            if path.is_file() and not path.is_symlink():
                (Path(session) / CODEX_LOGS).mkdir(exist_ok=True)
                shutil.copy2(path, Path(session) / CODEX_LOGS / path.name)
                copied.append(path.name)
    return copied


def keep_codex_logs(codex_home, session):
    """copy_codex_logs, then the copies are scanned for credential-like content (credential_like_hits): on any hit they
    are deleted. Returns policy_finished.json `codex_logs`: {"kept": true, "files": [...]}, or {"kept": false,
    "reason": "credential-like content (N hits)"} / {"kept": false, "reason": "no logs database"}. An OSError of a copy
    propagates (the caller keeps the home)."""
    copied = copy_codex_logs(codex_home, session)
    if not copied:
        return {"kept": False, "reason": "no logs database"}
    folder = Path(session) / CODEX_LOGS
    hits = credential_like_hits(folder / name for name in copied)
    if hits:
        for name in copied:
            with contextlib.suppress(FileNotFoundError):
                (folder / name).unlink()
        with contextlib.suppress(OSError):
            folder.rmdir()                        # only when nothing else is in it
        return {"kept": False, "reason": f"credential-like content ({hits} hits)"}
    return {"kept": True, "files": copied}


def remove_codex_home(codex_home):
    """Delete an episode's CODEX_HOME; None, or the paths that could not be deleted (with the error type)."""
    errors = []

    def record(function, path, info):
        kind = type(info).__name__ if isinstance(info, BaseException) else info[0].__name__
        errors.append(f"{path}: {kind}")
    if sys.version_info >= (3, 12):
        shutil.rmtree(codex_home, onexc=record)
    else:
        shutil.rmtree(codex_home, onerror=record)
    return errors or None


def own_rollout(path, session):
    """Whether a codex rollout file belongs to the session (its recorded cwd)."""
    try:
        with Path(path).open() as handle:
            meta = json.loads(handle.readline())
    except (OSError, ValueError):
        return False
    return Path(str((meta.get("payload") or {}).get("cwd", ""))).resolve() == Path(session).resolve()


def copy_own_rollout(codex_home, session):
    """Copy the latest rollout of the session from an episode home into session/policy_rollout.jsonl (and its source
    path into policy_rollout_source.txt); the copied name or None. An OSError propagates."""
    sessions = Path(codex_home) / "sessions"
    found = [p for p in sessions.rglob("rollout-*.jsonl") if own_rollout(p, session)] if sessions.is_dir() else []
    if not found:
        return None
    latest = max(found, key=lambda p: p.stat().st_mtime)
    shutil.copy2(latest, Path(session) / ROLLOUT)
    (Path(session) / "policy_rollout_source.txt").write_text(str(latest) + "\n")
    return latest.name


def mark_kept_home(codex_home, session, reason, *, logs_done, rollout_done, pid=None):
    """agp_core/run_agent.py keeps the episode's home: KEPT_MARK names the session, the launcher (pid and start time) and
    what still has to be copied, so release_kept_homes() can remove the home once the launcher is gone. Returns
    {"mark": KEPT_MARK} or {"mark_error": ...} (never raises; no secret)."""
    pid = os.getpid() if pid is None else pid
    doc = {"format": "agp-codex-home-kept/1", "session": str(session), "pid": pid, "pid_start": _pid_start(pid),
           "reason": reason, "logs_done": bool(logs_done), "rollout_done": bool(rollout_done), "at": iso(_now())}
    try:
        _write_json(Path(codex_home) / KEPT_MARK, doc)
    except OSError as exc:
        return {"mark_error": f"{type(exc).__name__}: {exc}"}
    return {"mark": KEPT_MARK}


def release_kept_homes(home):
    """Remove the episode homes agp_core/run_agent.py kept (KEPT_MARK) once their launcher is gone and their login copy is
    settled (no auth.json left: recover_stale settles a lease's copy first). What the session still lacks is copied
    before, as at the end of an episode: the logs database (keep_codex_logs, with its credential scan) and the rollout.
    A home whose copy fails, whose session folder is gone or that still holds a login copy stays; homes without the mark
    (such as one of a launcher killed before its `finally`) are never touched. The session gets
    codex_home_release.json. Returns [{home, result, ...}] (never raises; no secret)."""
    out = []
    for mark in sorted((Path(home) / "private" / "codex_episodes").glob(f"*/{KEPT_MARK}")):
        folder = mark.parent
        entry = {"home": folder.name}
        try:
            doc = _read_json(mark)
            if not isinstance(doc, dict):
                entry["result"] = "kept: unreadable mark"
            elif alive(doc.get("pid"), doc.get("pid_start")):
                continue                                     # its launcher still runs
            elif (folder / "auth.json").exists():
                entry["result"] = "kept: login copy not settled yet"
            elif not doc.get("session") or not Path(str(doc["session"])).is_dir():
                entry["result"] = "kept: session folder missing"
            else:
                session = Path(str(doc["session"]))
                if not doc.get("logs_done"):
                    entry["codex_logs"] = keep_codex_logs(folder, session)
                if not doc.get("rollout_done") and not (session / ROLLOUT).exists():
                    entry["rollout"] = copy_own_rollout(folder, session)
                errors = remove_codex_home(folder)
                entry["result"] = "removed" if not errors and not folder.exists() else \
                    f"remove errors ({len(errors or [])})"
                record = dict(entry, released=iso(_now()), reason_kept=doc.get("reason"),
                              by="agp_core/accounts.py release_kept_homes")
                with contextlib.suppress(OSError):
                    (session / "codex_home_release.json").write_text(json.dumps(record, indent=1) + "\n")
        except OSError as exc:
            entry["result"] = f"kept: {type(exc).__name__}: {exc}"
        out.append(entry)
    return out


# ----------------------------------------------------------------------------- the pinned codex (CLI)
def _codex_env(codex_home):
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"),
           "CODEX_HOME": str(codex_home), "LANG": os.environ.get("LANG", "C.UTF-8")}
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY", "all_proxy",
                 "ALL_PROXY", "SSL_CERT_FILE", "SSL_CERT_DIR"):
        if name in os.environ:
            env[name] = os.environ[name]
    return env


def redact(text):
    """Defensive: long token-like runs are cut out of any codex text this module prints."""
    return re.sub(r"[A-Za-z0-9_\-.=+/]{28,}", "<cut>", str(text))


def app_server_call(codex, codex_home, method, params, timeout_s=90.0):
    """One request to `codex app-server` (stdio, one JSON message per line) after the initialize handshake; returns
    the result (the caller reads only what it needs). Raises PoolError on an error answer, an exit or the timeout."""
    process = subprocess.Popen([str(codex), "app-server", "--listen", "stdio://"], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=str(codex_home),
                               env=_codex_env(codex_home))
    deadline = time.monotonic() + timeout_s
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    buffer = b""

    def send(message):
        process.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        process.stdin.flush()

    def answer(request_id):
        nonlocal buffer
        while True:
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if isinstance(message, dict) and message.get("id") == request_id and "method" not in message:
                    if "error" in message:
                        error = message["error"] if isinstance(message["error"], dict) else {}
                        raise PoolError(f"codex app-server {method}: {redact(error.get('message'))}")
                    return message.get("result")
            left = deadline - time.monotonic()
            if left <= 0:
                raise PoolError(f"codex app-server gave no answer within {timeout_s:.0f} s")
            if not selector.select(timeout=left):
                continue
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                raise PoolError(f"codex app-server exited (code {process.poll()})")
            buffer += chunk

    try:
        send({"id": 1, "method": "initialize",
              "params": {"clientInfo": {"name": "agp_accounts", "title": None, "version": "1"}}})
        answer(1)
        send({"method": "initialized"})
        send({"id": 2, "method": method, "params": params})
        return answer(2)
    finally:
        selector.close()
        with contextlib.suppress(OSError):
            process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _maintenance(home, label, codex, action):
    """Lease `label`, run `action(codex_home)` on a temporary CODEX_HOME holding the copy, settle like an episode.
    Returns (action result, account_end, lease)."""
    root = pool_root(home)
    if label not in slot_labels(root):
        raise PoolError(f"no slot {label!r}")
    codex_home = _mkdir(root / "tmp" / f"{label}-{os.getpid()}")
    try:
        record = lease(home, codex_home / "auth.json", owner=f"accounts.py {action.__name__}", label=label,
                       maintenance=True)
    except NoAccount as exc:
        shutil.rmtree(codex_home, ignore_errors=True)
        raise PoolError(f"slot {label} is not usable now: {exc}") from None
    result, reading, error = None, None, None
    try:
        result, reading = action(codex, codex_home)
    except PoolError as exc:
        error = str(exc)
    end = finish(home, record, reading=reading)
    shutil.rmtree(codex_home, ignore_errors=True)
    if error:
        raise PoolError(f"{error} (login file: {end.get('write_back')})")
    return result, end, record


def _refresh_action(codex, codex_home, timeout_s=90.0):
    result = app_server_call(codex, codex_home, "getAuthStatus", {"includeToken": False, "refreshToken": True},
                             timeout_s=timeout_s)
    method = result.get("authMethod") if isinstance(result, dict) else None
    return {"auth_method": method}, None


_refresh_action.__name__ = "refresh"


def _ratelimits_action(codex, codex_home):
    result = app_server_call(codex, codex_home, "account/rateLimits/read", {"excludeResetCreditDetails": True})
    snapshot = result.get("rateLimits") if isinstance(result, dict) else None
    reading = reading_from_snapshot(snapshot, iso(_now()))
    return {"reading": reading}, reading


_ratelimits_action.__name__ = "ratelimits"


# ----------------------------------------------------------------------------- CLI
def _load_paths():
    spec = importlib.util.spec_from_file_location("agp_accounts_paths", Path(__file__).resolve().with_name("paths.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_codex(paths_module):
    """deploy.yml agp.codex_binary of the policy folder (agp_core/paths.py deploy_pins)."""
    deploy = Path(paths_module.POLICY_DIR) / "deploy.yml"
    try:
        text = deploy.read_text()
    except OSError:
        raise PoolError(f"no {deploy}; pass --codex") from None
    try:
        import yaml
        agp = (yaml.safe_load(text) or {}).get("agp") or {}
    except ImportError:
        match = re.search(r"^\s+codex_binary:\s*(\S+)", text, re.M)
        agp = {"codex_binary": match.group(1)} if match else {}
    binary = paths_module.deploy_pins(agp, deploy.parent).get("codex_binary")
    if not isinstance(binary, str) or not os.path.isfile(binary):
        raise PoolError(f"deploy.yml agp.codex_binary {binary!r} is not a file; pass --codex")
    return binary


def _mode(path):
    try:
        return oct(os.stat(path).st_mode & 0o777)
    except OSError:
        return None


def cmd_add(home, label, staging, replace=False):
    if not LABEL.match(label or ""):
        raise PoolError(f"label {label!r}: lower-case letters, digits, - and _, at most 32")
    root = pool_root(home)
    staging = Path(staging)
    source = staging / "auth.json" if staging.is_dir() else staging
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise PoolError(f"cannot read {source} ({type(exc).__name__})") from None
    found = describe_bytes(data)
    if not found["ok"]:
        raise PoolError(f"{source}: {found['error']}")
    with locked(root):
        state = load_state(root)
        labels = slot_labels(root)
        if label in labels and not replace:
            raise PoolError(f"slot {label} exists (use --replace for a new login of it)")
        if any(r.get("label") == label and alive(r.get("pid"), r.get("pid_start")) for r in leases(root).values()):
            raise PoolError(f"slot {label} is leased now")
        same_account = 0
        for other in labels:
            if other == label:
                continue
            seen = describe(slot_file(root, other))
            if seen.get("file_sha256") == found["file_sha256"] or (
                    found.get("session_hash") and seen.get("session_hash") == found["session_hash"]):
                raise PoolError(f"this login is already slot {other}: one login per slot (log in again for a "
                                f"second slot)")
            if seen.get("account_hash") == found["account_hash"]:
                same_account += 1
        if same_account >= MAX_SLOTS_PER_ACCOUNT:
            raise PoolError(f"account {found['account_hash']} has {same_account} slots already (at most "
                            f"{MAX_SLOTS_PER_ACCOUNT})")
        _mkdir(root / "slots")
        _mkdir(root / "slots" / label)
        _write_atomic(slot_file(root, label), data)
        state["slots"][label] = {"account_hash": found["account_hash"], "session_hash": found["session_hash"],
                                 "plan": found["plan"], "added": iso(_now()), "disabled": None}
        save_state(root, state)
    source.unlink()
    if staging.is_dir() and paths_inside(staging, root / "staging"):
        shutil.rmtree(staging, ignore_errors=True)
    print(f"slot {label}: account {found['account_hash']}, session {found['session_hash']}, plan {found['plan']}, "
          f"last_refresh {found['last_refresh']}, access expires {found['access_expires']}")


def paths_inside(path, root):
    return Path(os.path.realpath(path)).is_relative_to(os.path.realpath(root))


def cmd_check(home, labels=None):
    root = pool_root(home)
    present = slot_labels(root)
    wanted = labels or present
    state = load_state(root)
    holders = _holders(root)
    per_login = settings(root)["episodes_per_login"]
    horizon = share_horizon_s()
    problems, sessions, per_account = [], {}, {}
    now = _now()
    if _mode(root) not in (None, "0o700"):
        problems.append(f"{root} has mode {_mode(root)} (0700 wanted)")
    for name in wanted:
        if name not in present:
            problems.append(f"{name}: no slot")
            continue
        path = slot_file(root, name)
        found = describe(path)
        mark = (state["slots"].get(name) or {}).get("disabled")
        records = holders.get(name, [])
        account = state["accounts"].get(found.get("account_hash"), {})
        hours = None
        expires = epoch_of(found.get("access_expires"))
        if expires is not None:
            hours = round((expires - now) / 3600, 1)
        sharing = (f"shared (up to {per_login} episodes) until {iso(expires - horizon)}"
                   if per_login > 1 and shareable(found, now, horizon) else
                   "one episode at a time" if per_login == 1 else
                   "near its access expiry: leased exclusively, or renewed first when free")
        print(f"{name}: {'ok' if found['ok'] else 'PROBLEM ' + str(found['error'])}; file {_mode(path)}, folder "
              f"{_mode(path.parent)}; account {found.get('account_hash')}, session {found.get('session_hash')}, plan "
              f"{found.get('plan')}; last_refresh {found.get('last_refresh')}; access expires "
              f"{found.get('access_expires')} ({hours} h); {sharing}; "
              + (f"{_held_mark(records, per_login)} (processes alive: "
                 f"{sum(alive(r.get('pid'), r.get('pid_start')) for r in records)})" if records else "free")
              + (f"; DISABLED ({mark.get('reason')})" if mark else "")
              + f"; account used {_effective_used(account, now):.0f}%")
        if not found["ok"]:
            problems.append(f"{name}: {found['error']}")
        if _mode(path) != "0o600" or _mode(path.parent) != "0o700":
            problems.append(f"{name}: modes {_mode(path)} / {_mode(path.parent)} (0600 / 0700 wanted)")
        if found.get("session_hash"):
            sessions.setdefault(found["session_hash"], []).append(name)
        per_account.setdefault(found.get("account_hash"), []).append(name)
        if hours is not None and hours < 0:
            print(f"  note: {name}'s access expired; codex renews it at the next use (the login stays valid while its "
                  f"refresh is accepted)")
    for session_hash, names in sessions.items():
        if len(names) > 1:
            problems.append(f"slots {names} hold the same login (session {session_hash}): one login per slot")
    for account_hash, names in per_account.items():
        if len(names) > MAX_SLOTS_PER_ACCOUNT:
            problems.append(f"account {account_hash} has {len(names)} slots (at most {MAX_SLOTS_PER_ACCOUNT})")
    if not present:
        print(f"no pool in {root}: episodes use {home_login()} "
              f"({'present' if home_login().is_file() else 'missing'})")
    for problem in problems:
        print(f"PROBLEM {problem}")
    return 1 if problems else 0


def cmd_status(home):
    root = pool_root(home)
    state = load_state(root)
    now = _now()
    held = leases(root)
    summary = pool_summary(home)
    print(f"pool {root}: {summary['slots']} slots, {len(summary['accounts'])} accounts, settings {settings(root)}")
    for account_hash, names in sorted(summary["accounts"].items()):
        account = state["accounts"].get(account_hash, {})
        cooling = account.get("cooling_until")
        busy = [r.get("label") for r in held.values() if r.get("label") in names]
        print(f"account {account_hash}: slots {names}, leased {busy}, used {_effective_used(account, now):.0f}% "
              f"(read {account.get('observed_at')}), resets {iso(account.get('resets_at'))}, plan "
              f"{account.get('plan_type')}"
              + (f", cooling until {iso(cooling)} ({account.get('cooling_reason')})"
                 if isinstance(cooling, (int, float)) and cooling > now else ""))
    for lease_id, record in sorted(held.items()):
        print(f"lease {lease_id}: owner {record.get('owner')}, since {record.get('started')}, process "
              f"{'alive' if alive(record.get('pid'), record.get('pid_start')) else 'gone (release-stale settles it)'}")
    return 0


def cmd_refresh(home, label, codex):
    root = pool_root(home)
    before = describe(slot_file(root, label)).get("last_refresh") if label in slot_labels(root) else None
    result, end, _ = _maintenance(home, label, codex, _refresh_action)
    after = describe(slot_file(root, label)).get("last_refresh")
    renewed = end.get("write_back") == "written_back" and after != before
    print(f"slot {label}: {'renewed' if renewed else 'NOT renewed'}; last_refresh {before} -> {after}; login file "
          f"{end.get('write_back')}; auth method {result.get('auth_method')}")
    return 0 if renewed else 1


def cmd_ratelimits(home, label, codex):
    result, end, record = _maintenance(home, label, codex, _ratelimits_action)
    reading = result.get("reading") or {}
    print(f"slot {label} (account {record.get('account_hash')}): used {reading.get('used_percent')}% of a "
          f"{reading.get('window_minutes')}-minute window, resets {iso(reading.get('resets_at'))}, plan "
          f"{reading.get('plan_type')}, reached {reading.get('reached')}; login file {end.get('write_back')}")
    return 0 if reading else 1


def _orphan_copies(home):
    """Login copies in episode homes (private/codex_episodes/<...>-<pid>/auth.json) whose process is gone and that no
    lease names: left by launchers killed before this pool existed."""
    named = {r.get("copy") for r in leases(pool_root(home)).values()}
    out = []
    for path in sorted((Path(home) / "private" / "codex_episodes").glob("*/auth.json")):
        tail = path.parent.name.rsplit("-", 1)[-1]
        if str(path) in named or not tail.isdigit() or alive(int(tail)):
            continue
        out.append(path)
    return out


def cmd_release_stale(home):
    root = pool_root(home)
    with locked(root):
        state = load_state(root)
        settled = recover_stale(root, state)
        busy = {r.get("label") for r in leases(root).values() if r.get("source") == "pool"}
        targets = [slot_file(root, name) for name in slot_labels(root) if name not in busy] + [home_login()]
        orphans = []
        for path in _orphan_copies(home):
            found = describe(path)
            entry = {"copy": path.parent.name, "write_back": "deleted"}
            for target in targets:
                current = describe(target)
                if not (found.get("ok") and current.get("ok") and found.get("session_hash")
                        and found["session_hash"] == current.get("session_hash")
                        and found.get("account_hash") == current.get("account_hash")):
                    continue
                newer = (epoch_of(found.get("last_refresh")) or 0) > (epoch_of(current.get("last_refresh")) or 0)
                if newer:                     # the copy holds the renewed login: it replaces the older one
                    _write_atomic(target, Path(path).read_bytes())
                    entry["write_back"] = f"written_back to {target.parent.name}"
                break
            path.unlink()
            orphans.append(entry)
        save_state(root, state)
        released = release_kept_homes(home)           # after the settling above (their login copies are gone)
    for entry in settled:
        print(f"lease {entry['id']}: settled ({entry.get('write_back') or entry.get('error')})")
    for entry in orphans:
        print(f"old copy {entry['copy']}: {entry['write_back']}")
    for entry in released:
        logs = entry.get("codex_logs")
        print(f"kept home {entry['home']}: {entry['result']}"
              + (f"; logs {'kept' if logs.get('kept') else 'not kept: ' + str(logs.get('reason'))}" if logs else "")
              + (f"; rollout {entry['rollout']}" if entry.get("rollout") else ""))
    if not settled and not orphans and not released:
        print("nothing stale")
    return 0


def cmd_enable(home, label):
    root = pool_root(home)
    with locked(root):
        state = load_state(root)
        if label not in slot_labels(root):
            raise PoolError(f"no slot {label!r}")
        state["slots"].setdefault(label, {})["disabled"] = None
        account_hash = describe(slot_file(root, label)).get("account_hash")
        if account_hash in state["accounts"]:
            state["accounts"][account_hash].pop("cooling_until", None)
            state["accounts"][account_hash].pop("cooling_reason", None)
        save_state(root, state)
    print(f"slot {label}: enabled")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--home", help="AGP_HOME (default: agp_core/paths.py)")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add")
    add.add_argument("label")
    add.add_argument("staging")
    add.add_argument("--replace", action="store_true")
    check = sub.add_parser("check")
    check.add_argument("labels", nargs="*")
    sub.add_parser("status")
    for name in ("refresh", "ratelimits"):
        one = sub.add_parser(name)
        one.add_argument("label")
        one.add_argument("--codex", help="the pinned codex binary (default deploy.yml agp.codex_binary)")
    sub.add_parser("release-stale")
    enable = sub.add_parser("enable")
    enable.add_argument("label")
    args = parser.parse_args(argv)
    paths_module = _load_paths()
    home = Path(os.path.abspath(os.path.expanduser(args.home))) if args.home else paths_module.AGP_HOME
    os.umask(0o077)
    try:
        if args.command == "add":
            return cmd_add(home, args.label, args.staging, args.replace) or 0
        if args.command == "check":
            return cmd_check(home, args.labels)
        if args.command == "status":
            return cmd_status(home)
        if args.command in ("refresh", "ratelimits"):
            codex = args.codex or default_codex(paths_module)
            return (cmd_refresh if args.command == "refresh" else cmd_ratelimits)(home, args.label, codex)
        if args.command == "release-stale":
            return cmd_release_stale(home)
        return cmd_enable(home, args.label)
    except PoolError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
