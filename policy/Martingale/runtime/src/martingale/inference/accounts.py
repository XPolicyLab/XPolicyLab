# Copyright © 2026 Martingale. All Rights Reserved.
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import uuid


class AccountError(RuntimeError):
    """Safe operator-facing configuration or credential error."""


class AccountBusy(AccountError):
    """Another operation is updating account registry metadata."""


class CredentialConflict(AccountError):
    """An external login changed the source credentials during an invocation."""


@dataclass(frozen=True)
class Account:
    alias: str
    credential_home: Path
    expected_email: str | None = None
    runtime_root: Path | None = None

    @property
    def resolved_runtime_root(self):
        return Path(self.runtime_root or Path(self.credential_home) / '.martingale-runtime').expanduser().resolve()


@dataclass(frozen=True)
class Identity:
    email: str
    account_id: str | None = None
    subject: str | None = None


_ALIAS = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z')
_EPISODE = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')
_MAX_AUTH = 1024 * 1024


def _alias(value):
    if not isinstance(value, str) or not _ALIAS.fullmatch(value):
        raise AccountError('Account alias must contain only letters, numbers, underscores or hyphens')
    return value


def _forbidden_roots(extra=()):
    source = Path(__file__).resolve().parents[3]
    roots = [source, *(parent for parent in source.parents if (parent / '.git').exists())]
    roots.extend(Path(path).expanduser().resolve() for path in extra)
    if os.environ.get('MARTINGALE_OUTPUT_ROOT'):
        roots.append(Path(os.environ['MARTINGALE_OUTPUT_ROOT']).expanduser().resolve())
    return roots


def _private_path(value, *, forbidden_roots=(), kind='Private account path'):
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise AccountError(f'{kind} is required')
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise AccountError(f'{kind} must be absolute')
    path = path.resolve()
    if any(path == root or root in path.parents for root in _forbidden_roots(forbidden_roots)):
        raise AccountError(f'{kind} must be outside source checkouts and episode outputs')
    return path


def _private_directory(path):
    """Create a private directory; never chmod an unrelated existing directory."""
    path = Path(path)
    if path.is_symlink():
        raise AccountError('Private runtime directories must not be symbolic links')
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = path.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise AccountError('Private runtime directory must be owned by this user with mode 0700')
    except OSError:
        raise AccountError('Cannot create or access private account directory') from None


def _read_private(path):
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise AccountError('Credential file must be owned by this user with mode 0600')
        with os.fdopen(descriptor, 'rb') as stream:
            descriptor = None
            data = stream.read(_MAX_AUTH + 1)
        if len(data) > _MAX_AUTH:
            raise AccountError('Credential file is too large')
        return data
    except OSError:
        raise AccountError('Operator subscription credential file is unavailable') from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _identity(data, expected_email=None):
    try:
        auth = json.loads(data)
        if not isinstance(auth, dict) or auth.get('OPENAI_API_KEY') or auth.get('auth_mode') not in (None, 'chatgpt'):
            raise ValueError
        tokens = auth['tokens']
        if not all(isinstance(tokens.get(key), str) and tokens[key] for key in ('access_token', 'refresh_token', 'id_token')):
            raise ValueError
        parts = tokens['id_token'].split('.')
        if len(parts) != 3:
            raise ValueError
        claims = json.loads(base64.urlsafe_b64decode(parts[1] + '=' * (-len(parts[1]) % 4)))
        email = claims.get('email')
        if not isinstance(email, str) or not email or '@' not in email or any(c.isspace() for c in email):
            raise ValueError
        provider_claims = claims.get('https://api.openai.com/auth', {})
        claim_account = provider_claims.get('chatgpt_account_id')
        token_account = tokens.get('account_id')
        if claim_account and token_account and claim_account != token_account:
            raise ValueError
        account_id = token_account or claim_account
        subject = claims.get('sub')
        if any(value is not None and (not isinstance(value, str) or not value) for value in (account_id, subject)):
            raise ValueError
    except (ValueError, KeyError, TypeError, AttributeError, UnicodeError):
        raise AccountError('A valid file-based ChatGPT login is required; API-key credentials are unsupported') from None
    if expected_email and email.casefold() != expected_email.casefold():
        raise AccountError('Credential identity does not match the configured account email')
    return Identity(email.casefold(), account_id, subject)


def validate_auth(account):
    """Return safe identity metadata; never return tokens or include them in errors."""
    return _identity(_read_private(Path(account.credential_home) / 'auth.json'), account.expected_email)


def _registry_path(registry_path=None):
    value = registry_path or os.environ.get('MARTINGALE_ACCOUNT_REGISTRY')
    if not value:
        raise AccountError('Set MARTINGALE_ACCOUNT_REGISTRY or provide --registry')
    return _private_path(value, kind='Account registry')


def _registry(registry_path=None, *, forbidden_roots=()):
    path = _private_path(_registry_path(registry_path), forbidden_roots=forbidden_roots, kind='Account registry')
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError):
        raise AccountError('Account registry is unavailable or malformed') from None
    if not isinstance(data, dict) or data.get('version') != 1 or set(data) != {'version', 'active_account', 'runtime_root', 'accounts'}:
        raise AccountError('Unsupported account registry schema; expected version 1')
    runtime = _private_path(data['runtime_root'], forbidden_roots=forbidden_roots, kind='Credential runtime root')
    if not isinstance(data['accounts'], dict):
        raise AccountError('Account registry accounts must be a mapping')
    accounts = {}
    homes = set()
    for alias, spec in data['accounts'].items():
        _alias(alias)
        if not isinstance(spec, dict) or set(spec) != {'credential_home', 'expected_email'}:
            raise AccountError('Every registered account requires credential_home and expected_email')
        home = _private_path(spec['credential_home'], forbidden_roots=forbidden_roots, kind='Credential home')
        email = spec['expected_email']
        if not isinstance(email, str) or '@' not in email or any(c.isspace() for c in email):
            raise AccountError('Every registered account requires its expected email')
        if home in homes:
            raise AccountError('Account aliases must use separate credential directories')
        homes.add(home)
        accounts[alias] = Account(alias, home, email, runtime)
    active = data['active_account']
    if active is not None and (not isinstance(active, str) or active not in accounts):
        raise AccountError('Selected account is absent from the registry')
    if accounts and active is None:
        raise AccountError('The registry must select one registered account')
    return path, data, accounts


def select_account(registry_path=None, *, forbidden_roots=()):
    """Pin metadata for the selected account, without logging in or reading auth."""
    if registry_path or os.environ.get('MARTINGALE_ACCOUNT_REGISTRY'):
        _, data, accounts = _registry(registry_path, forbidden_roots=forbidden_roots)
        if data['active_account'] is None:
            raise AccountError('The account registry is empty; add an authenticated account')
        return accounts[data['active_account']]
    value = os.environ.get('CODEX_HOME') or os.environ.get('MARTINGALE_CODEX_HOME') or str(Path.home() / '.codex')
    home = _private_path(value, forbidden_roots=forbidden_roots, kind='Credential home')
    runtime = _private_path(home / '.martingale-runtime', forbidden_roots=forbidden_roots, kind='Credential runtime root')
    return Account('environment', home, runtime_root=runtime)


def load_account(alias, registry_path=None, *, forbidden_roots=()):
    """Resolve only an operator-registered alias, with no fallback account choice."""
    _alias(alias)
    _, _, accounts = _registry(registry_path, forbidden_roots=forbidden_roots)
    if alias not in accounts:
        raise AccountError('Requested account is absent from the registry')
    return accounts[alias]


def _atomic_write(path, data):
    """Fsync a new private file before an atomic rename in the same directory."""
    path = Path(path)
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
        with os.fdopen(descriptor, 'wb') as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


@contextmanager
def _exclusive_lock(path, *, busy='Account registry is being modified; retry'):
    """Protect registry metadata only; credential operations never call this."""
    descriptor = None
    try:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise AccountError('Registry lock must be a private file owned by this user')
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AccountBusy(busy) from None
        except OSError:
            raise AccountError('Cannot access the private registry lock') from None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _resolution_path(recovery_file):
    return Path(recovery_file).with_suffix('.resolution.json')


def _resolution(account, recovery_file):
    """Fingerprint the exact private files an operator chooses to reconcile."""
    recovery_file = Path(recovery_file)
    return {'version': 1, 'disposition': 'keep-source', 'account': account.alias,
            'source_sha256': hashlib.sha256(_read_private(Path(account.credential_home) / 'auth.json')).hexdigest(),
            'recovery_sha256': hashlib.sha256(_read_private(recovery_file)).hexdigest()}


def reconcile_refresh(account, recovery_file):
    """Explicitly choose the existing source login over a pending rotated token.

    This never installs the pending credential or prints tokens. The live episode
    validates and consumes this exact choice on its next same-account invocation.
    """
    alias = _alias(account.alias)
    recovery_file = _private_path(recovery_file, kind='Private recovery file')
    expected_parent = account.resolved_runtime_root / 'recovery'
    if (recovery_file.parent != expected_parent or recovery_file.suffix != '.json'
            or not re.fullmatch(re.escape(alias) + r'-[0-9a-f]{32}\.json', recovery_file.name)):
        raise AccountError('Recovery file does not belong to this registered account runtime')
    validate_auth(account)
    value = _resolution(account, recovery_file)
    _atomic_write(_resolution_path(recovery_file), (json.dumps(value) + '\n').encode())
    return {'account': alias, 'recovery_file': str(recovery_file), 'resolution': 'keep-source',
            'note': 'Resume the same live account to apply; source credentials were not changed.'}


class AccountRuntime:
    """One episode's persistent private Codex home, pinned to one identity.

    Episodes can use the same account concurrently without credential locks.
    Each invocation stages credentials in its own isolated home. Refreshes use
    best-effort source comparisons and atomic writes, not atomic compare-and-swap;
    concurrent refreshes can still race. Detected conflicts retain the rotated
    credentials in an unmounted recovery directory.
    """

    def __init__(self, runtime_root, episode_id, *, forbidden_roots=()):
        if not isinstance(episode_id, str) or not _EPISODE.fullmatch(episode_id):
            raise AccountError('Invalid private runtime episode identifier')
        self.forbidden_roots = tuple(forbidden_roots)
        self.root = _private_path(runtime_root, forbidden_roots=forbidden_roots, kind='Credential runtime root')
        _private_directory(self.root)
        episodes = self.root / 'episodes'
        _private_directory(episodes)
        self.home = episodes / episode_id
        try:
            self.home.mkdir(mode=0o700, exist_ok=False)
        except FileExistsError:
            raise AccountError('Private episode runtime already exists; each live episode requires a unique identifier') from None
        except OSError:
            raise AccountError('Cannot create private episode runtime') from None
        self.pending_refresh = None
        self._pinned_identity = None
        self._invocations = {}

    def _preserve_refresh(self, account, data):
        recovery = self.root / 'recovery'
        _private_directory(recovery)
        path = recovery / f'{_alias(account.alias)}-{uuid.uuid4().hex}.json'
        _atomic_write(path, data)
        self.pending_refresh = path

    def _consume_resolution(self, account):
        if self.pending_refresh is None:
            return
        marker = _resolution_path(self.pending_refresh)
        try:
            chosen = json.loads(_read_private(marker))
        except (AccountError, ValueError, UnicodeError):
            raise CredentialConflict('Uncommitted credential refresh needs operator diagnosis and accounts recover --keep-source before live resume') from None
        if chosen != _resolution(account, self.pending_refresh):
            raise CredentialConflict('Credential recovery choice no longer matches this account and its files; select it again')
        validate_auth(account)
        marker.unlink()
        self.pending_refresh.unlink()
        self.pending_refresh = None

    @contextmanager
    def lease(self, account):
        home = _private_path(account.credential_home, forbidden_roots=self.forbidden_roots, kind='Credential home')
        # Reject accidental reentry into this one runtime without any credential
        # or filesystem lock. Distinct runtime instances remain independent.
        invocation = object()
        if self._invocations.setdefault('active', invocation) is not invocation:
            raise AccountError('This episode already has an active provider invocation')
        try:
            try:
                self._consume_resolution(account)
            except (AccountError, OSError) as exc:
                if isinstance(exc, CredentialConflict):
                    raise
                raise CredentialConflict('Pending credential recovery could not be validated or applied; operator diagnosis is required') from None
            auth = home / 'auth.json'
            original = _read_private(auth)
            identity = _identity(original, account.expected_email)
            if self._pinned_identity is None:
                self._pinned_identity = identity
            elif identity != self._pinned_identity:
                raise CredentialConflict('Episode credential identity is pinned; account handoff is disabled')
            staged = self.home / 'auth.json'
            if staged.exists() or staged.is_symlink():
                try:
                    self._preserve_refresh(account, _read_private(staged))
                    staged.unlink()
                except (AccountError, OSError):
                    raise CredentialConflict('Private runtime contains credentials from an unfinished invocation; recovery storage remains unavailable') from None
                raise CredentialConflict('Credentials from an unfinished invocation retained privately; use accounts recover --keep-source before resuming')
            _atomic_write(staged, original)
            try:
                yield self.home
            finally:
                updated = None
                safe_to_remove = False
                try:
                    updated = _read_private(staged)
                    updated_identity = _identity(updated, account.expected_email)
                    if updated_identity != identity:
                        raise CredentialConflict('Refreshed credential identity changed; source credentials were not overwritten')
                    if updated != original:
                        current = _read_private(auth)
                        if hashlib.sha256(current).digest() != hashlib.sha256(original).digest():
                            self._preserve_refresh(account, updated)
                            raise CredentialConflict('Source credentials changed during refresh; rotated credentials retained privately for operator diagnosis')
                        try:
                            _atomic_write(auth, updated)
                        except OSError:
                            try:
                                self._preserve_refresh(account, updated)
                            except (AccountError, OSError):
                                raise CredentialConflict('Refresh persistence and recovery storage failed; rotated credentials remain in the private episode runtime') from None
                            raise CredentialConflict('Could not persist refreshed credentials; retained privately for operator diagnosis') from None
                    safe_to_remove = True
                except (AccountError, OSError) as exc:
                    if self.pending_refresh is None and updated is not None:
                        try:
                            self._preserve_refresh(account, updated)
                        except (AccountError, OSError):
                            raise CredentialConflict('Credential recovery storage failed; credentials remain in the private episode runtime') from None
                    safe_to_remove = self.pending_refresh is not None
                    if isinstance(exc, CredentialConflict):
                        raise
                    raise CredentialConflict('Refreshed credential identity or storage could not be validated; retained credentials require operator diagnosis') from None
                finally:
                    if safe_to_remove:
                        try:
                            staged.unlink(missing_ok=True)
                        except OSError:
                            raise CredentialConflict('Private runtime credential cleanup failed; operator diagnosis is required') from None
        finally:
            self._invocations.pop('active', None)


def add_parser(subparsers):
    parser = subparsers.add_parser('accounts', help='Manage private operator account selection')
    parser.add_argument('--registry', type=Path, default=None)
    commands = parser.add_subparsers(dest='accounts_command', required=True)
    init = commands.add_parser('init', help='Create an empty private registry')
    init.add_argument('--runtime-root', type=Path, required=True)
    add = commands.add_parser('add', help='Register an existing authenticated credential store')
    add.add_argument('alias')
    add.add_argument('--home', type=Path, required=True)
    add.add_argument('--email', required=True)
    commands.add_parser('list', help='List registered account metadata without reading credentials')
    select = commands.add_parser('select', help='Select the account for NEW episodes only')
    select.add_argument('alias')
    status = commands.add_parser('status', help='Validate login identity without exposing credentials')
    status.add_argument('alias', nargs='?')
    recover = commands.add_parser('recover', help='Choose the current source login after a credential-refresh conflict')
    recover.add_argument('alias')
    recover.add_argument('--recovery-file', type=Path, required=True)
    recover.add_argument('--keep-source', action='store_true', required=True,
                         help='Explicitly discard the pending refresh on next live resume; authenticate the source first if needed')
    for child in (init, add, select, status, recover, commands.choices['list']):
        child.add_argument('--registry', type=Path, default=argparse.SUPPRESS)
    return parser


def run(args):
    action = args.accounts_command
    path = _registry_path(args.registry)
    if action == 'init':
        runtime = _private_path(args.runtime_root, kind='Credential runtime root')
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with _exclusive_lock(path.with_name(path.name + '.lock'), busy='Account registry is being modified; retry'):
            if path.exists() or path.is_symlink():
                raise AccountError('Account registry already exists; refusing to replace it')
            _private_directory(runtime)
            data = {'version': 1, 'active_account': None, 'runtime_root': str(runtime), 'accounts': {}}
            _atomic_write(path, (json.dumps(data, indent=2) + '\n').encode())
        result = {'registry': str(path), 'active_account': None, 'runtime_root': str(runtime)}
    elif action in ('add', 'select'):
        with _exclusive_lock(path.with_name(path.name + '.lock'), busy='Account registry is being modified; retry'):
            _, data, accounts = _registry(path)
            alias = _alias(args.alias)
            if action == 'add':
                if alias in accounts:
                    raise AccountError('Account alias is already registered')
                home = _private_path(args.home, kind='Credential home')
                if any(account.credential_home == home for account in accounts.values()):
                    raise AccountError('Credential home is already registered')
                if not isinstance(args.email, str) or '@' not in args.email or any(c.isspace() for c in args.email):
                    raise AccountError('Expected account email is required')
                account = Account(alias, home, args.email, Path(data['runtime_root']))
                validate_auth(account)
                data['accounts'][alias] = {'credential_home': str(Path(args.home).expanduser()), 'expected_email': args.email}
                if data['active_account'] is None:
                    data['active_account'] = alias
            else:
                if alias not in accounts:
                    raise AccountError('Requested account is absent from the registry')
                validate_auth(accounts[alias])
                data['active_account'] = alias
            _atomic_write(path, (json.dumps(data, indent=2) + '\n').encode())
        result = {'active_account': data['active_account'], 'applies_to': 'new episodes only'}
    else:
        _, data, accounts = _registry(path)
        if action == 'recover':
            if args.alias not in accounts:
                raise AccountError('Requested account is absent from the registry')
            result = reconcile_refresh(accounts[args.alias], args.recovery_file)
        elif action == 'list':
            result = {'active_account': data['active_account'], 'accounts': [
                {'alias': account.alias, 'credential_home': str(account.credential_home), 'expected_email': account.expected_email}
                for account in accounts.values()]}
        elif action == 'status':
            alias = args.alias or data['active_account']
            if alias not in accounts:
                raise AccountError('No registered account is selected')
            identity = validate_auth(accounts[alias])
            result = {'alias': alias, 'email': identity.email, 'auth_method': 'chatgpt', 'active_for_new_episodes': alias == data['active_account']}
        else:
            raise AccountError('Unknown account command')
    print(json.dumps(result, indent=2))
    return result
