# Copyright © 2026 Martingale. All Rights Reserved.
import os
from pathlib import Path
import shutil


def command(argv, workspace, client_home, credential_home):
    """Expose one episode and operator auth, never research or other sessions.

    Network is shared for the provider and local MCP. Tool configuration disables
    shell/browser/apps; this filesystem boundary also excludes host research if
    a client advertises an unexpected built-in filesystem tool.
    """
    binary=Path(shutil.which(argv[0]) or argv[0]).resolve()
    if not binary.is_file() or not (binary.parent/'codex-code-mode-host').is_file():
        raise RuntimeError('Codex binary and sibling code-mode host are required')
    workspace=Path(workspace).resolve();client_home=Path(client_home).resolve()
    client_home.mkdir(mode=0o700,parents=True,exist_ok=True)
    auth=Path(credential_home).expanduser().resolve()/'auth.json'
    if not auth.is_file():raise RuntimeError('Operator subscription credential file unavailable')
    wrapper=[os.environ.get('MARTINGALE_BWRAP','bwrap'),'--die-with-parent','--unshare-all','--share-net','--new-session']
    for path in ('/usr','/lib','/lib64','/bin','/etc/ssl','/etc/resolv.conf','/etc/hosts','/etc/nsswitch.conf'):
        if Path(path).exists():wrapper+=['--ro-bind',path,path]
    wrapper+=['--proc','/proc','--dev','/dev','--tmpfs','/tmp',
              '--ro-bind',str(binary.parent),str(binary.parent),
              '--bind',str(workspace),str(workspace),
              '--ro-bind',str(Path(__file__).with_name('library')),str(workspace/'library'),
              '--bind',str(client_home),str(client_home),
              '--ro-bind',str(auth),str(client_home/'auth.json'),
              '--setenv','HOME','/tmp','--setenv','CODEX_HOME',str(client_home),
              '--setenv','TMPDIR','/tmp','--chdir',str(workspace)]
    return wrapper+[str(binary),*argv[1:]]


def runtime_command(argv, workspace, client_home, *, include_library=True):
    """Expose a leased private home so Codex can atomically refresh auth.json.

    AccountRuntime owns credential staging and persistence. The account's host
    credential directory and every other episode remain outside the sandbox.
    """
    from .accounts import AccountError, _private_path
    binary = Path(shutil.which(argv[0]) or argv[0]).resolve()
    if not binary.is_file() or not (binary.parent / 'codex-code-mode-host').is_file():
        raise RuntimeError('Codex binary and sibling code-mode host are required')
    workspace = Path(workspace).resolve()
    client_home = _private_path(client_home, forbidden_roots=(workspace.parent,), kind='Private Codex home')
    if not client_home.is_dir() or not (client_home / 'auth.json').is_file():
        raise AccountError('A staged credential lease is required before launching Codex')
    if client_home.stat().st_mode & 0o077:
        raise AccountError('Private Codex home must have mode 0700')
    wrapper = [os.environ.get('MARTINGALE_BWRAP', 'bwrap'), '--die-with-parent', '--unshare-all', '--share-net', '--new-session']
    for path in ('/usr', '/lib', '/lib64', '/bin', '/etc/ssl', '/etc/resolv.conf', '/etc/hosts', '/etc/nsswitch.conf'):
        if Path(path).exists():
            wrapper += ['--ro-bind', path, path]
    wrapper += ['--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
                '--ro-bind', str(binary.parent), str(binary.parent),
                '--bind', str(workspace), str(workspace),
                '--bind', str(client_home), str(client_home),
                '--setenv', 'HOME', '/tmp', '--setenv', 'CODEX_HOME', str(client_home),
                '--setenv', 'MARTINGALE_CODEX_HOME', str(client_home),
                '--setenv', 'TMPDIR', '/tmp', '--chdir', str(workspace)]
    if include_library:
        wrapper += ['--ro-bind', str(Path(__file__).with_name('library')), str(workspace / 'library')]
    else:
        # /usr may contain a wheel install of the policy even when the source
        # library is not mounted. Hide that copy from the calibration client.
        import sysconfig
        for directory in {sysconfig.get_path('purelib'), sysconfig.get_path('platlib')}:
            package=Path(directory)/'martingale'
            if package.is_dir():wrapper+=['--tmpfs',str(package)]
    return wrapper + [str(binary), *argv[1:]]
