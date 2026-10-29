"""Check distributable files for private paths, network addresses and caches."""
from pathlib import Path
import re
import sys

PATTERNS = {
    'machine_path': re.compile(r'/(?:mnt|home|root|Users)/[^\s\"\x27<>]+'),
    'windows_user_path': re.compile(r'[A-Za-z]:[\\/]+Users[\\/][^\s\"\x27]+'),
    'private_ipv4': re.compile(r'(?<!\d)(?:10\.(?:\d{1,3}\.){2}\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?!\d)'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
}


def scan(root: Path) -> list[tuple[str, int, str]]:
    findings = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or '.git' in path.relative_to(root).parts:
            continue
        relative = str(path.relative_to(root))
        if path.suffix in ('.pyc', '.pyo') or '__pycache__' in path.parts:
            findings.append((relative, 0, 'bytecode_cache'))
        for number, line in enumerate(path.read_bytes().decode('utf-8', errors='ignore').splitlines(), 1):
            for kind, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append((relative, number, kind))
    return findings


if __name__ == '__main__':
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    findings = scan(root)
    for path, number, kind in findings:
        print(f'{path}:{number}: {kind}')
    print(f'Privacy scan: {len(findings)} findings')
    raise SystemExit(bool(findings))
