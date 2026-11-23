"""Stage experience-backed library edits and publish them between episodes."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(content):
    return hashlib.sha256(content).hexdigest() if content is not None else None


def replace_file(path, content):
    """Replace one file atomically, preserving its permissions."""
    fd, name = tempfile.mkstemp(prefix='.self-play-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
        os.chmod(name, path.stat().st_mode & 0o777 if path.exists() else 0o644)
        if path.suffix == '.py':
            cache = path.parent / '__pycache__'
            if cache.is_symlink():
                raise ValueError('Library bytecode directory cannot be a symlink')
            for bytecode in cache.glob(path.stem + '.*.pyc'):
                bytecode.unlink()
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class LibraryUpdates:
    MAX_FILES = 12
    MAX_BYTES = 1024 * 1024

    def __init__(self, library, archive):
        self.library = Path(library).resolve()
        self.archive = Path(archive)
        self.changes = {}
        self.report = None
        # Detect edits made by another episode even before our first stage call.
        self.initial = {p.relative_to(self.library).as_posix(): digest(p.read_bytes())
                        for p in self.library.rglob('*')
                        if p.is_file() and not p.is_symlink() and p.suffix in ('.md', '.py')}

    def target(self, name):
        if not isinstance(name, str):
            raise ValueError('Expected a library-relative path')
        name = name.removeprefix('library/')
        parts = name.split('/')
        if (len(parts) < 2 or any(p in ('', '.', '..') for p in parts)
                or not all(p.isidentifier() for p in parts[:-1])):
            raise ValueError('Choose a file inside an existing library namespace')
        path = self.library.joinpath(*parts)
        if path.suffix not in ('.md', '.py') or path.name == '__init__.py':
            raise ValueError('Only lesson/document .md and skill .py files can be updated')
        if not path.stem.isidentifier() or not path.parent.is_dir():
            raise ValueError('Choose a valid file name in an existing library directory')
        if any(p.is_symlink() for p in (path, *path.parents) if p != self.library):
            raise ValueError('Library update paths cannot contain symlinks')
        if self.library not in path.resolve().parents or (path.exists() and not path.is_file()):
            raise ValueError('Expected a regular library file')
        return name, path

    def stage(self, name, content, *, summary, evidence):
        if self.report is not None:
            raise RuntimeError('Library updates already finalized')
        name, path = self.target(name)
        if not isinstance(content, str) or not content.strip() or len(content.encode()) > self.MAX_BYTES:
            raise ValueError('Content must be nonempty text, at most 1 MiB')
        if not isinstance(summary, str) or not summary.strip() or len(summary) > 4000:
            raise ValueError('Provide an observed reason and validation summary, at most 4000 characters')
        if not evidence:
            raise ValueError('An update needs completed episode-cell evidence')
        if path.suffix == '.py':
            # Compile without executing model-authored code in the host process.
            compile(content, name, 'exec', dont_inherit=True)
        if name not in self.changes and len(self.changes) >= self.MAX_FILES:
            raise ValueError('At most 12 library files may be staged per episode')
        before = path.read_bytes() if path.exists() else None
        if digest(before) != self.initial.get(name):
            raise RuntimeError('Library file changed since this episode started; do not overwrite it')
        after = content.encode()
        record = {'path': name, 'before_sha256': digest(before), 'after_sha256': digest(after),
                  'summary': summary, 'evidence': evidence, 'source': '--self-play mode',
                  'validation': 'Python compilation' if path.suffix == '.py' else 'nonempty text'}
        directory = self.archive / 'files' / name
        directory.mkdir(parents=True, exist_ok=True)
        if before is not None:
            (directory / 'before').write_bytes(before)
        (directory / 'after').write_bytes(after)
        (directory / 'evidence.json').write_text(json.dumps(record, indent=2))
        self.changes[name] = (before, after, record)
        return {'status': 'staged', **{k: record[k] for k in ('path', 'before_sha256', 'after_sha256')}}

    def pending(self):
        return {'status': self.report['status'] if self.report else 'staged',
                'files': [value[2]['path'] for value in self.changes.values()]}

    def finalize(self, *, eligible, verdict):
        if self.report is not None:
            return self.report
        report = {'status': 'empty', 'verdict': verdict, 'files': list(self.changes)}
        if self.changes and not eligible:
            report['status'] = 'withheld'
            report['reason'] = 'Episode interrupted, abandoned, or a cell outcome is uncertain'
        elif self.changes:
            applied = []
            try:
                # A nonblocking lock avoids delaying episode cleanup behind another writer.
                with (self.library.parent / '.self_play_updates.lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    targets = {name: self.target(name)[1] for name in self.changes}
                    conflicts = [name for name, path in targets.items()
                                 if digest(path.read_bytes() if path.exists() else None)
                                 != self.changes[name][2]['before_sha256']]
                    if conflicts:
                        report.update(status='conflict', conflicts=conflicts)
                    else:
                        try:
                            for name, path in targets.items():
                                replace_file(path, self.changes[name][1])
                                applied.append(name)
                        except Exception:
                            for name in reversed(applied):
                                before = self.changes[name][0]
                                if before is None:
                                    targets[name].unlink()
                                else:
                                    replace_file(targets[name], before)
                            raise
                        report['status'] = 'applied'
            except Exception as error:
                report.update(status='failed', error=f'{type(error).__name__}: {error}')
        self.archive.mkdir(parents=True, exist_ok=True)
        (self.archive / 'result.json').write_text(json.dumps(report, indent=2))
        self.report = report
        return report
