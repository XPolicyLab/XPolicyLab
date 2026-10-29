from __future__ import annotations
import hashlib
import json
import mmap
import os
import shutil
import tempfile
import time
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import Any, BinaryIO
import numpy as np
SCHEMA_VERSION = 3
LOGIN_USERNAME_ENV = 'GENIE_ML_LOGIN_USERNAME'
SNAPSHOT_MOUNT_ROOT = Path('/mnt')
RECORD_DTYPE = np.dtype([('offset', '<u8'), ('length', '<u4')])
RECAP_DTYPE = np.dtype([('hash', 'S32'), ('key_offset', '<u8'), ('key_length', '<u4'), ('value_offset', '<u8'), ('value_length', '<u4')])

def _scan_json_string_end(data: mmap.mmap, start: int) -> int:
    if start >= len(data) or data[start] != ord('"'):
        raise ValueError(f'expected JSON string at byte offset {start}')
    escaped = False
    index = start + 1
    while index < len(data):
        byte = data[index]
        if escaped:
            escaped = False
        elif byte == ord('\\'):
            escaped = True
        elif byte == ord('"'):
            return index + 1
        index += 1
    raise ValueError('unterminated JSON string')

def _skip_json_whitespace(data: mmap.mmap, start: int) -> int:
    index = start
    while index < len(data) and data[index] in b' \t\r\n':
        index += 1
    return index

def _scan_json_value_end(data: mmap.mmap, start: int) -> int:
    if start >= len(data):
        raise ValueError('missing JSON value')
    first = data[start]
    if first == ord('"'):
        return _scan_json_string_end(data, start)
    if first not in (ord('['), ord('{')):
        index = start
        while index < len(data) and data[index] not in b',}]\n\r \t':
            index += 1
        return index
    stack = [ord(']') if first == ord('[') else ord('}')]
    index = start + 1
    while index < len(data) and stack:
        byte = data[index]
        if byte == ord('"'):
            index = _scan_json_string_end(data, index)
            continue
        if byte == ord('['):
            stack.append(ord(']'))
        elif byte == ord('{'):
            stack.append(ord('}'))
        elif byte in (ord(']'), ord('}')):
            if byte != stack[-1]:
                raise ValueError(f'mismatched JSON delimiter at byte offset {index}')
            stack.pop()
        index += 1
    if stack:
        raise ValueError('unterminated JSON value')
    return index

class SnapshotError(RuntimeError):
    pass

def _login_username() -> str:
    username = os.environ.get(LOGIN_USERNAME_ENV)
    if not username or username != username.strip() or username in {'.', '..'} or (Path(username).name != username) or ('\x00' in username):
        raise SnapshotError(f'{LOGIN_USERNAME_ENV} must be set to one non-empty path segment when metadata_index_manifest is not configured')
    return username

def _default_snapshot_manifest_path(config_path: str | os.PathLike[str] | None) -> Path:
    if config_path is None:
        raise SnapshotError('config_path is required when metadata_index_manifest is not configured')
    config_name = Path(config_path).stem
    if not config_name:
        raise SnapshotError('config_path must have a filename when metadata_index_manifest is not configured')
    return SNAPSHOT_MOUNT_ROOT / _login_username() / 'snapshots' / config_name / 'manifest.json'

def resolve_snapshot_manifest_path(path: str | os.PathLike[str] | None, *, root: str | os.PathLike[str] | None=None, config_path: str | os.PathLike[str] | None=None) -> Path:
    if path is None:
        return _default_snapshot_manifest_path(config_path)
    path_value = os.fspath(path)
    if not path_value:
        raise SnapshotError('metadata_index_manifest must not be an empty path')
    manifest_path = Path(path_value)
    if manifest_path.is_absolute():
        return manifest_path.resolve()
    if root is None:
        raise SnapshotError('repository root is required for a relative metadata_index_manifest')
    return (Path(root) / manifest_path).resolve()

class _ArrayLRU:

    def __init__(self, maxsize: int) -> None:
        if int(maxsize) <= 0:
            raise ValueError('snapshot mmap cache size must be positive')
        self.maxsize = int(maxsize)
        self._items: OrderedDict[str, np.memmap] = OrderedDict()
        self.opens = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, path: str) -> np.memmap:
        path = os.fspath(path)
        try:
            item = self._items.pop(path)
        except KeyError:
            self.misses += 1
            item = np.load(path, mmap_mode='r', allow_pickle=False)
            self.opens += 1
        else:
            self.hits += 1
        self._items[path] = item
        while len(self._items) > self.maxsize:
            (_, mapped) = self._items.popitem(last=False)
            mmap_handle = getattr(mapped, '_mmap', None)
            if mmap_handle is not None:
                mmap_handle.close()
            self.evictions += 1
        return item

    def close(self) -> None:
        for mapped in self._items.values():
            mmap_handle = getattr(mapped, '_mmap', None)
            if mmap_handle is not None:
                mmap_handle.close()
        self._items.clear()

class _FileLRU:

    def __init__(self, maxsize: int) -> None:
        if int(maxsize) <= 0:
            raise ValueError('snapshot file cache size must be positive')
        self.maxsize = int(maxsize)
        self._items: OrderedDict[str, BinaryIO] = OrderedDict()
        self._pid = os.getpid()

    def _ensure_process_local(self) -> None:
        current_pid = os.getpid()
        if current_pid == self._pid:
            return
        self.close()
        self._pid = current_pid

    def get(self, path: str | os.PathLike[str]) -> BinaryIO:
        self._ensure_process_local()
        path = os.fspath(path)
        try:
            handle = self._items.pop(path)
        except KeyError:
            handle = open(path, 'rb')
        self._items[path] = handle
        while len(self._items) > self.maxsize:
            (_, stale_handle) = self._items.popitem(last=False)
            stale_handle.close()
        return handle

    def close(self) -> None:
        for handle in self._items.values():
            handle.close()
        self._items.clear()

def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')

def _sha256_file(path: Path, *, chunk_size: int=1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while (chunk := handle.read(chunk_size)):
            digest.update(chunk)
    return digest.hexdigest()

def _write_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)

def _write_recap_index(source_path: Path, index_path: Path) -> dict[str, Any]:
    import mmap
    stat_before = source_path.stat()
    rows = []
    with source_path.open('rb') as handle:
        with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as data:
            position = _skip_json_whitespace(data, 0)
            if position >= len(data) or data[position] != ord('{'):
                raise SnapshotError(f'recap JSON must contain an object: {source_path}')
            position += 1
            while True:
                position = _skip_json_whitespace(data, position)
                if position >= len(data):
                    raise SnapshotError(f'unterminated recap JSON: {source_path}')
                if data[position] == ord('}'):
                    position += 1
                    break
                key_start = position
                key_end = _scan_json_string_end(data, key_start)
                key = json.loads(bytes(data[key_start:key_end]))
                position = _skip_json_whitespace(data, key_end)
                if position >= len(data) or data[position] != ord(':'):
                    raise SnapshotError(f"missing ':' in recap JSON at {source_path}:{key_start}")
                value_start = _skip_json_whitespace(data, position + 1)
                value_end = _scan_json_value_end(data, value_start)
                rows.append((hashlib.blake2b(str(key).encode('utf-8'), digest_size=16).hexdigest().encode('ascii'), key_start, key_end - key_start, value_start, value_end - value_start))
                position = _skip_json_whitespace(data, value_end)
                if position < len(data) and data[position] == ord(','):
                    position += 1
                    continue
                if position < len(data) and data[position] == ord('}'):
                    position += 1
                    break
                raise SnapshotError(f'invalid recap JSON separator at {source_path}:{position}')
            if _skip_json_whitespace(data, position) != len(data):
                raise SnapshotError(f'unexpected trailing recap data: {source_path}')
    array = np.asarray(rows, dtype=RECAP_DTYPE)
    array.sort(order='hash')
    _write_array(index_path, array)
    checksum = _sha256_file(source_path)
    stat_after = source_path.stat()
    if stat_before.st_size != stat_after.st_size or stat_before.st_mtime_ns != stat_after.st_mtime_ns:
        raise SnapshotError(f'recap input changed while being indexed: {source_path}')
    return {'path': str(source_path), 'size': stat_after.st_size, 'mtime_ns': stat_after.st_mtime_ns, 'sha256': checksum, 'index_path': index_path.name, 'index_size': index_path.stat().st_size, 'index_sha256': _sha256_file(index_path), 'entries': len(array)}

def _absolute_path(path: str | os.PathLike[str], root: Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return str(candidate.resolve())

def _recap_paths_from_spec(spec: dict[str, Any], data_root: Path) -> dict[str, str]:
    recap = spec.get('recap') or {}
    if not isinstance(recap, dict):
        raise SnapshotError('dataset spec recap must contain an object')
    if any((not isinstance(value, str) or not value for value in recap.values())):
        raise SnapshotError('dataset spec recap paths must be non-empty strings')
    return {str(name): _absolute_path(path, data_root) for (name, path) in recap.items()}

def _recap_paths_from_manifest(manifest: dict[str, Any]) -> dict[str, str]:
    recap = manifest.get('recap') or {}
    if not isinstance(recap, dict):
        raise SnapshotError('snapshot manifest recap must contain an object')
    try:
        return {str(name): str(record['path']) for (name, record) in recap.items()}
    except (KeyError, TypeError) as exc:
        raise SnapshotError('snapshot manifest recap has an invalid path') from exc

def _load_spec(spec_path: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    path = Path(spec_path).resolve()
    with path.open('r', encoding='utf-8') as handle:
        spec = json.load(handle)
    if not isinstance(spec, dict):
        raise SnapshotError('dataset spec must contain an object')
    root = Path(spec.get('data_root', path.parent)).resolve()
    sources = spec.get('sources')
    if not isinstance(sources, list) or not sources:
        raise SnapshotError('dataset spec must define a non-empty sources list')
    return (root, spec)

def _validate_source_spec(source: Any) -> None:
    if not isinstance(source, dict) or 'name' not in source or 'jsonl' not in source:
        raise SnapshotError('each source requires name and jsonl')
    jsonl_values = source['jsonl'] if isinstance(source['jsonl'], list) else [source['jsonl']]
    if not jsonl_values or any((not isinstance(value, str) or not value for value in jsonl_values)):
        raise SnapshotError(f"source {source['name']!r} has an invalid jsonl list")
    dataset = source.get('dataset', {})
    if not isinstance(dataset, dict):
        raise SnapshotError(f"source {source['name']!r} dataset config must be an object")

class _ExclusiveBuildLock:

    def __init__(self, output: Path) -> None:
        self.path = output.parent / f'.{output.name}.lock'
        self._acquired = False

    def try_acquire(self) -> bool:
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 420)
        except FileExistsError:
            return False
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
                handle.write(f'pid={os.getpid()}\n')
        except OSError:
            with suppress(OSError):
                self.path.unlink()
            raise
        self._acquired = True
        return True

    def release(self) -> None:
        if self._acquired:
            with suppress(FileNotFoundError):
                self.path.unlink()
            self._acquired = False

def _input_fingerprint(path: Path) -> dict[str, Any]:
    stat_before = path.stat()
    checksum = _sha256_file(path)
    stat_after = path.stat()
    if stat_before.st_size != stat_after.st_size or stat_before.st_mtime_ns != stat_after.st_mtime_ns:
        raise SnapshotError(f'snapshot input changed while being read: {path}')
    return {'path': str(path.resolve()), 'size': stat_after.st_size, 'mtime_ns': stat_after.st_mtime_ns, 'sha256': checksum}

def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)

def _write_index_shard(*, staging: Path, jsonl_path: Path, source_id: int, file_id: int, shard_index: int, episode_start: int, record_rows: list[tuple[int, int]]) -> dict[str, Any]:
    shard_name = f'source_{source_id:05d}_file_{file_id:05d}_shard_{shard_index:05d}'
    record_path = staging / f'{shard_name}.records.npy'
    _write_array(record_path, np.asarray(record_rows, dtype=RECORD_DTYPE))
    return {'path': record_path.name, 'jsonl': str(jsonl_path.resolve()), 'source_id': source_id, 'file_id': file_id, 'episode_start': episode_start, 'episode_count': len(record_rows), 'size': record_path.stat().st_size, 'sha256': _sha256_file(record_path)}

def _index_jsonl_file(*, staging: Path, jsonl_path: Path, source_id: int, file_id: int, shard_bytes: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    record_rows: list[tuple[int, int]] = []
    shards: list[dict[str, Any]] = []
    file_episode_count = 0
    digest = hashlib.sha256()
    stat_before = jsonl_path.stat()

    def flush() -> None:
        nonlocal record_rows, file_episode_count
        if not record_rows:
            return
        shard = _write_index_shard(staging=staging, jsonl_path=jsonl_path, source_id=source_id, file_id=file_id, shard_index=len(shards), episode_start=file_episode_count, record_rows=record_rows)
        shards.append(shard)
        file_episode_count += len(record_rows)
        record_rows = []
    with jsonl_path.open('rb') as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            digest.update(line)
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SnapshotError(f'invalid JSONL record at {jsonl_path}:{offset}') from exc
            if not isinstance(record, dict):
                raise SnapshotError(f'JSONL record must be an object at {jsonl_path}:{offset}')
            uint32_max = np.iinfo(np.uint32).max
            if len(line) > uint32_max:
                raise SnapshotError(f'JSONL record is too large at {jsonl_path}:{offset}')
            record_rows.append((offset, len(line)))
            estimated_bytes = len(record_rows) * RECORD_DTYPE.itemsize
            if estimated_bytes >= int(shard_bytes):
                flush()
    flush()
    stat_after = jsonl_path.stat()
    if stat_before.st_size != stat_after.st_size or stat_before.st_mtime_ns != stat_after.st_mtime_ns:
        raise SnapshotError(f'snapshot input changed while being indexed: {jsonl_path}')
    return (shards, {'path': str(jsonl_path.resolve()), 'size': stat_after.st_size, 'mtime_ns': stat_after.st_mtime_ns, 'sha256': digest.hexdigest()})

def build_snapshot(spec_path: str | os.PathLike[str], output_dir: str | os.PathLike[str], *, shard_bytes: int=256 * 1024 * 1024, reuse_manifest: str | os.PathLike[str] | None=None, workers: int=1, wait_for_lock_seconds: float=0, lock_poll_seconds: float=1) -> Path:
    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if int(workers) <= 0:
        raise ValueError('workers must be positive')
    if int(shard_bytes) <= RECORD_DTYPE.itemsize:
        raise ValueError('shard_bytes is too small for one record row')
    wait_seconds = float(wait_for_lock_seconds)
    poll_seconds = float(lock_poll_seconds)
    if wait_seconds < 0:
        raise ValueError('snapshot lock wait time must be non-negative')
    if poll_seconds <= 0:
        raise ValueError('snapshot lock poll interval must be positive')
    existing = _existing_snapshot(spec_path, output)
    if existing is not None:
        return existing
    lock = _ExclusiveBuildLock(output)
    deadline = time.monotonic() + wait_seconds
    while not lock.try_acquire():
        existing = _existing_snapshot(spec_path, output)
        if existing is not None:
            return existing
        remaining = deadline - time.monotonic()
        if wait_seconds == 0 or remaining <= 0:
            raise SnapshotError(f'timed out waiting for snapshot build lock: {lock.path}; another builder may be running (remove only after confirming it is stale)')
        time.sleep(min(poll_seconds, remaining))
    try:
        existing = _existing_snapshot(spec_path, output)
        if existing is not None:
            return existing
        return _build_snapshot_locked(spec_path, output, shard_bytes=shard_bytes, reuse_manifest=reuse_manifest, workers=workers)
    finally:
        lock.release()

def _existing_snapshot(spec_path: str | os.PathLike[str], output: Path) -> Path | None:
    manifest_path = output / 'manifest.json'
    if not manifest_path.is_file():
        return None
    (data_root, spec) = _load_spec(spec_path)
    manifest = load_manifest(manifest_path)
    verify_manifest(manifest, manifest_path.parent, check_inputs=True)
    requested_sources = []
    for (source_id, source) in enumerate(spec['sources']):
        _validate_source_spec(source)
        jsonl_values = source['jsonl']
        if not isinstance(jsonl_values, list):
            jsonl_values = [jsonl_values]
        requested_sources.append({'name': str(source['name']), 'source_id': source_id, 'jsonl': [_absolute_path(value, data_root) for value in jsonl_values], 'dataset': dict(source.get('dataset') or {})})
    existing_sources = [{'name': source['name'], 'source_id': source['source_id'], 'jsonl': source.get('jsonl'), 'dataset': source.get('dataset', {})} for source in manifest['sources']]
    if existing_sources != requested_sources or _recap_paths_from_manifest(manifest) != _recap_paths_from_spec(spec, data_root):
        raise SnapshotError(f'snapshot output already exists for a different build request: {output}')
    return manifest_path

def _build_snapshot_locked(spec_path: str | os.PathLike[str], output_dir: str | os.PathLike[str], *, shard_bytes: int=256 * 1024 * 1024, reuse_manifest: str | os.PathLike[str] | None=None, workers: int=1) -> Path:
    (data_root, spec) = _load_spec(spec_path)
    output = Path(output_dir).resolve()
    manifest_path = output / 'manifest.json'
    if output.exists():
        raise SnapshotError(f'snapshot output exists without a complete manifest: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.staging-', dir=output.parent))
    try:
        sources_manifest: list[dict[str, Any]] = []
        shard_manifest: list[dict[str, Any]] = []
        input_files: list[dict[str, Any]] = []
        global_episode_count = 0
        seen_source_names: set[str] = set()
        reuse = None
        reuse_dir = None
        reuse_sources: dict[str, dict[str, Any]] = {}
        reuse_inputs: dict[str, dict[str, Any]] = {}
        if reuse_manifest is not None:
            reuse_path = Path(reuse_manifest).resolve()
            reuse = load_manifest(reuse_path)
            reuse_dir = reuse_path.parent
            verify_manifest(reuse, reuse_dir, check_inputs=False, full_checksum=True)
            reuse_sources = {str(source['name']): source for source in reuse['sources']}
            reuse_inputs = {str(item['path']): item for item in reuse.get('input_files', [])}
        for (source_id, source) in enumerate(spec['sources']):
            _validate_source_spec(source)
            name = str(source['name'])
            if name in seen_source_names:
                raise SnapshotError(f'duplicate source name: {name!r}')
            seen_source_names.add(name)
            jsonl_paths = source['jsonl'] if isinstance(source['jsonl'], list) else [source['jsonl']]
            resolved_jsonl_paths = [(data_root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve() for value in jsonl_paths]
            if any((not path.is_file() for path in resolved_jsonl_paths)):
                missing = next((path for path in resolved_jsonl_paths if not path.is_file()))
                raise FileNotFoundError(missing)
            source_shards: list[str] = []
            source_episode_count = 0
            reused_source = reuse_sources.get(name)
            current_fingerprints = None
            if reused_source is not None and reused_source.get('dataset', {}) == dict(source.get('dataset') or {}):
                current_fingerprints = [_input_fingerprint(path) for path in resolved_jsonl_paths]
                can_reuse = all((reuse_inputs.get(item['path']) == item for item in current_fingerprints))
                old_source_id = int(reused_source['source_id'])
                old_shards = [shard for shard in reuse['shards'] if int(shard['source_id']) == old_source_id]
                old_jsonl_paths = list(reused_source.get('jsonl', []))
                can_reuse = can_reuse and old_jsonl_paths == [item['path'] for item in current_fingerprints]
                if can_reuse:
                    assert reuse_dir is not None
                    input_files.extend(current_fingerprints)
                    per_file_shard_index: dict[int, int] = {}
                    for old_shard in old_shards:
                        file_id = int(old_shard['file_id'])
                        shard_index = per_file_shard_index.get(file_id, 0)
                        per_file_shard_index[file_id] = shard_index + 1
                        shard_name = f'source_{source_id:05d}_file_{file_id:05d}_shard_{shard_index:05d}'
                        record_path = staging / f'{shard_name}.records.npy'
                        _link_or_copy(reuse_dir / old_shard['path'], record_path)
                        shard_record = dict(old_shard)
                        shard_record.update({'path': record_path.name, 'source_id': source_id, 'episode_start': global_episode_count})
                        shard_manifest.append(shard_record)
                        source_shards.append(shard_record['path'])
                        source_episode_count += int(shard_record['episode_count'])
                        global_episode_count += int(shard_record['episode_count'])
                    sources_manifest.append({'name': name, 'source_id': source_id, 'episode_count': source_episode_count, 'shards': source_shards, 'jsonl': [item['path'] for item in current_fingerprints], 'dataset': dict(source.get('dataset') or {})})
                    continue
            index_kwargs = [{'staging': staging, 'jsonl_path': jsonl_path, 'source_id': source_id, 'file_id': file_id, 'shard_bytes': shard_bytes} for (file_id, jsonl_path) in enumerate(resolved_jsonl_paths)]
            if int(workers) == 1 or len(index_kwargs) == 1:
                file_results = [_index_jsonl_file(**kwargs) for kwargs in index_kwargs]
            else:
                with ThreadPoolExecutor(max_workers=int(workers)) as executor:
                    futures = [executor.submit(_index_jsonl_file, **kwargs) for kwargs in index_kwargs]
                    file_results = [future.result() for future in futures]
            for (file_shards, input_record) in file_results:
                input_files.append(input_record)
                for shard_record in file_shards:
                    shard_record['episode_start'] = global_episode_count
                    shard_manifest.append(shard_record)
                    source_shards.append(shard_record['path'])
                    episode_count = int(shard_record['episode_count'])
                    source_episode_count += episode_count
                    global_episode_count += episode_count
            sources_manifest.append({'name': name, 'source_id': source_id, 'episode_count': source_episode_count, 'shards': source_shards, 'jsonl': [str(path) for path in resolved_jsonl_paths], 'dataset': dict(source.get('dataset') or {})})
        recap_manifest = {}
        for (recap_name, recap_value) in (spec.get('recap') or {}).items():
            recap_path = (data_root / recap_value).resolve() if not Path(recap_value).is_absolute() else Path(recap_value).resolve()
            if not recap_path.is_file():
                raise FileNotFoundError(recap_path)
            recap_record = _write_recap_index(recap_path, staging / f'recap_{recap_name}.npy')
            recap_record['path'] = str(recap_path)
            recap_manifest[str(recap_name)] = recap_record
        manifest = {'schema_version': SCHEMA_VERSION, 'episode_count': global_episode_count, 'sources': sources_manifest, 'shards': shard_manifest, 'input_files': input_files, 'recap': recap_manifest}
        snapshot_id = hashlib.blake2b(_canonical_json(manifest), digest_size=16).hexdigest()
        manifest['snapshot_id'] = snapshot_id
        with (staging / 'manifest.json.tmp').open('wb') as handle:
            handle.write(_canonical_json(manifest))
        os.replace(staging / 'manifest.json.tmp', staging / 'manifest.json')
        verify_manifest(manifest, staging, check_inputs=True, full_checksum=True)
        os.replace(staging, output)
        return output / 'manifest.json'
    except Exception:
        with suppress(Exception):
            for child in staging.iterdir():
                if child.is_file():
                    child.unlink()
            staging.rmdir()
        raise

def load_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open('r', encoding='utf-8') as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise SnapshotError('manifest must contain an object')
    if int(manifest.get('schema_version', -1)) != SCHEMA_VERSION:
        raise SnapshotError(f"unsupported snapshot schema {manifest.get('schema_version')}, expected {SCHEMA_VERSION}")
    return manifest

def verify_manifest(manifest: dict[str, Any], manifest_dir: str | os.PathLike[str], *, check_inputs: bool=True, full_checksum: bool=False) -> None:
    root = Path(manifest_dir)
    canonical = dict(manifest)
    snapshot_id = canonical.pop('snapshot_id', None)
    expected_id = hashlib.blake2b(_canonical_json(canonical), digest_size=16).hexdigest()
    if snapshot_id != expected_id:
        raise SnapshotError('manifest snapshot_id does not match its content')
    sources = manifest.get('sources')
    shards = manifest.get('shards')
    if not isinstance(sources, list) or not sources:
        raise SnapshotError('snapshot manifest has no sources')
    if not isinstance(shards, list) or not shards:
        raise SnapshotError('snapshot manifest has no shards')
    expected_episode_start = 0
    source_ids = [int(source['source_id']) for source in sources]
    if source_ids != list(range(len(sources))):
        raise SnapshotError('snapshot source IDs must be contiguous and ordered')
    for source in sources:
        if any((not Path(path).is_absolute() for path in source.get('jsonl', []))):
            raise SnapshotError(f'snapshot schema {SCHEMA_VERSION} requires absolute source JSONL paths')
    for shard in shards:
        if int(shard['episode_start']) != expected_episode_start:
            raise SnapshotError('snapshot shard episode ranges are not contiguous')
        expected_episode_start += int(shard['episode_count'])
        record_path = root / shard['path']
        if not record_path.is_file():
            raise SnapshotError(f"missing snapshot shard for {shard.get('path')}")
        if record_path.stat().st_size != int(shard['size']):
            raise SnapshotError(f'record shard size mismatch: {record_path}')
        if full_checksum and _sha256_file(record_path) != shard['sha256']:
            raise SnapshotError(f'record shard checksum mismatch: {record_path}')
        if not Path(shard['jsonl']).is_absolute():
            raise SnapshotError(f'snapshot schema {SCHEMA_VERSION} requires absolute shard JSONL paths')
    if expected_episode_start != int(manifest.get('episode_count', -1)):
        raise SnapshotError('snapshot episode_count does not match its shards')
    for item in manifest.get('input_files', []):
        path = Path(item['path'])
        if not path.is_absolute():
            raise SnapshotError(f'snapshot schema {SCHEMA_VERSION} requires absolute input paths: {path}')
        if check_inputs:
            if not path.is_file():
                raise SnapshotError(f'missing snapshot input: {path}')
            stat = path.stat()
            if stat.st_size != int(item['size']) or stat.st_mtime_ns != int(item['mtime_ns']):
                raise SnapshotError(f'snapshot input changed: {path}')
            if full_checksum and _sha256_file(path) != item['sha256']:
                raise SnapshotError(f'snapshot input checksum mismatch: {path}')
    for recap in manifest.get('recap', {}).values():
        index_path = root / recap['index_path']
        if not index_path.is_file():
            raise SnapshotError(f'missing recap index: {index_path}')
        if index_path.stat().st_size != int(recap['index_size']):
            raise SnapshotError(f'recap index size mismatch: {index_path}')
        if full_checksum and _sha256_file(index_path) != recap['index_sha256']:
            raise SnapshotError(f'recap index checksum mismatch: {index_path}')
        source_path = Path(recap['path'])
        if not source_path.is_absolute():
            raise SnapshotError(f'snapshot schema {SCHEMA_VERSION} requires absolute recap paths: {source_path}')
        if check_inputs:
            if not source_path.is_file():
                raise SnapshotError(f'missing recap input: {source_path}')
            stat = source_path.stat()
            if stat.st_size != int(recap['size']) or stat.st_mtime_ns != int(recap['mtime_ns']):
                raise SnapshotError(f'snapshot recap input changed: {source_path}')
            if full_checksum and _sha256_file(source_path) != recap['sha256']:
                raise SnapshotError(f'snapshot recap checksum mismatch: {source_path}')

class SnapshotRecordStore:

    def __init__(self, manifest_path: str | os.PathLike[str], *, cache_size: int=128, mmap_cache_size: int=16, file_cache_size: int | None=None, verify_inputs: bool=True) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = load_manifest(self.manifest_path)
        self.manifest_dir = self.manifest_path.parent
        verify_manifest(self.manifest, self.manifest_dir, check_inputs=verify_inputs)
        self.cache_size = int(cache_size)
        if self.cache_size < 0:
            raise ValueError('snapshot episode cache size must be non-negative')
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._mmap_cache = _ArrayLRU(mmap_cache_size)
        self._file_cache = _FileLRU(mmap_cache_size if file_cache_size is None else file_cache_size)
        self._shards = tuple(self.manifest['shards'])
        self._starts = tuple((int(item['episode_start']) for item in self._shards))
        self._source_names = tuple((item['name'] for item in self.manifest['sources']))
        self._source_by_id = {int(item['source_id']): item for item in self.manifest['sources']}
        self._source_configs = tuple((dict(self._source_by_id[source_id].get('dataset', {})) for source_id in range(len(self._source_names))))
        self._source_shards: dict[int, list[int]] = {}
        for (shard_idx, shard) in enumerate(self._shards):
            self._source_shards.setdefault(int(shard['source_id']), []).append(shard_idx)

    @property
    def snapshot_id(self) -> str:
        return str(self.manifest['snapshot_id'])

    @property
    def schema_version(self) -> int:
        return int(self.manifest['schema_version'])

    @property
    def source_names(self) -> tuple[str, ...]:
        return self._source_names

    @property
    def source_count(self) -> int:
        return len(self._source_names)

    @property
    def episode_count(self) -> int:
        return int(self.manifest['episode_count'])

    def __len__(self) -> int:
        return self.episode_count

    def _shard_for_episode(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_idx = bisect_right(self._starts, index) - 1
        return (shard_idx, index - self._starts[shard_idx])

    def _array(self, shard_idx: int) -> np.ndarray:
        shard = self._shards[shard_idx]
        path = self.manifest_dir / shard['path']
        array = self._mmap_cache.get(str(path))
        if array.ndim != 1 or array.dtype != RECORD_DTYPE:
            raise SnapshotError(f'snapshot shard has unexpected dtype or shape: {path}')
        return array

    def source_id(self, index: int) -> int:
        (shard_idx, local_idx) = self._shard_for_episode(index)
        return int(self._shards[shard_idx]['source_id'])

    def _load_record(self, index: int) -> dict[str, Any]:
        (shard_idx, local_idx) = self._shard_for_episode(index)
        row = self._array(shard_idx)[local_idx]
        jsonl = Path(self._shards[shard_idx]['jsonl'])
        handle = self._file_cache.get(jsonl)
        handle.seek(int(row['offset']))
        raw = handle.read(int(row['length']))
        record = json.loads(raw)
        if not isinstance(record, dict):
            raise SnapshotError(f'snapshot record is not an object at episode {index}')
        return record

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if self.cache_size:
            try:
                value = self._cache.pop(index)
            except KeyError:
                value = self._load_record(index)
            self._cache[index] = value
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return value
        return self._load_record(index)

    def source_config(self, source_id: int) -> dict[str, Any]:
        return self._source_configs[int(source_id)]

    def cache_metrics(self) -> dict[str, int]:
        return {'mmap_open': self._mmap_cache.opens, 'mmap_hit': self._mmap_cache.hits, 'mmap_miss': self._mmap_cache.misses, 'mmap_eviction': self._mmap_cache.evictions, 'mmap_cache_size': len(self._mmap_cache._items), 'episode_cache_size': len(self._cache), 'jsonl_open': len(self._file_cache._items)}

    def close(self) -> None:
        self._cache.clear()
        self._mmap_cache.close()
        self._file_cache.close()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state['_cache'] = OrderedDict()
        state['_mmap_cache'] = _ArrayLRU(self._mmap_cache.maxsize)
        state['_file_cache'] = _FileLRU(self._file_cache.maxsize)
        return state

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

class SnapshotLogicalView:

    def __init__(self, physical_store: SnapshotRecordStore, repeats: Iterable[int]) -> None:
        self.physical_store = physical_store
        repeat_values = tuple(repeats)
        normalized_repeats = tuple((int(repeat) for repeat in repeat_values))
        if any((isinstance(repeat, bool) or normalized != repeat for (repeat, normalized) in zip(repeat_values, normalized_repeats))):
            raise ValueError('snapshot repeats must be positive integers')
        if len(normalized_repeats) != physical_store.source_count:
            raise ValueError(f'snapshot repeat count must match snapshot source count: repeats={len(normalized_repeats)}, sources={physical_store.source_count}')
        if any((repeat <= 0 for repeat in normalized_repeats)):
            raise ValueError('snapshot repeats must be positive integers')
        self.repeats = normalized_repeats
        physical_starts = []
        physical_counts = []
        logical_starts = []
        logical_ends = []
        physical_start = 0
        logical_end = 0
        for (source_id, repeat) in enumerate(self.repeats):
            physical_starts.append(physical_start)
            logical_starts.append(logical_end)
            source_count = int(physical_store.manifest['sources'][source_id]['episode_count'])
            source_shard_ids = physical_store._source_shards.get(source_id, [])
            source_shard_count = sum((int(physical_store._shards[shard_id]['episode_count']) for shard_id in source_shard_ids))
            source_shard_start = int(physical_store._shards[source_shard_ids[0]]['episode_start']) if source_shard_ids else physical_start
            if source_shard_count != source_count or source_shard_start != physical_start:
                raise SnapshotError('snapshot source episode ranges are not contiguous and ordered')
            physical_counts.append(source_count)
            physical_start += source_count
            logical_end += source_count * repeat
            logical_ends.append(logical_end)
        if physical_start != len(physical_store):
            raise SnapshotError('snapshot source episode counts do not match total episode count')
        self._source_physical_starts = tuple(physical_starts)
        self._source_physical_counts = tuple(physical_counts)
        self._source_logical_starts = tuple(logical_starts)
        self._source_logical_ends = tuple(logical_ends)

    @property
    def snapshot_id(self) -> str:
        return self.physical_store.snapshot_id

    @property
    def schema_version(self) -> int:
        return self.physical_store.schema_version

    @property
    def source_names(self) -> tuple[str, ...]:
        return self.physical_store.source_names

    @property
    def source_count(self) -> int:
        return self.physical_store.source_count

    @property
    def episode_count(self) -> int:
        return self.physical_store.episode_count

    @property
    def source_physical_counts(self) -> tuple[int, ...]:
        return self._source_physical_counts

    def __len__(self) -> int:
        return self._source_logical_ends[-1] if self._source_logical_ends else 0

    def _logical_source(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_id = bisect_right(self._source_logical_ends, index)
        return (source_id, index - self._source_logical_starts[source_id])

    def physical_index(self, logical_index: int) -> int:
        (source_id, source_local_logical_index) = self._logical_source(int(logical_index))
        source_count = self._source_physical_counts[source_id]
        return self._source_physical_starts[source_id] + source_local_logical_index % source_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.physical_store[self.physical_index(index)]

    def source_id(self, index: int) -> int:
        (source_id, _) = self._logical_source(int(index))
        return source_id

    def source_config(self, source_id: int) -> dict[str, Any]:
        return self.physical_store.source_config(source_id)

    def cache_metrics(self) -> dict[str, int]:
        return self.physical_store.cache_metrics()

    def close(self) -> None:
        self.physical_store.close()

class SnapshotRecapMap:

    def __init__(self, manifest_path: str | os.PathLike[str], recap_name: str, *, cache_size: int) -> None:
        manifest_path = Path(manifest_path).resolve()
        manifest = load_manifest(manifest_path)
        try:
            recap = manifest['recap'][recap_name]
        except KeyError:
            raise SnapshotError(f'snapshot has no recap index named {recap_name!r}') from None
        self.source_path = Path(recap['path'])
        self.index_path = manifest_path.parent / recap['index_path']
        self.cache_size = int(cache_size)
        if self.cache_size < 0:
            raise ValueError('recap cache size must be non-negative')
        self._index: np.memmap | None = None
        self._handle = None
        self._cache: OrderedDict[str, Any] = OrderedDict()

    def _get_index(self) -> np.memmap:
        if self._index is None:
            self._index = np.load(self.index_path, mmap_mode='r', allow_pickle=False)
            if self._index.ndim != 1 or self._index.dtype != RECAP_DTYPE:
                raise SnapshotError(f'recap index has unexpected dtype or shape: {self.index_path}')
        return self._index

    def _lookup(self, key: str) -> Any:
        digest = hashlib.blake2b(str(key).encode('utf-8'), digest_size=16).hexdigest().encode('ascii')
        index = self._get_index()
        hashes = index['hash']
        position = int(np.searchsorted(hashes, np.bytes_(digest), side='left'))
        while position < len(index) and bytes(index[position]['hash']) == digest:
            row = index[position]
            if self._handle is None:
                self._handle = self.source_path.open('rb')
            self._handle.seek(int(row['key_offset']))
            stored_key = json.loads(self._handle.read(int(row['key_length'])))
            if stored_key == key:
                self._handle.seek(int(row['value_offset']))
                return json.loads(self._handle.read(int(row['value_length'])))
            position += 1
        raise KeyError(key)

    def __getitem__(self, key: str) -> Any:
        key = str(key)
        if self.cache_size:
            try:
                value = self._cache.pop(key)
            except KeyError:
                value = self._lookup(key)
            self._cache[key] = value
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return value
        return self._lookup(key)

    def get(self, key: str, default: Any=None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        try:
            self._lookup(str(key))
        except KeyError:
            return False
        return True

    def close(self) -> None:
        if self._index is not None:
            mmap_handle = getattr(self._index, '_mmap', None)
            if mmap_handle is not None:
                mmap_handle.close()
            self._index = None
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._cache.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state['_index'] = None
        state['_handle'] = None
        state['_cache'] = OrderedDict()
        return state

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()
