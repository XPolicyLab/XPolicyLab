"""Local file/directory caching for fast model loading.

Copies from remote/NFS to local NVMe (/tmp) with flock coordination
for multi-process and multi-user safety.

Cross-user strategy:
- Cache path is a hash of the source, shared across all users.
- Reads are lock-free: atomic rename guarantees readers never see
  partial files, so any user can read without locks.
- Writes go through tmp + atomic rename under LOCK_EX — rename only
  needs parent dir write perm, so any user can replace another's
  cached file.
- Cleanup only touches files owned by the current user; orphans from
  inactive users stay until that user runs cleanup again (or root cleans).
"""

import fcntl
import hashlib
import logging
import os
import shutil
import stat
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = os.environ.get("LOCAL_CACHE_DIR", "/tmp/ckpt_cache")


def _open_lock(lock_path: str):
    """Open a lock file for flock coordination.

    Tries read-write; if the file is owned by another user and write is
    denied, falls back to read-only — flock() works on read-only fds too.
    """
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
    except PermissionError:
        fd = os.open(lock_path, os.O_RDONLY)
        return os.fdopen(fd, "r")
    try:
        if os.fstat(fd).st_uid == os.getuid():
            os.fchmod(fd, 0o666)
        return os.fdopen(fd, "r+")
    except OSError:
        os.close(fd)
        raise


def _share_path(path: str | Path, mode: int) -> None:
    """Repair cache permissions without chmod on already-shared foreign paths."""
    current = stat.S_IMODE(os.stat(path).st_mode)
    if current & 0o777 != mode:
        os.chmod(path, (current & 0o7000) | mode)


def _prepare_cache_dir(cache_dir: str) -> None:
    """Share the cache root and newly created parents, not existing ancestors."""
    path = Path(cache_dir)
    missing = []
    parent = path
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    path.mkdir(parents=True, exist_ok=True)
    for directory in [*reversed(missing), path]:
        _share_path(directory, 0o777)


def _share_tree(path: str) -> None:
    """Share cached directories and files without touching symlink targets."""
    _share_path(path, 0o777)
    for root, directories, files in os.walk(path):
        for name in directories:
            child = os.path.join(root, name)
            if not os.path.islink(child):
                _share_path(child, 0o777)
        for name in files:
            child = os.path.join(root, name)
            if not os.path.islink(child):
                _share_path(child, 0o666)


def cache_locally(src_path: str, cache_dir: str = CACHE_DIR, max_age_hours: int = 24) -> str:
    """Cache remote artifacts locally; use local sources where they already are.

    ``SOMA_ARTIFACT_CACHE`` selects the policy:

    - unset or ``1`` (default): copy remote (``scheme://``) artifacts into
      ``cache_dir``; read anything already on a local path in place.
    - ``all``: also copy local paths. Worth it when the weights live on a
      shared network filesystem and several workers would each stream them.
    - ``0``: never copy.
    """
    policy = os.environ.get("SOMA_ARTIFACT_CACHE", "1")
    if policy == "0":
        return src_path
    if not _is_remote(src_path) and policy != "all":
        return src_path
    try:
        _prepare_cache_dir(cache_dir)
        if not _is_remote(src_path) and os.path.isdir(src_path):
            return _cache_dir(src_path, cache_dir, max_age_hours)
        return _cache_file(src_path, cache_dir, max_age_hours)
    except PermissionError as error:
        logger.warning("Artifact cache permission denied; loading source %s directly: %s", src_path, error)
        return src_path


def _cache_file(src_path: str, cache_dir: str, max_age_hours: int) -> str:
    """缓存单个文件。读无锁，写才加排他锁（double-check）。"""
    remote = _is_remote(src_path)
    identity = src_path if remote else os.path.abspath(src_path)
    key = hashlib.md5(identity.encode()).hexdigest()
    ext = os.path.splitext(src_path)[1]
    local_path = os.path.join(cache_dir, f"{key}{ext}")
    if remote:
        import megfile

        src_size = megfile.smart_stat(src_path).st_size
    else:
        src_size = os.path.getsize(src_path)

    # 无锁快速路径：原子 rename 保证不会读到半截文件
    if os.path.exists(local_path) and os.path.getsize(local_path) == src_size:
        _share_path(local_path, 0o666)
        logger.info(f"Using cached: {local_path}")
        return local_path

    # 未命中，拿排他锁写入
    lock_path = local_path + ".lock"
    with _open_lock(lock_path) as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        cleanup_cache(cache_dir, max_age_hours)
        # double-check: 可能等锁期间别的进程已经写好了
        if os.path.exists(local_path) and os.path.getsize(local_path) == src_size:
            _share_path(local_path, 0o666)
            logger.info(f"Using cached: {local_path}")
            return local_path
        sz_gb = src_size / (1024**3)
        logger.info(f"Caching file ({sz_gb:.1f}GB): {src_path} -> {local_path}")
        t0 = time.time()
        fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(local_path) + ".tmp.", dir=cache_dir)
        os.close(fd)
        try:
            if remote:
                megfile.smart_copy(src_path, tmp_path)
            else:
                shutil.copyfile(src_path, tmp_path)
            _share_path(tmp_path, 0o666)
            # Refresh source-preserved mtime before publishing the cache entry.
            os.utime(tmp_path, None)
            os.rename(tmp_path, local_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        logger.info(f"Cached in {time.time() - t0:.1f}s")

    return local_path


def _is_remote(path: str) -> bool:
    return "://" in path


def _cache_dir(src_path: str, cache_dir: str, max_age_hours: int) -> str:
    """缓存整个目录。读无锁（通过 marker 判断完整性），写才加排他锁。"""
    key = hashlib.md5(os.path.abspath(src_path).encode()).hexdigest()
    local_path = os.path.join(cache_dir, f"dir_{key}")
    marker = os.path.join(local_path, ".cache_complete")

    # 无锁快速路径：marker 存在说明目录已完整写入
    if os.path.exists(marker):
        _share_tree(local_path)
        logger.info(f"Using cached dir: {local_path}")
        return local_path

    # 未命中，拿排他锁写入
    lock_path = local_path + ".lock"
    with _open_lock(lock_path) as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        # double-check
        if os.path.exists(marker):
            _share_tree(local_path)
            logger.info(f"Using cached dir: {local_path}")
            return local_path
        total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(src_path) for f in fns)
        logger.info(f"Caching dir ({total / (1024**3):.1f}GB): {src_path} -> {local_path}")
        t0 = time.time()
        tmp_path = tempfile.mkdtemp(prefix=os.path.basename(local_path) + ".tmp.", dir=cache_dir)
        try:
            shutil.copytree(src_path, tmp_path, copy_function=shutil.copyfile, dirs_exist_ok=True)
            _share_tree(tmp_path)
            tmp_marker = os.path.join(tmp_path, ".cache_complete")
            with open(tmp_marker, "w") as f:
                f.write(str(time.time()))
            _share_path(tmp_marker, 0o666)
            os.utime(tmp_path, None)
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
            os.rename(tmp_path, local_path)
        finally:
            if os.path.exists(tmp_path):
                shutil.rmtree(tmp_path)
        logger.info(f"Cached in {time.time() - t0:.1f}s")

    return local_path


def cleanup_cache(cache_dir: str = CACHE_DIR, max_age_hours: int = 24):
    """Remove cached entries older than max_age_hours.

    Only removes files/dirs owned by the current user to avoid cross-user
    permission errors. Other users' stale entries will be cleaned when
    *they* run cleanup.
    """
    if not os.path.isdir(cache_dir):
        return

    global_lock = os.path.join(cache_dir, ".cleanup.lock")
    try:
        with _open_lock(global_lock) as lf:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            now = time.time()
            my_uid = os.getuid()
            for fname in os.listdir(cache_dir):
                if fname.endswith(".lock") or fname.startswith("."):
                    continue
                fpath = os.path.join(cache_dir, fname)
                try:
                    st = os.stat(fpath)
                    if st.st_uid != my_uid:
                        continue
                    if now - st.st_mtime > max_age_hours * 3600:
                        if os.path.isdir(fpath):
                            shutil.rmtree(fpath)
                            logger.info(f"Cleaned up stale cache dir: {fpath}")
                        else:
                            sz_gb = st.st_size / (1024**3)
                            os.remove(fpath)
                            logger.info(f"Cleaned up stale cache ({sz_gb:.1f}GB): {fpath}")
                except OSError:
                    pass
    except (BlockingIOError, PermissionError):
        pass  # Another process is cleaning, or lock not accessible — skip
