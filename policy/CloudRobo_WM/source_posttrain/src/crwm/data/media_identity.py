"""Portable content identity, with the original stat tuple as a fast path."""
from functools import lru_cache
import hashlib,json
from pathlib import Path

@lru_cache(maxsize=1)
def media_index():
    p=Path(__file__).resolve().parents[3]/'data/media_sha256.json'
    value=json.loads(p.read_text())
    if value['version']!=1:raise ValueError('Unknown media index')
    return value['files']

@lru_cache(maxsize=128)
def _verify_content(path,size,mtime_ns,expected):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    if h.hexdigest()!=expected:raise ValueError('Media content differs from the published Clean index')
    current=Path(path).stat()
    if (current.st_size,current.st_mtime_ns)!=(size,mtime_ns):raise ValueError('Media changed during verification')

def verify_video_identity(path,spec):
    entry=media_index().get(spec['path'])
    if entry is None:raise ValueError('Media is not in the published Clean subset')
    path=Path(path);stat=path.stat()
    if stat.st_size!=entry['bytes']:raise ValueError('Media size changed')
    if stat.st_mtime_ns!=entry['mtime_ns']:
        _verify_content(str(path.resolve()),stat.st_size,stat.st_mtime_ns,entry['sha256'])
