from __future__ import annotations
from cloudrobo_wm.contracts import data_protocol
from functools import lru_cache
import json
import os
from cloudrobo_wm.storage import DataPath as Path

@lru_cache(maxsize=2)
def _receipt(path, mtime_ns):
    value = json.loads(Path(path).read_text())
    if value.get('schema') != data_protocol('protocol_cc4c3236de60'):
        raise ValueError('Invalid media replica receipt')
    return value['files']

def verify_video_identity(path, spec):
    path = Path(path)
    stat = path.stat()
    if 'size' not in spec:
        return
    expected = [spec['size'], spec['mtime_ns']]
    observed = [stat.st_size, stat.st_mtime_ns]
    if expected == observed:
        return
    receipt_path = os.environ.get('CLOUDROBO_WM_MEDIA_REPLICA_RECEIPT')
    if receipt_path:
        receipt = Path(receipt_path)
        records = _receipt(str(receipt), receipt.stat().st_mtime_ns)
        record = records.get(str(path.resolve()), {})
        if record.get('expected') == expected and record.get('observed') == observed and (len(record.get('sha256', '')) == 64):
            return
    raise ValueError(f'Video identity changed without a matching SHA256 replica receipt: {path}')
