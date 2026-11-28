"""Transport checks for immutable Qwen downloads; no external model is fetched."""
import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

import download_qwen


class Response:
    def __init__(self, body, status=206, content_range=None):
        self.body = body
        self.url = 'https://example.test/qwen'
        self.status = status
        self.headers = {}
        if content_range:
            self.headers['Content-Range'] = content_range

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self, size=-1):
        return self.body if size < 0 else self.body[:size]


def test_whole_file_checks_bounded_length_and_writes_once(tmp_path):
    body = b'qwen-small-file'
    target = tmp_path / 'small'
    with mock.patch.object(download_qwen.urllib.request, 'urlopen',
                           return_value=Response(body)):
        download_qwen._fetch_whole('https://example.test/qwen', target, len(body), 1)
    assert target.read_bytes() == body


def test_whole_file_rejects_truncation_without_leaving_target(tmp_path):
    target = tmp_path / 'small'
    with mock.patch.object(download_qwen.urllib.request, 'urlopen',
                           return_value=Response(b'truncated')):
        with pytest.raises(ValueError, match='length differs'):
            download_qwen._fetch_whole('https://example.test/qwen', target, 100, 1)
    assert not target.exists()


def test_range_checks_content_range_and_length(tmp_path):
    target = tmp_path / 'block'
    target.write_bytes(b'00000000')
    response = Response(b'abcd', content_range='bytes 2-5/8')
    with mock.patch.object(download_qwen.urllib.request, 'urlopen',
                           return_value=response):
        download_qwen._fetch_block('https://example.test/qwen', target, 2, 5, 1)
    assert target.read_bytes() == b'00abcd00'
    response = Response(b'abcd', content_range='bytes 0-3/8')
    with mock.patch.object(download_qwen.urllib.request, 'urlopen',
                           return_value=response):
        with pytest.raises(ValueError, match='not honored'):
            download_qwen._fetch_block('https://example.test/qwen', target, 2, 5, 1)
    with mock.patch.object(download_qwen.urllib.request, 'urlopen',
                           return_value=Response(b'bad', content_range='bytes 2-5/8')):
        with pytest.raises(ValueError, match='length differs'):
            download_qwen._fetch_block('https://example.test/qwen', target, 2, 5, 1)


def test_verification_binds_every_file_and_model_identity(tmp_path):
    records = {}
    for name in ('config.json', 'model.safetensors', 'chat_template.jinja'):
        body = b'{"model_type": "qwen3_5", "hidden_size": 4096}' if name == 'config.json' else name.encode()
        (tmp_path / name).write_bytes(body)
        records[name] = {'size': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
    identity = hashlib.sha256(json.dumps(
        {name: record['sha256'] for name, record in records.items()}, sort_keys=True,
    ).encode()).hexdigest()
    manifest = {'repository': 'Qwen/Qwen3.5-9B',
                'revision': 'c' * 40, 'model_id': identity, 'files': records}
    report = download_qwen.verify(tmp_path, manifest)
    assert report['model_id'] == identity
    assert report['revision'] == 'c' * 40
    from wall_x.model.vap.qwen35_online import model_identity
    assert model_identity(tmp_path) == (identity, 4096)
    provenance = tmp_path.with_name(tmp_path.name + '.provenance.json')
    provenance.write_text(json.dumps(report, indent=2) + '\n')
    assert download_qwen.verify(tmp_path, manifest) == report
    assert model_identity(tmp_path) == (identity, 4096)
    (tmp_path / 'config.json').write_text('different')
    with pytest.raises(ValueError):
        download_qwen.verify(tmp_path, manifest)
