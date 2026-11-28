"""Local fixture tests for resumable downloads from the replaceable public main branch."""
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from unittest import mock
import urllib.error

import pytest

PATH = Path(__file__).resolve().parents[1] / 'download_checkpoint.py'
SPEC = importlib.util.spec_from_file_location('vap_checkpoint_download', PATH)
downloader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = downloader
SPEC.loader.exec_module(downloader)
PAYLOADS = {name: f'published-{name}'.encode() for name in
            ('model.safetensors', 'config.yml', 'normalizer_action.pth')}


class Response:
    def __init__(self, payload, status=200, headers=None):
        self.payload = payload
        self.status = status
        self.headers = headers or {'Content-Length': str(len(payload)), 'ETag': '"fixture-etag"'}
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        if size < 0:
            block = self.payload[self.offset:]
            self.offset = len(self.payload)
            return block
        end = min(self.offset + max(size // 2, 1), len(self.payload))
        block = self.payload[self.offset:end]
        self.offset = end
        return block


def use_fixture(monkeypatch, tmp_path, payloads=None):
    payloads = dict(PAYLOADS if payloads is None else payloads)
    manifest = {'repository': 'x-square-robot/wall-wm-2.0-robodojo',
                'revision': 'main', 'files': list(payloads)}
    fixture = tmp_path / 'manifest.json'
    fixture.write_text(json.dumps(manifest))
    monkeypatch.setattr(downloader, 'MANIFEST', fixture)
    requests = []
    xet_etags = 'model.safetensors' in payloads

    def response(request, timeout):
        requests.append(request)
        if '/api/models/' in request.full_url:
            return Response(json.dumps([
                {'type': 'file', 'path': name, 'size': len(payload),
                 'lfs': None if name == 'config.yml' else {
                     'oid': hashlib.sha256(payload).hexdigest(), 'size': len(payload)}}
                for name, payload in payloads.items()]).encode())
        name = request.full_url.rsplit('/', 1)[-1]
        payload = payloads[name]
        etag = '"' + 'e' * 64 + '"' if (xet_etags and name == 'model.safetensors'
                and '/x-square-robot/wall-wm-2.0-robodojo/' in request.full_url) else f'"{name}-etag"'
        if request.get_method() == 'HEAD':
            return Response(b'', headers={'Content-Length': str(len(payload)), 'ETag': etag})
        match = re.fullmatch(r'bytes=(\d+)-(\d+)', request.get_header('Range', ''))
        assert match and request.get_header('If-range') == etag
        start, end = map(int, match.groups())
        block = payload[start:end + 1]
        return Response(block, status=206, headers={
            'Content-Length': str(len(block)), 'Content-Range': f'bytes {start}-{end}/{len(payload)}',
            'ETag': etag,
        })

    monkeypatch.setattr(downloader.urllib.request, 'urlopen', response)
    return payloads, manifest, requests


def download(destination, base_url=None, workers=1, resume=None):
    argv = ['download_checkpoint.py', '--workers', str(workers)]
    if base_url is not None:
        argv.extend(['--base-url', base_url])
    if resume is not None:
        argv.extend(['--resume', str(resume)])
    with mock.patch.object(sys, 'argv', [*argv, '--output', str(destination)]):
        downloader.main()
    return json.loads(destination.with_name(destination.name + '.provenance.json').read_text())


def only_partial(tmp_path, name='checkpoint'):
    partials = [path for path in tmp_path.glob(name + '.partial-*') if path.is_dir()]
    assert len(partials) == 1
    return partials[0]


def test_public_manifest_tracks_main_without_release_identities():
    manifest = json.loads(downloader.MANIFEST.read_text())
    assert manifest == {'repository': 'x-square-robot/wall-wm-2.0-robodojo',
                        'revision': 'main',
                        'files': list(PAYLOADS)}
    for name in ('README.md', 'INSTALLATION.md'):
        text = (PATH.parent / name).read_text()
        assert 'main' in text and 'environment/CHECKPOINT_SOURCES.json' in text
        for old_value in ('fdd4874767857352bfd681a8069705fdfbc4a63d',
                          'a37576aeabc531841a21210ec2e113b22db5a7f282c03afeb7430d7a2f6a1c60',
                          '2ae846b845ba134fdda12d03d52956174dc88d280d923617c926bfbb07ef4872',
                          'a8cd4380de11bc0e1447e6ca2df8b96aa5fda1c16a65b1c5a53bb0d9bbd57c33',
                          '24131440648'):
            assert old_value not in text


def test_main_download_is_atomic_and_verifiable(monkeypatch, tmp_path):
    payloads, manifest, requests = use_fixture(monkeypatch, tmp_path)
    destination = tmp_path / 'checkpoint'
    provenance = download(destination)
    assert [request.full_url for request in requests if request.get_method() == 'HEAD'] == [
        f'https://huggingface.co/{manifest["repository"]}/resolve/main/{name}' for name in payloads]
    assert provenance == {
        'repository': manifest['repository'], 'revision': 'main',
        'files': {name: {'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
                  for name, payload in payloads.items()}}
    assert downloader.verify(destination, manifest) == provenance
    with mock.patch.object(sys, 'argv', ['download_checkpoint.py', '--verify-only',
                                        '--output', str(destination)]):
        downloader.main()
    assert not list(tmp_path.glob('*.partial-*'))


def test_mirror_download_uses_main(monkeypatch, tmp_path):
    _, _, requests = use_fixture(monkeypatch, tmp_path)
    download(tmp_path / 'mirror', base_url='https://mirror.example')
    assert [request.full_url for request in requests if request.get_method() == 'HEAD'] == [
        f'https://mirror.example/x-square-robot/wall-wm-2.0-robodojo/resolve/main/{name}'
        for name in PAYLOADS]


def test_new_main_payload_needs_no_manifest_or_code_change(monkeypatch, tmp_path):
    payloads, _, _ = use_fixture(monkeypatch, tmp_path)
    first = download(tmp_path / 'first')
    replacement = b'a differently sized replacement checkpoint'
    payloads['model.safetensors'] = replacement
    second = download(tmp_path / 'second')
    assert first['files']['model.safetensors']['size'] != len(replacement)
    assert second['files']['model.safetensors'] == {
        'size': len(replacement), 'sha256': hashlib.sha256(replacement).hexdigest()}


def test_empty_payload_is_never_published(monkeypatch, tmp_path):
    payloads, _, _ = use_fixture(monkeypatch, tmp_path)
    payloads['config.yml'] = b''
    destination = tmp_path / 'checkpoint'
    with pytest.raises(ValueError, match='public checkpoint metadata'):
        download(destination)
    assert not destination.exists() and not destination.with_name('checkpoint.provenance.json').exists()
    assert only_partial(tmp_path).is_dir()


def test_interrupted_payload_is_never_published(monkeypatch, tmp_path):
    payloads, _, _ = use_fixture(monkeypatch, tmp_path)

    def interrupted(url, target, workers=1, completed_record=None, expected_file=None):
        target.write_bytes(payloads[url.rsplit('/', 1)[-1]][:4])
        raise urllib.error.URLError('interrupted')

    monkeypatch.setattr(downloader, 'fetch', interrupted)
    destination = tmp_path / 'interrupted'
    with pytest.raises(urllib.error.URLError):
        download(destination)
    assert not destination.exists()
    assert not destination.with_name('interrupted.provenance.json').exists()
    assert only_partial(tmp_path, 'interrupted').is_dir()


def test_missing_changed_or_extra_payload_fails_verification(monkeypatch, tmp_path):
    _, _, _ = use_fixture(monkeypatch, tmp_path)
    destination = tmp_path / 'checkpoint'
    download(destination)
    (destination / 'config.yml').write_bytes(b'locally changed')
    with mock.patch.object(sys, 'argv', ['download_checkpoint.py', '--verify-only',
                                        '--output', str(destination)]):
        with pytest.raises(ValueError, match='provenance differs'):
            downloader.main()
    (destination / 'config.yml').unlink()
    manifest = json.loads(downloader.MANIFEST.read_text())
    with pytest.raises(ValueError, match='inventory differs'):
        downloader.verify(destination, manifest)
    (destination / 'unexpected').mkdir()
    with pytest.raises(ValueError, match='inventory differs'):
        downloader.verify(destination, manifest)


def test_publish_failure_cleans_only_its_own_provenance(monkeypatch, tmp_path):
    _, _, _ = use_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(downloader.os, 'rename', lambda source, target: (_ for _ in ()).throw(OSError('busy')))
    destination = tmp_path / 'checkpoint'
    with pytest.raises(OSError, match='busy'):
        download(destination)
    assert not destination.exists()
    assert not destination.with_name('checkpoint.provenance.json').exists()
    assert only_partial(tmp_path).is_dir()
    other = tmp_path / 'other'
    provenance = other.with_name('other.provenance.json')
    provenance.write_text('another download record')
    with pytest.raises(FileExistsError):
        download(other)
    assert provenance.read_text() == 'another download record'
    assert not other.exists() and only_partial(tmp_path, 'other').is_dir()


def test_existing_destination_and_dangling_symlink_are_preserved(monkeypatch, tmp_path):
    _, _, _ = use_fixture(monkeypatch, tmp_path)
    destination = tmp_path / 'existing'
    destination.mkdir()
    (destination / 'keep').write_text('unchanged')
    dangling = tmp_path / 'dangling'
    dangling.symlink_to(tmp_path / 'absent')
    for output in (destination, dangling):
        with mock.patch.object(sys, 'argv', ['download_checkpoint.py', '--output', str(output)]):
            with pytest.raises(FileExistsError):
                downloader.main()
    assert (destination / 'keep').read_text() == 'unchanged' and dangling.is_symlink()


def test_new_payload_must_match_public_lfs_not_xet_etag(monkeypatch, tmp_path):
    payloads, _, _ = use_fixture(monkeypatch, tmp_path)
    etags = {name: f'"{hashlib.sha256(payload).hexdigest()}"' for name, payload in payloads.items()}
    lfs_rows = [{'type': 'file', 'path': name, 'size': len(payload),
                 'lfs': None if name == 'config.yml' else {
                     'oid': hashlib.sha256(payload).hexdigest(), 'size': len(payload)}}
                for name, payload in payloads.items()]

    def verified(request, timeout):
        if '/api/models/' in request.full_url:
            return Response(json.dumps(lfs_rows).encode())
        name = request.full_url.rsplit('/', 1)[-1]
        payload = payloads[name]
        etag = etags[name]
        if request.get_method() == 'HEAD':
            return Response(b'', headers={'Content-Length': str(len(payload)), 'ETag': etag})
        start, end = map(int, re.fullmatch(
            r'bytes=(\d+)-(\d+)', request.get_header('Range', '')).groups())
        assert request.get_header('If-range') == etag
        return Response(payload[start:end + 1], status=206, headers={
            'Content-Length': str(end - start + 1),
            'Content-Range': f'bytes {start}-{end}/{len(payload)}', 'ETag': etag})

    monkeypatch.setattr(downloader.urllib.request, 'urlopen', verified)
    report = download(tmp_path / 'verified')
    assert all(report['files'][name]['sha256'] == etags[name].strip('"') for name in payloads)

    payloads['model.safetensors'] = b'X' * len(payloads['model.safetensors'])
    etags['model.safetensors'] = '"' + '0' * 64 + '"'
    with pytest.raises(ValueError, match='remote LFS SHA256'):
        download(tmp_path / 'corrupt-response')
    assert not (tmp_path / 'corrupt-response').exists()
    assert only_partial(tmp_path, 'corrupt-response').is_dir()


def test_non_lfs_config_and_incomplete_or_invalid_metadata(monkeypatch, tmp_path):
    payloads, _, _ = use_fixture(monkeypatch, tmp_path)
    report = download(tmp_path / 'non-lfs-config')
    assert report['files']['config.yml']['sha256'] == hashlib.sha256(payloads['config.yml']).hexdigest()
    rows = [{'type': 'file', 'path': name, 'size': len(payload),
             'lfs': None if name == 'config.yml' else {
                 'oid': hashlib.sha256(payload).hexdigest(), 'size': len(payload)}}
            for name, payload in payloads.items()]

    def metadata(rows_override):
        with mock.patch.object(downloader.urllib.request, 'urlopen',
                               lambda request, timeout: Response(json.dumps(rows_override).encode())):
            return downloader.remote_lfs_metadata(
                downloader.ORIGIN, 'x-square-robot/wall-wm-2.0-robodojo', 'main', payloads)

    missing = [row for row in rows if row['path'] != 'config.yml']
    with pytest.raises(ValueError, match='missing required files'):
        metadata(missing)
    wrong_size = [dict(row) for row in rows]
    wrong_size[0]['size'] += 1
    with pytest.raises(ValueError, match='LFS checkpoint metadata'):
        metadata(wrong_size)
    bad_oid = [dict(row) for row in rows]
    bad_oid[1]['lfs'] = {'oid': 'not-a-sha256', 'size': len(payloads['model.safetensors'])}
    with pytest.raises(ValueError, match='LFS checkpoint metadata'):
        metadata(bad_oid)


def test_multichunk_parallel_download(monkeypatch, tmp_path):
    payloads, _, requests = use_fixture(monkeypatch, tmp_path,
                                       {'model.safetensors': bytes(range(256)) * 600})
    monkeypatch.setattr(downloader, 'CHUNK_SIZE', 32768)
    destination = tmp_path / 'checkpoint'
    report = download(destination, workers=3)
    assert (destination / 'model.safetensors').read_bytes() == payloads['model.safetensors']
    assert report['files']['model.safetensors']['size'] == len(payloads['model.safetensors'])
    assert len([request for request in requests if request.get_header('Range') is not None]) == 5


def test_resume_after_transport_interruption(monkeypatch, tmp_path):
    payloads, _, requests = use_fixture(monkeypatch, tmp_path,
                                       {'model.safetensors': b'0123456789abcdef' * 8192})
    monkeypatch.setattr(downloader, 'CHUNK_SIZE', 16384)
    destination = tmp_path / 'checkpoint'
    original_urlopen = downloader.urllib.request.urlopen
    failed = False

    def interrupted(request, timeout):
        nonlocal failed
        response = original_urlopen(request, timeout)
        if request.get_header('Range') is not None and not failed:
            failed = True
            return Response(response.payload[:1000], status=response.status, headers=response.headers)
        return response

    monkeypatch.setattr(downloader.urllib.request, 'urlopen', interrupted)
    monkeypatch.setattr(downloader, 'MAX_ATTEMPTS', 1)
    with pytest.raises(ValueError, match='Incomplete response'):
        download(destination)
    partial = only_partial(tmp_path)
    part = partial / 'model.safetensors.parts/00000.part'
    assert part.stat().st_size == 1000 and not destination.exists()
    requests.clear()
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', original_urlopen)
    report = download(destination, resume=partial)
    first_get = next(request for request in requests if request.get_header('Range') is not None)
    assert first_get.get_header('Range') == 'bytes=1000-16383'
    assert report['files']['model.safetensors']['sha256'] == hashlib.sha256(payloads['model.safetensors']).hexdigest()
    assert not partial.exists() and not list(partial.parent.glob('*.json.tmp'))


def test_head_metadata_retries_a_transient_missing_length(monkeypatch):
    responses = iter([Response(b'', headers={'ETag': '"model-etag"'}),
                      Response(b'', headers={'Content-Length': '128', 'ETag': '"model-etag"'})])
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', lambda request, timeout: next(responses))
    monkeypatch.setattr(downloader, 'MAX_ATTEMPTS', 2)
    sleeps = []
    monkeypatch.setattr(downloader.time, 'sleep', sleeps.append)
    assert downloader.remote_metadata('https://example.invalid/model') == (128, '"model-etag"')
    assert sleeps == [1]


def test_head_uses_repository_size_when_transfer_is_chunked(monkeypatch):
    monkeypatch.setattr(downloader.urllib.request, 'urlopen',
                        lambda request, timeout: Response(b'', headers={'ETag': 'W/"config-etag"'}))
    assert downloader.remote_metadata('https://example.invalid/config', 14700) == (14700, 'W/"config-etag"')


def test_weak_etag_single_file_accepts_exact_bounded_get(monkeypatch, tmp_path):
    payload = b'current config'
    requests = []

    def whole(request, timeout):
        requests.append(request)
        return Response(payload, status=200, headers={'ETag': 'W/"config-etag"'})

    monkeypatch.setattr(downloader.urllib.request, 'urlopen', whole)
    target = tmp_path / 'config'
    downloader.fetch_range('https://example.invalid/config', target, 0, len(payload) - 1,
                           len(payload), 'W/"config-etag"')
    assert requests[0].get_header('Range') is None
    assert target.read_bytes() == payload


def test_weak_etag_single_file_never_appends_an_interrupted_payload(monkeypatch, tmp_path):
    payload = b'current config'
    target = tmp_path / 'config'
    target.write_bytes(payload[:3])
    monkeypatch.setattr(downloader.urllib.request, 'urlopen',
                        lambda request, timeout: Response(payload, status=200,
                                                         headers={'ETag': 'W/"config-etag"'}))
    downloader.fetch_range('https://example.invalid/config', target, 0, len(payload) - 1,
                           len(payload), 'W/"config-etag"')
    assert target.read_bytes() == payload
    assert target.with_name('config.unverified').read_bytes() == payload[:3]


def test_socket_interruption_retries_from_the_retained_byte_offset(monkeypatch, tmp_path):
    payload = b'0123456789abcdef' * 8192
    url = f'{downloader.ORIGIN}/x-square-robot/wall-wm-2.0-robodojo/resolve/main/model.safetensors'
    target = tmp_path / 'model.part'
    requests = []

    def interrupted(request, timeout):
        requests.append(request)
        start, end = map(int, re.fullmatch(
            r'bytes=(\d+)-(\d+)', request.get_header('Range', '')).groups())
        response = Response(payload[start:end + 1], status=206, headers={
            'Content-Length': str(end - start + 1),
            'Content-Range': f'bytes {start}-{end}/{len(payload)}', 'ETag': '"model-etag"'})
        if len(requests) == 1:
            original_read = response.read

            def fail_after_prefix(size):
                if response.offset >= 1000:
                    raise BrokenPipeError(32, 'Broken pipe')
                return original_read(min(size, 1000 - response.offset))

            response.read = fail_after_prefix
        return response

    monkeypatch.setattr(downloader.urllib.request, 'urlopen', interrupted)
    monkeypatch.setattr(downloader.time, 'sleep', lambda seconds: None)
    downloader.fetch_range(url, target, 0, len(payload) - 1, len(payload), '"model-etag"')
    assert requests[1].get_header('Range') == 'bytes=1000-131071'
    assert target.read_bytes() == payload
    record = json.loads(target.with_name(target.name + '.json').read_text())
    assert record['sha256'] == hashlib.sha256(payload).hexdigest()


def test_http_incomplete_read_uses_the_same_bounded_retry(monkeypatch, tmp_path):
    target = tmp_path / 'model.part'
    response = Response(b'abcdefgh', status=206, headers={
        'Content-Length': '8', 'Content-Range': 'bytes 0-7/8', 'ETag': '"model-etag"'})
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', lambda request, timeout: response)
    monkeypatch.setattr(downloader, 'MAX_ATTEMPTS', 2)
    sleeps = []
    monkeypatch.setattr(downloader.time, 'sleep', sleeps.append)

    def interrupted_read(size):
        raise http.client.IncompleteRead(b'abc', 5)

    response.read = interrupted_read
    with pytest.raises(http.client.IncompleteRead):
        downloader.fetch_range('https://example.invalid/model', target, 0, 7, 8, '"model-etag"')
    assert sleeps == [1]


def test_unrecorded_assembled_file_must_match_current_lfs_not_xet_etag(tmp_path):
    payload = b'actual LFS bytes, distinct from the Xet object identifier'
    etag = '"' + 'e' * 64 + '"'
    target = tmp_path / 'model.safetensors'
    target.write_bytes(payload)
    with mock.patch.object(downloader.urllib.request, 'urlopen', lambda request, timeout: Response(
            b'', headers={'Content-Length': str(len(payload)), 'ETag': etag})):
        with pytest.raises(ValueError, match='recorded SHA256'):
            downloader.fetch('https://example.invalid/model.safetensors', target)
    assert target.read_bytes() == payload


def test_completed_chunk_rejects_a_mismatched_strong_etag(monkeypatch, tmp_path):
    payloads, _, _ = use_fixture(monkeypatch, tmp_path)
    etag = '"' + '0' * 64 + '"'

    def verified(request, timeout):
        payload = payloads['model.safetensors']
        start, end = map(int, re.fullmatch(
            r'bytes=(\d+)-(\d+)', request.get_header('Range', '')).groups())
        assert request.get_header('If-range') == etag
        return Response(payload[start:end + 1], status=206, headers={
            'Content-Length': str(end - start + 1),
            'Content-Range': f'bytes {start}-{end}/{len(payload)}', 'ETag': etag})

    monkeypatch.setattr(downloader.urllib.request, 'urlopen', verified)
    (tmp_path / 'completed.part').write_bytes(payloads['model.safetensors'])
    url = f'{downloader.ORIGIN}/x-square-robot/wall-wm-2.0-robodojo/resolve/main/model.safetensors'
    downloader.fetch_range(url, tmp_path / 'completed.part', 0, len(payloads['model.safetensors']) - 1,
                           len(payloads['model.safetensors']), etag)
    assert (tmp_path / 'completed.part.unverified').read_bytes() == payloads['model.safetensors']
    assert downloader.remote_sha256(etag) == '0' * 64
    assert downloader.remote_sha256('W/' + etag) is None


def test_completed_chunks_are_hashed_and_remote_changes_rejected(monkeypatch, tmp_path):
    payloads, _, requests = use_fixture(monkeypatch, tmp_path,
                                       {'model.safetensors': bytes(range(256)) * 600})
    monkeypatch.setattr(downloader, 'CHUNK_SIZE', 32768)
    destination = tmp_path / 'checkpoint'
    partial = tmp_path / 'checkpoint.partial-fixture'
    parts = partial / 'model.safetensors.parts'
    parts.mkdir(parents=True)
    (partial.with_name(partial.name + '.source_base_url')).write_text(downloader.ORIGIN + '\n')
    part = parts / '00000.part'
    part.write_bytes(payloads['model.safetensors'][:32768])
    url = f'{downloader.ORIGIN}/x-square-robot/wall-wm-2.0-robodojo/resolve/main/model.safetensors'
    downloader.save_metadata(part.with_name(part.name + '.json'), url, 0, 32767,
                             len(payloads['model.safetensors']), '"' + 'e' * 64 + '"',
                             hashlib.sha256(part.read_bytes()).hexdigest())
    retained = tmp_path / 'checkpoint-replacement.partial-retained'
    shutil.copytree(partial, retained)
    retained.with_name(retained.name + '.source_base_url').write_text(
        partial.with_name(partial.name + '.source_base_url').read_text()
        .replace('checkpoint', 'checkpoint-replacement', 1))
    download(destination, resume=partial)
    assert not any(request.get_header('Range') == 'bytes=0-32767' for request in requests)
    payloads['model.safetensors'] = b'new-payload'
    with pytest.raises(ValueError, match='does not match the current main payload'):
        download(tmp_path / 'checkpoint-replacement', resume=retained)


def test_unsafe_resume_and_transport_responses(monkeypatch, tmp_path):
    use_fixture(monkeypatch, tmp_path)
    destination = tmp_path / 'checkpoint'
    partial = tmp_path / 'checkpoint.partial-fixture'
    partial.mkdir()
    (partial.with_name(partial.name + '.source_base_url')).write_text(downloader.ORIGIN + '\n')
    with pytest.raises(ValueError, match='--resume'):
        download(tmp_path / 'wrong-name', resume=partial)
    with pytest.raises(ValueError, match='Resume source differs'):
        download(destination, base_url='https://different-mirror.example', resume=partial)
    destination.mkdir()
    with pytest.raises(FileExistsError):
        download(destination, resume=partial)
    with mock.patch.object(downloader.urllib.request, 'urlopen',
                           lambda request, timeout: Response(b'wrong', status=200)):
        with pytest.raises(ValueError, match='Unsafe range response'):
            downloader.fetch_range('https://unsafe', tmp_path / 'part', 0, 9, 10, '"etag"')
    error = urllib.error.HTTPError('https://forbidden', 403, 'Forbidden', {}, None)
    with mock.patch.object(downloader.urllib.request, 'urlopen', side_effect=error):
        with pytest.raises(urllib.error.HTTPError, match='403'):
            downloader.fetch_range('https://forbidden', tmp_path / 'part', 0, 9, 10, '"etag"')


def test_incomplete_chunk_metadata_and_corrupt_completed_chunk_are_rejected(tmp_path):
    url = 'https://example/model.safetensors'
    part = tmp_path / '00000.part'
    metadata = part.with_name(part.name + '.json')
    downloader.save_metadata(metadata, url, 0, 3, 4, '"etag"', hashlib.sha256(b'good').hexdigest())
    with pytest.raises(ValueError, match='has no payload'):
        downloader.fetch_range(url, part, 0, 3, 4, '"etag"')
    part.write_bytes(b'bad!')
    with pytest.raises(ValueError, match='recorded SHA256'):
        downloader.fetch_range(url, part, 0, 3, 4, '"etag"')


def test_resume_reuses_completed_file_and_rejects_changed_remote(monkeypatch, tmp_path):
    payloads, _, requests = use_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(downloader, 'MAX_ATTEMPTS', 1)
    original_urlopen = downloader.urllib.request.urlopen

    def interrupted(request, timeout):
        if request.get_method() == 'HEAD' and request.full_url.endswith('config.yml'):
            raise urllib.error.URLError('interrupt after assembling model')
        return original_urlopen(request, timeout)

    monkeypatch.setattr(downloader.urllib.request, 'urlopen', interrupted)
    destination = tmp_path / 'checkpoint'
    with pytest.raises(urllib.error.URLError):
        download(destination)
    partial = only_partial(tmp_path)
    assert (partial / 'model.safetensors').is_file()
    retained = tmp_path / 'checkpoint-replacement.partial-retained'
    shutil.copytree(partial, retained)
    retained.with_name(retained.name + '.model.safetensors.complete.json').write_bytes(
        partial.with_name(partial.name + '.model.safetensors.complete.json').read_bytes())
    retained.with_name(retained.name + '.source_base_url').write_text(
        partial.with_name(partial.name + '.source_base_url').read_text()
        .replace('checkpoint', 'checkpoint-replacement', 1))
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', original_urlopen)
    requests.clear()
    download(destination, resume=partial)
    assert not any(request.get_header('Range') is not None and request.full_url.endswith('model.safetensors')
                   for request in requests)
    assert not list(partial.parent.glob('checkpoint.partial-*.complete.json'))
    payloads['model.safetensors'] = b'changed after the model was assembled'
    with pytest.raises(ValueError, match='does not match current main'):
        download(tmp_path / 'checkpoint-replacement', resume=retained)


def test_kill_windows_resume_without_crossing_range_or_losing_source(tmp_path, monkeypatch):
    payloads, _, requests = use_fixture(monkeypatch, tmp_path)
    payload = payloads['model.safetensors']
    part = tmp_path / '00000.part'
    part.write_bytes(payload)
    downloader.fetch_range(f'{downloader.ORIGIN}/model.safetensors', part, 0, len(payload) - 1,
                           len(payload), '"model.safetensors-etag"')
    assert part.read_bytes() == payload
    assert part.with_name(part.name + '.unverified').read_bytes() == payload
    part.with_name(part.name + '.json').unlink()
    damaged = tmp_path / 'damaged.part'
    damaged.write_bytes(b'X' * len(payload))
    downloader.fetch_range(f'{downloader.ORIGIN}/model.safetensors', damaged, 0, len(payload) - 1,
                           len(payload), '"model.safetensors-etag"')
    assert damaged.with_name(damaged.name + '.unverified').read_bytes() == b'X' * len(payload)
    assert damaged.read_bytes() == payload
    strong = tmp_path / 'strong.part'
    strong.write_bytes(payload)
    etag = '"' + hashlib.sha256(payload).hexdigest() + '"'
    downloader.fetch_range(f'{downloader.ORIGIN}/model.safetensors', strong, 0, len(payload) - 1,
                           len(payload), etag)
    assert strong.with_name(strong.name + '.json').is_file()

    destination = tmp_path / 'publication-window'
    original_rename = downloader.os.rename

    def interrupted(source, target):
        if Path(target) == destination:
            raise OSError('interrupted immediately before publication')
        return original_rename(source, target)

    monkeypatch.setattr(downloader.os, 'rename', interrupted)
    with pytest.raises(OSError, match='before publication'):
        download(destination)
    partial = only_partial(tmp_path, 'publication-window')
    assert not (partial / '.source_base_url').exists()
    assert partial.with_name(partial.name + '.source_base_url').is_file()
    monkeypatch.setattr(downloader.os, 'rename', original_rename)
    requests.clear()
    report = download(destination, resume=partial)
    assert not [request for request in requests if request.get_header('Range') is not None]
    assert report['files']['model.safetensors']['sha256'] == hashlib.sha256(payload).hexdigest()
    assert not list(tmp_path.glob('*.source_base_url'))


def test_interrupted_assembly_reuses_verified_chunks(monkeypatch, tmp_path):
    payloads, _, requests = use_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(downloader, 'CHUNK_SIZE', 8)
    destination = tmp_path / 'assembly-window'
    original_copy = downloader.shutil.copyfileobj

    def interrupted_copy(source, target, length):
        original_copy(source, target, 7)
        raise OSError('interrupted during assembly')

    monkeypatch.setattr(downloader.shutil, 'copyfileobj', interrupted_copy)
    with pytest.raises(OSError, match='during assembly'):
        download(destination)
    partial = only_partial(tmp_path, 'assembly-window')
    assert 0 < (partial / 'model.safetensors').stat().st_size < len(payloads['model.safetensors'])
    assert (partial / 'model.safetensors.parts').is_dir()
    monkeypatch.setattr(downloader.shutil, 'copyfileobj', original_copy)
    requests.clear()
    download(destination, resume=partial)
    assert not [request for request in requests if request.get_header('Range') is not None
                and request.full_url.endswith('model.safetensors')]
    assert not (destination / 'model.safetensors.parts').exists()
    assert (destination / 'model.safetensors').read_bytes() == payloads['model.safetensors']


def test_legacy_metadata_migrates_outside_payload_before_verification(monkeypatch, tmp_path):
    _, _, requests = use_fixture(monkeypatch, tmp_path)
    destination = tmp_path / 'legacy-window'
    original_rename = downloader.os.rename

    def interrupted(source, target):
        if Path(target) == destination:
            raise OSError('interrupted immediately before publication')
        return original_rename(source, target)

    monkeypatch.setattr(downloader.os, 'rename', interrupted)
    with pytest.raises(OSError, match='before publication'):
        download(destination)
    monkeypatch.setattr(downloader.os, 'rename', original_rename)
    partial = only_partial(tmp_path, 'legacy-window')
    partial.with_name(partial.name + '.source_base_url').rename(partial / '.source_base_url')
    partial.with_name(partial.name + '.download_state.json').unlink()
    for name in PAYLOADS:
        record = partial.with_name(partial.name + '.' + name + '.complete.json')
        record.rename(partial / (name + '.complete.json'))
    requests.clear()
    download(destination, resume=partial)
    assert not [request for request in requests if request.get_header('Range') is not None]
    assert not list(tmp_path.glob('legacy-window.partial-*'))


def test_resume_accepts_only_matching_orphaned_publication_provenance(monkeypatch, tmp_path):
    _, _, requests = use_fixture(monkeypatch, tmp_path)
    destination = tmp_path / 'orphan-provenance'
    provenance = destination.with_name(destination.name + '.provenance.json')
    code = '''
import importlib.util, hashlib, os, re, signal, sys, urllib.request
from pathlib import Path
source = Path(sys.argv[1])
root = Path(sys.argv[2])
output = Path(sys.argv[3])
spec = importlib.util.spec_from_file_location("kill_fixture", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.MANIFEST = root / "manifest.json"
class Response:
    def __init__(self, body, status=200, headers=None):
        self.body, self.status, self.headers, self.offset = body, status, headers, 0
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, size=-1):
        if size < 0:
            size = len(self.body) - self.offset
        block = self.body[self.offset:self.offset + size]
        self.offset += len(block)
        return block
def fetch(request, timeout):
    name = request.full_url.rsplit("/", 1)[-1]
    payload = payloads[name]
    etag = '"' + ('e' * 64 if name == 'model.safetensors' else name + '-etag') + '"'
    if request.get_method() == "HEAD":
        return Response(b"", headers={"Content-Length": str(len(payload)), "ETag": etag})
    start, end = map(int, re.fullmatch(r"bytes=(\\d+)-(\\d+)", request.get_header("Range")).groups())
    return Response(payload[start:end + 1], 206, {"Content-Length": str(end-start+1),
        "Content-Range": f"bytes {start}-{end}/{len(payload)}", "ETag": etag})
payloads = {name: ("published-" + name).encode() for name in
            ("model.safetensors", "config.yml", "normalizer_action.pth")}
urllib.request.urlopen = fetch
def public_lfs_metadata(base_url, repository, revision, required_files):
    return {name: {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in payloads.items()}
module.remote_lfs_metadata = public_lfs_metadata
def killed_rename(source, target):
    if Path(target) == output:
        os.kill(os.getpid(), signal.SIGKILL)
    return original_rename(source, target)
original_rename = module.os.rename
module.os.rename = killed_rename
sys.argv = ["download_checkpoint.py", "--output", str(output)]
module.main()
'''
    def killed_download(output):
        process = subprocess.run([sys.executable, '-I', '-c', code,
                                  str(PATH), str(tmp_path), str(output)],
                                 capture_output=True, text=True, timeout=30)
        assert process.returncode == -signal.SIGKILL, process.stderr

    killed_download(destination)
    expected = json.loads(provenance.read_text())
    partial = only_partial(tmp_path, destination.name)
    requests.clear()
    download(destination, resume=partial)
    assert not [request for request in requests if request.get_header('Range') is not None]
    assert json.loads(provenance.read_text()) == expected

    retained = tmp_path / 'different-provenance'
    killed_download(retained)
    different = retained.with_name(retained.name + '.provenance.json')
    different.write_text('{}\n')
    with pytest.raises(ValueError, match='provenance differs'):
        download(retained, resume=only_partial(tmp_path, retained.name))
    assert different.read_text() == '{}\n'
