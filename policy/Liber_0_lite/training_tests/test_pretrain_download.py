import hashlib
import json
from pathlib import Path
import sys

import huggingface_hub
import pytest

from XPolicyLab.policy.Liber_0_lite import download_checkpoint


@pytest.mark.parametrize('corrupt', [False, True])
@pytest.mark.parametrize('override', [False, True])
def test_pretraining_download_does_not_fetch_sft_weights(monkeypatch, tmp_path, corrupt, override):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        if len(calls) > 1:
            return
        root = Path(kwargs['local_dir'])
        root.mkdir()
        (root / 'pretrain').mkdir()
        root_hashes = {}
        for name in ('config.yaml', 'dataset_stats.json'):
            data = name.encode()
            (root / name).write_bytes(data)
            root_hashes[name] = hashlib.sha256(data).hexdigest()
        (root / 'manifest.json').write_text(json.dumps(dict(sha256=root_hashes)))
        weights = b'pretraining'
        (root / 'pretrain/model.pt').write_bytes(b'corrupt' if corrupt else weights)
        (root / 'pretrain/manifest.json').write_text(json.dumps(dict(sha256={'model.pt':hashlib.sha256(weights).hexdigest()})))

    monkeypatch.setattr(huggingface_hub, 'snapshot_download', download)
    arguments = ['download_checkpoint.py', '--pretrained', '--destination', str(tmp_path / 'checkpoint'),
                 '--assets-dir', str(tmp_path / 'assets')]
    if override:
        arguments += ['--revision', 'test-pinned-revision']
    monkeypatch.setattr(sys, 'argv', arguments)
    if corrupt:
        with pytest.raises(ValueError, match='checksum mismatch'):
            download_checkpoint.main()
        assert len(calls) == 1
    else:
        download_checkpoint.main()
        assert len(calls) == 2
    assert calls[0]['revision'] == ('test-pinned-revision' if override else '697c4792180cd38fb2045fb94d70f7981d4638b2')
    assert 'model.pt' not in calls[0]['allow_patterns']
    assert 'pretrain/model.pt' in calls[0]['allow_patterns']
    assert calls[0]['repo_id'] == 'LiberAI/Liber0-Lite-Robodojo'
