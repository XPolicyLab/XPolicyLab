"""Weight-only downloads: checked-in source and no archive/source downloads."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import patch
import shutil

from .joint96_assets import download_weights, sha256, verify_remote_weights, prepare_tokenizer, TOKENIZER_PATH


class WeightDownloadTests(unittest.TestCase):
    def test_parallel_tokenizer_initialization_never_exposes_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            (source / 'spiece.model').write_bytes(b'complete vocabulary')
            started, release = Event(), Event()
            copy = shutil.copy2

            def delayed_copy(src, dst):
                if not started.is_set():
                    Path(dst).write_bytes(b'partial')
                    started.set()
                    if not release.wait(10):
                        raise TimeoutError('Second initializer did not complete')
                return copy(src, dst)

            with patch('shutil.copy2', side_effect=delayed_copy), ThreadPoolExecutor(1) as pool:
                first = pool.submit(prepare_tokenizer, root / 'weights', source)
                try:
                    self.assertTrue(started.wait(10))
                    prepare_tokenizer(root / 'weights', source)
                finally:
                    release.set()
                first.result()
            target = root / 'weights' / TOKENIZER_PATH
            self.assertEqual((target / 'spiece.model').read_bytes(), b'complete vocabulary')
            self.assertFalse(list(target.glob('.tokenizer-*')))

    def fixture(self, root):
        policy = root / 'policy'
        source = policy / 'joint96_source'
        (source / 'tokenizer').mkdir(parents=True)
        token = source / 'tokenizer/vocab.json'
        token.write_text('{}')
        (source / 'source_manifest.json').write_text(json.dumps(dict(schema='phoenix-joint96-source-v1',
            paths={'tokenizer_root': 'tokenizer'}, sha256={'tokenizer/vocab.json': sha256(token)})))
        specs, entries, cached = {}, [], {}
        for key in ('policy', 'encoder', 'arx_x5', 'piper_x'):
            file = root / (key + '.pt')
            file.write_bytes(key.encode())
            spec = dict(repo_id='fixture/model', revision='master', filename=file.name,
                        path=file.name, sha256=sha256(file))
            specs[key] = spec
            entries.append(dict(Path=file.name, Type='blob', Sha256=spec['sha256']))
            cached[file.name] = file
        (policy / 'phoenix_release.json').write_text(json.dumps(dict(
            schema='phoenix-joint96-release-v2', checkpoints=dict(
                piper=specs['policy'], arx_x5=specs['arx_x5'], piper_x=specs['piper_x']),
            weights={'encoder': specs['encoder']})))
        return policy, entries, cached

    def test_individual_weight_download_and_offline_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy, entries, cached = self.fixture(root)
            calls = []
            api = SimpleNamespace(get_model_files=lambda **kw: entries)
            def download(**kw):
                self.assertEqual(kw['revision'], 'master')
                self.assertTrue(kw['file_path'].endswith('.pt'))
                calls.append(kw['file_path'])
                return cached[kw['file_path']]
            target = root / 'weights'
            download_weights(target, policy_dir=policy, robot_type='piper', api=api, download_file=download)
            self.assertEqual(calls, ['policy.pt', 'encoder.pt'])
            self.assertEqual((target / 'policy.pt').read_bytes(), b'policy')
            self.assertTrue((target / 'base_models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/vocab.json').is_file())
            self.assertFalse((target / 'joint96_source').exists())
            self.assertFalse(json.loads((target / 'WEIGHTS_READY.json').read_text())['model_loaded'])
            def offline(**kw):
                self.fail('Existing correct weights must not require network access')
            download_weights(target, policy_dir=policy, robot_type='piper',
                             api=SimpleNamespace(get_model_files=offline), download_file=offline)

    def test_download_all_and_reuse_shared_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy, entries, cached = self.fixture(root)
            calls = []
            def download(**kw):
                calls.append(kw['file_path'])
                return cached[kw['file_path']]
            api = SimpleNamespace(get_model_files=lambda **kw: entries)
            target = root / 'weights'
            download_weights(target, policy_dir=policy, robot_type='piper',
                             api=api, download_file=download)
            self.assertEqual(calls, ['policy.pt', 'encoder.pt'])
            calls.clear()
            download_weights(target, policy_dir=policy, api=api, download_file=download)
            self.assertEqual(calls, ['arx_x5.pt', 'piper_x.pt'])
            result = verify_remote_weights(policy_dir=policy, api=api)
            self.assertEqual(set(result), {'checkpoint_arx_x5', 'checkpoint_piper',
                                          'checkpoint_piper_x', 'encoder'})

    def test_single_robot_does_not_require_other_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy, entries, cached = self.fixture(root)
            api = SimpleNamespace(get_model_files=lambda **kw: entries[:2])
            download_weights(root / 'weights', policy_dir=policy, robot_type='piper',
                             api=api, download_file=lambda **kw: cached[kw['file_path']])
            with self.assertRaisesRegex(ValueError, 'Unsupported checkpoint robot'):
                verify_remote_weights(policy_dir=policy, robot_type='unknown', api=api)

    def test_remote_release_requires_all_pinned_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            policy, entries, _ = self.fixture(Path(directory))
            result = verify_remote_weights(policy_dir=policy, robot_type='piper',
                api=SimpleNamespace(get_model_files=lambda **kw: entries))
            self.assertEqual(set(result), {'checkpoint', 'encoder'})
            with self.assertRaises(FileNotFoundError):
                verify_remote_weights(policy_dir=policy, robot_type='piper',
                    api=SimpleNamespace(get_model_files=lambda **kw: entries[:1]))
            entries[1]['Sha256'] = 'different'
            with self.assertRaisesRegex(ValueError, 'differs from pinned'):
                verify_remote_weights(policy_dir=policy, robot_type='piper',
                    api=SimpleNamespace(get_model_files=lambda **kw: entries))

    def test_missing_published_weight_fails_before_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy, entries, _ = self.fixture(root)
            with self.assertRaisesRegex(FileNotFoundError, 'encoder.pt'):
                download_weights(root / 'weights', policy_dir=policy, robot_type='piper',
                    api=SimpleNamespace(get_model_files=lambda **kw: entries[:1]),
                    download_file=lambda **kw: self.fail('Do not download until all weights are published'))
            self.assertFalse((root / 'weights').exists())

    def test_existing_and_downloaded_corruption_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy, entries, cached = self.fixture(root)
            target = root / 'weights'
            target.mkdir()
            wrong = target / 'policy.pt'
            wrong.write_bytes(b'bad')
            api = SimpleNamespace(get_model_files=lambda **kw: entries)
            with self.assertRaisesRegex(ValueError, 'Refusing to replace'):
                download_weights(target, policy_dir=policy, robot_type='piper', api=api, download_file=lambda **kw: None)
            self.assertEqual(wrong.read_bytes(), b'bad')
            wrong.unlink()
            with self.assertRaisesRegex(ValueError, 'Downloaded weight checksum'):
                download_weights(target, policy_dir=policy, robot_type='piper', api=api,
                                 download_file=lambda **kw: cached['encoder.pt'])
            self.assertFalse(wrong.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
