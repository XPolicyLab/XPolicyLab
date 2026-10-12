"""Weight-only downloads: checked-in source and no archive/source downloads."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from .joint96_assets import download_weights, sha256, verify_remote_weights


class WeightDownloadTests(unittest.TestCase):
    def fixture(self, root):
        policy = root / 'policy'
        source = policy / 'joint96_source'
        (source / 'tokenizer').mkdir(parents=True)
        token = source / 'tokenizer/vocab.json'
        token.write_text('{}')
        (source / 'source_manifest.json').write_text(json.dumps(dict(schema='phoenix-joint96-source-v1',
            paths={'tokenizer_root': 'tokenizer'}, sha256={'tokenizer/vocab.json': sha256(token)})))
        specs, entries, cached = {}, [], {}
        for key in ('policy', 'encoder'):
            file = root / (key + '.pt')
            file.write_bytes(key.encode())
            spec = dict(repo_id='fixture/model', revision='master', filename=file.name,
                        path=file.name, sha256=sha256(file))
            specs[key] = spec
            entries.append(dict(Path=file.name, Type='blob', Sha256=spec['sha256']))
            cached[file.name] = file
        (policy / 'phoenix_release.json').write_text(json.dumps(dict(step=10000, weights=specs)))
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
            download_weights(target, policy_dir=policy, api=api, download_file=download)
            self.assertEqual(calls, ['policy.pt', 'encoder.pt'])
            self.assertEqual((target / 'policy.pt').read_bytes(), b'policy')
            self.assertTrue((target / 'base_models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/vocab.json').is_file())
            self.assertFalse((target / 'joint96_source').exists())
            self.assertFalse(json.loads((target / 'WEIGHTS_READY.json').read_text())['model_loaded'])
            def offline(**kw):
                self.fail('Existing correct weights must not require network access')
            download_weights(target, policy_dir=policy,
                             api=SimpleNamespace(get_model_files=offline), download_file=offline)

    def test_remote_release_requires_all_pinned_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            policy, entries, _ = self.fixture(Path(directory))
            result = verify_remote_weights(policy_dir=policy,
                api=SimpleNamespace(get_model_files=lambda **kw: entries))
            self.assertEqual(set(result), {'policy', 'encoder'})
            with self.assertRaises(FileNotFoundError):
                verify_remote_weights(policy_dir=policy,
                    api=SimpleNamespace(get_model_files=lambda **kw: entries[:1]))
            entries[1]['Sha256'] = 'different'
            with self.assertRaisesRegex(ValueError, 'differs from pinned'):
                verify_remote_weights(policy_dir=policy,
                    api=SimpleNamespace(get_model_files=lambda **kw: entries))

    def test_missing_published_weight_fails_before_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy, entries, _ = self.fixture(root)
            with self.assertRaisesRegex(FileNotFoundError, 'encoder.pt'):
                download_weights(root / 'weights', policy_dir=policy,
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
                download_weights(target, policy_dir=policy, api=api, download_file=lambda **kw: None)
            self.assertEqual(wrong.read_bytes(), b'bad')
            wrong.unlink()
            with self.assertRaisesRegex(ValueError, 'Downloaded weight checksum'):
                download_weights(target, policy_dir=policy, api=api,
                                 download_file=lambda **kw: cached['encoder.pt'])
            self.assertFalse(wrong.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
