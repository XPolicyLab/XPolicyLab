"""Small files exercise model binding admission without loading/downloading weights."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

from rsi.model_assets import verify


class ModelAssetPreflightTests(unittest.TestCase):
    def test_published_downloads_match_the_model_loader_and_pin_one_revision(self):
        root = Path(__file__).resolve().parents[2]
        policy = json.loads((root/'policies/riemann_1/policy.json').read_text())
        manifest = json.loads((root/'configs/models/downloads.json').read_text())
        published = [item for item in manifest['files'] if item['id'].startswith('riemann1.0:')]
        self.assertEqual(len(published), 7)
        self.assertEqual({item['path'].removeprefix('Riemann1.0/'): {
            key:item[key] for key in ('bytes','sha256')} for item in published}, policy['checkpoint_files'])
        self.assertIn(policy['model_directory']+'/config.json', policy['checkpoint_files'])
        self.assertIn(policy['norm_stats'], policy['checkpoint_files'])
        self.assertIn(policy['train_config'], policy['checkpoint_files'])
        revisions = set()
        for item in published:
            self.assertEqual(item['author_model'], 'bolwang/Riemann-RSI-Preview_robodojo')
            revision = item['author_revision']; self.assertRegex(revision, r'^[0-9a-f]{40}$')
            revisions.add(revision)
            url = urlsplit(item['url'])
            self.assertEqual((url.scheme,url.hostname), ('https','huggingface.co'))
            self.assertEqual(url.path, '/bolwang/Riemann-RSI-Preview_robodojo/resolve/'+revision+'/'+item['path'])
        self.assertEqual(len(revisions), 1)

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.checkpoint, self.encoder = self.root / 'checkpoint', self.root / 'encoder'
        self.files = {'config.json': b'{}', 'model-01.safetensors': b'shard',
                      'normalization.jsonl': b'norm'}
        self.checkpoint.mkdir(); self.encoder.mkdir()
        policy = self.root / 'policies/riemann_1/policy.json'
        policy.parent.mkdir(parents=True)
        records = {}
        for name, data in self.files.items():
            (self.checkpoint / name).write_bytes(data)
            records[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        policy.write_text(json.dumps({'checkpoint_files': records}))
        (self.encoder / 'tokenizer.json').write_bytes(b'tokenizer')
        (self.encoder / 't5.pth').write_bytes(b'encoder')
        manifest = {'files': [{'id': 'public-encoder:' + name, 'path': 'encoder/' + name,
                               'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                              for name, data in [('tokenizer.json', b'tokenizer'), ('t5.pth', b'encoder')]]}
        (self.root / 'configs/models').mkdir(parents=True)
        (self.root / 'configs/models/downloads.json').write_text(json.dumps(manifest))

    def check(self):
        return verify(wam_checkpoint=self.checkpoint, encoder_path=self.encoder, root=self.root)

    def test_complete_files_pass_without_models_or_large_hash_reads(self):
        with patch('rsi.model_records.hashlib.file_digest', side_effect=AssertionError('No deep read in completeness preflight')):
            result = self.check()
        self.assertEqual(result['files_size_checked'], {'wam_checkpoint': 3, 'encoder_path': 2})
        self.assertFalse(result['content_hashes_verified'])
        self.assertFalse(result['models_loaded'])

    def test_empty_checkpoint_is_rejected(self):
        for name in self.files: (self.checkpoint / name).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or incomplete'):
            self.check()

    def test_empty_encoder_is_rejected(self):
        for name in ('tokenizer.json', 't5.pth'): (self.encoder / name).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or incomplete'):
            self.check()

    def test_missing_shard_is_rejected(self):
        (self.checkpoint / 'model-01.safetensors').unlink()
        with self.assertRaises(ValueError): self.check()

    def test_missing_normalization_is_rejected(self):
        (self.checkpoint / 'normalization.jsonl').unlink()
        with self.assertRaises(ValueError): self.check()

    def test_incomplete_tokenizer_is_rejected(self):
        (self.encoder / 'tokenizer.json').write_text('x')
        with self.assertRaises(ValueError): self.check()

    def test_missing_directory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing asset directory'):
            verify(encoder_path=self.root / 'missing', root=self.root)

    def test_empty_encoder_declaration_is_rejected(self):
        (self.root / 'configs/models/downloads.json').write_text('{"files": []}')
        with self.assertRaisesRegex(ValueError, 'nonempty file declarations'):
            self.check()

    def test_escaping_member_is_rejected(self):
        p = self.root / 'configs/models/downloads.json'
        d = json.loads(p.read_text()); d['files'][0].update(id='public-encoder:../outside', path='encoder/../outside')
        p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, 'inside their declared root'):
            self.check()

    def test_real_configuration_cli_checks_assets_before_writing_binding(self):
        from tools import configure_official_policy as cli
        from rsi.rule_configuration import settings
        runtime = settings()
        (self.root / 'configs/official-rule-evaluation.json').write_text(json.dumps(runtime))
        python = self.root / 'eval/bin/python'
        python.parent.mkdir(parents=True); python.touch()
        arguments = ['--robodojo', str(self.root), '--eval-python', str(python),
                     '--policy-python', str(python),
                     '--wam-checkpoint', str(self.checkpoint), '--wam-encoder', str(self.encoder), '--check-only']
        with patch.object(cli, 'ROOT', self.root), \
                patch('subprocess.check_output', return_value=settings()['official_commit']), \
                patch.object(cli, 'python_identity', return_value={
                    'version': runtime['official_python_version'], 'prefix': str(python.parent.parent.resolve())}), \
                patch.object(cli, 'policy_preflight', return_value={'models_loaded': False}), \
                patch('rsi.policy_catalog.catalog', return_value={}), \
                patch('tools.official_rule_integrity.verify_policy', return_value=1), \
                patch('rsi.model_assets.verify_rule', return_value={'models_loaded':False}), \
                patch('builtins.print'):
            self.assertEqual(cli.main(arguments), 0)
            (self.encoder / 't5.pth').unlink()
            with patch('sys.stderr'), self.assertRaises(SystemExit) as result:
                cli.main(arguments)
            self.assertEqual(result.exception.code, 2)
        self.assertFalse((self.root / 'deploy.local.json').exists())


if __name__ == '__main__':
    unittest.main()
