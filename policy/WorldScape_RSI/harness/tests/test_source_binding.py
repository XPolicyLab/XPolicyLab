"""The frozen Harness must load the same WSP source it verifies."""
from pathlib import Path
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from test_harness import backends, framework, release
from r2si_under_test import wam, verify_checkpoints


@unittest.skipUnless(framework, 'Requires XPolicyLab CPU dependencies')
class SourceBindingTests(unittest.TestCase):
    def setUp(self):
        backends.register_vendor()
        from XPolicyLab.policy.WorldScape_RSI.xpolicylab_adapter import runtime_config
        self.runtime = runtime_config
        # This fixture constructs a temporary source tree. Preserve, then isolate
        # modules imported by other tests, just as a fresh policy process would.
        self.enterContext(patch.dict(sys.modules))
        for name in ('worldscape_policy', 'evals'):
            sys.modules.pop(name, None)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.own = self.root/'WorldScape_RSI/harness'; self.own.mkdir(parents=True)
        self.policy = self.own.parent
        self.source = self.policy/'worldscape-policy'
        self.foreign = self.root/'foreign-source'
        for directory in (self.source, self.foreign):
            (directory/'src/worldscape_policy').mkdir(parents=True)
            (directory/'evals').mkdir()
            (directory/'src/worldscape_policy/__init__.py').write_text('# '+directory.name+'\n')
        self.entry = self.policy/'wam_model.py'; self.entry.write_text('# pinned adapter\n')
        self.weights = self.root/'weights'; self.weights.mkdir()
        (self.weights/'model.bin').write_bytes(b'fixture')
        self.spec = {
            'policy_name':'WorldScape_RSI','profile':'fixture','config':{},
            'checkpoint_env':'TEST_BINDING_CHECKPOINT','worldscape_root_env':'TEST_BINDING_SOURCE',
            'device_env':'TEST_BINDING_DEVICE',
            'checkpoint_files':{'model.bin':release.digest(self.weights/'model.bin')},
            'source_files':{'wam_model.py':release.digest(self.entry),
                            'worldscape-policy/src/worldscape_policy/__init__.py':
                                release.digest(self.source/'src/worldscape_policy/__init__.py')},
        }
        self.enterContext(patch.object(wam,'ROOT',self.own))
        self.enterContext(patch.dict(os.environ,{'TEST_BINDING_CHECKPOINT':str(self.weights),
                                               'TEST_BINDING_SOURCE':'','WORLDSCAPE_POLICY_ROOT':'',
                                               'TEST_BINDING_DEVICE':'cpu'}))
        original_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None),original_path))

    def construct(self):
        runtime = self.runtime
        class Probe:
            def __init__(self,cfg):
                self.config = cfg
                self.source = runtime.prepare_source(cfg)
            def get_runtime_info(self): return {'requires_explicit_commit':True}
            def close(self): pass
        with patch.object(wam.inspect,'getfile',return_value=str(self.entry)), \
             patch.object(wam.importlib,'import_module',return_value=types.SimpleNamespace(Model=Probe)):
            return wam.create_model(self.spec,{'seed':7})

    def test_unset_environment_supplies_checked_source_to_native_resolver(self):
        model = self.construct()
        self.assertEqual(model.config['worldscape_root'],str(self.source.resolve()))
        self.assertEqual(model.source,self.source.resolve())
        self.assertNotIn('task_name',model.config)

    def test_previously_imported_foreign_source_is_rejected(self):
        sys.modules['worldscape_policy'] = types.SimpleNamespace(
            __file__=str(self.foreign/'src/worldscape_policy/__init__.py'))
        with self.assertRaisesRegex(RuntimeError, 'already imported from another checkout'):
            self.construct()

    def test_each_external_override_is_rejected_before_model_import(self):
        for key in ('TEST_BINDING_SOURCE','WORLDSCAPE_POLICY_ROOT'):
            with self.subTest(key=key), patch.dict(os.environ,{key:str(self.foreign)}), \
                 patch.object(wam.importlib,'import_module') as load:
                with self.assertRaisesRegex(ValueError,'hash-verified WSP source'):
                    wam.create_model(self.spec,{'seed':7})
                load.assert_not_called()

    def test_approved_explicit_source_cannot_hide_foreign_fallback(self):
        with patch.dict(os.environ,{'TEST_BINDING_SOURCE':str(self.source),
                                    'WORLDSCAPE_POLICY_ROOT':str(self.foreign)}):
            with self.assertRaisesRegex(ValueError,'WORLDSCAPE_POLICY_ROOT'):
                wam.verified_source_root(self.spec)

    def test_equivalent_paths_to_the_bundled_tree_are_accepted(self):
        alias = self.source/'..'/'worldscape-policy'
        with patch.dict(os.environ,{'TEST_BINDING_SOURCE':str(alias),
                                    'WORLDSCAPE_POLICY_ROOT':str(self.source)}):
            self.assertEqual(self.construct().source,self.source.resolve())

    def test_tampered_bundled_source_is_rejected_before_model_import(self):
        (self.source/'src/worldscape_policy/__init__.py').write_text('# changed\n')
        with patch.object(wam.importlib,'import_module') as load:
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'):
                wam.create_model(self.spec,{})
            load.assert_not_called()

    def test_preflight_rejects_the_same_external_source_override(self):
        frozen = types.SimpleNamespace(sha256='fixture',data={'wam_skills':{'fixture':self.spec}})
        with patch.object(verify_checkpoints,'FrozenRelease',return_value=frozen), \
             patch.dict(os.environ,{'WORLDSCAPE_POLICY_ROOT':str(self.foreign)}):
            with self.assertRaisesRegex(ValueError,'hash-verified WSP source'):
                verify_checkpoints.verify()

    def test_preflight_receipt_identifies_the_checked_source(self):
        frozen = types.SimpleNamespace(sha256='fixture',data={'wam_skills':{'fixture':self.spec}})
        with patch.object(verify_checkpoints,'FrozenRelease',return_value=frozen):
            receipt = verify_checkpoints.verify()
        self.assertEqual(receipt['checkpoints']['fixture']['verified_source_root'],str(self.source.resolve()))


if __name__ == '__main__': unittest.main()
