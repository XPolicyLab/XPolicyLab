"""A submitted adapter may advance Git without changing the official runtime."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


class OfficialProtocolAdmission(unittest.TestCase):
    def git(self, root, *args):
        return subprocess.check_output(['git', '-C', str(root), *args],
                                       text=True, stderr=subprocess.PIPE).strip()

    def initialize(self, root):
        root.mkdir(parents=True, exist_ok=True)
        self.git(root, 'init', '-q')
        self.git(root, 'config', 'user.name', 'Protocol contract fixture')
        self.git(root, 'config', 'user.email', 'fixture@example.invalid')

    def write(self, relative, text):
        path = self.protocol / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def commit(self):
        self.git(self.protocol, 'add', '.')
        self.git(self.protocol, 'commit', '-qm', 'Fixture change')

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        self.sdk = root / 'sdk'
        self.initialize(self.sdk)
        (self.sdk / 'README.md').write_text('Official SDK fixture\n')
        self.git(self.sdk, 'add', 'README.md')
        self.git(self.sdk, 'commit', '-qm', 'SDK baseline')
        self.protocol = self.sdk / 'XPolicyLab'
        self.initialize(self.protocol)
        self.write('README.md', 'Policy catalog\n')
        self.write('README_zh.md', '策略目录\n')
        self.write('model_template.py', 'class ModelTemplate: pass\n')
        protected = {
            'client_server/ws/protocol/codec.py': "FORMAT = 'official RGB'\n",
            'utils/process_data.py': "ACTION_FORMAT = 'official'\n",
            'policy/demo_policy/deploy.py': 'def eval_one_episode(*args): pass\n',
        }
        for name, content in protected.items():
            self.write(name, content)
        self.write('policy/demo_policy/helper.py', 'ORIGINAL = True\n')
        self.commit()
        source = root / 'source'
        (source / 'configs').mkdir(parents=True)
        data = dict(official_commit=self.git(self.sdk, 'rev-parse', 'HEAD'),
                    actual_XPolicyLab_commit=self.git(self.protocol, 'rev-parse', 'HEAD'),
                    official_files={'XPolicyLab/' + name: hashlib.sha256(content.encode()).hexdigest()
                                    for name, content in protected.items()})
        (source / 'configs/official-runtime-inputs.json').write_text(json.dumps(data))
        from tools import official_rule_integrity
        self.integrity = official_rule_integrity
        location = patch.object(official_rule_integrity, 'ROOT', source)
        location.start()
        self.addCleanup(location.stop)

    def verify(self):
        return self.integrity.verify_policy(sdk=self.sdk)

    def test_pinned_protocol_remains_accepted(self):
        self.assertEqual(self.verify(), 3)

    def test_committed_submission_and_bilingual_catalog_preserve_runtime(self):
        self.write('policy/Riemann-RSI-Preview/model.py', 'class Model: pass\n')
        self.write('README.md', 'Policy catalog includes Riemann-RSI-Preview\n')
        self.write('README_zh.md', '策略目录包含 Riemann-RSI-Preview\n')
        self.commit()
        self.assertEqual(self.verify(), 3)

    def test_other_policy_additions_do_not_change_selected_protocol(self):
        self.write('policy/another_policy/model.py', 'class Model: pass\n')
        self.commit()
        self.assertEqual(self.verify(), 3)

    def test_data_download_tools_do_not_change_the_policy_protocol(self):
        self.write('scripts/RoboDojo/download_robodojo_data.sh', '#!/bin/bash\nexit 0\n')
        self.write('docs/data-processing.md', 'Data preparation instructions\n')
        self.commit()
        self.assertEqual(self.verify(), 3)

    def test_new_shared_runtime_helper_is_rejected_without_a_file_pin(self):
        self.write('utils/additional_runtime.py', 'CHANGED_RUNTIME = True\n')
        self.commit()
        with self.assertRaisesRegex(ValueError, 'shared runtime changed'):
            self.verify()

    def test_uncommitted_reference_loop_helper_is_rejected(self):
        self.write('policy/demo_policy/helper.py', 'ORIGINAL = False\n')
        with self.assertRaisesRegex(ValueError, 'uncommitted changes'):
            self.verify()

    def test_protected_codec_change_is_rejected(self):
        self.write('client_server/ws/protocol/codec.py', "FORMAT = 'changed'\n")
        self.commit()
        with self.assertRaisesRegex(ValueError, 'closure changed'):
            self.verify()

    def test_committed_shared_helper_change_is_rejected(self):
        self.write('model_template.py', 'class DifferentTemplate: pass\n')
        self.commit()
        with self.assertRaises(ValueError):
            self.verify()

    def test_uncommitted_shared_helper_change_is_rejected(self):
        self.write('model_template.py', 'class DifferentTemplate: pass\n')
        with self.assertRaises(ValueError):
            self.verify()

    def test_reference_loop_cannot_hide_inside_allowed_policy_directory(self):
        self.write('policy/demo_policy/deploy.py', 'def eval_one_episode(*args): return 100\n')
        self.commit()
        with self.assertRaisesRegex(ValueError, 'closure changed'):
            self.verify()

    def test_unrelated_history_is_rejected_even_with_matching_files(self):
        self.git(self.protocol, 'checkout', '--orphan', 'unrelated')
        self.write('README.md', 'Unrelated history, matching protected runtime\n')
        self.commit()
        with self.assertRaises(ValueError):
            self.verify()

    def test_official_sdk_commit_requirement_remains_exact(self):
        (self.sdk / 'README.md').write_text('Different official SDK revision\n')
        self.git(self.sdk, 'add', 'README.md')
        self.git(self.sdk, 'commit', '-qm', 'Different SDK revision')
        with self.assertRaisesRegex(ValueError, 'Official SDK commit changed'):
            self.verify()
