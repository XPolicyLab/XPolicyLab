from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packed_client import client_checkout


class NativeClientTests(unittest.TestCase):
    def test_legacy_client_records_only_explicit_native_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            benchmark = Path(temporary)
            source = Path("src/eval_client")
            (benchmark / source).mkdir(parents=True)
            main = benchmark / source / "main.py"
            main.write_text("def finish(env):\n    _delete_resume_manifest(env)\n")
            env = benchmark / source / "eval_env.py"
            original = """\
def build():
    class Env:
        def reset_outcome(self, unstable_envs):
            self.unstable_nums += len(unstable_envs)
        def episode_outcome(self, unstable_in_batch):
            if unstable_in_batch:
                self.unstable_nums += len(unstable_in_batch)
    return Env()
"""
            env.write_text(original)
            (benchmark / "Assets").mkdir()
            with client_checkout(benchmark) as root:
                self.assertTrue((root / "Assets").samefile(benchmark / "Assets"))
                module = {}
                exec((root / source / "eval_env.py").read_text(), module)
                client = module["build"]()
                client.unstable_nums, client.abandoned_seeds = 0, set()
                client.get_seeds_for_envs = lambda slots: {19 + i for i in slots}
                client.persist_resume_manifest = Mock()
                client.reset_outcome([])
                client.episode_outcome([])
                client.persist_resume_manifest.assert_not_called()
                client.reset_outcome([0])
                client.episode_outcome([2])
                self.assertEqual(client.abandoned_seeds, {19, 21})
                self.assertEqual(client.unstable_nums, 2)
                exec((root / source / "main.py").read_text(), module)
                module["finish"](client)
                self.assertEqual(client.persist_resume_manifest.call_count, 3)
            self.assertEqual(env.read_text(), original)
            self.assertIn("_delete_resume_manifest", main.read_text())

    def test_client_which_retains_manifests_uses_original_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            src = root / "src/eval_client"
            src.mkdir(parents=True)
            (src / "main.py").write_text("def finish(env):\n    env.persist_resume_manifest()\n")
            with client_checkout(root) as path:
                self.assertEqual(path, root)
