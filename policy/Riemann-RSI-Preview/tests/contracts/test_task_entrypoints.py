"""Actual task CLIs keep one official route, independent of caller cwd."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class TaskEntrypoints(unittest.TestCase):
    def test_all54_published_entries_offer_the_same_official_interface(self):
        from rsi.policy_catalog import catalog
        modes = catalog(ROOT)
        self.assertEqual(len(modes), 54)
        with tempfile.TemporaryDirectory() as cwd:
            for mode in modes:
                with self.subTest(mode=mode):
                    result = subprocess.run([sys.executable, '-I', '-S', '-B',
                        str(ROOT / 'tasks' / mode / 'entry.py'), '--help'],
                        cwd=cwd, capture_output=True, text=True, timeout=20)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('All 54 registered modes share this client', result.stdout)

    def test_entry_rejects_another_public_task_before_sdk_startup(self):
        result = subprocess.run([sys.executable, '-I', '-S', '-B',
            str(ROOT / 'tasks/plug_in_charger/entry.py'), '--task_name', 'insert_key', '--help'],
            capture_output=True, text=True, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Task entry and requested official task differ', result.stderr)
