"""Every bench worker module imports without its simulator (they start it only when a
worker is created), so a broken import shows here and not first on a GPU machine."""

import importlib
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


class BenchImportTests(unittest.TestCase):
    def test_every_bench_module_imports(self):
        modules = sorted(path.relative_to(ROOT).with_suffix("").as_posix().replace("/", ".")
                         for path in (ROOT / "benches").glob("*/*.py"))
        self.assertIn("benches.robotwin.runtime", modules)
        for name in modules:
            with self.subTest(module=name):
                importlib.import_module(name)


if __name__ == "__main__":
    unittest.main()
