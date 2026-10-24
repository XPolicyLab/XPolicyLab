"""The documentation stays true to the code: commands parse, paths exist, every
configuration key the code reads is described, README's generated tables are current,
every page is in the navigation, and the site builds strictly (when MkDocs is installed)."""

import importlib.util
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

DOCS = ROOT / "docs"
PAGES = sorted(DOCS.rglob("*.md")) + [ROOT / "README.md"] + sorted(
    path for path in ROOT.glob("*/README.md") if path.parent.name != "third_party")
# Top-level directories whose paths the docs may name; anything under them must exist.
CHECKED_ROOTS = ("configs", "docs", "environment", "scripts", "services", "skills", "src", "tests", "benches",
                 "prompts", "embodiments")
# Files the docs ask the reader to create.
CREATED_BY_THE_READER = {"configs/deploy/mine.toml", "skills/wiggle/skill.toml", "skills/wiggle/__init__.py"}


def _generator():
    spec = importlib.util.spec_from_file_location("gen_docs_reference", ROOT / "scripts" / "gen_docs_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shell_blocks(text):
    for match in re.finditer(r"```(?:bash|sh|shell)\n(.*?)```", text, re.S):
        yield match.group(1).replace("\\\n", " ")


def _harness_commands(path):
    """Each `harness ...` command of the page's shell blocks, as arguments."""
    for block in _shell_blocks(path.read_text(encoding="utf-8")):
        for line in block.splitlines():
            line = line.split(" #", 1)[0].strip()                        # a trailing comment
            line = re.split(r"\s(?:\|\||&&|\||2?>)\s", line, maxsplit=1)[0]      # a pipe or redirection
            if not line.startswith("harness "):
                continue
            try:
                words = shlex.split(line)[1:]
            except ValueError:
                words = line.split()[1:]
            arguments = []
            for word in words:
                if word == "...":
                    continue
                word = re.sub(r"<[^>]*>", "x", word).strip("[]")
                if word:
                    arguments.append(word)
            yield line, arguments


class DocsTests(unittest.TestCase):
    def test_harness_commands_parse(self):
        from harness_policy.cli import parser

        failures, checked = [], 0
        for path in PAGES:
            for line, arguments in _harness_commands(path):
                checked += 1
                try:
                    parser().parse_args(arguments)
                except SystemExit:
                    failures.append(f"{path.relative_to(ROOT)}: {line}")
        self.assertEqual(failures, [])
        self.assertGreater(checked, 40)

    def test_named_paths_exist(self):
        missing = []
        pattern = re.compile(r"`((?:%s)/[A-Za-z0-9_./-]+)`" % "|".join(CHECKED_ROOTS))
        for path in PAGES:
            for match in pattern.finditer(path.read_text(encoding="utf-8")):
                named = match.group(1).rstrip(".")
                if named not in CREATED_BY_THE_READER and not (ROOT / named).exists():
                    missing.append(f"{path.relative_to(ROOT)}: {named}")
        self.assertEqual(missing, [])

    def test_every_page_is_in_the_navigation(self):
        summary = (DOCS / "SUMMARY.md").read_text(encoding="utf-8")
        linked = set(re.findall(r"\]\(([^)]+\.md)\)", summary))
        pages = {path.relative_to(DOCS).as_posix() for path in DOCS.rglob("*.md") if path.name != "SUMMARY.md"}
        self.assertEqual(sorted(pages - linked), [])

    def test_readme_tables_are_current(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertEqual(_generator().readme_with_tables(text), text,
                         "run python scripts/gen_docs_reference.py --readme")

    def test_every_runtime_key_the_code_reads_is_described(self):
        from harness_policy.config.agent import RUNTIME_KEYS

        used = set()
        for root in ("src", "skills", "benches", "scripts"):
            for path in (ROOT / root).rglob("*.py"):
                used |= set(re.findall(r"runtime(?:\.get\(|\[)\"([a-z_]+)\"", path.read_text(encoding="utf-8")))
        self.assertEqual(sorted(used - set(RUNTIME_KEYS)), [])

    def test_every_model_key_the_client_reads_is_described(self):
        from harness_policy.runtime.chat import MODEL_KEYS

        source = (ROOT / "src" / "harness_policy" / "runtime" / "chat.py").read_text(encoding="utf-8")
        used = set(re.findall(r"config(?:\.get\(|\[)\"([a-z_]+)\"", source))
        self.assertEqual(sorted(used - set(MODEL_KEYS)), [])

    def test_reference_pages_generate(self):
        pages = _generator().pages()
        self.assertIn("reference/cli.md", pages)
        self.assertIn("## harness export", pages["reference/cli.md"])
        self.assertIn("`contract`", pages["reference/configuration.md"])
        self.assertIn("::: skills.motion.Motion.move_eef", pages["reference/skills/motion.md"])

    @unittest.skipUnless(shutil.which("mkdocs") or importlib.util.find_spec("mkdocs"), "MkDocs is not installed")
    def test_the_site_builds_strictly(self):
        with tempfile.TemporaryDirectory() as site:
            result = subprocess.run([sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", site],
                                    cwd=ROOT, capture_output=True, text=True, timeout=600,
                                    env={**os.environ, "DISABLE_MKDOCS_2_WARNING": "true"})
            self.assertEqual(result.returncode, 0, result.stderr[-3000:])


if __name__ == "__main__":
    unittest.main()
