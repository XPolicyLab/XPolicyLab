# Copyright © 2026 Martingale. All Rights Reserved.
"""RLM-style root context, external memory retrieval and REPL result storage.

Python variables remain in the isolated worker; this interface stores its cell
receipts and the root prompt fragments. It does not make model calls.
"""
from pathlib import Path

from . import catalog


class RLMContext:
    def __init__(self):
        self.root_prompt_parts = ()
        self.repl_results = {}

    def read_prompt_source(self, name, *, library=False):
        """Read a host-selected prompt document with its original text semantics."""
        root = catalog.ROOT if library else Path(__file__).parent
        return (root / name).read_text()

    def build_root_prompt(self, parts):
        """Retain the ordered fragments and join without transforming their text."""
        self.root_prompt_parts = tuple(parts)
        return '\n\n'.join(self.root_prompt_parts)

    def read_memory(self, *args, **kwargs):
        return catalog.read(*args, **kwargs)

    def inspect_memory(self, *args, **kwargs):
        return catalog.doc(*args, **kwargs)

    def list_memory(self, *args, **kwargs):
        return catalog.catalog(*args, **kwargs)

    def memory_namespaces(self):
        return sorted(p.name for p in catalog.ROOT.iterdir()
                      if p.is_dir() and p.name != '__pycache__')

    def memory_fingerprint(self):
        return catalog.fingerprint()
