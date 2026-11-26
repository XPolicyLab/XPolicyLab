"""A versioned library of what the harness has learned, frozen for evaluation.

Layout of a library directory:

    notes/*.md          high-level notes, appended to the system prompt
    memory/*.md         high-level experience entries, not in the prompt; programs
                        look them up with recall(query, k) (layer: notes)
    skills/*.py         low-level skills and handlers, executed in the kernel
                        namespace before the first program (layer: low)
    reactive/*.py       guard/controller helpers, executed only when the reactive
                        layer is enabled (layer: reactive)

Each entry is one file. The library version is the sha256 over its entries'
relative paths and bytes; every episode records it. Evaluation uses a single
version, and a library may be restricted to some layers (Tab. 8 ablation).
"""

import hashlib
from pathlib import Path


SECTIONS = {"notes": ".md", "memory": ".md", "skills": ".py", "reactive": ".py"}
# Library layers (Tab. 8 ablation) and the sections each one enables.
LAYER_OF = {"notes": "notes", "memory": "notes", "skills": "skills", "reactive": "reactive"}
LAYERS = ("notes", "skills", "reactive")


class Library:
    def __init__(self, directory, layers=("notes", "skills", "reactive")):
        self.directory = Path(directory).resolve()
        unknown = set(layers) - set(LAYERS)
        if unknown:
            raise ValueError(f"Unknown library layers: {sorted(unknown)}")
        self.layers = tuple(layers)
        if not self.directory.is_dir():
            raise ValueError(f"Library directory not found: {self.directory}")

    def entries(self, section):
        if LAYER_OF[section] not in self.layers:
            return []
        folder = self.directory / section
        return sorted(folder.glob(f"*{SECTIONS[section]}")) if folder.is_dir() else []

    def version(self) -> str:
        digest = hashlib.sha256()
        for section in SECTIONS:
            for path in self.entries(section):
                digest.update(str(path.relative_to(self.directory)).encode() + b"\0")
                digest.update(path.read_bytes() + b"\0")
        return digest.hexdigest()

    def manifest(self) -> dict:
        return {"directory": str(self.directory), "layers": list(self.layers), "version": self.version(),
                "entries": {section: [p.name for p in self.entries(section)] for section in SECTIONS}}

    def notes(self) -> str:
        texts = [p.read_text().strip() for p in self.entries("notes")]
        if not texts:
            return ""
        return "\nLessons from earlier episodes (validated on development seeds):\n" + \
            "\n".join(f"- {text}" for text in texts) + "\n"

    def memory_entries(self) -> list[dict]:
        """Memory entries for recall(): [{"name", "text"}]."""
        return [{"name": p.stem, "text": p.read_text().strip()} for p in self.entries("memory")]

    def kernel_code(self, reactive_enabled: bool) -> str:
        """Source run in the kernel before the first program; skills may use the API."""
        parts = []
        for section in ("skills", "reactive"):
            if section == "reactive" and not reactive_enabled:
                continue
            for path in self.entries(section):
                parts.append(f"# --- library {section}/{path.name}\n{path.read_text()}")
        return "\n\n".join(parts)

    def api_summary(self, reactive_enabled: bool) -> str:
        """First docstring line of each library function, for the prompt."""
        import ast

        lines = []
        for section in ("skills", "reactive"):
            if section == "reactive" and not reactive_enabled:
                continue
            for path in self.entries(section):
                tree = ast.parse(path.read_text())
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                        args = ", ".join(a.arg for a in node.args.args)
                        doc = (ast.get_docstring(node) or "").strip().splitlines()
                        lines.append(f"  {node.name}({args})" + (f"  {doc[0]}" if doc else ""))
        return ("\nLibrary skills (already defined in your kernel):\n" + "\n".join(lines) + "\n") if lines else ""
