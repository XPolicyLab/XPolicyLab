# Skills

What a model's programs can call: one directory per skill, a `skill.toml` manifest and an
`__init__.py` with a mixin class. A system turns skills on by interface layer or by name;
a skill whose services are missing drops out, and the prompt drops its text with it.

- How it works: [docs/concepts/skills.md](../docs/concepts/skills.md)
- Write one: [docs/tutorials/write-a-skill.md](../docs/tutorials/write-a-skill.md)
- Every skill with its functions: the documentation site's *Reference → Skills*
  (generated from these directories by `scripts/gen_docs_reference.py`)
