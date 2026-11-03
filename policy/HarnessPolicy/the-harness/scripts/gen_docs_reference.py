#!/usr/bin/env python3
"""The documentation's reference pages, generated from the code at build time.

mkdocs-gen-files runs this file during `mkdocs build` / `mkdocs serve` (mkdocs.yml) and
the pages it writes exist only in the built site:

    reference/cli.md                 the CLI, from its argument parser
    reference/configuration.md       configuration keys, from the dictionaries the validators use
    reference/skills/<name>.md       each skill: manifest, functions (docstrings), what the model reads
    reference/services/<name>.md     each service: manifest, placement facts, weights
    reference/agent.md               the agent and each ablation resolved: skills, services, identity
    reference/configs.md             models, experiments and deploys
    reference/embodiments.md         the robots
    reference/api/<module>.md        the Python API, from docstrings

It also renders the tables README.md embeds between `<!-- generated: NAME -->` markers:

    python scripts/gen_docs_reference.py --readme     rewrite them
    python scripts/gen_docs_reference.py --list       print the pages it generates

Standard library and the harness's own (standard-library) modules only; nothing is
started and no model is called.
"""

import argparse
import os
from pathlib import Path
import re
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

TEST_DEPLOY = ROOT / "tests" / "fixtures" / "deploy-test.toml"
PACK = ROOT / "prompts" / "robodojo"


def _cell(text) -> str:
    """A table cell: pipes escaped, and <...> outside code spans shown, not taken as HTML."""
    parts = str(text).replace("\n", " ").split("`")
    parts = [part if index % 2 else part.replace("<", "&lt;").replace(">", "&gt;")
             for index, part in enumerate(parts)]
    return "`".join(parts).replace("|", "\\|")


def _table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _facts(rows) -> str:
    """(label, value) pairs as a list: **label**: value."""
    return "\n".join(f"- **{label}**: {_cell(value)}" for label, value in rows) + "\n"


def _keys(mapping) -> str:
    return _table(["Key", "Meaning"], [(f"`{key}`", value) for key, value in mapping.items()])


# -- CLI ------------------------------------------------------------------------------

def _subcommands(parser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            helps = {choice.dest: choice.help for choice in action._choices_actions}
            return [(name, sub, helps.get(name, "")) for name, sub in action.choices.items()]
    return []


def _arguments(parser):
    rows = []
    for action in parser._actions:
        if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)):
            continue
        name = ", ".join(f"`{flag}`" for flag in action.option_strings) or f"`{action.metavar or action.dest}`"
        default = action.default
        shown = "" if default in (None, False, [], argparse.SUPPRESS) or action.option_strings == [] else f"`{default}`"
        rows.append((name, shown, action.help or ""))
    return rows


def cli_page() -> str:
    os.environ["COLUMNS"] = "100"
    from harness_policy.cli import parser

    root = parser()
    out = ["# CLI\n", "`harness` (or `python scripts/harness` from a checkout). Every command prints JSON or a "
           "table; errors go to stderr as `{\"error\": ...}` with exit status 1.\n"]
    out.append(_table(["Command", "What it does"], [(f"[`{name}`](#harness-{name})", text)
                                                    for name, _, text in _subcommands(root)]))
    for name, sub, text in _subcommands(root):
        out += [f"\n## harness {name}\n", f"{text}\n" if text else "", "```text\n" + sub.format_usage() + "```\n"]
        rows = _arguments(sub)
        if rows:
            out.append(_table(["Argument", "Default", "Meaning"], rows))
        for nested, subsub, subtext in _subcommands(sub):
            out += [f"\n### harness {name} {nested}\n", f"{subtext}\n", "```text\n" + subsub.format_usage() + "```\n",
                    _table(["Argument", "Default", "Meaning"], _arguments(subsub))]
    return "\n".join(part for part in out if part)


# -- configuration keys ----------------------------------------------------------------

PACK_KEYS = {
    "sections": "the files of the pack in order: { file, skills = [...] (included when one is enabled), "
                "optional = true (may be missing) }",
    "undocumented": "exported names the pack does not describe (the documentation check skips them)",
}


def configuration_page() -> str:
    from harness_policy.config.agent import AGENT_KEYS, ROLE_DOCS, RUNTIME_KEYS
    from harness_policy.config.experiments import EXPERIMENT_KEYS
    from harness_policy.runtime.chat import MODEL_KEYS
    from harness_policy.runtime.prompts import FEATURES
    from harness_policy.runtime.skills import SKILL_KEYS
    from harness_policy.services.deploy import DEPLOY_KEYS, PLACEMENT_KEYS
    from harness_policy.services.registry import MANIFEST_KEYS

    return "\n".join([
        "# Configuration keys\n",
        "Generated from the dictionaries the validators use: a key not listed here is an error in its file "
        "(except under `[runtime]`, which is open). Paths are relative to the file that names them; files "
        "may `include` others ([Configure runs](../how-to/configure-runs.md)).\n",
        "## The agent\n", "`configs/default.toml`, and each ablation of `configs/ablations/` (which "
        "includes it)\n", _keys(AGENT_KEYS),
        "`[skills]`: `layers = { learned, reactive, handlers }` (defaults false, true, true), or "
        "`names = [...]`; `disable = [...]` removes skills. `[services]`: `segmenter = \"sam3\"`, "
        "`policies = [...]`, `world_models = [...]`; each a service name, or `{ service = \"...\", ... }` with "
        "client keys on top. `[budget]`: `max_model_calls`.\n",
        "## Models\n", "`configs/models/<name>.toml`: one table per role, plus `[runtime]` (settings the "
        "model needs, merged into the agent's) and `description`.\n",
        _table(["Role", "What for"], [(f"`[{role}]`", text) for role, text in ROLE_DOCS.items()]),
        "### Keys of a role\n", _keys(MODEL_KEYS),
        "## Runtime\n", "`[runtime]` of the agent (or of a model file, or an experiment's overrides). Some keys are "
        "written by the resolution itself.\n", _keys(dict(sorted(RUNTIME_KEYS.items()))),
        "## Experiments\n", "`configs/experiments/<name>.toml`\n", _keys(EXPERIMENT_KEYS),
        "## Deploys\n", "`configs/deploy/<machine>.toml`\n", _keys(DEPLOY_KEYS),
        "### Placement of a service\n", "`[services.<name>]` of a deploy.\n", _keys(PLACEMENT_KEYS),
        "## Service manifests\n", "`services/<name>.toml`\n", _keys(MANIFEST_KEYS),
        "## Skills\n", "`skills/<name>/skill.toml`\n", _keys(SKILL_KEYS),
        "## Prompt packs\n", "`prompts/<pack>/pack.toml`. Text inside a section is kept only when every skill or "
        "feature an `<!-- if: a, b -->` marker names is enabled (`<!-- else -->`, `<!-- end -->`; markers nest). "
        f"Features: {', '.join(f'`{name}`' for name in FEATURES)}.\n", _keys(PACK_KEYS),
    ])


# -- skills ------------------------------------------------------------------------------

def _full_prompt(skill=None):
    """The RoboDojo prompt with every layer on; with an optional `skill`, with it on in place
    of the skills it conflicts with."""
    from harness_policy.runtime.prompts import FEATURES, assemble_prompt
    from harness_policy.runtime.skills import skills_for_layers

    skills = skills_for_layers({"learned": True, "reactive": True, "handlers": True})
    if skill is not None and skill.optional:
        skills = [name for name in skills if name not in skill.conflicts] + [skill.name]
    return assemble_prompt(PACK, skills, features=FEATURES)


def _api_entry(prompt: str, name: str) -> str:
    """The lines of the prompt that document `name(...)`: its line and the deeper-indented
    lines after it."""
    lines = prompt.splitlines()
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(f"{name}(") or stripped.startswith(f"{name} "):
            indent = len(line) - len(stripped)
            block = [line]
            for follow in lines[index + 1:]:
                if follow.strip() and len(follow) - len(follow.lstrip()) > indent + 1:
                    block.append(follow)
                else:
                    break
            return "\n".join(block)
    return ""


def _methods(skill) -> set:
    import ast

    module, _, cls = skill.mixin.partition(":")
    source = ROOT / (module.replace(".", "/") + ".py")
    if not source.exists():
        source = ROOT / module.replace(".", "/") / "__init__.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return {item.name for item in node.body if isinstance(item, ast.FunctionDef)}
    return set()


def _needs(skill) -> str:
    parts = []
    if skill.needs:
        parts.append(", ".join(skill.needs))
    if skill.needs_any:
        parts.append("one of " + ", ".join(skill.needs_any))
    return "; ".join(parts) or "–"


def skills_table(link_prefix=None) -> str:
    from harness_policy.runtime.skills import load_skills

    rows = []
    for skill in load_skills().values():
        name = f"[{skill.name}]({link_prefix}{skill.name}.md)" if link_prefix is not None else skill.name
        rows.append((name, skill.layer, ", ".join(f"`{e}`" for e in skill.public), _needs(skill)))
    return _table(["Skill", "Layer", "Exports", "Needs"], rows)


def skill_pages() -> dict:
    from harness_policy.runtime.skills import load_skills

    prompt = _full_prompt()
    pages = {"reference/skills/index.md": "\n".join([
        "# Skills\n",
        "What a model's programs can call. The agent turns skills on by interface layer or by name; a skill "
        "whose services are missing drops out, and the prompt drops its text with it "
        "([Skills and prompt packs](../../concepts/skills.md)).\n",
        skills_table(link_prefix=""),
    ])}
    nav = []
    for skill in load_skills().values():
        methods = _methods(skill)
        module, _, cls = skill.mixin.partition(":")
        facts = [("Layer", skill.layer + (" (optional: only where a configuration enables it)" if skill.optional
                                            else "")), ("Needs", _needs(skill)),
                 ("Requires", ", ".join(skill.requires) or "–"),
                 ("Conflicts with", ", ".join(skill.conflicts) or "–"),
                 ("Never dropped", "yes" if skill.required else "no"), ("Mixin", f"`{skill.mixin}`"),
                 ("Manifest", f"`skills/{skill.name}/skill.toml`")]
        out = [f"# {skill.name}\n", f"{skill.summary}\n", _facts(facts)]
        out.append("## Functions\n")
        others = []
        for name in skill.public:
            # A skill whose names clash with another's prefixes its methods (exports(session) maps them).
            method = name if name in methods else f"{skill.name}_{name}" if f"{skill.name}_{name}" in methods else None
            if method is not None:
                out.append(f"::: {module}.{cls}.{method}\n    options:\n      show_root_full_path: false\n"
                           f"      heading_level: 3\n")
            else:
                others.append(name)
        if others:
            out.append("Also in the namespace: " + ", ".join(f"`{name}`" for name in others)
                       + " (constants and names the skill's `exports(session)` provides).\n")
        own = _full_prompt(skill) if skill.optional else prompt
        entries = [(name, _api_entry(own, name)) for name in skill.public]
        entries = [(name, text) for name, text in entries if text]
        if entries:
            out.append("## What the model reads\n")
            out.append("From the RoboDojo prompt pack (`prompts/robodojo/`), with every skill and feature on"
                       + (f" ({skill.name} in place of {', '.join(skill.conflicts)})" if skill.optional else "")
                       + ":\n")
            out.append("```text\n" + "\n".join(text for _, text in entries) + "\n```\n")
        pages[f"reference/skills/{skill.name}.md"] = "\n".join(out)
        nav.append((skill.name, f"reference/skills/{skill.name}.md"))
    NAV["reference/skills/"] = ("reference/skills/index.md", nav)
    return pages


# -- services ----------------------------------------------------------------------------

def _users():
    """Agent configurations that name each service, and skills that need each kind."""
    from harness_policy.config import load_layered
    from harness_policy.config.agent import load_model
    from harness_policy.runtime.skills import load_skills

    systems = {}
    for path in _agent_files():
        tree = load_layered(path)
        services = tree.get("services", {})
        names = []
        for key in ("segmenter",):
            if services.get(key):
                entry = services[key]
                names.append(entry if isinstance(entry, str) else entry["service"])
        for key in ("policies", "world_models"):
            names += [entry if isinstance(entry, str) else entry["service"] for entry in services.get(key, [])]
        roles = load_model(tree.get("model")) if tree.get("model") else {}
        names += [spec["service"] for spec in roles.values() if isinstance(spec, dict) and "service" in spec]
        for name in names:
            systems.setdefault(name, []).append(path.stem)
    kinds = {}
    for skill in load_skills().values():
        for kind in (*skill.needs, *skill.needs_any):
            kinds.setdefault(kind, []).append(skill.name)
    return systems, kinds


def services_table(link_prefix=None) -> str:
    from harness_policy.services.registry import load_services

    rows = []
    for spec in load_services().values():
        name = f"[{spec.name}]({link_prefix}{spec.name}.md)" if link_prefix is not None else spec.name
        rows.append((name, spec.kind, spec.summary, f"{spec.vram_gb:g} GB" if spec.vram_gb else "–"))
    return _table(["Service", "Kind", "What it is", "VRAM"], rows)


def service_pages() -> dict:
    from harness_policy.services.registry import load_services

    systems, kinds = _users()
    pages = {"reference/services/index.md": "\n".join([
        "# Services\n",
        "The model processes skills and model roles call; a deploy says where each runs "
        "([Services and deploys](../../concepts/services.md)). VRAM is measured on a 24 GB RTX 3090.\n",
        services_table(link_prefix=""),
    ])}
    nav = []
    for spec in load_services().values():
        facts = [("Kind", spec.kind), ("Protocol", spec.protocol), ("Default port", spec.port),
                 ("VRAM", f"{spec.vram_gb:g} GB"), ("Load time", f"about {spec.load_seconds:g} s"),
                 ("Skills that use a " + spec.kind, ", ".join(kinds.get(spec.kind, [])) or "–"),
                 ("Configurations that name it", ", ".join(systems.get(spec.name, [])) or "–")]
        out = [f"# {spec.name}\n", f"{spec.summary}\n", _facts(facts)]
        description = (spec.client or {}).get("description")
        if description:
            out += ["## What the model is told\n", f"> {description}\n"]
        if spec.weights:
            rows = []
            for entry in spec.weights:
                source = next(key for key in ("modelscope_model", "modelscope_dataset", "hf_model", "hf_dataset",
                                              "local") if key in entry)
                rows.append((f"`<weights>/{entry['dir']}`", source, entry[source] + (
                    f" (`{entry['root']}`)" if entry.get("root") else "")))
            out += ["## Weights\n", f"`harness fetch {spec.name} --deploy <machine>`\n",
                    _table(["Directory", "Source", "From"], rows)]
        if spec.notes:
            out += ["## Notes\n", f"{spec.notes}\n"]
        out += ["## Manifest\n", f"```toml\n--8<-- \"services/{spec.name}.toml\"\n```\n"]
        pages[f"reference/services/{spec.name}.md"] = "\n".join(out)
        nav.append((spec.name, f"reference/services/{spec.name}.md"))
    NAV["reference/services/"] = ("reference/services/index.md", nav)
    return pages


# -- the agent, models, experiments, deploys -------------------------------------------------

def _service_names(value) -> str:
    """A [services] entry by name: "sam3", {service = "sam3", ...} or a list of them."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("service", "")
    return ", ".join(_service_names(item) for item in value if item)


def _agent_files() -> list:
    """configs/default.toml, then the ablations."""
    return [ROOT / "configs" / "default.toml", *sorted((ROOT / "configs" / "ablations").glob("*.toml"))]


def _agent_label(path) -> str:
    return "default" if path.stem == "default" else f"ablations/{path.stem}"


def _resolve_all():
    from harness_policy.config.agent import resolve_agent

    return {path: resolve_agent(path, deploy=str(TEST_DEPLOY), name=path.stem) for path in _agent_files()}


def agent_table() -> str:
    from harness_policy.config import load_layered

    rows = [(f"`{_agent_label(path)}`", load_layered(path).get("description", "")) for path in _agent_files()]
    return _table(["Configuration", "What it is"], rows)


def agent_page() -> str:
    from harness_policy import AGENT_VERSION
    from harness_policy.config import load_layered
    from harness_policy.runtime.skills import load_skills

    every = list(load_skills())
    out = ["# The agent\n",
           f"`configs/default.toml` (agent version {AGENT_VERSION}) and each ablation of `configs/ablations/`, "
           "resolved against a deploy that runs every service (`tests/fixtures/deploy-test.toml`): what it turns "
           "on, and its identity there. The identity is the same on any machine that runs the services it "
           "names, wherever they run ([Configuration and identity](../concepts/configuration.md)).\n",
           agent_table()]
    for path, agent in _resolve_all().items():
        tree = load_layered(path)
        skills = agent.runtime.get("skills", [])
        dropped = agent.runtime.get("skills_dropped", {})
        services = {key: _service_names(value) for key, value in tree.get("services", {}).items() if value}
        relative = path.relative_to(ROOT).as_posix()
        facts = [("Mode", agent.lock["mode"]), ("Main model (its default model file)", f"`{agent.main.get('model')}`"),
                 ("Observation contract", agent.runtime.get("contract", "native")),
                 ("Skills on", ", ".join(skills)),
                 ("Skills off", ", ".join(f"{skill} ({dropped.get(skill, 'not in this configuration')})"
                                         for skill in every if skill not in skills) or "–"),
                 ("Services", ", ".join(f"{key}: {value}" for key, value in services.items()) or "–"),
                 ("Model calls per episode", agent.max_model_calls),
                 ("Identity", f"`{agent.identity}`")]
        out += [f"## {_agent_label(path)}\n", f"{tree.get('description', '')}\n", _facts(facts),
                f"??? note \"{relative}\"\n\n    ```toml\n    --8<-- \"{relative}\"\n    ```\n"]
    return "\n".join(out)


def _files(directory, title, intro) -> list:
    from harness_policy.config import read_toml

    out = [f"## {title}\n", intro + "\n"]
    rows = []
    for path in sorted((ROOT / "configs" / directory).glob("*.toml")):
        rows.append((f"`{path.stem}`", read_toml(path).get("description", _first_comment(path))))
    out.append(_table(["File", "What it is"], rows))
    for path in sorted((ROOT / "configs" / directory).glob("*.toml")):
        out.append(f"??? note \"configs/{directory}/{path.name}\"\n\n    ```toml\n    --8<-- "
                   f"\"configs/{directory}/{path.name}\"\n    ```\n")
    return out


def _first_comment(path) -> str:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        lines.append(line.lstrip("# ").strip())
    return " ".join(lines)


def configs_page() -> str:
    out = ["# Models, experiments and deploys\n", "The files under `configs/` besides the agent.\n"]
    out += _files("models", "Models", "Model roles: `configs/models/<name>.toml`; `--model NAME` picks one.")
    out += _files("experiments", "Experiments", "The agent with a model on a bench: `configs/experiments/<name>.toml`.")
    out += _files("deploy", "Deploys", "Where services run on one machine: `configs/deploy/<machine>.toml`.")
    out += _files("workers", "Worker options", "Options for `harness serve --options` (a worker started by hand).")
    return "\n".join(out)


# -- embodiments ---------------------------------------------------------------------------

def embodiments_page() -> str:
    from harness_policy.embodiments import EMBODIMENTS_ROOT, load_embodiment

    out = ["# Embodiments\n",
           "`embodiments/<name>/embodiment.toml`: what a policy may know about a robot without asking a "
           "simulator. Simulator workers report the same facts live; the XPolicyLab connector and workers "
           "under an observation contract take them from here.\n"]
    for path in sorted(EMBODIMENTS_ROOT.glob("*/embodiment.toml")):
        data = load_embodiment(path.parent.name)
        groups = ", ".join(f"{name} (joints {group['joints'][0]}-{group['joints'][-1]}, gripper {group['gripper']})"
                           for name, group in data["groups"].items())
        cameras = ", ".join(data["cameras"])
        facts = [("Robot", data.get("robot", "")), ("Control rate", f"{data['control_hz']:g} Hz"),
                 ("Joint groups", groups), ("Cameras", cameras),
                 ("Fingertip offset", f"{data.get('tip_offset_m', '–')} m along link6 +x"),
                 ("Empty closed gripper reads", data.get("gripper_empty_opening", "–")),
                 ("Table top", f"z = {data.get('table_z', '–')} m"),
                 ("Arms back to the start pose after finish", "yes" if data.get("return_to_start") else "no"),
                 ("Shield", ", ".join(f"{key} {value}" for key, value in data.get("shield", {}).items()))]
        out += [f"## {data['name']}\n", _facts(facts),
                f"??? note \"embodiments/{path.parent.name}/embodiment.toml\"\n\n    ```toml\n    --8<-- "
                f"\"embodiments/{path.parent.name}/embodiment.toml\"\n    ```\n"]
    return "\n".join(out)


# -- Python API ------------------------------------------------------------------------------

API = {
    "Configuration": ["harness_policy.config", "harness_policy.config.agent", "harness_policy.config.experiments"],
    "Episode runtime": ["harness_policy.runtime.driver", "harness_policy.runtime.kernel",
                        "harness_policy.runtime.session", "harness_policy.runtime.events",
                        "harness_policy.runtime.reactive", "harness_policy.runtime.shield",
                        "harness_policy.runtime.chat"],
    "Skills and prompts": ["harness_policy.runtime.skills", "harness_policy.runtime.prompts"],
    "Suites and reports": ["harness_policy.runtime.suite", "harness_policy.runtime.report"],
    "Services": ["harness_policy.services.registry", "harness_policy.services.deploy",
                 "harness_policy.services.supervisor", "harness_policy.services.broker",
                 "harness_policy.services.clients"],
    "Robots and workers": ["harness_policy.core.robot", "harness_policy.core.worker",
                           "harness_policy.core.contract", "harness_policy.embodiments"],
    "Evaluation by others": ["harness_policy.connectors.xpolicylab", "harness_policy.connectors.package",
                             "harness_policy.export.xpolicylab"],
}


def api_pages() -> dict:
    pages, nav = {}, []
    index = ["# Python API\n", "Generated from the docstrings. Most users need only the CLI and the "
             "configuration files; these are the modules to build on.\n"]
    for group, modules in API.items():
        slug = re.sub(r"[^a-z]+", "-", group.lower()).strip("-")
        body = [f"# {group}\n"]
        for module in modules:
            body.append(f"::: {module}\n    options:\n      heading_level: 2\n      members_order: source\n")
        pages[f"reference/api/{slug}.md"] = "\n".join(body)
        nav.append((group, f"reference/api/{slug}.md"))
        index.append(f"- [{group}]({slug}.md): " + ", ".join(f"`{module}`" for module in modules))
    pages["reference/api/index.md"] = "\n".join(index) + "\n"
    NAV["reference/api/"] = ("reference/api/index.md", nav)
    return pages


# -- all pages ---------------------------------------------------------------------------------

# Directory entries of docs/SUMMARY.md that generated pages fill in:
# directory -> (index page, [(title, page)]).
NAV = {}


def summary_with_generated(text: str) -> str:
    """docs/SUMMARY.md with each `- [Title](reference/<dir>/)` entry expanded into its
    index page and the pages generated in that directory (call pages() first)."""
    lines = []
    for line in text.splitlines():
        match = re.match(r"^(\s*)- \[(.+)\]\((reference/[a-z]+/)\)\s*$", line)
        if match and match.group(3) in NAV:
            indent, title, directory = match.groups()
            index, entries = NAV[directory]
            lines.append(f"{indent}- [{title}]({index})")
            lines += [f"{indent}    - [{name}]({page})" for name, page in entries]
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"

def pages() -> dict:
    out = {"reference/cli.md": cli_page(), "reference/configuration.md": configuration_page(),
           "reference/agent.md": agent_page(), "reference/configs.md": configs_page(),
           "reference/embodiments.md": embodiments_page()}
    out.update(skill_pages())
    out.update(service_pages())
    out.update(api_pages())
    return out


README_TABLES = {"skills": lambda: skills_table(), "services": lambda: services_table(),
                 "agent": lambda: agent_table()}


def readme_with_tables(text: str) -> str:
    """README.md with each `<!-- generated: NAME -->` ... `<!-- /generated: NAME -->` block
    rendered afresh."""
    for name, render in README_TABLES.items():
        pattern = re.compile(rf"(<!-- generated: {name} -->\n).*?(<!-- /generated: {name} -->)", re.S)
        text = pattern.sub(lambda match: match.group(1) + render() + match.group(2), text)
    return text


def _write_pages():
    import mkdocs_gen_files

    for path, content in pages().items():
        with mkdocs_gen_files.open(path, "w") as handle:
            handle.write(content)
        mkdocs_gen_files.set_edit_path(path, "../scripts/gen_docs_reference.py")
    summary = (ROOT / "docs" / "SUMMARY.md").read_text(encoding="utf-8")
    with mkdocs_gen_files.open("SUMMARY.md", "w") as handle:
        handle.write(summary_with_generated(summary))
    # The rewritten navigation is a new file: keep it out of the pages, as mkdocs.yml's
    # exclude_docs keeps the original (literate-nav still reads it).
    from mkdocs.structure.files import InclusionLevel

    mkdocs_gen_files.FilesEditor.current().files.get_file_from_path("SUMMARY.md").inclusion = InclusionLevel.EXCLUDED


if __name__ == "<run_path>":     # mkdocs-gen-files
    _write_pages()
elif __name__ == "__main__":
    command = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    command.add_argument("--readme", action="store_true", help="rewrite the generated tables in README.md")
    command.add_argument("--list", action="store_true", help="print the pages this generates")
    options = command.parse_args()
    if options.readme:
        readme = ROOT / "README.md"
        readme.write_text(readme_with_tables(readme.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    if options.list or not options.readme:
        print("\n".join(sorted(pages())))
