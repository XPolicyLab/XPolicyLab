"""Extract observation-only helpers into a new, reviewable asset revision.

This is a source transformation, not a qualification or runtime sandbox. It
removes simulator readers and their static dependents while preserving pure
geometry helpers. Review the report and validate each affected skill before
selecting the exported revision.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil

PRIVILEGED_ATTRIBUTES = frozenset({
    'scene_manager', 'layout_manager', 'reward_manager', 'get_label_pose',
    'get_instance_pose', 'get_instance_metadata', 'get_scene_object',
    'get_functional_points', 'env_seeds', 'end_flag', 'take_action_cnt',
})


def reads_privileged(node):
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr in PRIVILEGED_ATTRIBUTES:
            return True
        if (isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                and child.func.id == 'getattr' and len(child.args) > 1
                and isinstance(child.args[1], ast.Constant)
                and child.args[1].value in PRIVILEGED_ATTRIBUTES):
            return True
    return False


def normalize_owned_runtime(source, module_name):
    """Distinguish the RGB charger's own action counter from evaluator state.

    The adapter constructs this namespace from RobotFrames and a new defaultdict;
    the RGB backend receives that namespace, never the simulator environment.
    Keep the whitelist narrow so evaluator counters elsewhere remain forbidden.
    """
    if module_name == 'rdj_rgb_adapters.plug_in_charger':
        tree = ast.parse(source)
        owners = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == 'SimpleNamespace'
                  and any(item.arg == 'take_action_cnt' for item in node.keywords)]
        if len(owners) != 1:
            raise ValueError('RGB charger counter ownership requires review')
        counter = next(item.value for item in owners[0].keywords if item.arg == 'take_action_cnt')
        if ast.unparse(counter) != 'defaultdict(int)':
            raise ValueError('RGB charger counter must be locally initialized')
    elif module_name == 'rdj_master.evaluation.charger_server_backend':
        readers = [node for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.Attribute) and node.attr == 'take_action_cnt']
        if any(not isinstance(node.value, ast.Name) or node.value.id != 'runtime' for node in readers):
            raise ValueError('RGB charger backend counter ownership requires review')
    else:
        return source
    # Both audited modules communicate through the adapter-owned namespace.
    import io
    import tokenize
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return tokenize.untokenize(token._replace(string='action_counts')
                              if token.type == tokenize.NAME and token.string == 'take_action_cnt'
                              else token for token in tokens)


def extract(source, *, forbidden_imports=None, module_name="", is_package=False):
    """Remove top-level definitions that read or depend on simulator-only state."""
    tree = ast.parse(source)
    forbidden_imports = forbidden_imports or {}
    removed, blocked = set(), set()
    names = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names[node] = {node.name}
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            names[node] = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        elif isinstance(node, ast.ImportFrom):
            names[node] = {alias.asname or alias.name for alias in node.names}
        else:
            names[node] = set()
        if reads_privileged(node):
            removed.add(node)
            blocked.update(names[node])
        if isinstance(node, ast.ImportFrom):
            module = node.module or ''
            if node.level:
                parts = module_name.split('.') if is_package else module_name.split('.')[:-1]
                if node.level > 1:
                    parts = parts[:-(node.level - 1)]
                module = '.'.join([*parts, *([module] if module else [])])
            unavailable = forbidden_imports.get(module, set())
            if unavailable and any(alias.name == '*' for alias in node.names):
                raise ValueError('Cannot audit a wildcard import from a simulator-reader module')
            blocked.update(alias.asname or alias.name for alias in node.names if alias.name in unavailable)
    while True:
        previous = len(removed)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                continue
            uses = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            if uses & blocked:
                removed.add(node)
                blocked.update(names[node])
        if len(removed) == previous:
            break
    edits, report = [], []
    for node in tree.body:
        if node in removed:
            start = min([node.lineno, *(d.lineno for d in getattr(node, 'decorator_list', []))])
            edits.append((start, node.end_lineno, ''))
            report.append(dict(line=start, end_line=node.end_lineno, symbols=sorted(names[node]),
                               reason='simulator_state_or_static_dependency'))
        elif isinstance(node, ast.ImportFrom):
            kept = [alias for alias in node.names if (alias.asname or alias.name) not in blocked]
            if len(kept) != len(node.names):
                replacement = ''
                if kept:
                    replacement = ast.unparse(ast.ImportFrom(module=node.module, names=kept, level=node.level)) + '\n'
                edits.append((node.lineno, node.end_lineno, replacement))
    lines = source.splitlines(keepends=True)
    for start, end, replacement in sorted(edits, reverse=True):
        lines[start-1:end] = [replacement]
    result = ''.join(lines)
    compile(result, '<observation-only-export>', 'exec')
    return result, blocked, report


def _write(path, data):
    temporary = path.with_name(path.name + '.reviewing')
    temporary.write_bytes(data)
    shutil.copymode(path, temporary)
    temporary.replace(path)


def export(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError('Use a new export directory outside the source bundle')
    # Reflinks are not assumed. Hard links retain large immutable assets; every
    # changed text file is replaced atomically, never edited through a link.
    shutil.copytree(source, destination, copy_function=os.link,
                    ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache', '.git'), symlinks=True)
    return rewrite_sources(source, destination)


def rewrite_sources(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    candidates = {}
    for original in (source / 'implementations').rglob('*.py'):
        path = destination / original.relative_to(source)
        relative = path.relative_to(destination / 'implementations')
        # Include bundled legacy evaluation entrypoints, even when the public
        # adapter does not invoke them. Keep provider inference trees untouched;
        # this one reviewed provider module mixes inference helpers with an old
        # evaluator callback, so extract only its evaluator-dependent definitions.
        provider_evaluator = (relative.parts[0] == 'pi05_sparse_mem'
                              and relative.parts[2:] == (
                                  'pi05_sparse_mem', 'src', 'openpi', 'integrations',
                                  'rmbench_sparse_memory.py'))
        if not provider_evaluator and relative.parts[0] not in {'rdj_master', 'align_blocks_rgb_straightedge', 'rdj_pour',
                                     'rdj_v1_classify_objects_by_language', 'rdj_plug_policy',
                                     'rdj_rgb_adapters', 'deposit_policy', 'arrange_rgb',
                                     'pour_by_language'}:
            continue
        content = original.read_text()
        module_parts = list(relative.parts[2:])
        module_parts[-1] = Path(module_parts[-1]).stem
        if module_parts[-1] == '__init__': module_parts.pop()
        module = '.'.join(module_parts)
        candidates[path] = dict(source=normalize_owned_runtime(content, module),
                                parent=content, module=module, is_package=path.name=='__init__.py')
    forbidden = {}
    while True:
        before = sum(map(len, forbidden.values()))
        for entry in candidates.values():
            _, blocked, _ = extract(entry['source'], forbidden_imports=forbidden, module_name=entry['module'], is_package=entry['is_package'])
            forbidden.setdefault(entry['module'], set()).update(blocked)
        if sum(map(len, forbidden.values())) == before:
            break
    changes = []
    for path, entry in candidates.items():
        content, _, removals = extract(entry['source'], forbidden_imports=forbidden, module_name=entry['module'], is_package=entry['is_package'])
        if content != entry['parent']:
            _write(path, content.encode())
            changes.append(dict(path=str(path.relative_to(destination)),
                                parent_sha256=hashlib.sha256(entry['parent'].encode()).hexdigest(),
                                sha256=hashlib.sha256(content.encode()).hexdigest(), removals=removals))
    report = dict(schema='physicalrsi.asset-source-cleanup/v1', changes=changes,
                  scope='Static extraction only. No policy promotion or task qualification.',
                  parent_programs_sha256=hashlib.sha256((source/'programs.json').read_bytes()).hexdigest(),
                  qualified=False)
    (destination/'source-cleanup-review.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args=parser.parse_args()
    report=export(args.source,args.destination)
    print(json.dumps({'changed_files':len(report['changes']), 'qualified':False}))
