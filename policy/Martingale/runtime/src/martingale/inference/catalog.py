# Copyright © 2026 Martingale. All Rights Reserved.
import ast
import hashlib
from pathlib import Path
ROOT = Path(__file__).with_name('library')

def path(name):
    if not isinstance(name,str): raise TypeError('Expected a library-relative path')
    relative = name.removeprefix('library/')
    if not relative.endswith(('.py','.md','.json','.png','.jpg')):
        raise ValueError('Unsupported library file type')
    result = ROOT / relative
    if result.is_symlink() or ROOT.resolve() not in result.resolve().parents:
        raise ValueError('Only regular library files are readable')
    if not result.is_file(): raise FileNotFoundError(relative)
    return result

def read(name):
    p = path(name)
    if p.suffix in ('.png','.jpg'): return p.read_bytes()
    return p.read_text()

def catalog(capability=None):
    rows=[]
    if capability is not None and (not isinstance(capability,str) or not capability.isidentifier() or not (ROOT/capability).is_dir()):
        raise ValueError('Choose a catalog capability namespace')
    search=ROOT if capability is None else ROOT/capability
    for p in sorted(search.rglob('*.py')):
        if p.name=='__init__.py': continue
        nodes=ast.parse(p.read_text()).body
        functions={n.name:n for n in nodes if isinstance(n,ast.FunctionDef)}
        rows.append({'module':'library.'+'.'.join(p.relative_to(ROOT).with_suffix('').parts),
                     'file':p.relative_to(ROOT).as_posix(),
                     'classes':[{'name':n.name,'summary':(ast.get_docstring(n) or '').split('\n')[0]} for n in nodes if isinstance(n,ast.ClassDef) and not n.name.startswith('_')],
                     'functions':[{'name':n.name,'signature':ast.unparse(n.args),
                                   'summary':(ast.get_docstring(n) or 'Inspect implementation').split('\n')[0]}
                                  for n in functions.values() if not n.name.startswith('_')]})
    return rows

def doc(module, function):
    p=path(module)
    for n in reversed(ast.parse(p.read_text()).body):
        if isinstance(n,ast.FunctionDef) and n.name==function:
            return {'signature':n.name+'('+ast.unparse(n.args)+')','docstring':ast.get_docstring(n) or '',
                    'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    raise ValueError('Function not found')

def fingerprint():
    digest=hashlib.sha256()
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            if p.is_symlink(): raise ValueError('Library symlinks forbidden')
            digest.update(p.relative_to(ROOT).as_posix().encode()+b'\0'+p.read_bytes()+b'\0')
    return digest.hexdigest()
