# Copyright © 2026 Martingale. All Rights Reserved.
"""Inference inspection, account selection and explicit live control."""
import argparse
import importlib
import json
import pkgutil
import os
from pathlib import Path
import tempfile
from . import catalog
from .config import InferenceConfig

def preflight(isolated=False):
    from .import_checks import check_internal_imports, check_external_imports
    internal_imports_checked=check_internal_imports()
    external_imports_checked=check_external_imports()
    loaded=[]
    for row in catalog.catalog():
        importlib.import_module('martingale.inference.'+row['module']);loaded.append(row['module'])
    from . import extensions
    extensions_loaded=[]
    for item in pkgutil.walk_packages(extensions.__path__, extensions.__name__+'.'):
        importlib.import_module(item.name);extensions_loaded.append(item.name)
    if isolated:
        from .isolation import CellWorker
        with tempfile.TemporaryDirectory(prefix='martingale-check-') as root:
            worker=CellWorker(root)
            try:
                code='import importlib\n'+ '\n'.join(f'importlib.import_module({m!r})' for m in loaded)+'\nprint("all library modules imported")'
                result=worker.execute(code,lambda *args: (_ for _ in ()).throw(AssertionError('No robot call allowed')))
                if 'exception' in result:raise RuntimeError(result['exception'])
                denied=worker.execute('import os\nprint(os.path.exists("/host-private-config"), os.path.exists("/host-private-data"))',lambda *args:None)
                if denied['stdout'].strip()!='False False':raise RuntimeError('Isolation root check failed')
            finally:worker.close()
    return {'modules_imported':len(loaded),'isolated_imports':isolated,'library_sha256':catalog.fingerprint(),
            'static_internal_imports_checked':internal_imports_checked,
            'static_external_imports_checked':external_imports_checked,
            'host_extension_modules_imported':len(extensions_loaded),
            'robot_validation':'not performed','model_calls':0}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('preflight');p.add_argument('--isolated',action='store_true')
    p=sub.add_parser('catalog')
    p=sub.add_parser('resume');p.add_argument('episode',type=Path)
    p=sub.add_parser('status');p.add_argument('episode',type=Path)
    from . import accounts
    accounts.add_parser(sub)
    args=parser.parse_args(argv)
    if args.command=='preflight':print(json.dumps(preflight(args.isolated),indent=2))
    elif args.command=='catalog':print(json.dumps(catalog.catalog(),indent=2))
    elif args.command=='accounts':accounts.run(args)
    elif args.command=='status':
        from .control import read_status
        print(json.dumps(read_status(args.episode),indent=2))
    else:
        from .control import request_resume
        print(json.dumps(request_resume(args.episode),indent=2))
if __name__=='__main__':main()
