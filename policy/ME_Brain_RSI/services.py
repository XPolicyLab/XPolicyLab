"""Start an auxiliary perception service through the task module loader."""
import argparse
import json
import os
import sys
from .core.modules import TaskModules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--module', required=True)
    args, remaining = parser.parse_known_args()
    values = dict(equation_assets_root=os.environ['ME_BRAIN_EQUATION_ASSETS'])
    for path in json.loads(os.environ.get('ME_BRAIN_EQUATION_DEPENDENCIES', '[]')):
        sys.path.insert(0, path.format(**values))
    module = TaskModules(args.task).import_module(args.module)
    sys.argv = [args.module, *remaining]
    return module.main()


if __name__ == '__main__':
    raise SystemExit(main())
