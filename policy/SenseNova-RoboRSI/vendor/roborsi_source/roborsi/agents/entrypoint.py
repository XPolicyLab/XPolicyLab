# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a declared Agent source module from a checkout or an installed bundle."""

from __future__ import annotations

import argparse
import importlib
import sys
from importlib.machinery import PathFinder
from pathlib import Path

from roborsi.agents.config import AgentConfig


def main(argv: list[str] | None = None) -> int:
    """Load a trusted preset, then invoke main in one of its declared modules.

    The core wheel does not include robot extensions. Agent bundles keep their
    source under a private package directory; its authenticated source root
    must be added before Python can import a robot-specific CLI module.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-config", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    config = AgentConfig.load(args.agent_config)
    if not all(part.isidentifier() for part in args.module.split(".")):
        raise ValueError("Agent entry module must be a Python module name")
    candidate = config.source_root.joinpath(*args.module.split("."))
    declared = set(config.input_files)
    if not any(
        path.resolve() in declared
        for path in (candidate.with_suffix(".py"), candidate / "__init__.py")
    ):
        raise ValueError(
            "Agent entry module is not one of this preset's declared inputs"
        )
    # Inspect package origins without executing package initializers. A cached
    # package from another bundle must not redirect import to unrelated code.
    search = [str(config.source_root)]
    components = args.module.split(".")
    for index in range(1, len(components) + 1):
        name = ".".join(components[:index])
        cached = sys.modules.get(name)
        if cached is not None:
            origin = getattr(cached, "__file__", None)
            paths = getattr(cached, "__path__", ())
            if (origin is not None and Path(origin).resolve() not in declared) or any(
                not Path(path).resolve().is_relative_to(config.source_root)
                for path in paths
            ):
                raise ValueError(
                    "Agent entry package is already loaded from another source"
                )
        # Search a child directly in its verified parent directory. Using a
        # qualified name here would require importing a namespace parent first.
        specification = PathFinder.find_spec(components[index - 1], search)
        if specification is None:
            raise ValueError("Agent entry package is unavailable in declared source")
        if (
            specification.origin is not None
            and Path(specification.origin).resolve() not in declared
        ):
            raise ValueError("Agent entry package is not declared by this preset")
        search = list(specification.submodule_search_locations or ())
    sys.path.insert(0, str(config.source_root))
    module = importlib.import_module(args.module)
    location = getattr(module, "__file__", None)
    if location is None or Path(location).resolve() not in config.input_files:
        raise ValueError(
            "Agent entry module is not one of this preset's declared inputs"
        )
    entry = getattr(module, "main", None)
    if not callable(entry):
        raise ValueError("Agent entry module has no callable main")
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    return entry(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
