# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

from awomo.awomo.training.config.loader import ConfigNode, load_yaml_config, translate_value


def parse_cli_argv(default_cfg_path: str | None = None, argv: list[str] | None = None) -> tuple[str, list[str]]:
    if argv is None:
        argv = sys.argv[1:]
    overrides = []
    i = 0
    cfg_path = default_cfg_path
    while i < len(argv):
        tok = argv[i]

        if tok in ("--config", "-c"):
            if i + 1 >= len(argv):
                raise ValueError("Expected a path after --config")
            cfg_path = argv[i + 1]
            i += 2
            continue

        if tok.startswith("--"):
            key = tok.lstrip("-")
            if "=" in key:
                overrides.append(f"{key}")
                i += 1
            else:
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    val = argv[i + 1]
                    overrides.append(f"{key}={val}")
                    i += 2
                else:
                    overrides.append(f"{key}=True")
                    i += 1
            continue

        i += 1

    if not cfg_path:
        raise ValueError("You must specify --config <path>")
    return cfg_path, overrides


def parse_args_and_load_config(default_cfg_path: str | None = None, argv: list[str] | None = None) -> ConfigNode:
    cfg_path, overrides = parse_cli_argv(default_cfg_path, argv=argv)
    print(f"cfg-path: {cfg_path}")
    cfg = load_yaml_config(cfg_path)

    for kv in overrides:
        dotted, val_str = kv.split("=", 1)
        cfg.set_by_dotted(dotted, translate_value(val_str))

    return cfg
