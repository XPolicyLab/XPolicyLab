# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import os

_WORK_DIR: str | None = None
_DEFAULT_WORK_DIR = "./runs/"


def register_work_dir(path: str | os.PathLike | None) -> None:
    global _WORK_DIR
    _WORK_DIR = str(path) if path is not None else None
    os.makedirs(path, exist_ok=True)


def get_work_dir() -> str | None:
    return _WORK_DIR if _WORK_DIR is not None else _DEFAULT_WORK_DIR
