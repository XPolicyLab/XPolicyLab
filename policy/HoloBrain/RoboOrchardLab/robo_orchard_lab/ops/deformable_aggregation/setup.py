# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
)


def make_cuda_ext(
    name,
    module,
    sources,
    sources_cuda: list | None = None,
    extra_args: list | None = None,
    extra_include_path: list | None = None,
):
    if sources_cuda is None:
        sources_cuda = []
    if extra_args is None:
        extra_args = []
    if extra_include_path is None:
        extra_include_path = []
    define_macros = []
    extra_compile_args = {"cxx": [] + extra_args}

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros += [("WITH_CUDA", None)]
        extension = CUDAExtension
        extra_compile_args["nvcc"] = extra_args + [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        sources += sources_cuda
    else:
        print("Compiling {} without CUDA".format(name))
        extension = CppExtension

    return extension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if __name__ == "__main__":
    setup(
        name="deformable_aggregation_op",
        ext_modules=[
            make_cuda_ext(
                "deformable_aggregation_with_depth_ext",
                module=".",
                sources=[
                    "src/deformable_aggregation_with_depth.cpp",
                    "src/deformable_aggregation_with_depth_cuda.cu",
                ],
            ),
            make_cuda_ext(
                "deformable_aggregation_ext",
                module=".",
                sources=[
                    "src/deformable_aggregation.cpp",
                    "src/deformable_aggregation_cuda.cu",
                ],
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
    )
