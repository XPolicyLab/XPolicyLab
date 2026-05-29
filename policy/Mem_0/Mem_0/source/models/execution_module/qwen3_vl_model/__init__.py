# Copyright 2025 MemoryMatters Team. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Yuran Wang / Peking University] in [2025].

"""
Qwen3-VL Model Module

This module provides a lightweight encapsulation of the Qwen3-VL vision-language model.
It handles image and text inputs, and provides hidden states for downstream tasks.
"""

from .qwen3_vl_encapsulation import Qwen3VL_Encapsulation

__all__ = ["Qwen3VL_Encapsulation"]

