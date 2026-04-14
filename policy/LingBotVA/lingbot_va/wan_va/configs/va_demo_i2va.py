# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_demo_cfg import va_demo_cfg

va_demo_i2va_cfg = EasyDict(__name__='Config: VA demo i2va')
va_demo_i2va_cfg.update(va_demo_cfg)

va_demo_i2va_cfg.input_img_path = 'example/demo'
va_demo_i2va_cfg.num_chunks_to_infer = 10
va_demo_i2va_cfg.prompt = 'Pick the green cube and place it inside the blue box'
va_demo_i2va_cfg.infer_mode = 'i2va'