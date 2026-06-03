# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build as build_vae
from .detr_vae import build_cnnmlp as build_cnnmlp
from .detr_vae import build_vae_head
from .droid_unet_diffusion import ConditionalUnet1D
def build_ACT_model(args):
    return build_vae(args)

def build_ACT_head(args):
    return build_vae_head(args)
def build_CNNMLP_model(args):
    return build_cnnmlp(args)
