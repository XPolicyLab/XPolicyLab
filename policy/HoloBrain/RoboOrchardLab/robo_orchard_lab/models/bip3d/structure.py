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

import torch
from torch import nn

from robo_orchard_lab.models.mixin import (
    ClassType_co,
    ModelMixin,
    TorchModuleCfg,
    TorchModuleCfgType_co,
)
from robo_orchard_lab.utils.build import (
    DelayInitDictType,
    build,
)

__all__ = ["BIP3D", "BIP3DConfig"]


class BIP3D(ModelMixin):
    cfg: "BIP3DConfig"  # for type hint

    def __init__(self, cfg: "BIP3DConfig"):
        super().__init__(cfg)
        self.backbone = build(self.cfg.backbone)
        self.decoder = build(self.cfg.decoder)
        self.neck = build(self.cfg.neck)
        self.text_encoder = build(self.cfg.text_encoder)
        self.feature_enhancer = build(self.cfg.feature_enhancer)
        self.spatial_enhancer = build(self.cfg.spatial_enhancer)
        self.data_preprocessor = build(self.cfg.data_preprocessor)
        self.backbone_3d = build(self.cfg.backbone_3d)
        self.neck_3d = build(self.cfg.neck_3d)
        self.input_2d = self.cfg.input_2d
        self.input_3d = self.cfg.input_3d
        self.embed_dims = self.cfg.embed_dims
        self.pre_spatial_enhancer = self.cfg.pre_spatial_enhancer

        if self.cfg.text_encoder is not None:
            self.text_feat_map = nn.Linear(
                self.text_encoder.language_backbone.body.language_dim,  # type: ignore
                self.embed_dims,
                bias=True,
            )

    def save_model(
        self, directory: str, model_prefix: str = "model", **kwargs
    ):
        super().save_model(directory, model_prefix, **kwargs)
        for module in [self.text_encoder, self.decoder]:
            if hasattr(module, "save_metadata") and callable(
                module.save_metadata
            ):
                module.save_metadata(directory, model_prefix)

    def extract_feat(self, inputs):
        imgs = inputs.get(self.input_2d)
        if imgs.dim() == 5:
            bs, num_cams = imgs.shape[:2]
            imgs = imgs.flatten(end_dim=1)
        else:
            bs = imgs.shape[0]
            num_cams = 1

        feature_maps = self.backbone(imgs)
        if self.neck is not None:
            feature_maps = self.neck(feature_maps)
        feature_maps = [x.unflatten(0, (bs, num_cams)) for x in feature_maps]

        input_3d = inputs.get(self.input_3d)
        if self.backbone_3d is not None and input_3d is not None:
            if "depth" in self.input_3d and input_3d.dim() == 5:
                assert input_3d.shape[1] == num_cams
                input_3d = input_3d.flatten(end_dim=1)
            feature_3d = self.backbone_3d(input_3d)
            if self.neck_3d is not None:
                feature_3d = self.neck_3d(feature_3d)
            feature_3d = [x.unflatten(0, (bs, num_cams)) for x in feature_3d]
        else:
            feature_3d = None
        return feature_maps, feature_3d

    def extract_text_feature(self, inputs):
        if self.text_encoder is not None:
            text_dict = self.text_encoder(inputs["text"])
            text_dict["embedded"] = self.text_feat_map(text_dict["embedded"])
        else:
            text_dict = None
        return text_dict

    def forward(self, inputs):
        if self.data_preprocessor is not None:
            device = next(self.parameters()).device
            inputs = self.data_preprocessor(inputs, device)
        if self.training:
            return self.loss(inputs)
        else:
            return self.predict(inputs)

    def loss(self, inputs):
        assert self.training, "Requried training mode"
        model_outs, text_dict, loss_depth = self._forward(inputs)  # type: ignore
        loss = self.decoder.loss(model_outs, inputs, text_dict=text_dict)
        if loss_depth is not None:
            loss["loss_depth"] = loss_depth
        return loss

    @torch.no_grad()
    def predict(self, inputs):
        assert not self.training, "Required evaluation mode"
        model_outs, text_dict = self._forward(inputs)  # type: ignore
        results = self.decoder.post_process(
            model_outs, inputs, text_dict=text_dict
        )
        return results

    def _forward(self, inputs):
        feature_maps, feature_3d = self.extract_feat(inputs)
        text_dict = self.extract_text_feature(inputs)
        if self.spatial_enhancer is not None and self.pre_spatial_enhancer:
            feature_maps, depth_prob, loss_depth = self.spatial_enhancer(
                feature_maps=feature_maps,
                feature_3d=feature_3d,
                text_dict=text_dict,
                inputs=inputs,
            )
        else:
            depth_prob = loss_depth = None
        if self.feature_enhancer is not None:
            feature_maps, text_feature = self.feature_enhancer(
                feature_maps=feature_maps,
                feature_3d=feature_3d,
                text_dict=text_dict,
                inputs=inputs,
            )
            text_dict["embedded"] = text_feature  # type: ignore
        if self.spatial_enhancer is not None and not self.pre_spatial_enhancer:
            feature_maps, depth_prob, loss_depth = self.spatial_enhancer(
                feature_maps=feature_maps,
                feature_3d=feature_3d,
                text_dict=text_dict,
                inputs=inputs,
            )
        else:
            depth_prob = loss_depth = None
        model_outs = self.decoder(
            feature_maps=feature_maps,
            feature_3d=feature_3d,
            text_dict=text_dict,
            inputs=inputs,
            depth_prob=depth_prob,
        )
        if self.training:
            return model_outs, text_dict, loss_depth
        return model_outs, text_dict


MODULE_TPYE = TorchModuleCfgType_co | DelayInitDictType  # noqa: E501


class BIP3DConfig(TorchModuleCfg[BIP3D]):
    class_type: ClassType_co[BIP3D] = BIP3D
    backbone: MODULE_TPYE  # type: ignore
    decoder: MODULE_TPYE  # type: ignore
    neck: MODULE_TPYE | None = None
    text_encoder: MODULE_TPYE | None = None
    feature_enhancer: MODULE_TPYE | None = None
    spatial_enhancer: MODULE_TPYE | None = None
    data_preprocessor: MODULE_TPYE | None = None
    backbone_3d: MODULE_TPYE | None = None
    neck_3d: MODULE_TPYE | None = None
    input_2d: str = "imgs"
    input_3d: str = "depths"
    embed_dims: int = 256
    pre_spatial_enhancer: bool = False
