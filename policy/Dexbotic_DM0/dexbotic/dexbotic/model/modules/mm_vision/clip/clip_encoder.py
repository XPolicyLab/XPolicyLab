import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self._meta_initialized = False

        self.vision_tower_name = vision_tower
        self.select_layer = -2

        if not delay_load and not torch.empty(0).is_meta:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
            if torch.empty(0).is_meta:
                # Under meta-device context (transformers>=5.0 from_pretrained),
                # from_pretrained is not allowed. Create module structure from
                # config only; actual weights are filled by the outer state_dict load.
                self.vision_tower = CLIPVisionModel(self.cfg_only)
                self._meta_initialized = True

    def load_model(self):
        if self.is_loaded:
            return
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name)

        if getattr(self, "_meta_initialized", False):
            # Weights were already loaded via outer from_pretrained state_dict;
            # only finalize non-parameter state.
            self.vision_tower.requires_grad_(False)
        else:
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
            self.vision_tower.requires_grad_(False)

        if hasattr(self, "cfg_only"):
            del self.cfg_only
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]

        image_features = image_features[:, 1:]

        return image_features

    def forward(self, images):
        if isinstance(images, list):
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(
                        device=self.device,
                        dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(
                    device=self.device,
                    dtype=self.dtype),
                output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
