import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CONFIG_MAPPING

from dexbotic.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from dexbotic.model.dexbotic_arch import (
    CausalLMOutputDexbotic,
    DexboticConfig,
    DexboticForCausalLM,
    DexboticVLMModel,
)
from dexbotic.model.uninavid.constants import NAVIGATION_IDENTIFIER
from dexbotic.model.uninavid.eva_vit import EVAVisionTowerLavis


class DexboticUniNaVidConfig(DexboticConfig):
    model_type = "dexbotic_uninavid"
    chat_template: str | None = "vicuna"
    tie_word_embeddings: bool = False
    image_processor: str | None = None
    mm_vision_select_layer: int = -1
    mm_vision_select_feature: str = "patch"
    compress_type: str | None = "grid:2"
    run_type: str = "train"
    tokenizer_padding_side: str = "right"
    tokenizer_model_max_length: int | None = None


class DexboticUniNaVidModel(DexboticVLMModel):
    def __init__(self, config: DexboticUniNaVidConfig):
        if isinstance(config.llm_config, dict):
            config.llm_config = CONFIG_MAPPING[config.llm_config["model_type"]](
                **config.llm_config
            )
        super().__init__(config)
        self.initialize_online_inference_nav_feat_cache()

    def initialize_online_inference_nav_feat_cache(self):
        self.feat_cache = None
        self.long_feat_cache = None
        self.weight = 1
        self.new_frames = 0

    def _build_mm_vision_module(self, _config) -> nn.Module:
        if getattr(self, "mm_vision_tower", None) is not None:
            return self.mm_vision_tower
        vision_tower = EVAVisionTowerLavis(
            None,
            getattr(self.config, "image_processor", None),
            self.config,
            use_checkpoint=False,
            drop_path_rate=0.0,
            dtype=torch.float32,
        )
        self.mm_vision_tower = vision_tower
        self.config.mm_hidden_size = vision_tower.hidden_size
        return vision_tower

    @property
    def embed_tokens(self):
        return self.backbone.embed_tokens

    def get_vision_tower(self):
        return self.mm_vision_tower

    @property
    def mm_vision_prefix(self) -> str:
        return "mm_vision_tower"


class DexboticUniNaVidForCausalLM(DexboticForCausalLM):
    config_class = DexboticUniNaVidConfig
    _tied_weights_keys = []

    def _real_init(self, config: DexboticUniNaVidConfig):
        self.model = DexboticUniNaVidModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        
    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def online_process_tensor(
        self, nav_size, length_threshold=64, similarity_threshold=0.985):
        # Keep compressed long-term memory plus recent high-resolution frames.
        k, m, c = self.get_model().feat_cache.shape

        assert m % nav_size == 0, f"m ({m}) must be divisible by nav_size ({nav_size})"
        result_list = []

        if k <= length_threshold:
            result_list = [nav_size] * k
            return self.get_model().feat_cache.reshape(-1, c), result_list

        cos = torch.nn.CosineSimilarity(dim=0)

        for frame_index in range(self.get_model().new_frames - 1, -1, -1):

            if k - length_threshold - frame_index - 1 < 0:
                continue

            oldest_short_mem_token = (
                self.get_model()
                .feat_cache[k - length_threshold - frame_index - 1, :, :]
                .mean(dim=0)
            )
            long_term_cache = self.get_model().long_feat_cache

            if long_term_cache is not None:
                assert (
                    long_term_cache[-1, :].shape == oldest_short_mem_token.shape
                ), f"Shape mismatch: long_term_cache[-1, :] {long_term_cache[-1, :].shape} vs oldest_short_mem_token {oldest_short_mem_token.shape}"

                similarity = cos(long_term_cache[-1, :], oldest_short_mem_token)

                if similarity > similarity_threshold:
                    new_mean = (
                        long_term_cache[-1, :] * self.get_model().weight
                        + oldest_short_mem_token
                    ) / (self.get_model().weight + 1)
                    self.get_model().weight = self.get_model().weight + 1
                    long_term_cache[-1] = new_mean
                    self.get_model().long_feat_cache = long_term_cache
                else:
                    self.get_model().long_feat_cache = torch.cat(
                        [long_term_cache, oldest_short_mem_token[None, :]], dim=0
                    )
                    self.get_model().weight = 1
            else:
                self.get_model().long_feat_cache = oldest_short_mem_token[None, :]

        result_list = [1] * self.get_model().long_feat_cache.shape[0] + [
            nav_size
        ] * length_threshold

        result_tensor = torch.cat(
            [
                self.get_model().long_feat_cache,
                self.get_model().feat_cache[k - length_threshold :].reshape(-1, c),
            ],
            dim=0,
        )

        assert result_tensor.shape[0] == sum(
            result_list
        ), f"The sum of the list does not match the tensor dimension {result_tensor.shape[0]}, {sum(result_list)}"

        return result_tensor, result_list

    def process_tensor(
        self, tensor, nav_size, length_threshold=64, similarity_threshold=0.985):
        """Compress older frames while keeping recent frames at full resolution."""
        n, m, t = tensor.shape

        if m % nav_size != 0:
            raise ValueError("m must be divisible by nav_size")

        k = m // nav_size

        if k <= length_threshold:
            result_list = [nav_size] * k
            return tensor, result_list

        elif k == length_threshold + 1:
            split_tensors = tensor.view(n, k, nav_size, t)
            means = split_tensors[:, : k - length_threshold, :, :].mean(dim=2)
            remaining_tensors = split_tensors[:, k - length_threshold :, :, :].reshape(
                n, -1, t
            )
            result_tensor = torch.cat([means, remaining_tensors], dim=1)
            result_list = [1] + [nav_size] * length_threshold
            return result_tensor, result_list

        split_tensors = tensor.view(n, k, nav_size, t)

        means_tensor = split_tensors[:, : k - length_threshold, :, :].mean(dim=2)
        cos = torch.nn.CosineSimilarity(dim=2)

        means = [means_tensor[:, 0:1, :]]
        weights = [1]

        for i in range(1, k - length_threshold):
            last_mean = means[-1]
            current_tensor = means_tensor[:, i : i + 1, :]

            similarity = cos(last_mean, current_tensor).mean(dim=0)

            if similarity > similarity_threshold:
                new_weight = weights[-1] + 1
                new_mean = (last_mean * weights[-1] + current_tensor) / new_weight
                means[-1] = new_mean
                weights[-1] = new_weight
            else:
                means.append(current_tensor)
                weights.append(1)

        means = torch.cat(means, dim=1)

        remaining_tensors = split_tensors[:, k - length_threshold :, :, :].reshape(
            n, -1, t
        )

        result_tensor = torch.cat([means, remaining_tensors], dim=1)
        result_list = [1] * means.shape[1] + [nav_size] * length_threshold

        assert result_tensor.shape[1] == sum(
            result_list
        ), "The sum of the list does not match the tensor dimension"

        return result_tensor, result_list

    def encode_images(self, images, prompts=None, image_counts=None, long_video=False):
        """Encode images and project them into language-model tokens."""
        if long_video:
            # Reuse precomputed features for long-video inference.
            image_features = images
        else:
            image_features = self.get_model().get_vision_tower()(images)

        image_features, video_or_not, nav_or_not, final_token_length_lst = (
            self.vlm_attention(
                image_features,
                prompts=prompts,
                image_counts=image_counts,
                long_video=long_video,
            )
        )
        return image_features, video_or_not, nav_or_not, final_token_length_lst

    def vlm_attention(
        self, image_features, prompts=None, image_counts=None, long_video=False):
        """Build per-sample visual token layouts for images, videos, and navigation."""
        compress_type = self.config.compress_type
        compress_grid_sizes = {"grid:2": 4, "grid:4": 16, "mean": 1}

        nav_size = compress_grid_sizes.get(compress_type)
        if nav_size is None:
            raise ValueError(f"Unsupported compress type: {compress_type}")

        if image_counts is None:
            assert len(image_features) == len(
                prompts
            ), f"Size mismatch! image_features: {len(image_features)}, prompts: {len(prompts)}"
        else:
            assert len(prompts) == len(
                image_counts
            ), f"Size mismatch! prompts: {len(prompts)}, image_counts: {len(image_counts)}"

        img_feat_lst = []
        video_or_not = []
        nav_or_not = []
        final_token_length_lst = []
        total_count = 0

        for _idx, prompt in enumerate(prompts):
            assert isinstance(
                prompt, list
            ), f"Prompt should be a list, but got {type(prompt)}"

            if image_counts is None:
                img_feat_prompt = image_features[_idx, None]
            else:
                img_feat_prompt = image_features[
                    total_count : total_count + image_counts[_idx]
                ]
                total_count += image_counts[_idx]

            is_navigation = NAVIGATION_IDENTIFIER in prompt[0]
            if is_navigation:
                if image_counts is None or image_counts[_idx] < 1 or len(prompt) != 1:
                    raise ValueError("[Navigation] wrong")

            if (
                self.config.mm_vision_select_feature == "patch"
                and img_feat_prompt.shape[1] % 2 == 1
            ):
                img_feat_prompt = img_feat_prompt[:, 1:]

            final_token, final_token_nav = self.token_generation(
                img_feat_prompt,
                image_counts=None if image_counts is None else image_counts[_idx],
                navigation=is_navigation,
            )

            if is_navigation and final_token_nav is None:
                raise ValueError("[Navigation] wrong")

            final_token = (
                final_token[None].expand(len(prompt), -1, -1, -1).flatten(1, 2)
            )
            if image_counts is not None:
                if is_navigation:
                    final_token_nav = (
                        final_token_nav[None]
                        .expand(len(prompt), -1, -1, -1)
                        .flatten(1, 2)
                    )
                    assert (
                        final_token_nav.shape[0] == 1
                        and final_token_nav.shape[1] == 64
                        and final_token.shape[0] == 1
                    )

                    if self.config.run_type == "eval":
                        final_token, lengths_list = self.online_process_tensor(nav_size)
                        final_token = final_token.unsqueeze(0)
                    else:
                        final_token, lengths_list = self.process_tensor(
                            final_token, nav_size
                        )

                    video_or_not.append(True)
                    final_token_length_lst.append(lengths_list)
                    nav_or_not.append(final_token_nav)

                elif not is_navigation and image_counts[_idx] > 1:
                    final_token, lengths_list = self.process_tensor(
                        final_token, nav_size
                    )
                    video_or_not.append(True)
                    final_token_length_lst.append(lengths_list)
                    nav_or_not.append(None)

                elif not is_navigation and image_counts[_idx] == 1:
                    video_or_not.append(False)
                    final_token_length_lst.append(None)
                    nav_or_not.append(None)

                else:
                    raise ValueError("unexpected case")

            else:
                assert final_token.shape[1] == 64
                video_or_not.append(False)
                nav_or_not.append(None)
                final_token_length_lst.append(None)

            img_feat_lst.append(final_token)

        return img_feat_lst, video_or_not, nav_or_not, final_token_length_lst

    def token_generation(self, vis_embed, image_counts=None, navigation=False):
        """Pool vision features, project them, and update eval caches when needed."""

        def process_grid(vis_embed, grid_size):
            cur_shape = int(vis_embed.shape[1] ** 0.5)
            assert (
                grid_size > 1
            ), f"Grid size should be larger than 1, but got {grid_size}"
            vis_embed = vis_embed.reshape(vis_embed.shape[0], cur_shape, cur_shape, -1)

            grid_stride = cur_shape // grid_size
            vis_embed = F.avg_pool2d(
                vis_embed.permute(0, 3, 1, 2),
                padding=0,
                kernel_size=grid_stride,
                stride=grid_stride,
            )
            return vis_embed.permute(0, 2, 3, 1).flatten(1, 2)

        grid_size = int(self.config.compress_type.split("grid:")[-1])

        if image_counts is None or (image_counts == 1 and not navigation):
            vis_embed = process_grid(vis_embed, 8)
        elif navigation:

            vis_embed_nav = vis_embed[-1:]
            vis_embed_nav = process_grid(vis_embed_nav, 8)
            vis_embed = process_grid(vis_embed, grid_size)

        else:
            vis_embed = process_grid(vis_embed, grid_size)

        vis_embed = self.get_model().mm_projector(vis_embed)

        if self.config.run_type == "eval":
            temp_embed = self.get_model().feat_cache
            vis_embed = (
                torch.cat([temp_embed, vis_embed], dim=0)
                if temp_embed is not None
                else vis_embed
            )
            self.get_model().feat_cache = vis_embed

        vis_embed_nav = (
            self.get_model().mm_projector(vis_embed_nav) if navigation else None
        )

        return vis_embed, vis_embed_nav

    def update_prompt(self, prompts=None):
        """Store prompts for the next multimodal generation call."""
        self.prompts = prompts

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, prompts=None):
        """Replace image placeholders with visual embeddings and align masks and labels."""
        def validate_compress_type():
            compress_type = getattr(self.config, "compress_type", None)
            if compress_type is None:
                raise ValueError(
                    "config.compress_type is None; set model_args.compress_type (e.g. 'grid:2')."
                )
            if "grid" in compress_type:
                grid_size = int(compress_type.split("grid:")[-1])
                if grid_size not in {2, 4}:
                    raise ValueError
                return
            if "mean" in compress_type:
                return
            raise ValueError

        def get_sample_image_features(sample_image_idx, token_idx):
            if isinstance(image_features, list):
                return image_features[sample_image_idx][token_idx]
            return image_features[sample_image_idx]

        def build_video_segments(cur_input_ids, image_token_start, cur_image_features, token_lengths):
            separator_token = self.get_model().embed_tokens(
                cur_input_ids[image_token_start - 1, None]
            )
            assert token_lengths is not None

            segments = []
            feature_offset = 0
            for segment_idx, segment_length in enumerate(token_lengths):
                segments.append(
                    cur_image_features[feature_offset : feature_offset + segment_length]
                )
                if segment_idx == len(token_lengths) - 1:
                    break
                segments.append(separator_token)
                feature_offset += segment_length
            return segments

        def build_visual_segments(cur_input_ids, image_token_start, sample_image_idx, token_idx):
            cur_image_features = get_sample_image_features(sample_image_idx, token_idx)
            cur_nav_features = nav_or_not[sample_image_idx]
            is_navigation = cur_nav_features is not None
            is_video = video_or_not[sample_image_idx]
            token_lengths = final_token_length_lst[sample_image_idx]

            segments = [
                self.get_model().embed_tokens(cur_input_ids[:image_token_start])
            ]
            ignore_lengths = [cur_image_features.shape[0]]
            consumed_input_tokens = 1

            if not is_video:
                segments.append(cur_image_features)
                assert cur_image_features.shape[0] == 64
                return segments, ignore_lengths, consumed_input_tokens

            segments.extend(
                build_video_segments(
                    cur_input_ids,
                    image_token_start,
                    cur_image_features,
                    token_lengths,
                )
            )
            ignore_lengths.append(len(token_lengths) - 1)

            if is_navigation:
                assert token_idx == 0
                nav_features = cur_nav_features[token_idx]
                assert nav_features.shape[0] == 64
                segments.append(
                    self.get_model().embed_tokens(
                        cur_input_ids[image_token_start + 1 : image_token_start + 3]
                    )
                )
                segments.append(nav_features)
                ignore_lengths.append(nav_features.shape[0] + 2)
                consumed_input_tokens = 3

            return segments, ignore_lengths, consumed_input_tokens

        def build_ignore_labels(length):
            return torch.full(
                (length,),
                IGNORE_INDEX,
                device=labels.device,
                dtype=labels.dtype,
            )

        def append_visual_labels(cur_new_labels, cur_labels, image_token_start, ignore_lengths):
            cur_new_labels.append(cur_labels[:image_token_start])
            for ignore_length in ignore_lengths:
                if ignore_length > 0:
                    cur_new_labels.append(build_ignore_labels(ignore_length))

        validate_compress_type()
        if prompts is None and hasattr(self, "prompts"):
            prompts = self.prompts

        vision_tower = self.get_vision_tower()
        # Skip image fusion for text-only inputs and single-token decode steps.
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels

        long_video = False

        if isinstance(images, list) or images.ndim == 5:
            # Keep prebatched long-video inputs unchanged.
            if not long_video:
                images = [
                    image if len(image.shape) == 4 else image.unsqueeze(0)
                    for image in images
                ]
            image_counts = [image.shape[0] for image in images]
            concat_images = torch.cat(images, dim=0)
            image_features, video_or_not, nav_or_not, final_token_length_lst = (
                self.encode_images(
                    concat_images, prompts, image_counts, long_video=long_video
                )
            )
        else:
            image_features, video_or_not, nav_or_not, final_token_length_lst = (
                self.encode_images(images, prompts, long_video=long_video)
            )

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        # Expand per-sample multimodal layouts and keep labels aligned with inserted tokens.
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                if isinstance(image_features, list):
                    cur_image_features = image_features[cur_image_idx][0]
                else:
                    cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(
                    cur_input_ids[:half_len]
                )
                cur_input_embeds_2 = self.get_model().embed_tokens(
                    cur_input_ids[half_len:]
                )
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2],
                    dim=0,
                )
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            if not long_video:
                token_idx = 0
                while image_token_indices.numel() > 0:
                    image_token_start = image_token_indices[0]
                    (
                        visual_segments,
                        ignore_lengths,
                        consumed_input_tokens,
                    ) = build_visual_segments(
                        cur_input_ids, image_token_start, cur_image_idx, token_idx
                    )
                    cur_new_input_embeds.extend(visual_segments)

                    if labels is not None:
                        append_visual_labels(
                            cur_new_labels,
                            cur_labels,
                            image_token_start,
                            ignore_lengths,
                        )
                        cur_labels = cur_labels[image_token_start + consumed_input_tokens :]

                    cur_input_ids = cur_input_ids[image_token_start + consumed_input_tokens :]
                    image_token_indices = torch.where(
                        cur_input_ids == IMAGE_TOKEN_INDEX
                    )[0]
                    token_idx += 1

                # Advance to the next visual sample after finishing this sequence.
                cur_image_idx += 1
                if cur_input_ids.numel() > 0:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids)
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels)
                cur_new_input_embeds = [
                    x.to(device=self.device) for x in cur_new_input_embeds
                ]
                cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
                new_input_embeds.append(cur_new_input_embeds)
                if labels is not None:
                    cur_new_labels = torch.cat(cur_new_labels, dim=0)
                    assert cur_new_input_embeds.shape[0] == cur_new_labels.shape[0]
                    new_labels.append(cur_new_labels)
            else:
                cur_new_input_embeds = torch.Tensor(
                    len(cur_input_ids), self.config.hidden_size
                ).to(dtype=self.dtype, device=self.device)
                text_token_indices = torch.where(cur_input_ids != IMAGE_TOKEN_INDEX)[0]
                if (
                    not self.training
                    and self.get_model().embed_tokens.weight.device
                    != cur_input_ids.device
                ):
                    model_device = self.get_model().embed_tokens.weight.device
                    data_device = cur_input_ids.device
                    cur_input_ids_text = cur_input_ids[text_token_indices].to(
                        device=model_device
                    )
                    cur_new_input_embeds[text_token_indices] = (
                        self.get_model()
                        .embed_tokens(cur_input_ids_text)
                        .to(device=data_device)
                    )
                else:
                    cur_new_input_embeds[text_token_indices] = (
                        self.get_model().embed_tokens(cur_input_ids[text_token_indices])
                    )
                cur_image_features = image_features[cur_image_idx]
                cur_new_input_embeds[image_token_indices] = cur_image_features
                new_input_embeds.append(cur_new_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_label.dtype,
                                device=cur_new_label.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            # This path is only used with right-padding tokenizers.
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                    attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat(
                        (
                            new_attn_mask_pad_left,
                            cur_attention_mask,
                            new_attn_mask_pad_right,
                        ),
                        dim=0,
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            # This path is only used with right-padding tokenizers.
            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (
                        attention_mask.shape[0],
                        new_input_embeds.shape[1] - input_ids.shape[1],
                    ),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask), dim=1
                )
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        images: torch.FloatTensor | None = None,
        prompts: list[str] | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        actions: torch.LongTensor | None = None,
        states: torch.LongTensor | None = None,
    ) -> CausalLMOutputDexbotic:
        del actions, states, position_ids, cache_position
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if prompts is not None:
            self.update_prompt(prompts)

        if not self.training and images:
            if images[0].device != self.device:
                images[0] = images[0].to(device=self.device)
            if input_ids is not None and input_ids.device != self.device:
                input_ids = input_ids.to(device=self.device)

        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            prompts=prompts,
        )

        # Must request hidden states: lm_head needs the last layer; default config has
        # output_hidden_states=False so hidden_states would be None (see DexboticForCausalLM).
        outputs = self.model.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        hidden_states = outputs.hidden_states[-1]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config.vocab_size)

        return CausalLMOutputDexbotic(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
