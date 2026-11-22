"""
Local WanPrompter implementation - removed diffsynth dependency
"""

import logging
import torch
import ftfy
import html
import string
import regex as re
from transformers import AutoTokenizer


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def canonicalize(text, keep_punctuation_exact_string=None):
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans("", "", string.punctuation))
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class HuggingfaceTokenizer:

    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        assert clean in (None, "whitespace", "lower", "canonicalize")
        self.name = name
        self.seq_len = seq_len
        self.clean = clean

        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence, **kwargs):
        return_mask = kwargs.pop("return_mask", False)

        # arguments
        _kwargs = {"return_tensors": "pt"}
        if self.seq_len is not None:
            _kwargs.update(
                {
                    "padding": "max_length",
                    "truncation": True,
                    "max_length": self.seq_len,
                }
            )
        _kwargs.update(**kwargs)

        # tokenization
        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [self._clean(u) for u in sequence]
        ids = self.tokenizer(sequence, **_kwargs)

        # output
        if return_mask:
            return ids.input_ids, ids.attention_mask
        else:
            return ids.input_ids

    def _clean(self, text):
        if self.clean == "whitespace":
            text = whitespace_clean(basic_clean(text))
        elif self.clean == "lower":
            text = whitespace_clean(basic_clean(text)).lower()
        elif self.clean == "canonicalize":
            text = canonicalize(basic_clean(text))
        return text


class BasePrompter:
    def __init__(self):
        self.refiners = []
        self.extenders = []

    def load_prompt_refiners(self, model_manager, refiner_classes=[]):
        for refiner_class in refiner_classes:
            refiner = refiner_class.from_model_manager(model_manager)
            self.refiners.append(refiner)

    def load_prompt_extenders(self, model_manager, extender_classes=[]):
        for extender_class in extender_classes:
            extender = extender_class.from_model_manager(model_manager)
            self.extenders.append(extender)

    @torch.no_grad()
    def process_prompt(self, prompt, positive=True):
        if isinstance(prompt, list):
            prompt = [
                self.process_prompt(prompt_, positive=positive) for prompt_ in prompt
            ]
        else:
            for refiner in self.refiners:
                prompt = refiner(prompt, positive=positive)
        return prompt

    @torch.no_grad()
    def extend_prompt(self, prompt: str, positive=True):
        extended_prompt = dict(prompt=prompt)
        for extender in self.extenders:
            extended_prompt = extender(extended_prompt)
        return extended_prompt


class WanPrompter(BasePrompter):

    def __init__(
        self,
        tokenizer_path=None,
        text_len=512,
        use_qwen25=False,
        qwen25_path=None,
        use_qwen35=False,
        qwen35_path=None,
        qwen35_system_prompt=None,
        qwen_proj_out_dim=4096,
        qwen25_cascade_t5=False,
        qwen35_cascade_t5=False,
        skip_text_encoder: bool = False,
    ):
        super().__init__()
        if use_qwen25 and use_qwen35:
            raise ValueError("use_qwen25 and use_qwen35 cannot both be True")
        self.text_len = text_len
        self.text_encoder = None
        self.t5_text_encoder = None
        self.use_qwen25 = use_qwen25
        self.qwen25_path = qwen25_path
        self.use_qwen35 = use_qwen35
        self.qwen35_path = qwen35_path
        self.qwen35_system_prompt = qwen35_system_prompt
        self.qwen_proj_out_dim = qwen_proj_out_dim
        self.qwen25_cascade_t5 = qwen25_cascade_t5
        self.qwen35_cascade_t5 = qwen35_cascade_t5
        self.skip_text_encoder = skip_text_encoder
        self.tokenizer = None
        self.text_feature_cache = {}
        self.cache_online_encoded_features = False

        if not self.skip_text_encoder:
            if (
                self.use_qwen25
                and self.qwen25_path is not None
                and not self.qwen25_cascade_t5
            ):
                self.fetch_tokenizer(self.qwen25_path)
            elif (
                self.use_qwen35
                and self.qwen35_path is not None
                and not self.qwen35_cascade_t5
            ):
                self.fetch_tokenizer(self.qwen35_path)
            else:
                self.fetch_tokenizer(tokenizer_path)

    def fetch_tokenizer(self, tokenizer_path=None):
        if tokenizer_path is not None:
            if (
                self.use_qwen25
                and self.qwen25_path is not None
                and tokenizer_path != self.qwen25_path
                and not self.qwen25_cascade_t5
            ):
                return
            if (
                self.use_qwen35
                and self.qwen35_path is not None
                and tokenizer_path != self.qwen35_path
                and not self.qwen35_cascade_t5
            ):
                return
            self.tokenizer = HuggingfaceTokenizer(
                name=tokenizer_path, seq_len=self.text_len, clean="whitespace"
            )

    def fetch_models(self, text_encoder=None):
        if self.use_qwen25:
            if self.qwen25_path is None:
                raise ValueError("use_qwen25=True but qwen25_path is not provided")
            # Import locally to avoid dependency when not using qwen25
            try:
                from diffsynth.models.qwen25_text_encoder import (
                    Qwen25TextEncoderAdapter,
                )

                self.text_encoder = Qwen25TextEncoderAdapter(
                    pretrained_path=self.qwen25_path,
                    proj_out_dim=self.qwen_proj_out_dim,
                    torch_dtype=torch.bfloat16,
                    device="cpu",
                )
                if self.qwen25_cascade_t5:
                    logging.info(
                        "[WanPrompter] Using Qwen2.5 -> T5 Cascade mode. Qwen2.5 path: %s",
                        self.qwen25_path,
                    )
                else:
                    logging.info(
                        "[WanPrompter] Using Qwen2.5 Text Encoder. Path: %s",
                        self.qwen25_path,
                    )
            except ImportError:
                raise ImportError(
                    "Qwen25 requires diffsynth.models.qwen25_text_encoder"
                )
            if self.qwen25_cascade_t5:
                self.t5_text_encoder = text_encoder
        elif self.use_qwen35:
            if self.qwen35_path is None:
                raise ValueError("use_qwen35=True but qwen35_path is not provided")
            # Import locally to avoid dependency when not using qwen35
            from wall_x.model.vga.modeling.qwen35_text_encoder import (
                Qwen35TextEncoderAdapter,
            )

            self.text_encoder = Qwen35TextEncoderAdapter(
                pretrained_path=self.qwen35_path,
                proj_out_dim=self.qwen_proj_out_dim,
                torch_dtype=torch.bfloat16,
                device="cpu",
                system_prompt=self.qwen35_system_prompt,
            )
            if self.qwen35_cascade_t5:
                logging.info(
                    "[WanPrompter] Using Qwen3.5 -> T5 Cascade mode. Qwen3.5 path: %s",
                    self.qwen35_path,
                )
                self.t5_text_encoder = text_encoder
            else:
                logging.info(
                    "[WanPrompter] Using Qwen3.5 Text Encoder. Path: %s",
                    self.qwen35_path,
                )
        else:
            self.text_encoder = text_encoder
            logging.info("[WanPrompter] Using T5 Text Encoder.")

    def load_text_feature_cache(self, cache_path):
        """Load pre-computed text feature cache from a .pt file.

        Args:
            cache_path: Path to the cache file (dict mapping prompt_str -> [1, seq, dim] tensor).
        """
        import os

        if not os.path.exists(cache_path):
            logging.warning("[WanPrompter] Cache file not found: %s", cache_path)
            return
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if isinstance(cache, dict):
            self.text_feature_cache = cache
            logging.info(
                "[WanPrompter] Loaded %d cached text features from %s",
                len(cache),
                cache_path,
            )
        else:
            logging.warning("[WanPrompter] Invalid cache format in %s", cache_path)

    def _lookup_text_feature_cache(self, prompt, device):
        """Try to look up all prompts in the cache.

        Returns stacked tensor on success, None on any cache miss.
        """
        if not self.text_feature_cache:
            return None

        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        cached_embs = []
        for p in prompts:
            if p in self.text_feature_cache:
                cached_embs.append(self.text_feature_cache[p])
            else:
                return None

        return torch.cat(cached_embs, dim=0).to(device)

    def has_cached_prompt(self, prompt, positive=True):
        """Return whether every processed prompt already has an embedding."""

        processed = self.process_prompt(prompt, positive=positive)
        prompts = [processed] if isinstance(processed, str) else list(processed)
        return bool(self.text_feature_cache) and all(
            value in self.text_feature_cache for value in prompts
        )

    def _remember_online_encoding(self, prompt, prompt_emb):
        """Memoize one online encoding without changing the returned tensor."""

        if not self.cache_online_encoded_features:
            return prompt_emb
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if prompt_emb.ndim != 3 or prompt_emb.shape[0] != len(prompts):
            raise RuntimeError(
                "online text feature cache expects [B,L,C] with one row per "
                f"prompt, got prompts={len(prompts)} shape={tuple(prompt_emb.shape)}"
            )
        inserted = []
        for index, value in enumerate(prompts):
            if value in self.text_feature_cache:
                continue
            embedding = prompt_emb[index : index + 1].detach().to(
                device="cpu", dtype=torch.bfloat16
            ).contiguous()
            if not torch.isfinite(embedding.float()).all():
                raise RuntimeError(
                    f"online text encoder returned non-finite features for {value!r}"
                )
            self.text_feature_cache[value] = embedding
            inserted.append(value)
        if inserted:
            logging.info(
                "[WanPrompter] Memoized %d online text feature(s); cache_size=%d "
                "examples=%r",
                len(inserted),
                len(self.text_feature_cache),
                inserted[:3],
            )
        return prompt_emb

    def encode_prompt(
        self,
        prompt,
        positive=True,
        device="cuda",
        first_frame_image=None,
        print_qwen_output=False,
    ):
        prompt = self.process_prompt(prompt, positive=positive)
        cached = self._lookup_text_feature_cache(prompt, device)
        if cached is not None:
            return cached
        if self.skip_text_encoder:
            prompts_list = [prompt] if isinstance(prompt, str) else list(prompt)
            missing = [p for p in prompts_list if p not in self.text_feature_cache]
            logging.error(
                "encode_prompt: text feature cache miss while skip_text_encoder=True. "
                "positive=%s missing=%d/%d examples=%r cache_size=%d",
                positive,
                len(missing),
                len(prompts_list),
                missing[:5],
                len(self.text_feature_cache),
            )
            raise RuntimeError(
                f"encode_prompt: cache miss; first missing prompt (repr): {missing[0]!r}"
            )

        if (self.use_qwen25 and self.qwen25_cascade_t5) or (
            self.use_qwen35 and self.qwen35_cascade_t5
        ):
            # Cascade mode: Qwen generates text -> T5 encodes
            generated_text = self.text_encoder.generate_text(
                prompt, image=first_frame_image
            )
            if print_qwen_output:
                logging.info("[Qwen->T5 Cascade] Generated text: %s", generated_text)
            generated_text = generated_text.split("\nassistant\n", 1)[-1].strip()

            combined_text = f"{prompt.split('请你')[0]}在画面保持物理规则合理的情况下，{generated_text}"

            ids, mask = self.tokenizer(
                combined_text, return_mask=True, add_special_tokens=True
            )
            ids = ids.to(device)
            mask = mask.to(device)
            seq_lens = mask.gt(0).sum(dim=1).long()
            prompt_emb = self.t5_text_encoder(ids, mask)
            for i, v in enumerate(seq_lens):
                prompt_emb[i, v:] = 0
            return self._remember_online_encoding(prompt, prompt_emb)
        elif self.use_qwen25 or self.use_qwen35:
            # Qwen path: pass first frame image + instruction, output embedding directly
            prompt_emb = self.text_encoder.encode(
                prompt, image=first_frame_image, print_generated=print_qwen_output
            )
            return self._remember_online_encoding(prompt, prompt_emb)
        else:
            # T5 path
            ids, mask = self.tokenizer(
                prompt, return_mask=True, add_special_tokens=True
            )
            ids = ids.to(device)
            mask = mask.to(device)
            seq_lens = mask.gt(0).sum(dim=1).long()
            prompt_emb = self.text_encoder(ids, mask)
            for i, v in enumerate(seq_lens):
                prompt_emb[i, v:] = 0
            return self._remember_online_encoding(prompt, prompt_emb)
