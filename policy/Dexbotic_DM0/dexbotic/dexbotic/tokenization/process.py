import copy
import random
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import transformers

from dexbotic.constants import (
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from dexbotic.model.uninavid.constants import (
    IMAGE_SEPARATOR,
    IMAGE_END_TOKEN,
    IMAGE_START_TOKEN,
    NAVIGATION_IDENTIFIER,
    NAVIGATION_SPECIAL_TOKEN,
    VIDEO_END_SPECIAL_TOKEN,
    VIDEO_START_SPECIAL_TOKEN,
)
from dexbotic.data.dataset.tokenization import Tokenization
from dexbotic.tokenization import conversation as conversation_lib
from dexbotic.tokenization.tokenization import tokenizer_image_token
from dexbotic.tokenization import tokenization as tokenization_lib


def _process(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    chat_template: str = "dexbotic",
):
    if chat_template in ["dexbotic", "step"]:
        return tokenization_lib.tokenize_dexbotic(
            sources=sources,
            tokenizer=tokenizer,
            has_image=has_image,
            chat_template=chat_template,
        )
    else:
        raise ValueError(f"Unsupported chat template: {chat_template}")


def llava_multi_image_map_fn(conversations, mode="dexbotic"):
    messages = conversations

    for msg in messages:
        if DEFAULT_IMAGE_TOKEN in msg["value"]:
            # move the image token to the beginning of the sentence
            msg["value"] = msg["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            if mode == "step":
                msg["value"] = msg["value"] + f"<im_start>{DEFAULT_IMAGE_TOKEN}<im_end>"
            else:
                msg["value"] = DEFAULT_IMAGE_TOKEN + "\n" + msg["value"]
            msg["value"] = msg["value"].strip()

    return conversations


def process_data_item(
    conversations: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    chat_template: str,
    has_image: bool,
) -> Dict:
    conversations = llava_multi_image_map_fn(conversations, mode=chat_template)
    text_dict = _process(
        sources=[conversations],
        tokenizer=tokenizer,
        has_image=has_image,
        chat_template=chat_template,
    )
    data_dict = dict(input_ids=text_dict["input_ids"][0], labels=text_dict["labels"][0])
    return data_dict


def _tokenizer_encode(
    tokenizer: transformers.PreTrainedTokenizer,
    text: str,
    add_bos: bool = False,
    add_eos: bool = False,
) -> list[int]:
    if hasattr(tokenizer, "sp_model"):
        return tokenizer.sp_model.encode(text, add_bos=add_bos, add_eos=add_eos)

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if add_bos and tokenizer.bos_token_id is not None:
        token_ids = [tokenizer.bos_token_id] + token_ids
    if add_eos and tokenizer.eos_token_id is not None:
        token_ids = token_ids + [tokenizer.eos_token_id]
    return token_ids


class LLMTokenization(Tokenization):
    def __init__(self, tokenizer, data_args):
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __call__(self, conversations: List[Dict], has_image: bool) -> Dict:
        data_dict = process_data_item(
            conversations=conversations,
            tokenizer=self.tokenizer,
            chat_template=self.data_args.chat_template,
            has_image=has_image,
        )
        return data_dict


class NaVILATokenization(Tokenization):
    """
    Tokenization class for NaVILA dataset.

    Directly processes conversation format without using chat templates.
    Preserves all <image> tokens in their original positions.
    """

    def __init__(self, tokenizer, data_args):
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __call__(self, conversations: List[Dict], has_image: bool) -> Dict:
        from dexbotic.constants import IGNORE_INDEX
        from dexbotic.tokenization.tokenization import tokenizer_image_token

        human_msg = conversations[0]["value"]
        gpt_msg = conversations[1]["value"] if len(conversations) > 1 else ""

        # Tokenize full prompt: human_msg + gpt_msg + "\n"
        prompt = human_msg + gpt_msg + "\n"
        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
        # Ensure 1D tensor [seq_len] for DataCollator (it will add batch dimension)
        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze()

        # Create labels and mask human question part
        labels = input_ids.clone()
        human_len = len(tokenizer_image_token(human_msg, self.tokenizer))
        labels[:human_len] = IGNORE_INDEX

        # Mask padding tokens
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if pad_token_id is not None:
            labels[input_ids == pad_token_id] = IGNORE_INDEX

        return {"input_ids": input_ids, "labels": labels}


class UniNaVidTokenization(Tokenization):
    """Imgsp supervision batching with video special tokens; used by ``DexUniNaVidDataset``."""

    def __init__(self, tokenizer, data_args):
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __call__(self, conversations: List[Dict], has_image: bool) -> Dict:
        sources = [copy.deepcopy(conversations)]
        mm_sources = self._format_multimodal_sources(sources, self.data_args)
        full = self._build_imgsp_supervised_batch(
            mm_sources,
            self.tokenizer,
            has_image=has_image,
        )
        data_dict = {
            "input_ids": full["input_ids"][0],
            "labels": full["labels"][0],
        }
        if "prompt" in full and full["prompt"] is not None:
            data_dict["prompt"] = full["prompt"]
        return data_dict

    @staticmethod
    def _format_multimodal_sources(sources: Sequence, data_args: Any):
        """Normalize `<image>` placeholders for imgsp-style multimodal prompts."""
        if not data_args.is_multimodal:
            return sources

        for source in sources:
            for sentence in source:
                if DEFAULT_IMAGE_TOKEN not in sentence["value"]:
                    continue
                text = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = (DEFAULT_IMAGE_TOKEN + "\n" + text).strip()

        return sources

    @staticmethod
    def _build_imgsp_supervised_batch(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
    ) -> Dict:
        """Build ``input_ids``, ``labels``, and optional ``prompt`` for imgsp (vicuna-style) supervision."""
        conv = conversation_lib.conv_templates["vicuna"].copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        conversations = []
        guided_prompt = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                source = source[1:]

            conv.messages = []
            img_in_text = False
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"

                if role == conv.roles[0]:
                    guided_sent = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").replace("\n", "")
                    guided_prompt.append(guided_sent)
                if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                    img_in_text = True
                if role == conv.roles[0] and img_in_text and DEFAULT_IMAGE_TOKEN not in sentence["value"]:
                    if random.randint(0, 1) == 0:
                        img_conv = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                    else:
                        img_conv = sentence["value"] + "\n" + DEFAULT_IMAGE_TOKEN
                    conv.append_message(role, img_conv)
                else:
                    conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        def _lookup_special_token(token: str) -> torch.Tensor:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id == tokenizer.unk_token_id:
                raise ValueError(f"Special token {token!r} is missing from tokenizer vocab.")
            return torch.tensor([token_id], dtype=torch.long)

        image_start_special_token = _lookup_special_token(IMAGE_START_TOKEN)
        image_end_special_token = _lookup_special_token(IMAGE_END_TOKEN)
        video_start_special_token = _lookup_special_token(VIDEO_START_SPECIAL_TOKEN)
        video_end_special_token = _lookup_special_token(VIDEO_END_SPECIAL_TOKEN)
        navigation_special_token = _lookup_special_token(NAVIGATION_SPECIAL_TOKEN)
        image_separator = _lookup_special_token(IMAGE_SEPARATOR)

        if has_image:
            new_list_all = []
            for prompt in conversations:
                is_nav = NAVIGATION_IDENTIFIER in prompt
                token_prompt = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                indices_to_replace = torch.where(token_prompt == -200)[0]
                new_list = []
                while indices_to_replace.numel() > 0:
                    idx = indices_to_replace[0]
                    new_list.append(token_prompt[:idx])
                    new_list.append(video_start_special_token)
                    new_list.append(image_separator)
                    new_list.append(token_prompt[idx : idx + 1])
                    new_list.append(video_end_special_token)
                    if is_nav:
                        new_list.append(image_start_special_token)
                        new_list.append(image_end_special_token)
                        new_list.append(navigation_special_token)
                    token_prompt = token_prompt[idx + 1 :]
                    indices_to_replace = torch.where(token_prompt == -200)[0]
                if token_prompt.numel() > 0:
                    new_list.append(token_prompt)
                new_list_all.append(torch.cat(new_list, dim=0))
            input_ids = torch.stack(new_list_all, dim=0)

        else:
            input_ids = tokenizer(
                conversations,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).input_ids

        targets = input_ids.clone()

        assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())
            extra = (6 if NAVIGATION_IDENTIFIER in conversation else 3) if has_image else None

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                if has_image:
                    round_len = len(tokenizer_image_token(rou, tokenizer)) + extra
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) + extra - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs {total_len}. "
                        f"conversation:{conversation}, video_or_not:True, "
                        f"identifier:{NAVIGATION_IDENTIFIER in conversation} (ignored)"
                    )
        return dict(
            input_ids=input_ids,
            labels=targets,
            prompt=guided_prompt,
        )


class Pi0Tokenization(Tokenization):
    def __init__(self, tokenizer: transformers.GemmaTokenizer, *args, **kwargs):
        self.tokenizer = tokenizer
        self._max_len = tokenizer.model_max_length

    def __call__(self, conversations: List[Dict], **kwargs):
        prompt = conversations[0]["value"]
        cleaned_prompt = prompt.strip().replace("\n", " ").replace("_", " ")
        tokens = _tokenizer_encode(self.tokenizer, cleaned_prompt, add_bos=True) + _tokenizer_encode(self.tokenizer, "\n")
        tokens = tokens[: self._max_len]
        tokens += [0] * (self._max_len - len(tokens))
        return {"input_ids": np.asarray(tokens), "labels": np.asarray(tokens)}


class Pi05Tokenization(Tokenization):
    def __init__(self, tokenizer: transformers.GemmaTokenizer, *args, **kwargs):
        self.tokenizer = tokenizer
        self._max_len = tokenizer.model_max_length

    def clean_the_text(self, text):
        return text.strip().replace("\n", " ").replace("_", " ").replace("<image>", "")

    def __call__(self, conversations: List[Dict], **kwargs):
        all_tokens = []
        all_labels = []
        for msg in conversations:
            role = msg["from"]
            if role not in ["human", "gpt"]:
                continue
            text = self.clean_the_text(msg["value"])
            if text == "":
                continue

            if role == "human":
                text = f"User: {text}\n"
                tokens = _tokenizer_encode(self.tokenizer, text, add_bos=True)
                labels = [IGNORE_INDEX] * len(tokens)
            else:  # role == "gpt"
                role_text = "Assistant: "
                role_tokens = _tokenizer_encode(self.tokenizer, role_text)
                role_labels = [IGNORE_INDEX] * len(role_tokens)
                text_tokens = _tokenizer_encode(self.tokenizer, text, add_eos=True)
                text_labels = text_tokens
                tokens = role_tokens + text_tokens
                labels = role_labels + text_labels

            all_tokens.extend(tokens)
            all_labels.extend(labels)

        assert len(all_tokens) == len(all_labels), "Tokens and labels length mismatch"
        if len(all_tokens) > self._max_len:
            print(
                f"Warning: Truncating input from {len(all_tokens)} to {self._max_len} tokens."
            )
        all_tokens = all_tokens[: self._max_len]
        all_labels = all_labels[: self._max_len]
        padding_length = self._max_len - len(all_tokens)
        all_tokens += [0] * padding_length
        all_labels += [IGNORE_INDEX] * padding_length

        return {"input_ids": np.asarray(all_tokens), "labels": np.asarray(all_labels)}


class DM0Tokenization(Tokenization):
    """DM0 tokenization matching OpenPI's DM0Tokenizer format for SFT mode.

    Uses "step" conversation template with USER/ASSISTANT roles.
    Format: "System prompt USER: prompt ASSISTANT: <empty>"
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        chat_template: str = "step",
        *args,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self._max_len = tokenizer.model_max_length
        self.chat_template = chat_template

    def __call__(self, conversations: List[Dict], **kwargs) -> Dict:
        """Tokenize conversations in SFT format.

        Args:
            conversations: List of conversation turns, e.g. [{"from": "human", "value": "prompt"}]

        Returns:
            Dict with input_ids, labels, token_mask, ar_mask, loss_mask
        """
        # Get conversation template
        conv = conversation_lib.conv_templates[self.chat_template].copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        seps = {conv.roles[0]: conv.sep, conv.roles[1]: conv.sep2}

        # Build system prompt
        system_prompt = f"{conv.system}{conv.sep}"
        tokens = list(self.tokenizer.encode(system_prompt, add_special_tokens=False))
        token_mask = [True] * len(tokens)
        ar_mask = [1] * len(tokens)  # Causal attention for all
        loss_mask = [False] * len(tokens)  # No loss on system prompt

        # Remove empty trailing assistant turn if present (requested for OpenPI alignment)
        conversations = list(conversations)
        if (
            conversations
            and conversations[-1].get("from") == "gpt"
            and not conversations[-1].get("value")
        ):
            conversations.pop()

        # Process each conversation turn
        for i, msg in enumerate(conversations):
            role_key = msg.get("from", "human")
            if role_key not in roles:
                continue
            role = roles[role_key]
            text = msg.get("value", "")
            if text is None:
                text = ""
            text = text.strip().replace("\n", " ")
            sep = seps[role]

            # Role token
            role_str = f"{role}: "
            role_tokens = list(
                self.tokenizer.encode(role_str, add_special_tokens=False)
            )
            tokens.extend(role_tokens)
            token_mask.extend([True] * len(role_tokens))
            ar_mask.extend([1] * len(role_tokens))
            loss_mask.extend([False] * len(role_tokens))

            # Content + separator
            if text:
                content_str = f"{text}{sep}"
            else:
                content_str = ""  # Empty response for assistant in SFT
            content_tokens = list(
                self.tokenizer.encode(content_str, add_special_tokens=False)
            )
            tokens.extend(content_tokens)
            token_mask.extend([True] * len(content_tokens))
            ar_mask.extend([1] * len(content_tokens))

            # Loss only on assistant responses
            if role == roles["gpt"]:
                loss_mask.extend([True] * len(content_tokens))
            else:
                loss_mask.extend([False] * len(content_tokens))

        # Pad or truncate to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [self.tokenizer.pad_token_id] * (self._max_len - tokens_len)
            pad_mask = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + pad_mask
            ar_mask = ar_mask + pad_mask
            loss_mask = loss_mask + pad_mask
        else:
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        # Create labels (same as input_ids, with IGNORE_INDEX where loss_mask is False)
        from dexbotic.constants import IGNORE_INDEX

        input_ids = np.asarray(tokens)
        labels = np.where(np.asarray(loss_mask), input_ids, IGNORE_INDEX)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "token_mask": np.asarray(token_mask),
            "ar_mask": np.asarray(ar_mask),
            "loss_mask": np.asarray(loss_mask),
        }

class GR00TN1Tokenization(Tokenization):
    def __init__(self, tokenizer, data_args):
        self.tokenizer = tokenizer
        self.data_args = data_args

    @staticmethod
    def tokenizer_image_token(
        prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
    ):
        # 以<image>为锚点split prompt
        prompt_chunks = [
            tokenizer(chunk).input_ids for chunk in prompt.split("<image>")
        ]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == "pt":
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f"Unsupported tensor type: {return_tensors}")
        return input_ids

    def __call__(self, conversations: List[Dict], has_image: bool) -> Dict:
        conv = conversation_lib.conv_templates[self.data_args.chat_template].copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates - this part is correct (lines 144-155)
        sources = [conversations]
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        if has_image:
            input_ids = torch.stack(
                [
                    self.tokenizer_image_token(
                        prompt, self.tokenizer, return_tensors="pt"
                    )
                    for prompt in conversations
                ],
                dim=0,
            )
        else:
            input_ids = self.tokenizer(
                conversations,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        data_dict = dict(input_ids=input_ids[0], labels=labels[0])

        return data_dict
