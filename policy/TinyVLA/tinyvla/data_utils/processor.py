import copy
from dataclasses import dataclass, field, fields, asdict
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import sys
import torch

import transformers

from llava_pythia.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN
from torch.utils.data import DataLoader, Dataset, Subset
from llava_pythia.train.llava_pythia_trainer import LLaVAPythiaTrainer

from llava_pythia import conversation as conversation_lib
from llava_pythia.model import *
from llava_pythia.mm_utils import tokenizer_image_token
from transformers import CLIPVisionConfig, SiglipVisionConfig, CLIPImageProcessor, SiglipImageProcessor
from PIL import Image
import numpy as np
import os

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
        sources: Sequence[str],
        data_args,
) -> Dict:
    """
    Preprocesses a list of multimodal sources by modifying image tokens.

    This function checks if the data is multimodal based on the `data_args` parameter.
    If it is, it processes each source and its sentences to handle image tokens.
    Specifically, it replaces the default image token with a formatted version,
    optionally adding start and end tokens around it.

    Args:
        sources (Sequence[str]): A sequence of source data, where each source is a list of sentences.
        data_args: An object containing data arguments, including whether the data is multimodal
                   and whether to use start and end tokens for images.

    Returns:
        Dict: The processed sources with modified image tokens.
    """
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_v0(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    """
    Preprocesses a list of conversation sources for tokenization.

    This function processes a list of conversation sources, applying prompt templates
    and tokenizing the conversations. It handles both text and multimodal data (if images are present).
    The function also masks certain parts of the tokenized data to ignore them during training.

    Args:
        sources (list): A list of conversation sources, where each source is a list of sentences.
        tokenizer (transformers.PreTrainedTokenizer): A tokenizer to convert text into token IDs.
        has_image (bool): A flag indicating whether the data includes images.

    Returns:
        Dict: A dictionary containing tokenized input IDs and labels.
    """
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
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

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
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

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum()) # in phi-2, pad_token_id == eos_token_id
        if 'phi' in tokenizer.name_or_path.lower():
            total_len +=1
        rounds = conversation.split(conv.sep2)
        cur_len = 0
        if cur_len > 0:
            target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) + 1  # +1 for <|endoftext|>
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids) + 1  # +1 for <|endoftext|>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(conversation)
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Preprocesses a list of conversation sources for tokenization in a plain format.

    This function processes a list of conversation sources, ensuring that each source
    contains exactly two elements and that the first element includes a default image token.
    It concatenates the values of the two elements with a separator and tokenizes the resulting
    conversation. The function also masks certain parts of the tokenized data to ignore them
    during training.

    Args:
        sources (Sequence[str]): A sequence of conversation sources, where each source is a list
                                 of two sentences. The first sentence must contain a default image token.
        tokenizer (transformers.PreTrainedTokenizer): A tokenizer to convert text into token IDs.

    Returns:
        Dict: A dictionary containing tokenized input IDs and labels, with certain parts masked.
    """
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=targets)


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    """
    Preprocesses a list of conversation sources for tokenization.

    This function processes a list of conversation sources, applying different preprocessing
    strategies based on the conversation separator style and version. It handles both text
    and multimodal data (if images are present). The function also masks certain parts of
    the tokenized data to ignore them during training.

    Args:
        sources (Sequence[str]): A sequence of conversation sources, where each source is a list of sentences.
        tokenizer (transformers.PreTrainedTokenizer): A tokenizer to convert text into token IDs.
        has_image (bool): A flag indicating whether the data includes images.

    Returns:
        Dict: A dictionary containing tokenized input IDs and labels.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    elif conversation_lib.default_conversation.version.startswith("v0"):
        return preprocess_v0(sources, tokenizer, has_image=has_image)
    else:
        raise ValueError(f"Invalid version: {conversation_lib.default_conversation.version}")
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_type: str,
                 data_ratio: int,
                 concat: str,
                 data_args,):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        # if data_type == 'train':
        #     list_data_dict = list_data_dict[:int(data_ratio*len(list_data_dict))]
        # elif data_type == 'eval':
        #     list_data_dict = list_data_dict[int(data_ratio*len(list_data_dict)):]
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.concat = concat

        image_file = self.list_data_dict[0]['image']
        image_folder = self.data_args.image_folder
        image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        image_r = Image.open(os.path.join(image_folder, image_file.replace('left_cap2', 'right_cap2'))).convert('RGB')
        print(
            f"{data_type}:Formatting inputs...Skip in lazy mode:{len(list_data_dict)} Size of left single image:{np.array(image).shape};Size of right single image:{np.array(image_r).shape}")

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def parse_image(self, i, image_file):
        # image_file = self.list_data_dict[i]['image']
        if isinstance(image_file, str):
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        elif isinstance(image_file, torch.Tensor):
            image = Image.fromarray(image_file.numpy())
        if self.data_args.image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            # print("##"*50)
            # print(processor.image_mean)
            # exit(0)

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        return image

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        Retrieves a sample from the dataset at the specified index.

        This function processes the sample, including tokenizing text and processing images,
        and returns a dictionary containing the processed data.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing tokenized input IDs, labels, and image data.
        """
        sources = self.list_data_dict[i]
        # print("#@"*100)
        # print(sources)
        try:
            state = sources['state']
            action = sources['action']
        except:
            pass
        # exit(0)
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image = self.parse_image(i, image_file)
            if self.concat != 'single':
                assert 'left_cap2' in self.list_data_dict[i][
                    'image'], f"Wrong data, no left_cap2 in the path {self.list_data_dict[i]['image']}"
                image_file_right = self.list_data_dict[i]['image'].replace('left_cap2', 'right_cap2')
                image_right = self.parse_image(i, image_file_right)
            #             image_file = self.list_data_dict[i]['image']
            #             image_folder = self.data_args.image_folder
            #             processor = self.data_args.image_processor
            #             image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            #             if self.data_args.image_aspect_ratio == 'pad':
            #                 def expand2square(pil_img, background_color):
            #                     width, height = pil_img.size
            #                     if width == height:
            #                         return pil_img
            #                     elif width > height:
            #                         result = Image.new(pil_img.mode, (width, width), background_color)
            #                         result.paste(pil_img, (0, (width - height) // 2))
            #                         return result
            #                     else:
            #                         result = Image.new(pil_img.mode, (height, height), background_color)
            #                         result.paste(pil_img, ((height - width) // 2, 0))
            #                         return result

            #                 # print("##"*50)
            #                 # print(processor.image_mean)
            #                 # exit(0)

            #                 image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            #                 image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            #             else:
            #                 image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            if self.concat != 'single' and self.concat != 'direct_cat':
                data_dict['image_r'] = image_right
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            try:
                crop_size = self.data_args.image_processor.crop_size
            except:
                crop_size = self.data_args.image_processor.size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        # process robot-related data，e.g. state，action
        try:
            data_dict['state'] = state
            data_dict['action'] = action
            # print("#@"*50)
            # print(action)
            # exit(0)
        except:
            pass
        # print("#@"*100)
        # print(data_dict['image_r'].shape)
        # print(action, data_dict.keys())
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    Collate examples for supervised fine-tuning.

    This class is responsible for preparing batches of data for supervised training.
    It processes a sequence of instances, each containing input IDs, labels, and potentially
    other data like actions, states, and images. The class ensures that all sequences are
    padded to the same length and that any missing or invalid data is handled appropriately.

    Attributes:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to convert text into token IDs.
    """
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        # temp_pad_token_id = 51000
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
            # padding_value=temp_pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        if not isinstance(instances[0]['action'], torch.Tensor):
            actions = torch.tensor(np.array([instance['action'] for instance in instances]))
            states = torch.tensor(np.array([instance['state'][0:] for instance in instances]))
        else:
            actions = torch.stack([instance['action'] for instance in instances])
            states = torch.stack([instance['state'][0:] for instance in instances])

        is_pad_all = torch.stack([instance['is_pad'] for instance in instances])
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            actions=actions,
            states=states,
            images_r=None,
            is_pad=is_pad_all
            # attention_mask=input_ids.ne(temp_pad_token_id),
        )
        if 'image' in instances[0]:
            # keys.append('images')
            images = [instance['image'].squeeze() for instance in instances]
            if 'image_r' in instances[0].keys():
                images_right = [instance['image_r'].squeeze() for instance in instances]
            if 'image_top' in instances[0].keys():
                images_top = [instance['image_top'].squeeze() for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
                if 'image_r' in instances[0].keys():
                    batch['images_r'] = torch.stack(images_right)
                if 'image_top' in instances[0].keys():
                    batch['images_top'] = torch.stack(images_top)
            else:
                batch['images'] = images
        # print("9"*50)
        # print(batch[images_r.shape])
        for key in ['actions', 'images', 'images_r']:
            batch[key] = torch.nan_to_num(batch[key])

        # for k,v in batch.items():
        #     batch[k] = v.to(dtype=torch.bfloat16)
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, concat="None") -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    train_eval_split = 0.9
    # print("$"*50)
    # print(concat)
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_ratio=train_eval_split,
                                          data_type='train',
                                          data_path=data_args.data_path,
                                          data_args=data_args,
                                          concat=concat)
    assert 'train' in data_args.data_path or 'eval' in data_args.data_path, "Please use train eval split data!!!!!"
    eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                         data_ratio=train_eval_split,
                                         data_type='eval',
                                         data_path=data_args.data_path.replace('train', 'eval'),
                                         data_args=data_args,
                                         concat=concat)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                # eval_dataset=None,
                data_collator=data_collator)