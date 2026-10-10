# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import base64
import copy
import io
import json
import os
import os.path as osp
import random
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import PIL
import torch
import transformers
from PIL import Image, ImageFile
from torch.utils.data import Dataset, default_collate
from transformers import PreTrainedTokenizer

import llava.data.datasets_mixture as datasets_mixture
from llava import conversation as conversation_lib
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.mm_utils import (
  dynamic_process_images_and_prompt, process_image, tokenizer_image_token,
  process_image_ndarray, process_image_bytes,
)
from llava.train.args import DataArguments, TrainingArguments
from llava.train.sequence_parallel import (
    extract_local_from_list,
    extract_local_input_ids,
    extract_local_position_ids,
    get_pg_manager,
)
from llava.utils.tokenizer import preprocess_conversation

ImageFile.LOAD_TRUNCATED_IMAGES = True
PIL.Image.MAX_IMAGE_PIXELS = 1000000000


# import lmdb
import cv2

import datasets
import pickle

def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        concat_values = "".join([sentence["value"] for sentence in source])
        for sid, sentence in enumerate(source):
            # In multimodal conversations, we automatically prepend '<image>' at the start of the first sentence if it doesn't already contain one.
            if sid == 0 and DEFAULT_IMAGE_TOKEN not in concat_values:
                sentence["value"] = f"{DEFAULT_IMAGE_TOKEN}\n" + sentence["value"]
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence_chunks = [chunk.strip() for chunk in sentence["value"].split(DEFAULT_IMAGE_TOKEN)]
                sentence_chunks = [
                    chunk + " " if not (chunk.endswith("\n")) else chunk for chunk in sentence_chunks[:-1]
                ] + [sentence_chunks[-1]]
                sentence["value"] = f"{DEFAULT_IMAGE_TOKEN}\n".join(sentence_chunks).strip()

                replace_token = DEFAULT_IMAGE_TOKEN
                if "mmtag" in conversation_lib.default_conversation.version:
                    replace_token = "<Image>" + replace_token + "</Image>"
                if data_args.mm_use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
            # ensure every DEFAULT_IMAGE_TOKEN is followed by a newline character.
            # If it has one already, we don't add another one.
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, f"{DEFAULT_IMAGE_TOKEN}\n")
                sentence["value"] = sentence["value"].replace(f"{DEFAULT_IMAGE_TOKEN}\n\n", f"{DEFAULT_IMAGE_TOKEN}\n")

    return sources


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    no_system_prompt: bool = False,
) -> Dict:
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    return default_collate(
        [
            preprocess_conversation(conversation, tokenizer, no_system_prompt=no_system_prompt)
            for conversation in sources
        ]
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is originally implemented by the LLaVA team and modified by
    Ji Lin and Haotian Tang.
    """

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        super().__init__()
        try:
            with open(data_path) as fp:
                list_data_dict = json.load(fp)
        except:
            with open(data_path) as fp:
                list_data_dict = [json.loads(q) for q in fp]

        # rank0_print("Formatting inputs...Skip in lazy mode")
        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.image_folder = image_folder

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    @staticmethod
    def _load_video(video_path, num_video_frames, loader_fps, data_args, fps=None, frame_count=None):
        from torchvision import transforms

        from llava.mm_utils import opencv_extract_frames

        # frames_loaded = 0
        if "shortest_edge" in data_args.image_processor.size:
            image_size = data_args.image_processor.size["shortest_edge"]
        elif "longest_edge" in data_args.image_processor.size:
            image_size = data_args.image_processor.size["longest_edge"]
        else:
            image_size = data_args.image_processor.size["height"]
        # toTensor = transforms.ToTensor()

        try:
            pil_imgs, frames_loaded = opencv_extract_frames(video_path, num_video_frames, loader_fps, fps, frame_count)
        except Exception as e:
            video_loading_succeed = False
            print(f"bad data path {video_path}")
            print(f"[DEBUG] Error processing {video_path}: {e}")
            # video_outputs = torch.zeros(3, 8, image_size, image_size, dtype=torch.uint8)
            empty_num_video_frames = int(random.uniform(2, num_video_frames))
            # pil_imgs = [torch.zeros(3, image_size, image_size, dtype=torch.float32)] * empty_num_video_frames
            pil_imgs = [Image.new("RGB", (448, 448), (0, 0, 0))] * empty_num_video_frames
            frames_loaded = 0

        return pil_imgs, frames_loaded

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        enable_dynamic_res = self.data_args.image_aspect_ratio == "dynamic"
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]

            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            if isinstance(image_file, list):
                if enable_dynamic_res:
                    processed_images, sources[0][0]["value"] = dynamic_process_images_and_prompt(
                        image_file, sources[0][0]["value"], self.data_args, self.image_folder
                    )
                else:
                    processed_images = torch.stack(
                        [process_image(img, self.data_args, self.image_folder) for img in image_file]
                    )
            else:
                if enable_dynamic_res:
                    processed_images, sources[0][0]["value"] = dynamic_process_images_and_prompt(
                        [image_file], sources[0][0]["value"], self.data_args, self.image_folder
                    )
                else:
                    processed_images = process_image(
                        image_file, self.data_args, self.image_folder, enable_dynamic_res=enable_dynamic_res
                    )

        elif "images" in sources[0]:
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            image_files = [
                image_file["path"] if isinstance(image_file, dict) else image_file
                for image_file in self.list_data_dict[i]["images"]
            ]
            if enable_dynamic_res:
                processed_images, sources[0][0]["value"] = dynamic_process_images_and_prompt(
                    image_files, sources[0][0]["value"], self.data_args, self.image_folder
                )
            else:
                all_images = []
                for image_file in self.list_data_dict[i]["images"]:
                    if isinstance(image_file, dict):
                        image_file = image_file["path"]
                    image = process_image(image_file, self.data_args, self.image_folder)
                    all_images.append(image)
                processed_images = torch.stack(all_images)
        elif ("video" in sources[0]) or ("video_id" in sources[0]):
            # num_video_frames = self.data_args.num_video_frames
            if "video_path" in sources[0]:
                video_file = sources[0]["video_path"]
            elif "video" in sources[0]:
                video_file = sources[0]["video"]
            else:
                video_file = sources[0]["video_id"] + ".mp4"
            video_folder = self.image_folder
            video_path = os.path.join(video_folder, video_file)
            num_video_frames = self.data_args.num_video_frames if hasattr(self.data_args, "num_video_frames") else 8
            loader_fps = self.data_args.fps if hasattr(self.data_args, "fps") else 0.0

            if "fps" in sources[0]:
                fps = sources[0]["fps"]
            else:
                fps = None
            if "frame_count" in sources[0]:
                frame_count = sources[0]["frame_count"]
            else:
                frame_count = None

            images, frames_loaded = self._load_video(
                video_path, num_video_frames, loader_fps, self.data_args, fps=fps, frame_count=frame_count
            )

            image_tensor = torch.stack([process_image(image, self.data_args, None) for image in images])

            if "captions" in sources[0]:
                question = "Elaborate on the visual and narrative elements of the video in detail."
                assert sources[0]["captions"][-1]["idx"] == "-1"
                answer = sources[0]["captions"][-1]["content"]
            elif "video" in sources[0]:
                question = sources[0]["conversations"][0]["value"].rstrip()
                if isinstance(sources[0]["conversations"][1]["value"], str):
                    answer = sources[0]["conversations"][1]["value"].rstrip()
                else:
                    answer = str(sources[0]["conversations"][1]["value"]).rstrip()
            else:
                question = sources[0]["q"]
                answer = sources[0]["a"]

            if frames_loaded == 0:
                answer = "Empty video."
            num_frames_loaded_successfully = len(images)

            question = question.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
            question = question.replace("<video>\n", "").replace("\n<video>", "").replace("<video>", "")
            question = "<image>\n" * num_frames_loaded_successfully + question
            conversation = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer},
            ]

            sources = [conversation]
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        # data_dict = preprocess(sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=(
                "image" in self.list_data_dict[i]
                or "images" in self.list_data_dict[i]
                or "video" in self.list_data_dict[i]
                or "video_id" in self.list_data_dict[i]
            ),
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            if processed_images is None or len(processed_images.shape) == 4:
                data_dict["image"] = processed_images
            else:
                data_dict["image"] = processed_images.unsqueeze(0)
        elif "images" in self.list_data_dict[i]:
            data_dict["image"] = processed_images
        elif ("video" in self.list_data_dict[i]) or ("video_id" in self.list_data_dict[i]):
            data_dict["image"] = image_tensor
            if frames_loaded == 0:
                data_dict["labels"][:] = IGNORE_INDEX
        else:
            # llava 1.5 way
            # image does not exist in the data, but the model is multimodal
            # crop_size = self.data_args.image_processor.crop_size
            # data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            # vila way
            data_dict["image"] = None
        return data_dict


class LazyMMC4Dataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Ji Lin and Haotian Tang."""

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        image_following_text_only=False,
        text_only=False,
    ):
        super().__init__()

        import pickle

        n_samples = []
        # actually shards and stats info
        n_shards = len(os.listdir(data_path)) // 2
        # n_shards = 100
        count_info_list = sorted([f for f in os.listdir(data_path) if f.endswith(".count")])[:n_shards]
        n_samples = [int(open(os.path.join(data_path, f)).read().strip()) for f in count_info_list]

        print("total MMC4 samples", sum(n_samples))  # 10,881,869

        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            import torch.distributed as dist

            sequence_parallel_size = training_args.seq_parallel_size
        else:
            sequence_parallel_size = 1
        print("sequence_parallel_size", sequence_parallel_size)
        rank = training_args.process_index // sequence_parallel_size  # int(os.environ["RANK"])
        world_size = training_args.world_size // sequence_parallel_size  # int(os.environ["WORLD_SIZE"])
        shared_size = n_shards // world_size

        gpu_samples = [sum(n_samples[i * shared_size : (i + 1) * shared_size]) for i in range(world_size)]
        self.n_samples = min(gpu_samples) * world_size  # total size
        self.idx_offset = rank * min(gpu_samples)
        shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
        print(f" * loading data from shard {shard_start}-{shard_end}")

        shard_names = [d.replace(".count", ".pkl") for d in count_info_list]
        shard_names = shard_names[shard_start:shard_end]

        full_data_list = []
        # now load data
        for shard_name in shard_names:
            # load shard
            with open(os.path.join(data_path, shard_name), "rb") as f:
                data_list = pickle.load(f)

            full_data_list.extend(data_list)

        print(f"* loaded totally {len(full_data_list)} samples")

        self.data_list = full_data_list

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

        self.image_following_text_only = image_following_text_only
        self.text_only = text_only

    def __len__(self):
        # return len(self.data_list)
        return self.n_samples

    @property
    def modality_lengths(self):
        # Estimate the number of tokens after tokenization, used for length-grouped sampling
        length_list = []
        for info in self.data_list:
            num_images = min(6, len(info["image_info"]))
            sentences = [info["text_list"][x["matched_text_index"]] for x in info["image_info"][:num_images]]
            # The unit of cur_len is "words". We assume 1 word = 2 tokens.
            cur_len = num_images * self.num_image_tokens // 2 + sum([len(x) for x in sentences])
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        info = self.data_list[i - self.idx_offset]

        sentences = info["text_list"]
        # kentang-mit@: remove existing <image> tokens in the sentences
        for ix in range(len(sentences)):
            # if this is an html tag, we still preserve its semantic meaning
            sentences[ix] = sentences[ix].replace("<image>", "<IMAGE>")
        sim_matrix = info["similarity_matrix"]  # we do not use this...

        # convert images from base64 to PIL and filter based on image-text similarity
        images, sentence_ixs = [], []
        if not self.text_only:
            for sample_image, sim_vec in zip(info["image_info"], sim_matrix):
                image_base64 = sample_image["image_base64"]
                rawbytes = base64.b64decode(image_base64)

                sim_ix = sample_image["matched_text_index"]
                # sim_ix = np.argmax(sim_vec)
                # sim_score = sim_vec[sim_ix]

                # filter to images >= 5KB
                # if len(rawbytes) // 1000 <= 5:
                #     continue
                # if sim_score < 0.24:
                #     continue
                image = Image.open(io.BytesIO(rawbytes)).convert("RGB")

                images.append(image)
                sentence_ixs.append(sim_ix)

        # constrain max num 6 images
        max_num_images = 6
        if len(images) > max_num_images:
            images = images[:max_num_images]
            sentence_ixs = sentence_ixs[:max_num_images]

        # reorder images according to text insertion
        images = [images[iii] for iii in np.argsort(sentence_ixs)]

        # preprocess and tokenize text
        for ix in sentence_ixs:
            sentences[ix] = f"<image>\n{sentences[ix]}"

        if self.image_following_text_only:
            # use pad tokens to divide sentence pieces
            text = self.tokenizer.pad_token.join(sentences)
        else:
            text = " ".join(sentences)
        # whitespace cleanup
        text = text.replace("<image> ", "<image>").replace(" <image>", "<image>")
        text = f"{text}{self.tokenizer.eos_token}"  # add eos token

        if len(images) > 0:
            if self.data_args.image_aspect_ratio == "dynamic":
                images, text = dynamic_process_images_and_prompt(images, text, self.data_args, self.image_folder)
            else:
                images = torch.stack([process_image(image, self.data_args, self.image_folder) for image in images])

            # the same size for all images, so we concat
            # cur_token_len = (
            #     images[0].shape[-2] // self.multimodal_cfg["patch_size"]
            # ) * (images[0].shape[-1] // self.multimodal_cfg["patch_size"])
            # cur_token_len += self.multimodal_cfg["n_extra_patch"]
        else:
            images = None
            # cur_token_len = 0

        # im_patch_token = self.tokenizer.convert_tokens_to_ids(
        #     [DEFAULT_IMAGE_PATCH_TOKEN]
        # )[0]
        # print(text, len(images))
        input_ids = tokenizer_image_token(
            text,
            self.tokenizer,
            return_tensors="pt",
        )

        # now check the case where the last token is image patch token
        if input_ids[-1] == IMAGE_TOKEN_INDEX:  # need to remove one last image
            last_non_im_patch_indices = torch.where(input_ids != IMAGE_TOKEN_INDEX)[0][-1] + 1
            input_ids = input_ids[:last_non_im_patch_indices]

        n_im_patch = (input_ids == IMAGE_TOKEN_INDEX).sum().item()

        images = images[:n_im_patch]
        assert len(images) == n_im_patch, print(text, input_ids)
        assert len(input_ids.shape) == 1, "Unexpected shape of 'input_ids' from MMC4."
        input_ids = (
            torch.concat([torch.tensor([self.tokenizer.bos_token_id]), input_ids])
            if self.tokenizer.bos_token_id is not None and input_ids[0] != self.tokenizer.bos_token_id
            else input_ids
        )
        targets = input_ids.clone()

        if self.image_following_text_only:  # keep only text after leading image token
            # remove loss for any token before the first <image> token
            label_idx = 0
            while label_idx < targets.shape[-1] and targets[label_idx] != IMAGE_TOKEN_INDEX:
                targets[label_idx] = IGNORE_INDEX
                label_idx += 1

            pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]

            pad_token_idxs = torch.where(targets == pad_token)[0]
            for pad_token_idx in pad_token_idxs:
                token_idx = pad_token_idx + 1
                while token_idx < targets.shape[-1] and targets[token_idx] != IMAGE_TOKEN_INDEX:
                    targets[token_idx] = IGNORE_INDEX
                    token_idx += 1
            # do not train on padding tokens
            targets[targets == pad_token] = IGNORE_INDEX

        # mask image tokens is unnecessary for llava-1.5
        # targets[targets == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
        # print(input_ids.shape)

        return dict(input_ids=input_ids, labels=targets, image=images)


class LazyCoyoDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Ji Lin and Haotian Tang."""

    num_image_tokens = 576

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        # kentang-mit@: balance the total number of tokens for Coyo and MMC4.
        n_samples_per_idx=4,
    ):
        super().__init__()

        import pickle

        n_samples = []
        # actually shards and stats info
        n_shards = len(os.listdir(data_path)) // 2
        # n_shards = 100
        count_info_list = sorted([f for f in os.listdir(data_path) if f.endswith(".count")])[:n_shards]
        n_samples = [int(open(os.path.join(data_path, f)).read().strip()) for f in count_info_list]

        print("total COYO samples", sum(n_samples))

        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            import torch.distributed as dist

            sequence_parallel_size = training_args.seq_parallel_size
        else:
            sequence_parallel_size = 1
        print("sequence_parallel_size", sequence_parallel_size)
        rank = training_args.process_index // sequence_parallel_size  # int(os.environ["RANK"])
        world_size = training_args.world_size // sequence_parallel_size  # int(os.environ["WORLD_SIZE"])
        shared_size = n_shards // world_size

        gpu_samples = [
            sum(n_samples[i * shared_size : (i + 1) * shared_size]) // n_samples_per_idx for i in range(world_size)
        ]
        self.n_samples = min(gpu_samples) * world_size  # total size
        self.idx_offset = rank * min(gpu_samples)

        shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
        print(f" * loading data from shard {shard_start}-{shard_end}")

        shard_names = [d.replace(".count", ".pkl") for d in count_info_list]
        shard_names = shard_names[shard_start:shard_end]

        full_data_list = []
        # now load data
        for shard_name in shard_names:
            # load shard
            with open(os.path.join(data_path, shard_name), "rb") as f:
                shard_data = pickle.load(f)
                random.seed(42)
                if "mmc4" in data_path:
                    random.shuffle(shard_data)  # shuffle for MMC4cap only
                full_data_list.extend(shard_data)

        print(f"* loaded totally {len(full_data_list)} samples")

        # now pack the samples into groups
        n_groups = len(full_data_list) // n_samples_per_idx
        full_data_list = [
            full_data_list[i : i + n_samples_per_idx] for i in range(0, len(full_data_list), n_samples_per_idx)
        ]
        if len(full_data_list[-1]) < n_samples_per_idx:
            full_data_list = full_data_list[:-1]
        assert len(full_data_list) == n_groups
        print(f"split into {n_groups} groups")

        self.data_list = full_data_list

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

    def __len__(self):
        # return len(self.data_list)
        return self.n_samples

    @property
    def modality_lengths(self):
        # Estimate the number of tokens after tokenization, used for length-grouped sampling
        length_list = []
        for samples in self.data_list:
            cur_len = sum([len(conv["text" if "text" in conv else "caption"].split()) for conv in samples])
            # The unit of cur_len is "words". We assume 1 word = 2 tokens.
            cur_len = cur_len + len(samples) * self.num_image_tokens // 2
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        CONCAT_SAMPLES = False
        info_list = self.data_list[i - self.idx_offset]

        text_list = []
        image_list = []

        for sample in info_list:
            caption_key = (
                "text" if "text" in sample else "caption"
            )  # kentang-mit@: remove existing <image> tokens in the sentences
            # kentang-mit@: remove existing <image> token.
            # if this is an html tag, we still preserve its semantic meaning
            sample[caption_key] = sample[caption_key].replace("<image>", "<IMAGE>")
            text_list.append(DEFAULT_IMAGE_TOKEN + "\n" + sample[caption_key] + self.tokenizer.eos_token)
            if "image" in sample:
                image_base64 = sample["image"]
                rawbytes = base64.b64decode(image_base64)
            else:
                rawbytes = sample["rawbytes"]
            image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
            image_list.append(image)

        image_list = torch.stack([process_image(image, self.data_args, self.image_folder) for image in image_list])

        # the same size for all images, so we concat
        # cur_token_len = (
        #     image_list[0].shape[-2] // self.multimodal_cfg["patch_size"]
        # ) * (image_list[0].shape[-1] // self.multimodal_cfg["patch_size"])
        # cur_token_len += self.multimodal_cfg["n_extra_patch"]

        # replace_token = DEFAULT_IMAGE_TOKEN
        # if self.multimodal_cfg["use_im_start_end"]:
        #     replace_token = (
        #         DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
        #     )
        # text_list = [
        #     text.replace(DEFAULT_IMAGE_TOKEN, replace_token) for text in text_list
        # ]

        if CONCAT_SAMPLES:
            # into <image>cap<eos><image>cap<eos>...
            text_list = "".join(text_list)

            input_ids = self.tokenizer(
                text_list,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids  # 4, seq_len

            input_ids = input_ids[0]

        else:
            input_ids = [
                tokenizer_image_token(
                    prompt,
                    self.tokenizer,
                    return_tensors="pt",
                )
                for prompt in text_list
            ]
            # print([x.shape[0] for x in input_ids], [len(x.split()) for x in text_list], [len(re.findall(r"<image[^>]*>", x)) for x in text_list])

            # input_ids = torch.nn.utils.rnn.pad_sequence(
            #     input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            # )

        targets = copy.deepcopy(input_ids)
        # mask image tokens is unnecessary for llava-1.5
        # targets[targets == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
        for i in range(len(targets)):
            targets[i][targets[i] == self.tokenizer.pad_token_id] = IGNORE_INDEX

        return dict(input_ids=input_ids, labels=targets, image=image_list)


class LazyWDSDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Ji Lin and Ligeng Zhu."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        image_folder: str,
        training_args: TrainingArguments,
    ):
        super().__init__()
        n_samples = []
        n_shards = len(os.listdir(data_path)) // 3
        for shard in range(n_shards):
            with open(os.path.join(data_path, f"{shard:05d}_stats.json")) as f:
                info = json.load(f)
                n_samples.append(info["successes"])

        # print(f"[DEBUG] {data_path} total samples", sum(n_samples))  # 10,881,869

        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            import torch.distributed as dist

            sequence_parallel_size = training_args.seq_parallel_size
        else:
            sequence_parallel_size = 1
        print("sequence_parallel_size", sequence_parallel_size)
        rank = training_args.process_index // sequence_parallel_size  # int(os.environ["RANK"])
        world_size = training_args.world_size // sequence_parallel_size  # int(os.environ["WORLD_SIZE"])
        shared_size = n_shards // world_size
        print("rank", rank, "world_size", world_size, "shared_size", shared_size)
        gpu_samples = [sum(n_samples[i * shared_size : (i + 1) * shared_size]) for i in range(world_size)]
        self.n_samples = min(gpu_samples) * world_size  # total size
        self.idx_offset = rank * min(gpu_samples)
        shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
        print(f" * loading data from shard {shard_start}-{shard_end}")

        tar_list = [f"{shard_idx:05d}.tar" for shard_idx in range(shard_start, shard_end)]

        self.data_list = []
        t1 = time.time()
        for tar in tar_list:
            tmp_path = f"/tmp/ccs{tar}"
            tar_path = os.path.join(data_path, tar)

            if PROCESS_GROUP_MANAGER is not None:
                dist.barrier()
                if PROCESS_GROUP_MANAGER.sp_rank == 0:
                    os.makedirs(tmp_path, exist_ok=True)
                    os.system(f"tar -xkf {tar_path} -C {tmp_path}")
                dist.barrier()
            else:
                os.makedirs(tmp_path, exist_ok=True)
                os.system(f"tar -xkf {tar_path} -C {tmp_path}")

            txt_list = [f for f in os.listdir(tmp_path) if f.endswith(".txt")]

            for txt in txt_list:
                caption = open(os.path.join(tmp_path, txt)).read().strip()
                image_path = os.path.join(tmp_path, txt.split(".")[0] + ".jpg")
                self.data_list.append({"caption": caption, "image": image_path})
        t2 = time.time()
        print(f"Loading done. Total time: {t2 - t1:.2f} seconds")

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        # print("i", i, "idx_offset", self.idx_offset, "len", len(self.data_list))
        info = self.data_list[i - self.idx_offset]
        caption, image_path = info["caption"], info["image"]

        rand_prompt = "<image>\n"
        sources = [
            {
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": rand_prompt},
                    {"from": "gpt", "value": caption},
                ],
            }
        ]

        # one example of sources
        # [{'id': 'GCC_train_001738742', 'image': 'GCC_train_001738742.jpg', 'conversations': [{'from': 'human', 'value': 'Provide a brief description of the given image.\n<image>'}, {'from': 'gpt', 'value': 'a sketch of an ostrich'}]}]
        if "image" in sources[0]:
            image = process_image(sources[0]["image"], self.data_args, self.image_folder)
            image = torch.unsqueeze(image, dim=0)
            # now random pick some context samples for training
            if hasattr(self.data_args, "num_shots"):
                if self.data_args.num_shots > 0:
                    raise NotImplementedError
        else:
            raise NotImplementedError

        data_dict = preprocess([sources[0]["conversations"]], self.tokenizer, has_image=True)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if image is not None:
            data_dict["image"] = image
        else:
            raise NotImplementedError

        return data_dict


class LazyCCSWebDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Ligeng Zhu."""

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        super().__init__()
        t1 = time.time()

        from llava.data.simple_vila_webdataset import VILAWebDataset

        print("[DEBUG] ", osp.abspath(data_path))
        self.dataset = VILAWebDataset(data_path=osp.abspath(data_path))

        t2 = time.time()
        print(f"Loading done. Total time: {t2 - t1:.2f} seconds")

        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # info = self.data_list[i - self.idx_offset]
        # caption, image_path = info["caption"], info["image"]
        info = self.dataset[i]
        if ".jpg" in info:
            caption, image_path = info[".txt"], info[".jpg"]
        elif ".png" in info:
            caption, image_path = info[".txt"], info[".png"]
        elif ".webp" in info:
            caption, image_path = info[".txt"], info[".webp"]
        elif ".bmp" in info:
            caption, image_path = info[".txt"], info[".bmp"]
        elif ".tiff" in info:
            caption, image_path = info[".txt"], info[".tiff"]
        else:
            print(info.keys())
            print(info)
            raise KeyError

        caption = caption.replace("<image>", "<IMAGE>")
        if isinstance(image_path, io.BytesIO):
            image_path = Image.open(image_path).convert("RGB")

        if not isinstance(image_path, PIL.Image.Image):
            print(image_path)
            print(info.keys())
            print(type(image_path))
            raise NotImplementedError

        rand_prompt = "<image>\n"
        sources = [
            {
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": rand_prompt},
                    {"from": "gpt", "value": caption},
                ],
            }
        ]

        # one example of sources
        # [{'id': 'GCC_train_001738742', 'image': 'GCC_train_001738742.jpg', 'conversations': [{'from': 'human', 'value': 'Provide a brief description of the given image.\n<image>'}, {'from': 'gpt', 'value': 'a sketch of an ostrich'}]}]
        if "image" in sources[0]:
            image = process_image(sources[0]["image"], self.data_args, image_folder=None)
            image = torch.unsqueeze(image, dim=0)
            # now random pick some context samples for training
            if hasattr(self.data_args, "num_shots"):
                if self.data_args.num_shots > 0:
                    raise NotImplementedError
        else:
            raise NotImplementedError

        data_dict = preprocess([sources[0]["conversations"]], self.tokenizer, has_image=True)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if image is not None:
            data_dict["image"] = image
        else:
            raise NotImplementedError

        return data_dict


from functools import lru_cache


@lru_cache(maxsize=16)
def lru_json_load(fpath):
    with open(fpath) as fp:
        return json.load(fp)


class LazyCoyoWebDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Ligeng Zhu."""

    num_image_tokens = 576

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        # kentang-mit@: balance the total number of tokens for Coyo and MMC4.
        n_samples_per_idx=4,
    ):
        super().__init__()

        from llava.data.simple_vila_webdataset import VILAWebDataset

        print("[DEBUG] ", osp.abspath(data_path))
        self.dataset = VILAWebDataset(data_path=osp.abspath(data_path), meta_path=data_args.meta_path)

        if data_args.start_idx >= 0 and data_args.end_idx >= 0:
            # Ligeng: support slicing for ablate different subsets.
            total = len(self.dataset)
            start_idx = int(total * data_args.start_idx)
            end_idx = int(total * data_args.end_idx)
            print(f"loading subset from {start_idx} to {end_idx}, total {total}")
            self.dataset = torch.utils.data.Subset(self.dataset, range(start_idx, end_idx))

        # For caption choice,
        #   if None: use original caption
        #   if a folder path: use specified caption to override original one (choice1)
        #   if a folder path: use specified caption and concat with original one (choice2)
        self.caption_choice = None
        self.caption_choice_2 = None
        self.data_path = data_path

        if data_args.caption_choice is not None:
            self.caption_choice = data_args.caption_choice
            print("[recap] Override coyo caption using ", self.caption_choice)

        if data_args.caption_choice_2 is not None:
            self.caption_choice_2 = data_args.caption_choice_2
            print("[recapv2] Override coyo caption using ", self.caption_choice_2)

        print("total samples", len(self.dataset))
        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            import torch.distributed as dist

            sequence_parallel_size = training_args.seq_parallel_size
            sequence_parallel_rank = PROCESS_GROUP_MANAGER.sp_rank
        else:
            sequence_parallel_size = 1
        print("sequence_parallel_size", sequence_parallel_size)
        rank = (
            training_args.process_index // sequence_parallel_size if "RANK" in os.environ else 2
        )  # int(os.environ["RANK"])
        world_size = (
            training_args.world_size // sequence_parallel_size if "WORLD_SIZE" in os.environ else 32
        )  # int(os.environ["WORLD_SIZE"])
        print(
            "rank",
            rank,
            "world_size",
            world_size,
        )

        self.n_samples_per_idx = n_samples_per_idx
        # self.n_samples = len(self.dataset) // n_samples_per_idx
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.dataset) // self.n_samples_per_idx

    @property
    def modality_lengths(self):
        # Estimate the number of tokens after tokenization, used for length-grouped sampling
        length_list = []
        for samples in self.data_list:
            cur_len = sum([len(conv["text" if "text" in conv else "caption"].split()) for conv in samples])
            # The unit of cur_len is "words". We assume 1 word = 2 tokens.
            cur_len = cur_len + len(samples) * self.num_image_tokens // 2
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        CONCAT_SAMPLES = False
        # info_list = self.dataset[i - self.idx_offset]

        begin_idx, end_idx = (
            i * self.n_samples_per_idx,
            (i + 1) * self.n_samples_per_idx,
        )
        end_idx = min(end_idx, len(self.dataset))

        text_list = []
        image_list = []

        for idx in range(begin_idx, end_idx):
            info = self.dataset[idx]
            if ".jpg" in info:
                caption, image_path = info[".txt"], info[".jpg"]
            elif ".png" in info:
                caption, image_path = info[".txt"], info[".png"]
            elif ".webp" in info:
                caption, image_path = info[".txt"], info[".webp"]
            elif ".bmp" in info:
                caption, image_path = info[".txt"], info[".bmp"]
            elif ".tiff" in info:
                caption, image_path = info[".txt"], info[".tiff"]
            else:
                print(info.keys())
                print(info)
                raise KeyError

            if self.caption_choice is not None:
                # load new captions
                shard = info["__shard__"]
                url = info[".json"]["url"]
                tar_name = osp.relpath(osp.realpath(shard), osp.realpath(self.data_path))
                # tar_name = osp.dirname(shard)
                shard_json_path = osp.join(self.caption_choice, tar_name + ".json")
                try:
                    shard_json = lru_json_load(shard_json_path)
                    try:
                        caption = shard_json[url]["output"]
                    except KeyError:
                        print(f"{url} not in caption. fallback to original caption temporarially")
                except:
                    print(f"shard_json_path {shard_json_path} not found. fallback to original caption temporarially")
            caption = caption.replace("<image>", "<IMAGE>")
            text_list.append(DEFAULT_IMAGE_TOKEN + caption + self.tokenizer.eos_token)

            if isinstance(image_path, io.BytesIO):
                image_path = Image.open(image_path).convert("RGB")

            if not isinstance(image_path, PIL.Image.Image):
                print(image_path)
                print(info.keys())
                print(type(image_path))
                raise NotImplementedError

            image_list.append(image_path)

        image_list = torch.stack([process_image(image, self.data_args, image_folder=None) for image in image_list])

        if CONCAT_SAMPLES:
            # into <image>cap<eos><image>cap<eos>...
            text_list = "".join(text_list)

            input_ids = self.tokenizer(
                text_list,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids  # 4, seq_len

            input_ids = input_ids[0]
        else:
            input_ids = [
                tokenizer_image_token(
                    prompt,
                    self.tokenizer,
                    return_tensors="pt",
                )
                for prompt in text_list
            ]
            input_ids = [
                (
                    torch.concat([torch.tensor([self.tokenizer.bos_token_id]), input_ids_i])
                    if input_ids_i[0] != self.tokenizer.bos_token_id
                    else input_ids_i
                )
                for input_ids_i in input_ids
            ]

        targets = copy.deepcopy(input_ids)
        # mask image tokens is unnecessary for llava-1.5
        # targets[targets == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
        for i in range(len(targets)):
            targets[i][targets[i] == self.tokenizer.pad_token_id] = IGNORE_INDEX

        return dict(input_ids=input_ids, labels=targets, image=image_list)


class LazyVideoWebDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        # cache_path: str,
        # n_samples_per_idx=4,
    ):
        super().__init__()

        # from llava.data.simple_video_dataset import SimpleVideoDataset

        from llava.data.simple_vila_webdataset import VILAWebDataset

        print("[DEBUG] ", osp.abspath(data_path))
        self.dataset = VILAWebDataset(
            data_path=osp.abspath(data_path),
            meta_path=f"{osp.abspath(data_path)}/wids-meta.json",
            # cache_dir=cache_path,
        )

        # None: use original caption
        # Folder path: use original caption
        self.caption_choice = None
        self.data_path = data_path

        if data_args.caption_choice is not None:
            self.caption_choice = data_args.caption_choice
            print("[recap] Override LazyVideo caption using ", self.caption_choice)

        print("total samples", len(self.dataset))
        # InternVid: TODO
        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            import torch.distributed as dist

            sequence_parallel_size = training_args.seq_parallel_size
            sequence_parallel_rank = PROCESS_GROUP_MANAGER.sp_rank
        else:
            sequence_parallel_size = 1
        print("sequence_parallel_size", sequence_parallel_size)
        rank = (
            training_args.process_index // sequence_parallel_size if "RANK" in os.environ else 2
        )  # int(os.environ["RANK"])
        world_size = (
            training_args.world_size // sequence_parallel_size if "WORLD_SIZE" in os.environ else 32
        )  # int(os.environ["WORLD_SIZE"])
        print(
            "rank",
            rank,
            "world_size",
            world_size,
        )
        self.rank = rank
        # rank = int(os.environ["RANK"]) if "RANK" in os.environ else 2
        # world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 32

        self.tokenizer = tokenizer
        self.data_args = data_args

        self.missing_uids = set()

    def __len__(self):
        return len(self.dataset)

    @property
    def modality_lengths(self):
        # Estimate the number of tokens after tokenization, used for length-grouped sampling
        length_list = []
        for samples in self.data_list:
            cur_len = sum([len(conv["text" if "text" in conv else "caption"].split()) for conv in samples])
            # The unit of cur_len is "words". We assume 1 word = 2 tokens.
            cur_len = cur_len + len(samples) * self.num_image_tokens // 2
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        ADD_TEXT_PROMPT = False
        num_video_frames = self.data_args.num_video_frames if hasattr(self.data_args, "num_video_frames") else 8
        loader_fps = self.data_args.fps if hasattr(self.data_args, "fps") else 0.0

        info = self.dataset[i]

        caption = ""
        # print(info)
        if ".mp4" in info:
            caption, video_path = info[".txt"], info[".mp4"]
        else:
            video_path = None
            caption = "Empty video."

        images, frames_loaded = LazySupervisedDataset._load_video(
            video_path, num_video_frames, loader_fps, self.data_args
        )

        if frames_loaded == 0:
            caption = "Empty video."

        if self.caption_choice is not None:
            shard = info["__shard__"]
            uuid = osp.join(info["__shard__"], info["__key__"])
            url = info["__key__"]
            tar_name = osp.basename(info["__shard__"])

            try:
                shard_json_path = osp.join(self.caption_choice, tar_name.replace(".tar", ".json"))
                shard_json = lru_json_load(shard_json_path)
                caption = shard_json[url]["summary"]["output"]
            except (KeyError, FileNotFoundError, json.decoder.JSONDecodeError):
                if uuid not in self.missing_uids:
                    print("override caption not found for ", uuid)
                    self.missing_uids.add(uuid)

            # print(f"[DEBUG {uuid}]", caption)

        frames_loaded_successfully = len(images)
        if caption is None:
            caption = ""
        prompt = "<image>\n" * frames_loaded_successfully + caption
        image_tensor = torch.stack([process_image(image, self.data_args, None) for image in images])

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            return_tensors="pt",
        )
        targets = copy.deepcopy(input_ids)
        data_dict = dict(input_ids=input_ids, labels=targets, image=image_tensor)

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning.
    This class is originally implemented by the LLaVA team and
    modified by Haotian Tang."""

    tokenizer: transformers.PreTrainedTokenizer
    data_args: DataArguments

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # input_ids, labels = tuple([instance[key] for instance in instances]
        #                           for key in ("input_ids", "labels"))
        input_ids, labels, images = [], [], []
        for instance in instances:
            if not isinstance(instance["input_ids"], list):
                input_ids.append(instance["input_ids"])
            else:
                input_ids += instance["input_ids"]
            if not isinstance(instance["labels"], list):
                labels.append(instance["labels"])
            else:
                labels += instance["labels"]
            # Note (kentang-mit@: we do not directly push tensors to
            # images, but list of tensors.
            if instance.get("image") is not None:
                cur_image = instance["image"]
                # assert len(cur_image.shape) == 4
                # n_images, 3, size, size
                if not isinstance(instance["input_ids"], list):
                    assert len(cur_image.shape) == 4
                    # datasets other than coyo, not packing >1 samples together
                    images.append(cur_image)
                else:
                    # coyo-like datasets
                    images.extend(cur_image)
            else:
                images.append([])
        # kentang-mit@: we need to make sure these two lists have
        # the same length. We will use input_ids to filter out images corresponding
        # to truncated <image> tokens later.
        for _images, _input_ids in zip(images, input_ids):
            assert (
                len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()
            ), f"Number mismatch between images and placeholder image tokens in 'len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()'.\
                Expect to have {len(_images)} images but only found {(_input_ids == IMAGE_TOKEN_INDEX).sum().item()} images in tokens. \
                Error input_ids: {_input_ids} {self.tokenizer.decode([x if x != -200 else 200 for x in _input_ids])}"

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        # print(input_ids.shape, labels.shape)
        new_images = []
        # kentang-mit@: it is possible that some <image> tokens get removed
        # after truncation. It is important to also remove corresponding images.
        # otherwise, text and image will mismatch in the model.
        for ix in range(len(input_ids)):
            num_images = (input_ids[ix] == IMAGE_TOKEN_INDEX).sum().item()
            cur_images = images[ix]
            cur_images = cur_images[:num_images]
            if len(cur_images) > 0:
                new_images.append(cur_images)
        if len(new_images) > 0:
            batch["images"] = torch.cat(new_images, dim=0)
        else:
            # the entire batch is text-only
            if hasattr(self.data_args.image_processor, "crop_size"):
                crop_size = self.data_args.image_processor.crop_size
            else:
                crop_size = self.data_args.image_processor.size
            # we still need 1 dummy image for the vision tower
            batch["images"] = torch.zeros(1, 3, crop_size["height"], crop_size["width"])

        raw_prioprio_inputs = []
        raw_action_labels = []
        raw_action_masks = []

        raw_prioprio_inputs_2d = []
        raw_prioprio_inputs_3d = []
        raw_prioprio_inputs_rot = []
        raw_prioprio_inputs_handdof = []
        raw_prioprio_inputs_hand_finger_tip = []
        raw_ee_movement_mask = []
        for instance in instances:
            if "raw_action_label" in instance:
                raw_action_labels.append(instance["raw_action_label"])
            if "raw_action_mask" in instance:
                raw_action_masks.append(instance["raw_action_mask"])
            if "proprio_input" in instance:
               raw_prioprio_inputs.append(instance["proprio_input"])
            if "proprio_input_2d" in instance:
               raw_prioprio_inputs_2d.append(instance["proprio_input_2d"])
            if "proprio_input_3d" in instance:
               raw_prioprio_inputs_3d.append(instance["proprio_input_3d"])
            if "proprio_input_rot" in instance:
               raw_prioprio_inputs_rot.append(instance["proprio_input_rot"])
            if "proprio_input_handdof" in instance:
               raw_prioprio_inputs_handdof.append(instance["proprio_input_handdof"])
            if "proprio_input_hand_finger_tip" in instance:
               raw_prioprio_inputs_hand_finger_tip.append(
                  instance["proprio_input_hand_finger_tip"]
                )
            if "ee_movement_mask" in instance:
                raw_ee_movement_mask.append(instance["ee_movement_mask"])
        if len(raw_action_labels) > 0:
            batch["raw_action_labels"] = torch.cat(raw_action_labels, dim=0)
            batch["raw_action_masks"] = torch.cat(raw_action_masks, dim=0)
            batch["raw_proprio_inputs"] = torch.cat(raw_prioprio_inputs, dim=0)
            batch["raw_proprio_inputs_2d"] = torch.cat(raw_prioprio_inputs_2d, dim=0)
            batch["raw_proprio_inputs_3d"] = torch.cat(raw_prioprio_inputs_3d, dim=0)
            batch["raw_proprio_inputs_rot"] = torch.cat(raw_prioprio_inputs_rot, dim=0)
            batch["raw_proprio_inputs_handdof"] = torch.cat(raw_prioprio_inputs_handdof, dim=0)
            batch["raw_proprio_inputs_hand_finger_tip"] = torch.cat(
                raw_prioprio_inputs_hand_finger_tip, dim=0
            )
            batch["raw_ee_movement_masks"] = torch.cat(raw_ee_movement_mask, dim=0)
        return batch


@dataclass
class DataCollatorForSupervisedDatasetSeqParallel:
    """Collate examples for supervised fine-tuning.
    This class is originally implemented by the LLaVA team and
    modified by Haotian Tang."""

    tokenizer: transformers.PreTrainedTokenizer
    data_args: DataArguments
    training_args: TrainingArguments
    sp_degree: int
    sp_rank: int
    ring_degree: int
    ring_type: str

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, images = [], [], []
        for instance in instances:
            if not isinstance(instance["input_ids"], list):
                input_ids.append(instance["input_ids"])
            else:
                input_ids += instance["input_ids"]
            if not isinstance(instance["labels"], list):
                labels.append(instance["labels"])
            else:
                labels += instance["labels"]
            # Note (kentang-mit@: we do not directly push tensors to
            # images, but list of tensors.
            if instance["image"] is not None:
                cur_image = instance["image"]
                assert len(cur_image.shape) == 4
                # n_images, 3, size, size
                if cur_image.shape[0] == 0:
                    warnings.warn("loaded one sample without images.")
                if not isinstance(instance["input_ids"], list):
                    # datasets other than coyo, not packing >1 samples together
                    images.append(cur_image)
                else:
                    # coyo-like datasets
                    images.extend(cur_image.chunk(cur_image.size(0), 0))
            else:
                warnings.warn("loaded one sample without images.")
                images.append([])
        # kentang-mit@: we need to make sure these two lists have
        # the same length. We will use input_ids to filter out images corresponding
        # to truncated <image> tokens later.
        max_num_images = max([len(_images) for _images in images])
        for _images, _input_ids in zip(images, input_ids):
            assert (
                len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()
            ), f"Number mismatch between images and placeholder image tokens in 'len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()'.\
                Expect to have {len(_images)} images but only found {(_input_ids == IMAGE_TOKEN_INDEX).sum().item()} images in tokens. \
                Error input_ids: {_input_ids} {self.tokenizer.decode([x if x != -200 else 200 for x in _input_ids])}"

        # TODO: Remove the hard coding of NUM_TOKENS_PER_IMAGE
        NUM_TOKENS_PER_IMAGE = 196
        if hasattr(self.data_args.image_processor, "crop_size"):
            crop_size = self.data_args.image_processor.crop_size
        else:
            crop_size = self.data_args.image_processor.size

        # Init the padding sample
        seq_id = 0
        while seq_id < len(input_ids):
            # Skip the samples without images
            dummy_image = torch.ones((1, 3, crop_size["height"], crop_size["width"]), device=input_ids[seq_id].device)
            # dummy input_ids include one bos, one image token, and one eos
            dummy_input_ids = torch.zeros_like(input_ids[seq_id][:3])
            dummy_input_ids[0] = self.tokenizer.bos_token_id
            dummy_input_ids[1] = IMAGE_TOKEN_INDEX
            dummy_input_ids[2] = self.tokenizer.eos_token_id
            dummy_labels = copy.deepcopy(dummy_input_ids)
            dummy_labels[:2] = IGNORE_INDEX
            dummy_seqlen = NUM_TOKENS_PER_IMAGE + 2  # TODO: Check the hard coding of 2
            dummy_position_ids = torch.arange(start=0, end=dummy_seqlen, dtype=torch.int32)
            break

        # Sort with the real length of the sequence
        combined = sorted(
            zip(input_ids, labels, images),
            key=lambda x: len(x[2]) * (NUM_TOKENS_PER_IMAGE - 1) + x[0].size(-1),
            reverse=True,  # Start Packing from the sequence with most images.
        )
        sorted_ids, sorted_labels, sorted_images = zip(*combined)
        sorted_ids, sorted_labels, sorted_images = list(sorted_ids), list(sorted_labels), list(sorted_images)
        max_seq_length = self.tokenizer.model_max_length  # len(sorted_ids[0])
        max_sample_len = 0

        batches = []
        label_batches = []
        position_ids = []
        batch_images = []
        seqlens_in_batch = []

        i = 0
        while i < len(sorted_ids):
            current_batch = torch.tensor([], dtype=torch.int32)
            current_label_batch = torch.tensor([], dtype=torch.int32)
            current_position_ids = torch.tensor([], dtype=torch.int32)
            current_batch_images = []
            current_num_images = 0
            current_len = 0
            current_num_samples = 0

            # Pack a few samples into one sample
            while i < len(sorted_ids):
                num_images = (sorted_ids[i] == IMAGE_TOKEN_INDEX).sum().item()
                num_image_tokens_added = num_images * (NUM_TOKENS_PER_IMAGE - 1)
                num_incoming_tokens = sorted_ids[i].size(-1) + num_image_tokens_added

                # Handle RingAttn_Varlen which requires `seqlens_in_batch` should be divisible by `ring_degree`
                if self.ring_degree > 1:
                    RING_PAD_TOKEN_INDEX = 2
                    if self.ring_type == "ring_varlen":
                        if num_incoming_tokens % self.sp_degree != 0:
                            pad_len = self.sp_degree - num_incoming_tokens % self.sp_degree
                            num_incoming_tokens += pad_len
                            # pad `input_ids`
                            pad_tensor = torch.full(
                                (pad_len,), RING_PAD_TOKEN_INDEX, dtype=sorted_ids[i].dtype, device=sorted_ids[i].device
                            )
                            sorted_ids[i] = torch.cat([sorted_ids[i], pad_tensor])

                            # pad `label`
                            pad_label_tensor = torch.full(
                                (pad_len,), IGNORE_INDEX, dtype=sorted_labels[i].dtype, device=sorted_labels[i].device
                            )
                            sorted_labels[i] = torch.cat([sorted_labels[i], pad_label_tensor])
                    elif self.ring_type == "zigzag_ring_varlen":
                        self.zigzag_sp_degree = self.sp_degree * 2
                        if num_incoming_tokens % self.zigzag_sp_degree != 0:
                            pad_len = self.zigzag_sp_degree - num_incoming_tokens % self.zigzag_sp_degree
                            num_incoming_tokens += pad_len
                            # pad `input_ids`
                            pad_tensor = torch.full(
                                (pad_len,), RING_PAD_TOKEN_INDEX, dtype=sorted_ids[i].dtype, device=sorted_ids[i].device
                            )
                            sorted_ids[i] = torch.cat([sorted_ids[i], pad_tensor])

                            # pad `label`
                            pad_label_tensor = torch.full(
                                (pad_len,), IGNORE_INDEX, dtype=sorted_labels[i].dtype, device=sorted_labels[i].device
                            )
                            sorted_labels[i] = torch.cat([sorted_labels[i], pad_label_tensor])
                    else:
                        raise ValueError(f"Invalid ring_type: {self.ring_type}")

                if num_incoming_tokens > max_seq_length:
                    print(
                        f"Warning: Skipping one packed sample with {num_incoming_tokens} tokens,\
                        please consider increase max seq len {max_seq_length}."
                    )
                    i += 1
                    continue

                if (
                    (current_num_images == 0)
                    or (current_num_images < self.sp_degree)
                    or (
                        (current_num_images + num_images <= max_num_images)
                        and (current_len + num_incoming_tokens <= max_sample_len)
                    )
                ) and (current_len + num_incoming_tokens <= max_seq_length):
                    current_num_images += num_images
                    current_len += num_incoming_tokens
                    current_num_samples += 1
                    current_position_ids = torch.cat(
                        (current_position_ids, torch.arange(start=0, end=num_incoming_tokens)), dim=0
                    )
                    current_batch = torch.cat((current_batch, sorted_ids[i]), dim=0)
                    sorted_labels[i][0] = IGNORE_INDEX
                    current_label_batch = torch.cat((current_label_batch, sorted_labels[i]), dim=0)
                    seqlens_in_batch.append(num_incoming_tokens)
                    current_batch_images.extend(sorted_images[i])
                    i += 1
                    assert current_num_images == len(current_batch_images)
                else:
                    break

            # Padding the sample with the dummy image sample, if there are no enough images
            MAX_RETRY = self.sp_degree
            num_retry = 0
            while current_num_images < self.sp_degree and current_len < max_seq_length and num_retry <= MAX_RETRY:
                current_num_images += dummy_image.size(0)
                current_len += dummy_seqlen
                current_num_samples += 1
                current_position_ids = torch.cat((current_position_ids, dummy_position_ids), dim=0)
                current_batch = torch.cat((current_batch, dummy_input_ids), dim=0)
                current_label_batch = torch.cat((current_label_batch, dummy_labels), dim=0)
                seqlens_in_batch.append(dummy_seqlen)
                current_batch_images.extend(dummy_image)
                # We pad from left side to ensure correct grad flow
                # current_batch = torch.cat((dummy_input_ids, current_batch), dim=0)
                # current_label_batch = torch.cat((dummy_labels, current_label_batch), dim=0)
                # seqlens_in_batch.insert(0, dummy_seqlen)
                # current_batch_images = torch.cat((dummy_image, current_batch_images), dim=0)
                num_retry += 1

            # Drop the samples that do not have enough images
            if current_num_images < self.sp_degree:
                print(f"Warning: Skipping one packed sample with {current_num_images} images")
                seqlens_in_batch = seqlens_in_batch[:-current_num_samples]
                continue

            max_sample_len = max(max_sample_len, current_len)
            batches.append(current_batch)
            label_batches.append(current_label_batch)
            position_ids.append(current_position_ids)
            batch_images.append(current_batch_images)

            try:
                assert current_num_images == len(torch.where(current_batch == IMAGE_TOKEN_INDEX)[0].tolist())
            except AssertionError:
                print(f"Error num_images on {self.sp_rank}", current_num_images)
                print("current_batch", current_batch)
                print(
                    f"Error len(torch.where(batches[i] == IMAGE_TOKEN_INDEX)[0].tolist() on {self.sp_rank}:",
                    len(torch.where(current_batch == IMAGE_TOKEN_INDEX)[0].tolist()),
                )
                print(f"Error len(current_batch_images) on {self.sp_rank}:", len(current_batch_images))
                raise AssertionError

        # Split for sequence parallelism
        for i in range(len(batches)):
            image_token_indices = torch.where(batches[i] == IMAGE_TOKEN_INDEX)[0].tolist()
            image_ids = torch.arange(0, len(image_token_indices), dtype=torch.int32)
            batches[i] = extract_local_input_ids(
                batches[i], image_token_indices, self.sp_rank, self.sp_degree, self.tokenizer.bos_token_id
            )
            label_batches[i] = extract_local_input_ids(
                label_batches[i], image_token_indices, self.sp_rank, self.sp_degree, self.tokenizer.bos_token_id
            )
            batch_images[i] = torch.concat(
                extract_local_from_list(batch_images[i], self.sp_rank, self.sp_degree), dim=0
            )
            H, W = batch_images[i].size(-2), batch_images[i].size(-1)
            batch_images[i] = batch_images[i].reshape(-1, 3, W, H)
            num_images = len(batch_images[i])

            try:
                assert num_images == len(torch.where(batches[i] == IMAGE_TOKEN_INDEX)[0].tolist())
            except AssertionError:
                print(f"Error num_images on {self.sp_rank}", num_images)
                print("batches[i]", batches[i])
                print(
                    f"Error len(torch.where(batches[i] == IMAGE_TOKEN_INDEX)[0].tolist() on {self.sp_rank}:",
                    len(torch.where(batches[i] == IMAGE_TOKEN_INDEX)[0].tolist()),
                )
                print(f"Error batch_images[i] on {self.sp_rank}:", batch_images[i].shape)
                raise AssertionError
            position_ids[i] = extract_local_position_ids(
                position_ids[i], image_token_indices, image_ids, self.sp_rank, self.sp_degree, NUM_TOKENS_PER_IMAGE - 1
            )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            batches, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(label_batches, batch_first=True, padding_value=IGNORE_INDEX)
        seqlens_in_batch = [torch.tensor(x) for x in seqlens_in_batch]
        seqlens_in_batch = torch.stack(seqlens_in_batch, axis=0)
        seqlens_in_batch = seqlens_in_batch.flatten()
        position_ids = torch.nn.utils.rnn.pad_sequence(position_ids, batch_first=True, padding_value=-1)

        if batch_images:
            flat_batch_images = torch.concat(batch_images, dim=0)
        else:
            flat_batch_images = None
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            # notice that we inject attention mask here
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            seqlens_in_batch=seqlens_in_batch,
            images=flat_batch_images,
            position_ids=position_ids,
        )

        return batch


from human_plan.preprocessing.preprocessing import (
  preprocess_vla,
  preprocess_vla_qa,
  preprocess_multimodal_vla,
  # preprocess_language_instruction
)
from human_plan.preprocessing.prompting_format import (
  preprocess_language_instruction,
#   preprocess_language_instruction_qa,
)
# from human_plan.preprocessing.preprocessing import preprocess_multimodal_vla
from human_plan.utils.action_tokenizer import ActionTokenizer
# from llava.train.train import rank0_print
from human_plan.utils.transformation import ee_from_frame_to_frame


def deserialize_item(value):
  """Deserialize a single item based on its type."""
  try:
    # Try to load using pickle (for numpy arrays, etc.)
    return pickle.loads(value)
  except (pickle.UnpicklingError, EOFError, ValueError):
    # If it's not pickle, try as string or image
    try:
      return value.decode('utf-8')
    except UnicodeDecodeError:
      # If it's not a string, assume it's an image
      image_array = np.frombuffer(value, dtype=np.uint8)
      return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def reconstruct_dict(txn, sample_id):
  """Reconstruct a nested dictionary from LMDB data."""
  result_dict = {}

  rgb_key = sample_id + "/" + "rgb_obs"
  result_dict["rgb_obs"] = deserialize_item(
      txn.get(rgb_key.encode('utf-8'))
  )
  language_label_key = sample_id + "/" + "language_label"
  result_dict["language_label"] = deserialize_item(
      txn.get(language_label_key.encode('utf-8'))
  )
  hand_structure = [
      "current_left_hand_pose",
      "current_right_hand_pose",
      "future_left_hand_pose",
      "future_right_hand_pose"
  ]
  per_hand_structure = [
      "active_flag",
      "hand_pose",
      "hand_trans_cam_frame",
      "valid_state",
      "track_state"
  ]
  for hand_s in hand_structure:
    # result_dict
    result_dict[hand_s] = {}
    for per_hand_s in per_hand_structure:
      key = sample_id + "/" + hand_s + "/" + per_hand_s
      result_dict[hand_s][per_hand_s] = deserialize_item(
        txn.get(key.encode('utf-8'))
      )
  return result_dict


@dataclass
class LazyVLAHoloAssistDataset(Dataset):
  def __init__(
      self,
      data_path: str,
      tokenizer: transformers.PreTrainedTokenizer,
      image_folder: str,
      data_args: DataArguments,
      training_args: TrainingArguments
  ):
    super(LazyVLAHoloAssistDataset, self).__init__()
    with open(data_path, "rb") as f:
      data_list = pickle.load(f)

    print("Formatting inputs...Skip in lazy mode")
    self.tokenizer = tokenizer
    self.action_tokenizer = data_args.action_tokenizer
    self.data_list = data_list
    self.data_args = data_args
    self.image_folder = image_folder
    self.training_args = training_args

    # Delay loading LMDB data until after initialization to avoid "can't pickle Environment Object error"
    self.env = None
    self.txn = None
    
    self.picked_index = torch.Tensor([1, 5, 10, 15, 20, 25]).long()
    
  def __len__(self):
    return len(self.data_list)

  @property
  def lengths(self):
    length_list = []
    for sample in self.data_list:
      img_tokens = 128
      length_list.append(len(sample[1]) + img_tokens)
    return length_list

  @property
  def modality_lengths(self):
    length_list = []
    for sample in self.data_list:
      cur_len = len(sample[1])
      cur_len = cur_len
      length_list.append(cur_len)
    return length_list

  def _init_db(self):
      self.env = lmdb.open(self.image_folder, subdir=os.path.isdir(self.image_folder),
          readonly=True, lock=False,
          readahead=False, meminit=False)
      self.txn = self.env.begin()

  def __getitem__(self, i) -> Dict[str, torch.Tensor]:
    # Delay loading LMDB data until after initialization
    if self.env is None:
        self._init_db()
        
    sample_id = self.data_list[i][0]
    data_dict = reconstruct_dict(
      self.txn, sample_id
    )
    language_instruction = f"<image>\nWhat should the robot do to: {data_dict['language_label']} ? A: "

    if self.data_args.ignore_language:
      language_instruction = f"<image>\nWhat should the robot do to finish the task ? A: "

    # image_processor = self.data_args.image_processor
  
    image = torch.stack([process_image_ndarray(
        data_dict["rgb_obs"], self.data_args,
    )], dim=0)

    future_idx = self.data_args.future_index

    hands = ["left_hand", "right_hand"]

    current_input_masks = []

    hand_inputs = []
    # hand_inputs_scaled = []
    for hand in hands:
      valid_mask = torch.tensor(
          data_dict["current_" + hand + "_pose"]["valid_state"]
      ).unsqueeze(-1) * torch.tensor(
          data_dict["current_" + hand + "_pose"]["track_state"]
      ).unsqueeze(-1)

      single_hand_trans = torch.tensor(
          data_dict["current_" + hand + "_pose"]["hand_trans_cam_frame"]
      ).reshape(-1, 26, 3) * valid_mask

      hand_inputs.append((
          single_hand_trans
      ).reshape(-1))

      # for key in hand_info_masks:
      current_input_masks.append(
          valid_mask.reshape(1, -1, 1)
      )

    hand_inputs = torch.concat(hand_inputs)
    # hand_inputs_scaled = torch.concat(hand_inputs_scaled)
    hand_input_masks = torch.concat(
        current_input_masks, dim=0
    )

    future_idx = self.data_args.future_index

    hand_labels = []
    hand_label_masks = []
    for hand in hands:
      # for key in hand_infos:
      single_hand_label_trans = torch.tensor(
          data_dict[
              "future_" + hand + "_pose"
          ]["hand_trans_cam_frame"][future_idx]
      ).reshape(-1, 26, 3)

      valid_mask = torch.tensor(
          data_dict["future_" + hand + "_pose"]["valid_state"][future_idx]
      ).unsqueeze(-1) * torch.tensor(
          data_dict["future_" + hand + "_pose"]["track_state"][future_idx]
      ).unsqueeze(-1)

      hand_labels.append((
          single_hand_label_trans * valid_mask
      ).reshape(-1))
      # for key in hand_info_masks:
      hand_label_masks.append(
          valid_mask.reshape(1, -1, 1)
      )

    # Raw Future
    hand_labels = torch.concat(hand_labels)

    # Filter the not valid data points
    hand_labels = torch.where(
        torch.isfinite(hand_labels),
        hand_labels, torch.tensor(0.0)
    )
    hand_inputs = torch.where(
        torch.isfinite(hand_inputs),
        hand_inputs, torch.tensor(0.0)
    )                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

    hand_labels = torch.clamp(
        hand_labels, -1, 1
    )
    hand_inputs = torch.clamp(
        hand_inputs, -1, 1
    )

    # Use Diff as label
    hand_labels = hand_labels - hand_inputs

    hand_label_masks = torch.concat(
        hand_label_masks, dim=0
    )
    # 2, 26, 3
    # 256, 2, 26, 3
    hand_label_masks = torch.repeat_interleave(
        (hand_label_masks * hand_input_masks).bool(),
        3, dim=-1
    ).reshape(-1)

    picked_index = self.picked_index.to(hand_inputs.device)
    # print(hand_inputs.shape)
    # print(hand_labels.shape)
    # print(hand_label_masks.shape)
    hand_inputs = hand_inputs.reshape(2, 26, 3)[:, picked_index].reshape(2 * 6 * 3)
    hand_labels = hand_labels.reshape(2, 26, 3)[:, picked_index].reshape(2 * 6 * 3)
    hand_label_masks = hand_label_masks.reshape(2, 26, 3)[:, picked_index].reshape(2 * 6 * 3)


    language_instruction = preprocess_multimodal_vla(
      language_instruction,
      self.data_args
    )

    data_dict = preprocess_vla(
      language_instruction,
      hand_labels,
      hand_label_masks,
      self.action_tokenizer,
      self.tokenizer,
      mask_input=self.data_args.mask_input,
      mask_ignore=self.data_args.mask_ignore,
      raw_action_label=self.data_args.raw_action_label,
      traj_action_output_dim=self.data_args.traj_action_output_dim,
      input_placeholder_diff_index=self.data_args.input_placeholder_diff_index
    )
    data_dict["image"] = image
    return data_dict


def to_ndarray(sample):
  skip_keys = [
    "rgb_obs", "language_", "frame_count",
    "raw_width","raw_height",
    "raw_w","raw_h",
  ]
  for key in sample.keys():
    if key in skip_keys or "language_" in key:
      continue
    try:
      sample[key] = [v if v is not None else 0 for v in sample[key]]
    except Exception:
      print(key)
      exit()

    sample[key] = [v if v is not None else 0 for v in sample[key]]
    sample[key] = np.array(sample[key])
    # sample[key][~np.isfinite(sample[key])] = 0


@dataclass
class LazyVLAHoloAssistHFDataset(Dataset):
  def __init__(
      self,
      data_path: str,
      tokenizer: transformers.PreTrainedTokenizer,
      image_folder: str,
      data_args: DataArguments,
      training_args: TrainingArguments,
      data_skip: int
  ):
    super().__init__()

    if os.path.isfile(os.path.join(data_path, "dataset_info.json")):
        self.dataset = datasets.load_from_disk(data_path)
    else:
        subdataset_list = [
            os.path.join(data_path, p) for p in os.listdir(data_path)
        ]
        subdatasets = [datasets.load_from_disk(path) for path in subdataset_list]
        self.dataset = concatenate_datasets(subdatasets)

    self.tokenizer = tokenizer
    self.action_tokenizer = data_args.action_tokenizer
    self.data_args = data_args
    self.training_args = training_args
    self.data_skip = data_skip

    self.picked_index = torch.Tensor([1, 5, 10, 15, 20, 25]).long()
    
  def __len__(self):
    return len(self.dataset) // self.data_skip

  def __getitem__(self, i) -> Dict[str, torch.Tensor]:
    sample = self.dataset[i * self.data_skip]

    to_ndarray(sample)

    language_instruction = f"<image>\nWhat should the robot do to: {sample['language_label']} ? A: "
    if self.data_args.ignore_language:
      language_instruction = f"<image>\nWhat should the robot do to finish the task ? A: "

    # image_processor = self.data_args.image_processor
  
    image = torch.stack([process_image_bytes(
        sample["rgb_obs"], self.data_args,
    )], dim=0)

    future_idx = self.data_args.future_index

    hands = ["left_hand", "right_hand"]

    current_input_masks = []

    relative_hand_pose = self.data_args.relative_hand_pose

    hand_inputs = []
    # hand_inputs_scaled = []
    for hand in hands:
      valid_mask = torch.tensor(
          sample["current_" + hand + "_pose/valid_state"].reshape(-1, 26)
      ).unsqueeze(-1) * torch.tensor(
          sample["current_" + hand + "_pose/track_state"].reshape(-1, 26)
      ).unsqueeze(-1)

      single_hand_trans = torch.tensor(
          sample["current_" + hand + "_pose/hand_trans_cam_frame"]
      ).reshape(-1, 26, 3)

      if relative_hand_pose:
        single_hand_trans[:, 1:, :] = (
          single_hand_trans[:, 2:, :] - single_hand_trans[:, 0:1, :]
        )

      hand_inputs.append((
          single_hand_trans * valid_mask
      ).reshape(-1))

      # for key in hand_info_masks:
      current_input_masks.append(
          valid_mask.reshape(1, -1, 1)
      )

    hand_inputs = torch.concat(hand_inputs)
    # hand_inputs_scaled = torch.concat(hand_inputs_scaled)
    hand_input_masks = torch.concat(
        current_input_masks, dim=0
    )

    future_idx = self.data_args.future_index

    hand_labels = []
    hand_label_masks = []

    for hand in hands:
      single_hand_label_trans = torch.tensor(
          sample[
              "future_" + hand + "_pose/hand_trans_cam_frame"
          ].reshape(-1, 26,3)[future_idx]
      ).reshape(-1, 26, 3)

      if relative_hand_pose:
        single_hand_label_trans[:, 1:, :] = (
          single_hand_label_trans[:, 2:, :] - single_hand_label_trans[:, 0:1, :]
        )

      valid_mask = torch.tensor(
          sample["future_" + hand + "_pose/valid_state"].reshape(-1, 26)[future_idx]
      ).unsqueeze(-1) * torch.tensor(
          sample["future_" + hand + "_pose/track_state"].reshape(-1, 26)[future_idx]
      ).unsqueeze(-1)

      hand_labels.append((
          single_hand_label_trans * valid_mask
      ).reshape(-1))
      # for key in hand_info_masks:
      hand_label_masks.append(
          valid_mask.reshape(1, -1, 1)
      )

    # Raw Future
    hand_labels = torch.concat(hand_labels)

    # Filter the not valid data points
    hand_labels = torch.where(
        torch.isfinite(hand_labels),
        hand_labels, torch.tensor(0.0)
    )

    hand_inputs = torch.where(
        torch.isfinite(hand_inputs),
        hand_inputs, torch.tensor(0.0)
    )

    hand_labels = torch.clamp(
        hand_labels, -1, 1
    )
    hand_inputs = torch.clamp(
        hand_inputs, -1, 1
    )

    # Use Diff as label
    hand_labels = hand_labels - hand_inputs

    hand_label_masks = torch.concat(
        hand_label_masks, dim=0
    )
    # 2, 26, 3
    # 256, 2, 26, 3
    hand_label_masks = torch.repeat_interleave(
        (hand_label_masks * hand_input_masks).bool(),
        3, dim=-1
    ).reshape(-1)

    picked_index = self.picked_index.to(hand_inputs.device)
    # print(hand_inputs.shape)
    # print(hand_labels.shape)
    # print(hand_label_masks.shape)
    hand_inputs = hand_inputs.reshape(2, 26, 3)[:, picked_index].reshape(2 * 6 * 3)
    hand_labels = hand_labels.reshape(2, 26, 3)[:, picked_index].reshape(2 * 6 * 3)
    hand_label_masks = hand_label_masks.reshape(2, 26, 3)[:, picked_index].reshape(2 * 6 * 3)


    language_instruction = preprocess_multimodal_vla(
      language_instruction,
      self.data_args
    )

    data_dict = preprocess_vla(
      language_instruction,
      hand_labels,
      hand_label_masks,
      self.action_tokenizer,
      self.tokenizer,
      mask_input=self.data_args.mask_input,
      mask_ignore=self.data_args.mask_ignore,
      raw_action_label=self.data_args.raw_action_label,
      traj_action_output_dim=self.data_args.traj_action_output_dim,
      input_placeholder_diff_index=self.data_args.input_placeholder_diff_index
    )
    data_dict["image"] = image
    return data_dict


@dataclass
class LazyVLAHoloAssistHFEEDataset(LazyVLAHoloAssistHFDataset):
  # ee_picked_index = torch.Tensor([1, 5, 10, 15, 20, 25]).long()
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def __getitem__(self, i) -> Dict[str, torch.Tensor]:
    sample = self.dataset[i * self.data_skip]

    to_ndarray(sample)

    language_instruction = f"<image>\nWhat should the robot do to: {sample['language_label']} ? A: "
    if self.data_args.ignore_language:
      language_instruction = f"<image>\nWhat should the robot do to finish the task ? A: "

    image = torch.stack([process_image_bytes(
        sample["rgb_obs"], self.data_args,
    )], dim=0)

    future_idx = self.data_args.future_index

    hands = ["left_hand", "right_hand"]

    current_input_masks = []

    relative_hand_pose = self.data_args.relative_hand_pose

    hand_inputs = []
    for hand in hands:
      valid_mask = torch.tensor(
          sample["current_" + hand + "_pose/valid_state"].reshape(-1, 26)
      ).unsqueeze(-1) * torch.tensor(
          sample["current_" + hand + "_pose/track_state"].reshape(-1, 26)
      ).unsqueeze(-1)

      single_hand_trans = torch.tensor(
          sample["current_" + hand + "_pose/hand_trans_cam_frame"]
      ).reshape(-1, 26, 3)

      hand_inputs.append((
          single_hand_trans * valid_mask
      ).reshape(-1))

      # for key in hand_info_masks:
      current_input_masks.append(
          valid_mask.reshape(1, -1, 1)
      )

    hand_inputs = torch.concat(hand_inputs)
    hand_input_masks = torch.concat(
        current_input_masks, dim=0
    )

    future_idx = self.data_args.future_index

    hand_labels = []
    hand_label_masks = []

    for hand in hands:
      single_hand_label_trans = torch.tensor(
          sample[
              "future_" + hand + "_pose/hand_trans_cam_frame"
          ].reshape(-1, 26,3)[future_idx]
      ).reshape(-1, 26, 3)

      if relative_hand_pose:
        single_hand_label_trans[:, 1:, :] = (
          single_hand_label_trans[:, 1:, :] - single_hand_label_trans[:, 0:1, :]
        )

      valid_mask = torch.tensor(
          sample["future_" + hand + "_pose/valid_state"].reshape(-1, 26)[future_idx]
      ).unsqueeze(-1) * torch.tensor(
          sample["future_" + hand + "_pose/track_state"].reshape(-1, 26)[future_idx]
      ).unsqueeze(-1)

      hand_labels.append((
          single_hand_label_trans * valid_mask
      ).reshape(-1))
      # for key in hand_info_masks:
      hand_label_masks.append(
          valid_mask.reshape(1, -1, 1)
      )

    # Raw Future
    hand_labels = torch.concat(hand_labels)
    # Filter the not valid data points
    hand_labels = torch.where(
        torch.isfinite(hand_labels),
        hand_labels, torch.tensor(0.0)
    )
    hand_inputs = torch.where(
        torch.isfinite(hand_inputs),
        hand_inputs, torch.tensor(0.0)
    )

    hand_labels = torch.clamp(
        hand_labels, -1, 1
    )
    hand_inputs = torch.clamp(
        hand_inputs, -1, 1
    )

    # Use Diff as label
    hand_labels = hand_labels - hand_inputs

    hand_label_masks = torch.concat(
        hand_label_masks, dim=0
    )
    # 2, 26, 3
    # 256, 2, 26, 3
    hand_label_masks = torch.repeat_interleave(
        (hand_label_masks * hand_input_masks).bool(),
        3, dim=-1
    ).reshape(-1)

    hand_inputs = hand_inputs.reshape(2, 26, 3)[:, 1, :].reshape(2 * 1 * 3)
    hand_labels = hand_labels.reshape(2, 26, 3)[:, 1, :].reshape(2 * 1 * 3)
    hand_label_masks = hand_label_masks.reshape(2, 26, 3)[:, 1, :].reshape(2 * 1 * 3)

    # print(hand_inputs)
    # print(hand_labels)
    # print(hand_label_masks)

    language_instruction = preprocess_multimodal_vla(
      language_instruction,
      self.data_args
    )

    data_dict = preprocess_vla(
      language_instruction,
      hand_labels,
      hand_label_masks,
      self.action_tokenizer,
      self.tokenizer,
      mask_input=self.data_args.mask_input,
      mask_ignore=self.data_args.mask_ignore,
      raw_action_label=self.data_args.raw_action_label,
      traj_action_output_dim=self.data_args.traj_action_output_dim,
      input_placeholder_diff_index=self.data_args.input_placeholder_diff_index
    )
    data_dict["image"] = image
    return data_dict


from human_plan.utils.hand_dof import (
  compute_hand_dof_5dim,
  convert_full_mano_to_pca_dof
)
from human_plan.utils.normalization import normalize_item

from scipy.spatial.transform import Rotation as R
from llava.data.utils import norm_hand_dof

@dataclass
class LazyVLAHFAbsDataset(Dataset):
  def __init__(
      self,
      data_path: str,
      tokenizer: transformers.PreTrainedTokenizer,
      image_folder: str,
      data_args: DataArguments,
      training_args: TrainingArguments,
      data_skip: int
  ):
    super().__init__()

    if os.path.isfile(os.path.join(data_path, "dataset_info.json")):
      self.dataset = datasets.load_from_disk(data_path)
    else:
      subdataset_list = [
          os.path.join(data_path, p) for p in os.listdir(data_path)
      ]
      from datasets import concatenate_datasets
      subdatasets = [datasets.load_from_disk(path) for path in subdataset_list]
      self.dataset = concatenate_datasets(subdatasets)

    self.tokenizer = tokenizer
    self.action_tokenizer = data_args.action_tokenizer
    self.data_args = data_args
    self.training_args = training_args
    self.data_skip = data_skip

    with open(data_args.image_mapping_path, "rb") as f:
      self.image_mapping_dict = pickle.load(f)

    self.img_dataset = datasets.load_from_disk(image_folder)

    assert self.data_args.use_mano
    self.hand_dof_dim = 15

    self.init_dataset_specific_info()

  def __len__(self):
    return len(self.dataset) // self.data_skip

  def init_dataset_specific_info(self):
    raise NotImplementedError

  def normalize_2d(self, ee_2d, sample):
    ee_2d_normalization = torch.Tensor([
      self.raw_image_width, self.raw_image_height
    ]).reshape(1, 1, 2)
    ee_2d  = ee_2d / ee_2d_normalization
    return ee_2d

  def get_current_hand_data(self, sample, hand) -> Sequence[torch.Tensor]:
    raise NotImplementedError

  def get_future_hand_data(
    self, sample, hand, future_step, future_idx
  ) -> Sequence[torch.Tensor]:
    raise NotImplementedError

  def get_current_language_label(self, sample):
    return sample["language_label"]

  def process_hand_label_masks(self, raw_hand_label_masks):
    dof_hand_label_masks = torch.repeat_interleave(
        raw_hand_label_masks.bool().squeeze(-1),
        self.hand_dof_dim, dim=-1
    ).reshape(-1, 2, self.hand_dof_dim)

    ee_hand_label_masks = torch.repeat_interleave(
        raw_hand_label_masks.bool(),
        3, dim=-1
    ).reshape(-1, 2, 3)

    rot_hand_label_masks = torch.repeat_interleave(
        raw_hand_label_masks.bool(),
        3, dim=-1
    ).reshape(-1, 2, 3)

    ee_2d_hand_label_masks = ee_hand_label_masks[..., :2]
    return ee_hand_label_masks, ee_2d_hand_label_masks, dof_hand_label_masks, rot_hand_label_masks

  def __getitem__(self, i) -> Dict[str, torch.Tensor]:
    sample = self.dataset[i * self.data_skip]

    to_ndarray(sample)

    image_list = []
    valid_his_len = 0
    seq_name = ''.join(sample["seq_name"])
    try:
      current_image_file_index = self.image_mapping_dict[seq_name][sample["frame_count"]]
    except Exception:
      current_image_file_index = None
      print(seq_name, sample["frame_count"])

    assert self.data_args.add_his_img_skip % self.frame_count_scaler == 0
    # assert self.data_args.add_his_img_skip % self.frame_count_scaler == 0
    for i in range(self.data_args.add_his_obs_step):
      # Need to fxxking align with dataset
      query_idx = max(
         0, 
        sample["frame_count"] - \
          (i + 1) * self.data_args.add_his_img_skip * \
          self.frame_count_scaler_up // self.frame_count_scaler
      )
      if not query_idx in self.image_mapping_dict[seq_name]:
          continue
      c_his_image_file_index = self.image_mapping_dict[
          seq_name
      ][query_idx]
      if current_image_file_index is None:
        current_image_file_index = c_his_image_file_index 
      image_list.append(process_image_bytes(
          self.img_dataset[c_his_image_file_index]["rgb_obs"],
          self.data_args,
          reverse_channel_order=self.reverse_channel_order
      ))
      valid_his_len += 1
    # print(current_image_file_index)
    image_list.append(process_image_bytes(
        self.img_dataset[current_image_file_index]["rgb_obs"],
        self.data_args,
        reverse_channel_order=self.reverse_channel_order
    ))
    image = torch.stack(image_list, dim=0)

    # future_idx = self.data_args.future_index // self.frame_count_scaler
    # Do not need to do scaler here actually, since already converted the in the future_data part during data preprocessing
    # All should be in the same hz for this part.
    future_idx = self.data_args.future_index

    hands = ["left", "right"]

    current_input_masks = []

    # Input
    ee_3d_inputs = []
    ee_2d_inputs = []
    ee_rot_inputs = []
    hand_dof_inputs = []
    handkp_3d_inputs = []
    hand_finger_tip_3d_inputs = []

    for hand in hands:
      single_hand_current_data = self.get_current_hand_data(
        sample, hand
      )
      # valid_mask, single_hand_trans, single_hand_trans_2d, \
      #   single_hand_pose, single_hand_rot = single_hand_current_data

      valid_mask, single_ee_trans_3d, single_ee_trans_2d, \
        single_ee_rot, single_handkp_3d, single_hand_finger_tip_3d, single_hand_dof = single_hand_current_data


      single_hand_dof = norm_hand_dof(single_hand_dof)
      single_hand_dof[..., self.training_args.hand_loss_dim:] = 0

      ee_3d_inputs.append((
        single_ee_trans_3d * valid_mask
      ).reshape(-1))

      ee_2d_inputs.append((
        single_ee_trans_2d * valid_mask
      ).reshape(-1))

      ee_rot_inputs.append((
        single_ee_rot * valid_mask
      ))

      handkp_3d_inputs.append((
        single_handkp_3d * valid_mask
      ))

      hand_finger_tip_3d_inputs.append((
        single_hand_finger_tip_3d * valid_mask
      ))

      hand_dof_inputs.append((
        single_hand_dof * valid_mask
      ))
      # for key in hand_info_masks:
      current_input_masks.append(
          valid_mask.reshape(1, -1)
      )

    ee_2d_inputs = torch.concat(ee_2d_inputs)
    ee_3d_inputs = torch.concat(ee_3d_inputs)
    ee_rot_inputs = torch.concat(ee_rot_inputs)

    hand_dof_inputs = torch.concat(hand_dof_inputs)
    handkp_3d_inputs = torch.concat(handkp_3d_inputs)
    hand_finger_tip_3d_inputs = torch.concat(hand_finger_tip_3d_inputs)
    # print(hand_finger_tip_3d_inputs.shape)

    # print("EE 2D Input Shape:", ee_2d_inputs.shape)
    # print("EE 3D Input Shape:", ee_3d_inputs.shape)
    # print("EE ROT Input Shape:", ee_rot_inputs.shape)
    # print("Hand KP Input Shape:", handkp_3d_inputs.shape)

    ee_2d_labels = []
    ee_3d_labels = []
    ee_rot_labels = []

    dof_hand_labels = []
    handkp_3d_labels = []

    raw_hand_label_masks = []

    for future_step in range(self.data_args.predict_future_step):
      for hand in hands:
        # Do not check length for now -> make the future length consistent
        future_hand_data = self.get_future_hand_data(
          sample, hand, future_step, future_idx
        )

        valid_mask, future_ee_3d, future_ee_2d, \
          future_hand_pose, future_ee_rot, future_handkp_3d = future_hand_data

        future_hand_pose = norm_hand_dof(future_hand_pose)
        future_hand_pose[..., self.training_args.hand_loss_dim:] = 0

        ee_3d_labels.append((
           future_ee_3d * valid_mask
        ).reshape(1, 3))

        ee_2d_labels.append((
            future_ee_2d * valid_mask
        ).reshape(1, 2))

        assert self.data_args.use_mano
        dof_hand_labels.append((
            future_hand_pose * valid_mask
        ).reshape(1, 15))
        # else:
        #   dof_hand_labels.append((
        #       future_hand_pose * valid_mask
        #   ).reshape(1, 5))

        # Use RotVec
        ee_rot_labels.append((
            future_ee_rot * valid_mask
        ).reshape(1, 3))

        # Fxxk it there is something wrong with the orders fxxk it
        # handkp_3d_labels.append((
        #     future_handkp_3d * valid_mask
        # ).reshape(1, 21, 3))

        raw_hand_label_masks.append(
            valid_mask.reshape(1, -1)
        )

    # Raw Future
    ee_3d_labels = torch.concat(ee_3d_labels, dim=0).reshape(-1, 2, 3)
    ee_2d_labels = torch.concat(ee_2d_labels, dim=0).reshape(-1, 2, 2)
    ee_rot_labels = torch.concat(ee_rot_labels, dim=0).reshape(-1, 2, 3)

    ee_movement_mask_idx = self.data_args.ee_movement_mask_idx
    ee_movement_mask = torch.linalg.norm(
       ee_3d_labels - ee_3d_inputs.reshape(-1, 2, 3), dim=-1
    )[ee_movement_mask_idx] >= 0.02

    # Fxxk it there is something wrong with the orders fxxk it
    # handkp_3d_labels = torch.concat(handkp_3d_labels, dim=0).reshape(-1, 2, 21, 3)

    ee_2d_labels = self.normalize_2d(
      ee_2d_labels, sample
    )
    ee_2d_labels = ee_2d_labels.clamp(0, 1)

    dof_hand_labels = torch.concat(
      dof_hand_labels, dim=0
    ).reshape(-1, 2, self.hand_dof_dim)

    # Filter the not valid data points
    ee_2d_labels = torch.where(
        torch.isfinite(ee_2d_labels),
        ee_2d_labels, torch.tensor(0.0)
    )
    ee_rot_labels = torch.where(
        torch.isfinite(ee_rot_labels),
        ee_rot_labels, torch.tensor(0.0)
    )
    ee_3d_labels = torch.where(
        torch.isfinite(ee_3d_labels),
        ee_3d_labels, torch.tensor(0.0)
    )
    dof_hand_labels = torch.where(
        torch.isfinite(dof_hand_labels),
        dof_hand_labels, torch.tensor(0.0)
    )

    # Fxxk it there is something wrong with the orders fxxk it
    # handkp_3d_labels = torch.where(
    #     torch.isfinite(handkp_3d_labels),
    #     handkp_3d_labels, torch.tensor(0.0)
    # )

    if self.data_args.use_relative_label:
      ee_2d_labels = ee_2d_labels - ee_2d_inputs
      ee_3d_labels = ee_3d_labels - ee_3d_inputs

    # -1, 1
    raw_hand_label_masks = torch.concat(
        raw_hand_label_masks, dim=0
    )

    # ee_3d_label_masks, ee_2d_label_masks, \
    #   dof_hand_label_masks, ee_rot_label_masks \
    #   handkp_3d_label_masks = self.process_hand_label_masks(
    #   raw_hand_label_masks
    # )

    ee_3d_label_masks, ee_2d_label_masks, \
      dof_hand_label_masks, ee_rot_label_masks  = self.process_hand_label_masks(
      raw_hand_label_masks
    )

    assert self.data_args.merge_hand
    ee_2d_labels = ee_2d_labels.reshape(-1, 4)
    ee_3d_labels = ee_3d_labels.reshape(-1, 6)
    ee_rot_labels = ee_rot_labels.reshape(-1, 6)

    dof_hand_labels = dof_hand_labels.reshape(-1, 2 * self.hand_dof_dim)

    ee_2d_label_masks = ee_2d_label_masks.reshape(-1, 4)
    ee_3d_label_masks = ee_3d_label_masks.reshape(-1, 6)
    ee_rot_label_masks = ee_rot_label_masks.reshape(-1, 6)
    dof_hand_label_masks = dof_hand_label_masks.reshape(-1, 2 * self.hand_dof_dim)

    label_list = []
    mask_list = []
    if self.data_args.include_2d_label:
      label_list.append(ee_2d_labels)
      mask_list.append(ee_2d_label_masks)
  
    label_list += [ee_3d_labels, dof_hand_labels]
    mask_list += [ee_3d_label_masks, dof_hand_label_masks]

    if self.data_args.include_rot_label:
      label_list.append(ee_rot_labels)
      mask_list.append(ee_rot_label_masks)

    if self.data_args.include_handkp:
      raise NotImplementedError
      label_list.append(handkp_3d_labels)
      # mask_list.append(handkp_3d_label_masks)

    hand_labels = torch.cat(label_list, dim=-1).reshape(-1)
    hand_label_masks = torch.cat(mask_list, dim=-1).reshape(-1)

    current_language_label = self.get_current_language_label(sample)

    language_instruction = preprocess_language_instruction(
      current_language_label, valid_his_len, self.data_args
    )

    language_instruction = preprocess_multimodal_vla(
      language_instruction,
      self.data_args
    )

    # print("EE 2D Input Shape:", ee_2d_inputs.shape)
    # print("EE 3D Input Shape:", ee_3d_inputs.shape)
    # print("EE ROT Input Shape:", ee_rot_inputs.shape)
    # print("Hand KP Input Shape:", handkp_3d_inputs.shape)

    # proprio_input = {
    #    "proprio_input": torch.cat([
    #       ee_2d_inputs.reshape(-1, 4),
    #       ee_3d_inputs.reshape(-1, 6),
    #       ee_rot_inputs.reshape(-1, 6)
    #    ], dim=-1)
    # }


    if self.data_args.input_hand_dof:
      proprio_input = torch.cat([
            ee_2d_inputs.reshape(-1, 4),
            ee_3d_inputs.reshape(-1, 6),
            ee_rot_inputs.reshape(-1, 6),
            hand_dof_inputs.reshape(-1, 30)
      ], dim=-1)
    else:
      proprio_input = torch.cat([
            ee_2d_inputs.reshape(-1, 4),
            ee_3d_inputs.reshape(-1, 6),
            ee_rot_inputs.reshape(-1, 6)
      ], dim=-1)


    # proprio_input = torch.repeat_interleave(
    #    proprio_input, self.data_args.predict_future_step, dim=0
    # )

    # print("Proprio Input Shape:", proprio_input.shape)
    data_dict = preprocess_vla(
      language_instruction,
      proprio_input,
      hand_labels,
      hand_label_masks,
      self.action_tokenizer,
      self.tokenizer,
      mask_input=self.data_args.mask_input,
      mask_ignore=self.data_args.mask_ignore,
      raw_action_label=self.data_args.raw_action_label,
      traj_action_output_dim=self.data_args.traj_action_output_dim,
      input_placeholder_diff_index=self.data_args.input_placeholder_diff_index,
      sep_query_token=self.data_args.sep_query_token,
      language_response=None,
      include_response=self.data_args.include_response,
      include_repeat_instruction=self.data_args.include_repeat_instruction,
      raw_language_label=current_language_label
    )
    raw_proprio_inputs = {
      "proprio_input_2d": ee_2d_inputs.reshape(-1, 4),
      "proprio_input_3d": ee_3d_inputs.reshape(-1, 6),
      "proprio_input_rot": ee_rot_inputs.reshape(-1, 6),
      "proprio_input_handdof": hand_dof_inputs.reshape(-1, 30),
      "proprio_input_hand_finger_tip": hand_finger_tip_3d_inputs.reshape(-1, 5 * 3 * 2),
    }
    data_dict.update(raw_proprio_inputs)

    data_dict["image"] = image
    data_dict["ee_2d_inputs"] = ee_2d_inputs
    data_dict["ee_3d_inputs"] = ee_3d_inputs
    data_dict["ee_rot_inputs"] = ee_rot_inputs
    data_dict["dof_hand_inputs"] = dof_hand_labels
    data_dict["handkp_3d_inputs"] = handkp_3d_inputs

    # for 2d
    data_dict["raw_width"] = sample["raw_width"]
    data_dict["raw_height"] = sample["raw_height"]

    data_dict["seq_name"] = sample["seq_name"]
    data_dict["frame_count"] = sample["frame_count"]

    if "current_kp_2d" in sample:
      data_dict["kp_2d"] = sample["current_kp_2d"]

    # data_dict["cam_intrinsics"] = sample["cam_intrinsics"]
    data_dict["language_label"] = current_language_label
    data_dict["ee_movement_mask"] = ee_movement_mask
    return data_dict


# from human_plan.utils.mano.mano_model import (
#   LEFT_AXIS_TRANSFORMATION,
#   RIGHT_AXIS_TRANSFORMATION,
#   holoassist_to_mano_joint_mapping,
#   obtain_mano_pelvis,
#   mano_left,
#   mano_right
# )

from human_plan.utils.mano.model import (
   mano_left,
   mano_right
)

from human_plan.utils.mano.constants import (
   LEFT_AXIS_TRANSFORMATION,
   RIGHT_AXIS_TRANSFORMATION,
   holoassist_to_mano_joint_mapping,
   LEFT_PELVIS,
   RIGHT_PELVIS
)

@dataclass
class LazyVLAHoloAssistHFAbsDataset(LazyVLAHFAbsDataset):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def init_dataset_specific_info(self):
    # 
    self.finger_tip_index = torch.Tensor([5, 10, 15, 20, 25]).long()
    self.frame_count_scaler_up = 1
    self.frame_count_scaler = 1
    self.reverse_channel_order = True

    self.raw_image_width = 1
    self.raw_image_height = 1

    self.dof_normalization = np.load(os.path.join(
        self.data_args.stats_path, "finger_stats.npy"
    ), allow_pickle=True).item()
    self.ee_normalization = np.load(os.path.join(
        self.data_args.stats_path, "pos_data_stats.npy"
    ), allow_pickle=True).item()

    for hand in ["left", "right", "full"]:
      for item in ["lower_bound", "upper_bound", "mean", "std"]:
        self.dof_normalization[hand][item] = torch.Tensor(self.dof_normalization[hand][item])
        self.ee_normalization[hand][item] = torch.Tensor(self.ee_normalization[hand][item])
        
    self.left_pelvis = LEFT_PELVIS.clone()
    self.right_pelvis = RIGHT_PELVIS.clone()

  def get_current_hand_data(self, sample, hand) -> Sequence[torch.Tensor]:
    hand = hand + "_hand"

    if hand == "left":
      hand_pelvis = self.left_pelvis
      hand_axis_transformation = LEFT_AXIS_TRANSFORMATION
    else:
      hand_pelvis = self.right_pelvis
      hand_axis_transformation = RIGHT_AXIS_TRANSFORMATION

    valid_mask = torch.tensor(
        sample["current_" + hand + "_pose/valid_state"].reshape(-1, 26)
    ) * torch.tensor(
        sample["current_" + hand + "_pose/track_state"].reshape(-1, 26)
    ) * torch.tensor(
        sample["current_" + hand + "_inframe_mask"].reshape(-1, 26)
    )
    valid_mask = valid_mask[:, 1:2]

    single_handkp_3d = torch.tensor(
        sample["current_" + hand + "_pose/hand_trans_cam_frame"]
    ).reshape(-1, 26, 3)

    # Usse Future to pretend current for now.
    single_ee_2d = torch.tensor(sample[
        "future_trans_" + hand + "_2d"
    ].reshape(-1, 26, 2)[0]).reshape(1, 26, 2)[:, 1, :]

    single_ee_3d = single_handkp_3d[:, 1, :] - hand_pelvis
    single_handkp_3d = single_handkp_3d - hand_pelvis

    single_hand_finger_tip_3d = single_handkp_3d[:, self.finger_tip_index, :]

    single_handkp_3d = single_handkp_3d[:, holoassist_to_mano_joint_mapping, :]
    
    single_hand_rot = torch.tensor(sample[
        "current_" + hand + "_pose/transformed_hand_pose_cam_frame"
    ]).reshape(-1, 26, 4, 4)[:, 1, :3, :3]

    current_rot = single_hand_rot @ hand_axis_transformation.detach().cpu().numpy()
    r = R.from_matrix(current_rot)
    single_hand_rot = r.as_rotvec()
    single_hand_rot = torch.tensor(single_hand_rot)

    hand_dof = torch.tensor(
      sample[
        "future_" + hand + "_pose/mano_parameters"
      ].reshape(-1, 15)[0]
    )

    return valid_mask, \
      single_ee_3d, \
      single_ee_2d, \
      single_hand_rot, \
      single_handkp_3d, \
      single_hand_finger_tip_3d, \
      hand_dof

    # return {
    #   "valid_mask": valid_mask, 
    #   "ee_3d": single_ee_3d, 
    #   "ee_2d": single_ee_2d,
    #   "ee_rot": single_hand_rot,
    #   "handkp_3d": single_handkp_3d
    # }


  def get_future_hand_data(
    self, sample, hand, future_step, future_idx
  ) -> Sequence[torch.Tensor]:
    
    hand = hand + "_hand"
    max_len = sample["future_" + hand + "_pose/transformed_hand_trans_cam_frame"].reshape(-1, 3).shape[0]

    if hand == "left":
      hand_pelvis = self.left_pelvis
      hand_axis_transformation = LEFT_AXIS_TRANSFORMATION
    else:
      hand_pelvis = self.right_pelvis
      hand_axis_transformation = RIGHT_AXIS_TRANSFORMATION


    target_idx = min(
        future_idx * (future_step + 1), max_len - 1
    )
    
    valid_mask = torch.tensor(
        sample["future_" + hand + "_pose/valid_state"].reshape(-1, 26)[target_idx]
    ).unsqueeze(-1) * torch.tensor(
        sample["future_" + hand + "_pose/track_state"].reshape(-1, 26)[target_idx]
    ).unsqueeze(-1) * torch.tensor(
        sample["future_" + hand + "_inframe_mask"].reshape(-1, 26)[target_idx]
    ).unsqueeze(-1)
    valid_mask = valid_mask.reshape(1, 26)
    valid_mask = valid_mask[:, 1:2]

    # if self.data_args.correct_transformation:
    #   # Raw Feature
    #   single_hand_label_trans = torch.tensor(
    #       sample[
    #           "future_" + hand + "_pose/transformed_hand_trans_cam_frame"
    #       ].reshape(-1, 26,3)[target_idx]
    #   ).reshape(1, 26, 3)
    # else:
    #   single_hand_label_trans = torch.tensor(
    #       sample[
    #           "future_" + hand + "_pose/hand_trans_cam_frame"
    #       ].reshape(-1, 26,3)[target_idx]
    #   ).reshape(1, 26, 3)
    assert self.data_args.correct_transformation
    single_handkp_3d_label = torch.tensor(
        sample[
            "future_" + hand + "_pose/transformed_hand_trans_cam_frame"
        ].reshape(-1, 26,3)[target_idx]
    ).reshape(1, 26, 3)

    assert self.data_args.use_mano
    hand_dof = torch.tensor(
      sample[
        "future_" + hand + "_pose/mano_parameters"
      ].reshape(-1, 15)[target_idx]
    )
    normalized_dof = hand_dof

    assert self.data_args.no_norm_ee_label
    single_ee_3d_label = single_handkp_3d_label[:, 1, :] - hand_pelvis

    single_handkp_3d_label = single_handkp_3d_label - hand_pelvis
    single_hand_tip_3d = single_handkp_3d_label[:, self.finger_tip_index, :]

    single_handkp_3d_label = single_handkp_3d_label[:, holoassist_to_mano_joint_mapping, :]

    # normalized_single_hand_label_trans_3d = normalized_single_hand_label_trans_3d[:, 1, :]

    single_ee_2d_label = torch.tensor(sample[
        "future_trans_" + hand + "_2d"
    ].reshape(-1, 26, 2)[target_idx]).reshape(1, 26, 2)[:, 1, :]

    future_rot = sample[
        "future_" + hand + "_pose/transformed_hand_pose_cam_frame"
    ].reshape(-1, 26, 4, 4)[target_idx][..., 1, :3, :3]

    # Convert to Mano coordinate: Shape (-1, 3, 3)
    future_rot = future_rot @ hand_axis_transformation.detach().cpu().numpy()

    r = R.from_matrix(future_rot)
    future_rot = r.as_rotvec()

    single_ee_rot_label = torch.tensor(
      future_rot
    ).reshape(-1, 3)

    return valid_mask, \
      single_ee_3d_label, \
      single_ee_2d_label, \
      normalized_dof, \
      single_ee_rot_label, \
      single_handkp_3d_label
      # single_hand_tip_3d, \

  def get_current_language_label(self, sample):
    current_language_label = sample['language_label']
    if self.data_args.use_short_language_label:
      if len(sample['language_label_short']) == 0:
        current_language_label = "manipulate the object"
      else:
        current_language_label = sample['language_label_short']
    elif self.data_args.use_empty_language_label:
      current_language_label = sample['language_label_empty']
    elif self.data_args.use_verb_only_short_language_label:
      if len(sample['language_label_verb']) == 0:
        current_language_label = "manipulate the object"
      else:
        current_language_label = sample['language_label_verb'] + " the object"
    elif self.data_args.use_noun_only_short_language_label:
      if len(sample['language_label_noun']) == 0:
        current_language_label = "manipulate the object"
      else:
        current_language_label = "manipulate the " + sample['language_label_noun']

    mix_lanugage_ratio = self.data_args.mix_language_ratio
    import random

    random_selection = random.random()
    if random_selection < mix_lanugage_ratio:
       current_language_label = sample['language_label_empty']
    return current_language_label


HOT3D_IMAGE_WIDTH=1408
HOT3D_IMAGE_HEIGHT=1408

@dataclass
class LazyVLAHOT3DHFAbsDataset(LazyVLAHFAbsDataset):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def init_dataset_specific_info(self):
    self.finger_tip_index = torch.Tensor([0, 1, 2, 3, 4]).long()
    self.frame_count_scaler_up = 1
    self.frame_count_scaler = 1
    self.reverse_channel_order = False

    self.raw_image_width = HOT3D_IMAGE_WIDTH
    self.raw_image_height = HOT3D_IMAGE_HEIGHT

    self.dof_normalization = np.load(os.path.join(
        self.data_args.stats_path, "finger_stats.npy"
    ), allow_pickle=True).item()
    self.ee_normalization = np.load(os.path.join(
        self.data_args.stats_path, "pos_data_stats.npy"
    ), allow_pickle=True).item()

    for hand in ["left", "right", "full"]:
      for item in ["lower_bound", "upper_bound", "mean", "std"]:
        self.dof_normalization[hand][item] = torch.Tensor(self.dof_normalization[hand][item])
        self.ee_normalization[hand][item] = torch.Tensor(self.ee_normalization[hand][item])

  def get_current_hand_data(self, sample, hand) -> Sequence[torch.Tensor]:

    valid_mask = torch.tensor(
        sample["current_" + hand + "_flag"].reshape(-1, 1)
    ) * torch.tensor(
        sample["current_mano_" + hand + "_inframe_mask"].reshape(-1, 1)
    )

    single_ee_3d = torch.tensor(
        sample["current_" + hand + "_wrist_3d"]
    ).reshape(-1, 3)

    single_ee_2d = torch.tensor(
        sample["current_mano_" + hand + "_wrist_2d"]
    )

    single_ee_rot = torch.tensor(
        sample["current_" + hand + "_wrist_rot"]
    ).reshape(-1, 3, 3)
    r = R.from_matrix(single_ee_rot)
    current_rot = r.as_rotvec()
    single_ee_rot = torch.tensor(
      current_rot
    ).reshape(-1, 3)
    # Fxxk it there is something wrong with the orders fxxk it
    handkp_3d_label = torch.tensor(
      sample[
        "current_" + hand + "_hand_kp"
      ].reshape(-1, 20, 3)
    )
    hand_finger_tip_3d = handkp_3d_label[:, self.finger_tip_index, :]

    hand_dof = torch.tensor(
      sample[
        "current_" + hand + "_hand_joint_pca"
      ].reshape(-1, 15)
    )
    return valid_mask, \
      single_ee_3d, \
      single_ee_2d, \
      single_ee_rot, \
      handkp_3d_label, \
      hand_finger_tip_3d, \
      hand_dof

    # return valid_mask, single_hand_trans, single_hand_trans_2d, \
    #   single_hand_pose, single_hand_rot

  def get_future_hand_data(
    self, sample, hand, future_step, future_idx
  ) -> Sequence[torch.Tensor]:

    max_len = sample["future_mano_" + hand + "_wrist_3d"].reshape(-1, 3).shape[0]

    target_idx = min(
        future_idx * (future_step + 1), max_len - 1
    )

    # Directly considered MANO stuff, no need for additional transformation
    single_ee_3d_label = torch.tensor(
      sample[
            "future_" + hand + "_wrist_3d"
      ].reshape(-1, 3)[target_idx]
    ).reshape(1, 3)

    single_ee_2d_label = torch.tensor(
        sample[
            "future_mano_" + hand + "_wrist_2d"
        ].reshape(-1, 2)[target_idx]
    ).reshape(1, 2)

    # Already considered MANO transformation
    future_rot = sample[
      "future_" + hand + "_wrist_rot"
    ].reshape(-1, 3, 3)[target_idx]

    r = R.from_matrix(future_rot)
    future_rot = r.as_rotvec()
    single_hand_label_rot = torch.tensor(
      future_rot
    ).reshape(1, 3)

    assert self.data_args.use_mano
    hand_dof = torch.tensor(
      sample[
        "future_" + hand + "_hand_joint_pca"
      ].reshape(-1, 15)[target_idx]
    )

    # Fxxk it there is something wrong with the orders fxxk it
    handkp_3d_label = torch.tensor(
      sample[
        "future_" + hand + "_hand_kp"
      ].reshape(-1, 20, 3)[target_idx]
    )
    # normalized_dof = hand_dof
    # else:
    #   single_hand_kp_3d = torch.tensor(
    #       sample[
    #           "future_" + hand + "_hand_kp"
    #       ].reshape(-1, 20, 3)[target_idx]
    #   ).reshape(1, 20, 3)

    #   hand_dof = compute_hand_dof_5dim(
    #     palm_center=(
    #       single_hand_kp_3d[:, 5:6, :] + single_hand_kp_3d[:, 11:12, :]
    #     ) / 2,
    #     wrist=single_hand_kp_3d[:, 5:6, :],
    #     finger_tip=single_hand_kp_3d[:, self.finger_tip_index, :]
    #   ).reshape(1, 5)

    #   normalized_dof = normalize_item(
    #     hand_dof,
    #     self.dof_normalization["full"]
    #   )

    assert self.data_args.no_norm_ee_label
    # single_ee_3d_label = single_ee_3d_label
    # else:
    #   normalized_single_hand_label_3d = normalize_item(
    #     single_hand_label_3d,
    #     self.ee_normalization["full"]
    #   )


    # valid_mask = torch.tensor(
    #     sample["future_" + hand + "_flag"].reshape(-1, 1)[target_idx]
    # ).unsqueeze(-1)

    valid_mask = torch.tensor(
        sample["future_mano_" + hand + "_inframe_mask"].reshape(-1, 1)[target_idx]
    ) * torch.tensor(
        sample["future_" + hand + "_flag"].reshape(-1, 1)[target_idx]
    )
    valid_mask = valid_mask.reshape(1, 1)

    return valid_mask, \
      single_ee_3d_label, \
      single_ee_2d_label, \
      hand_dof, \
      single_hand_label_rot, \
      handkp_3d_label

  def get_current_language_label(self, sample):
    # HOT3D provide no language label
    current_language_label = "manipulate the object"
    return current_language_label

  def normalize_2d(self, ee_2d, sample):    
    ee_2d_normalization = torch.Tensor([
      sample["raw_width"], sample["raw_height"]
    ]).reshape(1, 1, 2)

    ee_2d  = ee_2d / ee_2d_normalization
    # print(ee_2d_normalization, ee_2d)
    return ee_2d

HOI4D_IMAGE_WIDTH=1920
HOI4D_IMAGE_HEIGHT=1080

from human_plan.dataset_preprocessing.utils.mano_utils import (
  mano_right
)

@dataclass
class LazyVLAHOI4DHFAbsDataset(LazyVLAHFAbsDataset):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def init_dataset_specific_info(self):
    # 
    self.finger_tip_index = torch.Tensor([4, 8, 12, 16, 20]).long()
    self.frame_count_scaler_up = 1
    self.frame_count_scaler = 2
    self.reverse_channel_order = True

    self.raw_image_width = HOI4D_IMAGE_WIDTH
    self.raw_image_height = HOI4D_IMAGE_HEIGHT

    self.dof_normalization = np.load(os.path.join(
        self.data_args.stats_path, "finger_stats.npy"
    ), allow_pickle=True).item()
    self.ee_normalization = np.load(os.path.join(
        self.data_args.stats_path, "pos_data_stats.npy"
    ), allow_pickle=True).item()

    for hand in ["right", "full"]:
      for item in ["lower_bound", "upper_bound", "mean", "std"]:
        self.dof_normalization[hand][item] = torch.Tensor(self.dof_normalization[hand][item])
        self.ee_normalization[hand][item] = torch.Tensor(self.ee_normalization[hand][item])
        
    self.mano_hand_mean = mano_right.hand_mean.detach().cpu().numpy()
    self.mano_hand_components = mano_right.np_hand_components

  def __len__(self):
    return len(self.dataset) // self.data_skip

  def get_current_hand_data(self, sample, hand) -> Sequence[torch.Tensor]:
    if hand == "left":
      valid_mask = torch.zeros((1))
      single_ee_3d = torch.zeros((1, 3))
      single_ee_2d = torch.zeros((1, 2))
      single_ee_rot = torch.zeros((1, 3))
      single_handkp_3d = torch.zeros((1, 21, 3))
      single_hand_finger_tip_3d = torch.zeros((1, 5, 3))
      single_hand_dof = torch.zeros((1, 15))
      return valid_mask, \
        single_ee_3d, \
        single_ee_2d, \
        single_ee_rot, \
        single_handkp_3d, \
        single_hand_finger_tip_3d, \
        single_hand_dof

    valid_mask = torch.tensor(
        sample["current_" + hand + "_flag"].reshape(-1, 1)
    )

    single_ee_3d = torch.tensor(
        sample["current_" + hand + "_wrist_3d"]
    ).reshape(-1, 3)

    single_ee_2d = torch.tensor(
        sample["current_" + hand + "_wrist_2d"]
    ).reshape(-1, 2)

    current_rot = sample["current_" + hand + "_wrist_rot"].reshape(-1, 3, 3)
    r = R.from_matrix(current_rot)
    current_rot = r.as_rotvec()
    single_hand_rot = torch.tensor(
      current_rot
    ).reshape(-1, 3)

    single_handkp_3d = torch.tensor(
      sample["current_" + hand + "_hand_kp"]
    ).reshape(-1, 21, 3)

    hand_finger_tip_3d = single_handkp_3d[:, self.finger_tip_index, :]

    # future_right_hand_kp
    hand_dof = torch.tensor(
      sample[
        "current_" + hand + "_pose_theta"
      ].reshape(-1, 45)[0]
    )
    hand_dof = torch.tensor(self.get_pca_dim(hand_dof)).reshape(-1, 15)

    return valid_mask, \
      single_ee_3d, \
      single_ee_2d, \
      single_hand_rot, \
      single_handkp_3d, \
      hand_finger_tip_3d, \
      hand_dof
    # return valid_mask, single_hand_trans, single_hand_trans_2d, \
    #   single_hand_pose, single_hand_rot

  def get_pca_dim(self, raw_rotation) -> torch.Tensor:
    return convert_full_mano_to_pca_dof(
      raw_rotation, self.mano_hand_mean, self.mano_hand_components
    )

  def get_future_hand_data(
    self, sample, hand, future_step, future_idx
  ) -> Sequence[torch.Tensor]:

    if hand == "left":
      valid_mask = torch.zeros((1, 1))
      single_ee_3d_label = torch.zeros((1, 3))
      single_ee_2d_label = torch.zeros((1, 2))
      assert self.data_args.use_mano
      single_hand_pose_label = torch.zeros((1, 15))
      # else:
      #   single_hand_pose = torch.zeros((1, 5))
      single_ee_rot_label = torch.zeros((1, 3))
      single_handkp_3d_label = torch.zeros((1, 21, 3))
      return valid_mask, \
        single_ee_3d_label, \
        single_ee_2d_label, \
        single_hand_pose_label, \
        single_ee_rot_label, \
        single_handkp_3d_label

    max_len = sample["future_" + hand + "_wrist_3d"].reshape(-1, 3).shape[0]

    target_idx = min(
        future_idx * (future_step + 1), max_len - 1
    )

    single_ee_3d_label = torch.tensor(
        sample[
            "future_" + hand + "_wrist_3d"
        ].reshape(-1, 3)[target_idx]
    ).reshape(1, 3)

    single_ee_2d_label = torch.tensor(
        sample[
            "future_" + hand + "_kp_wrist_2d"
        ].reshape(-1, 2)[target_idx]
    ).reshape(1, 2)

    single_handkp_3d_label = torch.tensor(
        sample[
            "future_" + hand + "_hand_kp"
        ].reshape(-1, 21, 3)[target_idx]
    ).reshape(1, 21, 3)

    assert self.data_args.use_mano
    hand_dof = torch.tensor(
      sample[
        "future_right_pose_theta"
      ].reshape(-1, 45)[target_idx]
    )
    hand_dof = torch.tensor(self.get_pca_dim(hand_dof))
    #   normalized_dof = hand_dof
    # else:
    #   hand_dof = compute_hand_dof_5dim(
    #     palm_center=(
    #       single_hand_kp_3d[:, 0:1, :] + single_hand_kp_3d[:, 9:10, :]
    #     ) / 2,
    #     wrist=single_hand_kp_3d[:, 0:1, :],
    #     finger_tip=single_hand_kp_3d[:, self.finger_tip_index, :]
    #   ).reshape(1, 5)
    #   normalized_dof = normalize_item(
    #     hand_dof,
    #     self.dof_normalization["full"]
    #   )

    assert self.data_args.no_norm_ee_label
    # normalized_single_hand_label_3d = single_hand_label_3d
    # else:
    #   normalized_single_hand_label_3d = normalize_item(
    #     single_hand_label_3d,
    #     self.ee_normalization["full"]
    #   )

    valid_mask = torch.tensor(
      sample["future_" + hand + "_kp_flag"].reshape(-1, 1)[target_idx]
    ).unsqueeze(-1)
  
    valid_mask = valid_mask.reshape(1, 1)

    future_rot = sample[
        "future_" + hand + "_wrist_rot"
    ].reshape(-1, 3, 3)[target_idx]

    r = R.from_matrix(future_rot)
    future_rot = r.as_rotvec()
    single_ee_rot_label = torch.tensor(
        future_rot
    ).reshape(1, 3)

    single_handkp_3d_label = torch.tensor(
        sample["future_"+ hand + "_hand_kp"].reshape(-1, 21, 3)[target_idx]
    ).reshape(1, 21, 3)
    
    return valid_mask, \
      single_ee_3d_label, \
      single_ee_2d_label, \
      hand_dof, \
      single_ee_rot_label, \
      single_handkp_3d_label

  def get_current_language_label(self, sample):
    current_language_label = f"{sample['language_label_verb']} {sample['language_label_noun']}"
    return current_language_label


TACO_IMAGE_WIDTH=1920
TACO_IMAGE_HEIGHT=1080

@dataclass
class LazyVLATACOHFAbsDataset(LazyVLAHFAbsDataset):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def init_dataset_specific_info(self):
    # 
    self.finger_tip_index = torch.Tensor([4, 8, 12, 16, 20]).long()
    self.frame_count_scaler_up = 1
    self.frame_count_scaler = 1
    self.reverse_channel_order = True

    self.raw_image_width = TACO_IMAGE_WIDTH
    self.raw_image_height = TACO_IMAGE_HEIGHT

  def __len__(self):
    return len(self.dataset) // self.data_skip

  def get_current_hand_data(self, sample, hand) -> Sequence[torch.Tensor]:
    valid_mask = torch.tensor(
        sample["current_" + hand + "_flag"].reshape(-1, 1)
    )

    single_ee_3d = torch.tensor(
        sample["current_" + hand + "_ee_trans_3d"]
    ).reshape(-1, 3)

    single_ee_2d = torch.tensor(
        sample["current_" + hand + "_ee_trans_2d"]
    ).reshape(-1, 2)
    # print("Current Rot:", sample["current_" + hand + "_wrist_rot"].shape)
    current_rot = sample["current_" + hand + "_wrist_rot"].reshape(-1, 3, 3)
    r = R.from_matrix(current_rot)
    current_rot = r.as_rotvec()
    single_hand_rot = torch.tensor(
      current_rot
    ).reshape(-1, 3)

    single_handkp_3d = torch.tensor(
      sample["current_" + hand + "_hand_kp"]
    ).reshape(-1, 21, 3)
    # future_right_hand_kp
    hand_finger_tip_3d = single_handkp_3d[:, self.finger_tip_index, :]

    hand_dof = torch.tensor(
      sample[
        "current_" + hand + "_pose_theta"
      ].reshape(-1, 15)
    )
    return valid_mask, \
      single_ee_3d, \
      single_ee_2d, \
      single_hand_rot, \
      single_handkp_3d, \
      hand_finger_tip_3d, \
      hand_dof

  def get_future_hand_data(
    self, sample, hand, future_step, future_idx
  ) -> Sequence[torch.Tensor]:

    max_len = sample["future_" + hand + "_wrist_3d"].reshape(-1, 3).shape[0]

    target_idx = min(
        future_idx * (future_step + 1), max_len - 1
    )

    single_ee_3d_label = torch.tensor(
        sample[
            "future_" + hand + "_ee_trans_3d"
        ].reshape(-1, 3)[target_idx]
    ).reshape(1, 3)

    single_ee_2d_label = torch.tensor(
        sample[
            "future_" + hand + "_ee_trans_2d"
        ].reshape(-1, 2)[target_idx]
    ).reshape(1, 2)

    single_handkp_3d_label = torch.tensor(
        sample[
            "future_" + hand + "_hand_kp"
        ].reshape(-1, 21, 3)[target_idx]
    ).reshape(1, 21, 3)

    assert self.data_args.use_mano
    hand_dof = torch.tensor(
      sample[
        "future_" + hand + "_pose_theta"
      ].reshape(-1, 15)[target_idx]
    )
    # hand_dof = torch.tensor(self.get_pca_dim(hand_dof))

    assert self.data_args.no_norm_ee_label
    # normalized_single_hand_label_3d = single_hand_label_3d
    # else:
    #   normalized_single_hand_label_3d = normalize_item(
    #     single_hand_label_3d,
    #     self.ee_normalization["full"]
    #   )

    valid_mask = torch.tensor(
      sample["future_" + hand + "_kp_flag"].reshape(-1, 1)[target_idx]
    ).unsqueeze(-1)
  
    valid_mask = valid_mask.reshape(1, 1)

    future_rot = sample[
        "future_" + hand + "_wrist_rot"
    ].reshape(-1, 3, 3)[target_idx]

    r = R.from_matrix(future_rot)
    future_rot = r.as_rotvec()
    single_ee_rot_label = torch.tensor(
        future_rot
    ).reshape(1, 3)

    single_handkp_3d_label = torch.tensor(
        sample["future_"+ hand + "_hand_kp"].reshape(-1, 21, 3)[target_idx]
    ).reshape(1, 21, 3)
    
    return valid_mask, \
      single_ee_3d_label, \
      single_ee_2d_label, \
      hand_dof, \
      single_ee_rot_label, \
      single_handkp_3d_label

  def get_current_language_label(self, sample):
    # current_language_label = f"{sample['language_label_']} {sample['language_label_noun']}"
    current_language_label = sample['language_label_short']
    return current_language_label



def make_supervised_data_module(
    tokenizer: PreTrainedTokenizer,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning.
    This function is originally implemented by the LLaVA team and
    modified by Jason Lu, Haotian Tang and Ligeng Zhu."""
    datasets_mixture.register_datasets_mixtures()

    from .builder import build_dataset

    train_dataset = build_dataset(data_args.data_mixture, data_args, training_args, tokenizer)
    training_args.sample_lens = [len(d) for d in train_dataset.datasets]

    eval_dataset = build_dataset(data_args.eval_data_mixture, data_args, training_args=training_args, tokenizer=tokenizer)
    training_args.eval_sample_lens = [len(d) for d in eval_dataset.datasets]
    # training_args.sample_lens = [len(d) for d in train_dataset.datasets]

    PROCESS_GROUP_MANAGER = get_pg_manager()
    if PROCESS_GROUP_MANAGER is None:
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    else:
        sp_degree = training_args.seq_parallel_size
        sp_rank = PROCESS_GROUP_MANAGER.sp_rank
        ring_degree = PROCESS_GROUP_MANAGER.ring_degree
        ring_type = PROCESS_GROUP_MANAGER.ring_type
        data_collator = DataCollatorForSupervisedDatasetSeqParallel(
            tokenizer=tokenizer,
            data_args=data_args,
            training_args=training_args,
            sp_degree=sp_degree,
            sp_rank=sp_rank,
            ring_degree=ring_degree,
            ring_type=ring_type,
        )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
