# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import random
import torch
import json
import os
import pandas as pd
import traceback
import numpy as np
from PIL import Image
from typing import List
from BeingH.dataset.preprocess import dynamic_preprocess
from BeingH.utils.conversation import get_conv_template
from ..preprocess import build_vit_transform_base


def pil_img2rgb(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")

    return image


import os
from contextlib import contextmanager

@contextmanager
def suppress_ffmpeg_warnings():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        os.close(devnull)


class SftJSONLIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        dataset_name: str, 
        dataset_list: List[str],
        dataset_path_list: List[str],
        jsonl_path_list: List[str],
        vit_transform_args, 
        num_used_data: List[int] = None,
        is_train=True,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        logger = None,
        # Tokenizer and text processing
        tokenizer=None, 
        template_name=None, 
        force_image_size=448, 
        num_image_tokens=0,
        shuffle_lines=False, shuffle_seed=0,
        local_rank=0, world_size=1, num_workers=8, 
        **kwargs,
    ):
        """
        jsonl_path_list: list of jsonl file paths
        data_dir_list: list of image directories containing the images of each jsonl file
        num_used_data: list of number of sampled data points for each jsonl
        """
        self.is_train = is_train
        self.logger = logger
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.initial_seed = 42

        self.dataset_name = dataset_name

        self.dataset_id2name = {}
        self.sub_dataset_imgdir = {}
        self.lazy_load_subset_ids = []
        self.lazy_load_parquet_dirs = {}
        for _dataset in dataset_list:
            subset_id = len(self.dataset_id2name)
            self.dataset_id2name[subset_id] = _dataset
            if _dataset in ["as_v2_pretrain_10m", "finevision"]:
                self.lazy_load_subset_ids.append(subset_id)
                
        self.dataset_name2id = {value: key for key, value in self.dataset_id2name.items()}
        self.large_db_size = {"as_v2_pretrain_10m": 9_993_656, "finevision": 24_082_623} #
        self.force_image_size = force_image_size

        self.use_dynamic_size = True if vit_transform_args['transform_type']=="dynamic_size" else False
        self.min_dynamic_patch = vit_transform_args['min_dynamic_patch'] if 'min_dynamic_patch' in vit_transform_args else 1
        self.max_dynamic_patch = vit_transform_args['max_dynamic_patch'] if 'max_dynamic_patch' in vit_transform_args else 1
        self.max_frame_number = vit_transform_args['max_frame_number'] if 'max_frame_number' in vit_transform_args else 1

        self.pre_transform, self.vit_transform = build_vit_transform_base(is_train=self.is_train, force_image_size=force_image_size, **vit_transform_args)
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.num_image_tokens = num_image_tokens

        self.tokenizer = tokenizer
        self.template_name = template_name
        conv = get_conv_template(self.template_name)
        self.system_prompt = conv.system_message
   
        self.rng = random.Random(self.initial_seed)

        # For very large datasets, read num_used_data samples per epoch, with id list order incrementing for the next epoch
        self.data_paths = self.get_data_paths(
            jsonl_path_list, 
            dataset_list,
            dataset_path_list, 
            num_used_data, 
            shuffle_lines, 
            shuffle_seed,
        )
        self.logger.info(f"VLM dataset group '{self.dataset_name}' initialized with {self.__len__()/1_000_000:.3f}M total sample units.")

        self.set_epoch(seed=self.initial_seed)

    def __len__(self):
        return len(self.data_paths)
    
    def get_data_paths(
        self, 
        jsonl_path_list, 
        dataset_list,
        dataset_path_list, 
        num_used_data, 
        shuffle_lines, 
        shuffle_seed,
    ):
        data_paths = []
        for jsonl_path, dataset_name, image_dir, num_data_point in zip(
            jsonl_path_list, dataset_list, dataset_path_list, num_used_data
        ):
            subset_id = self.dataset_name2id[dataset_name]
            self.sub_dataset_imgdir[subset_id] = image_dir

            if subset_id in self.lazy_load_subset_ids:
                self.lazy_load_parquet_dirs[subset_id] = jsonl_path
                sub_data_paths = [(subset_id, idx) for idx in range(self.large_db_size[dataset_name]-1)]
                if shuffle_lines:
                    self.rng.shuffle(sub_data_paths)
                sub_data_paths = sub_data_paths[:num_data_point]
                data_paths.extend(sub_data_paths)
                self.logger.info(f"Sampling {num_data_point/1_000_000:.3f}M in {jsonl_path.split('/')[-1].split('.')[0]}")
            else:
                with open(jsonl_path, 'r') as f:
                    raw_data = f.readlines()
                total_len = len(raw_data)
          
                if shuffle_lines:
                    self.rng.seed(shuffle_seed)
                    self.rng.shuffle(raw_data)
                if num_data_point>0:
                    raw_data = raw_data[:num_data_point]
            
                sub_data_paths = [(subset_id, json_data) for json_data in raw_data]
                data_paths.extend(sub_data_paths)
            
                self.logger.info(f"Sampling {len(raw_data)/1_000_000:.3f}M  from {total_len/1_000_000:.3f}M data in {jsonl_path.split('/')[-1].split('.')[0]}")
                 
        return data_paths
    
    def set_epoch(self, seed):
        total_sample_idxs = [i for i in range(len(self))]
        self.rng.seed(seed)
        self.rng.shuffle(total_sample_idxs)

        num_files_per_rank = len(self) // self.world_size
        self.total_sample_idxs = total_sample_idxs
        local_start = self.local_rank * num_files_per_rank
        local_end = (self.local_rank + 1) * num_files_per_rank
        self.num_files_per_rank = num_files_per_rank
        self.sample_idxs_per_rank = self.total_sample_idxs[local_start: local_end]

    def get_data_paths_per_worker(self):
        if self.data_paths is None:
            return None

        info = torch.utils.data.get_worker_info()
        if info is None:
            # Single worker: Use all files assigned to the rank
            return self.data_paths_per_rank, 0

        worker_id = info.id
        num_files_per_worker = self.num_files_per_rank // info.num_workers
        start = num_files_per_worker * worker_id
        end = num_files_per_worker * (worker_id + 1)
        data_paths_per_worker = self.data_paths_per_rank[start:end]

        return data_paths_per_worker[::-1], worker_id

    def change_format(self, data, num_images, dynamic_patch_nums=[]):
        elements = []
        img_tag_id = 0
        for conversation in data['conversations']:
            conv_elements = []
            value = conversation['value']
            role = conversation['from']

            if role == 'human':
                if '<image>' not in conversation['value']:
                    # a pure-text sentence
                    conv_elements.append({
                        'type': 'text', 'has_loss': 0, 'text': f"user\n{value}",
                        'role': role, 'is_bos': True, 'is_eos': True
                    })
                else:
                    text_list = conversation['value'].split('<image>')

                    conv_elements.append({
                                'type': 'text', 'has_loss': 0, 'text': 'user\n', 
                                'role': role, 'is_bos': True, 'is_eos': False
                            })
                    
                    for idx, text in enumerate(text_list):
                        stripped_text = text.strip()
                        if stripped_text:
                            conv_elements.append({
                                'type': 'text', 'has_loss': 0,
                                'text': stripped_text,
                                'role': role, 'is_bos': False, 'is_eos': False
                            })
                        if (idx != len(text_list) - 1) and (idx < num_images):
                            if len(dynamic_patch_nums)==0:
                                conv_elements.append({'type': 'image', 'role': role, 
                                                    'is_bos': False, 'is_eos': False})
                            else:
                                for _ in range(dynamic_patch_nums[img_tag_id]):
                                    conv_elements.append({'type': 'image', 'role': role, 
                                                    'is_bos': False, 'is_eos': False})
                                img_tag_id += 1

                    conv_elements.append({
                                'type': 'text', 'has_loss': 0,
                                'text': '',
                                'role': role, 'is_bos': False, 'is_eos': True
                            })
            elif role == 'gpt':
                conv_elements.append({
                    'type': 'text', 'has_loss': 1, 'text': f'assistant\n{value}',
                    'role': role, 'is_bos': True, 'is_eos': True
                })

            conv_elements[0]['is_bos'] = conv_elements[-1]['is_eos'] = True

            elements.extend(conv_elements)
 
        return elements

    def get_frame_index_bymax(self, total_frames, duration, fps):
        if duration <= self.max_frame_number:
            frame_indices = []
            for second in range(int(duration)):
                frame_idx = int(second * fps + fps / 2)
                if frame_idx < total_frames:
                    frame_indices.append(frame_idx)
            if not frame_indices or frame_indices[-1] != total_frames - 1:
                frame_indices.append(total_frames - 1)
            return sorted(set(frame_indices))
        else:
            step = total_frames / self.max_frame_number
            frame_indices = []
            for i in range(self.max_frame_number):
                frame_idx = int(i * step)
                if frame_idx < total_frames:
                    frame_indices.append(frame_idx)
            if frame_indices[-1] != total_frames - 1:
                frame_indices[-1] = total_frames - 1
        
        return frame_indices
    
    def frame_sampler(self, video_path, frame_indexes=None, fps=None, duration=None) -> List[Image.Image]:
        if not os.path.exists(video_path):
            assert 1 == 0 # TODO, load from image dir

        # https://github.com/dmlc/decord/issues/208
        vr = None
        try:
            #vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            #vr = VideoReader(video_path)
            with suppress_ffmpeg_warnings():
                #vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
                vr = VideoReader(video_path, num_threads=1)
                duration = len(vr) / vr.get_avg_fps()
                if frame_indexes is None:
                    frame_indexes = self.get_frame_index_bymax(len(vr), duration, fps)
                if isinstance(frame_indexes, int):
                    frame_indexes = [frame_indexes]
                #valid_frame_indices = [min(max(idx, 0), len(video_reader) - 1) for idx in frame_indexes]
                frames = vr.get_batch(frame_indexes).asnumpy()
        
            pil_frames = [Image.fromarray(frame) for frame in frames]
        except:
            # avoid corrupted videos
            self.logger.info(f"{video_path} is corrupted, replacing with black!!!")
            num_frames = 1 if frame_indexes is None else len(frame_indexes)
            pil_frames = [Image.fromarray(np.zeros([256,256,3], dtype=np.uint8))] * num_frames
        finally:
            if vr is not None:
                del vr

        return pil_frames
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        epoch_count = 0
        
        while True:
            current_seed = self.initial_seed + epoch_count*100 #worker_id * 1000 + 
            epoch_count += 1
            self.set_epoch(current_seed)
            info = torch.utils.data.get_worker_info()

            if info is None:
                sample_ids_per_worker = self.sample_idxs_per_rank
            else:
                worker_id = info.id
                num_files_per_worker = self.num_files_per_rank // info.num_workers
                start = num_files_per_worker * worker_id
                end = num_files_per_worker * (worker_id + 1)
                sample_ids_per_worker = self.sample_idxs_per_rank[start:end][::-1]

            if not sample_ids_per_worker:
                return

            for sample_id in sample_ids_per_worker:
                subset_id, json_data = self.data_paths[sample_id]
                
                try:
                    if subset_id in self.lazy_load_subset_ids:
                        # for super large dataset
                        parquet_id, parquet_row = json_data//1000, json_data%1000   
                        parquet_file_path = f"{self.lazy_load_parquet_dirs[subset_id]}/{parquet_id:06d}.parquet"

                        try:
                            df = pd.read_parquet(parquet_file_path)
                            data_item = df.iloc[parquet_row].to_dict()
                            if "image" in data_item and data_item['image'] is not None and not isinstance(data_item['image'], str):
                                data_item['image'] = data_item['image'].tolist() 
                        finally:
                            del df
                    else:
                        data_item = json.loads(json_data)

                    image_dir = self.sub_dataset_imgdir[subset_id]

                    # each sample in the dataset
                    packet = {
                        'sequence_plan': [], 'num_tokens': 0,
                        'text_ids_list': [], 'image_tensor_list': [],
                        'is_und': True,
                    }

                    dynamic_patch_nums = []

                    raw_images = None
                    if 'image' in data_item and data_item['image'] is not None:
                        images = data_item['image'] if type(data_item['image']) == list else [data_item['image']]

                        for image in images:
                            if self.use_dynamic_size:
                                raw_img = self.pre_transform(Image.open(os.path.join(image_dir, image)))
                                img_patches = dynamic_preprocess(raw_img, self.min_dynamic_patch, self.max_dynamic_patch, self.force_image_size)
                                image_tensors = [self.vit_transform(img_patch).unsqueeze(0) for img_patch in img_patches]
                                #special_tokens = '<image>' * len(img_patches)
                                dynamic_patch_nums.append(len(img_patches))

                                packet['image_tensor_list'].extend(image_tensors)
                            else:       
                                image_tensor = self.vit_transform(
                                    Image.open(os.path.join(image_dir, image))
                                ).unsqueeze(0)
                                packet['image_tensor_list'].append(image_tensor)
                    
                    elif 'video' in data_item:
                        if "frame" in data_item:
                            raw_images = self.frame_sampler(os.path.join(image_dir, data_item['video']), frame_indexes=data_item['frame'])
                        elif "duration" in data_item:
                            raw_images = self.frame_sampler(os.path.join(image_dir, data_item['video']), fps=data_item['fps'], duration=data_item['duration'])
                        
                        special_tokens = '<image>' * len(raw_images)
                        for item in data_item['conversations']:
                            if item['from'] != 'human':
                                continue
                            if '<video>' in item['value']:
                                item['value'] = item['value'].replace('<video>', special_tokens)
                                break
                            elif '<image>' in item['value']:
                                continue
                            else:
                                raise ValueError("Cannot find <video> in the conversation!")
                        for image in raw_images:
                            image_tensor = self.vit_transform(image).unsqueeze(0)
                            packet['image_tensor_list'].append(image_tensor)
                    
                except:
                    traceback.print_exc()
                    continue

                for _ in packet['image_tensor_list']:
                    packet['num_tokens'] += self.num_image_tokens +2
                
                elements = self.change_format(data_item, len(packet['image_tensor_list']), dynamic_patch_nums)
   
                # add system_prompt if available
                system_prompt = f"system\n{self.system_prompt}"
                text_ids = self.tokenizer.encode(system_prompt)
                packet['text_ids_list'].append(text_ids)
                packet['sequence_plan'].append({'type': 'text', 'has_loss': 0, 'enable_cfg': 0,
                                                'special_token_loss': 0, 'special_token_label': None,
                                                'is_bos': True, 'is_eos': True})
                packet['num_tokens'] += len(text_ids) + 2 +1 # bos & eos\n
                
                if len(packet['image_tensor_list'])!=len([i for i in elements if i['type']=="image"]):
                    print(data_item)

                assert len(packet['image_tensor_list'])==len([i for i in elements if i['type']=="image"])

                for item in elements: # elements -> sequence_plans
                    if item['type'] == 'text':
                        text_ids = self.tokenizer.encode(item['text'])
                        packet['text_ids_list'].append(text_ids)  
                        packet['sequence_plan'].append({'type': 'text', 'has_loss': item['has_loss'], 'enable_cfg': 0,
                                                'special_token_loss': 0, 'special_token_label': None,
                                                'is_bos': item['is_bos'], 'is_eos': item['is_eos']})

                        packet['num_tokens'] += len(text_ids)
                        if item['is_bos']:
                            packet['num_tokens'] += 1 # bos
                        if item['is_eos']:
                            packet['num_tokens'] += 1+1 # eos\n
            
                    elif item['type'] == 'image':
                        packet['sequence_plan'].append({'type': 'vit_image', 'has_loss': 0, 'enable_cfg': 0, 
                                            'special_token_loss': 0, 'special_token_label': None,
                                            'num_image_tokens': self.num_image_tokens,
                                            'is_bos': False, 'is_eos': False})

                packet['sequence_plan'][-1]['is_end'] = True
                packet['num_tokens'] -= 1 # the last is eos w/o \n
         
                yield packet
           

           