
import sys
import os
import io
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import traceback
import json
import random
import math
import numpy as np
import pandas as pd

import signal
import torch
from torch.utils.data.dataset import Dataset
from einops import rearrange
import glob
from moviepy.editor import VideoFileClip
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F
import cv2
from PIL import Image
from collections import defaultdict
from data.utils.constants import NORM_SET

# from data.utils.get_actions import parse_h5

from utils import zero_rank_print
from data.utils.utils import intrinsic_transform, gen_crop_config, intrin_crop_transform

def load_jsonl(jsonl_path):
    """
    load jsonl file
    """
    data = []
    with open(jsonl_path, 'r', encoding='UTF-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException("pd.read_parquet timeout!")


def safe_read_parquet(path, timeout=10):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)  
    try:
        return pd.read_parquet(path)
    finally:
        signal.alarm(0) 



class CustomLeRobotDataset(Dataset):
    def __init__(self,
        data_roots,
        domains,
        task_recap_file = None,
        step_recap_file = None,
        sample_size=(192, 256), 
        sample_n_frames=64,
        preprocess = 'resize',
        valid_cam = ['observation.images.top_head', 'observation.images.hand_left', 'observation.images.hand_right'],
        chunk=1,
        action_chunk=None,
        n_previous=-1,
        previous_pick_mode='uniform',
        random_crop=True,
        dataset_info_cache_path = None,
        action_type = "absolute",
        action_space = "joint",
        train_dataset=True,
        action_key = "action",
        state_key = "observation.state",
        use_unified_prompt = False,
        unified_prompt = "best quality, consistent and smooth motion, realistic, clear and distinct.",
        fix_epiidx = None,
        fix_sidx = None,
        fix_mem_idx = None,
    ):
        """
        data_roots:              directory of LeRoBot dataset
        domains:                 name of your dataset, used to index different statistics
        task_recap_file:         json file of augmented task captions:
                                 {
                                    'ori_task_caption_1': ['new_caption_1', 'new_caption_2'...],
                                    'ori_task_caption_2': ['new_caption_1', 'new_caption_2'...],
                                 }
        step_recap_file:         json file of augmented step captions:
                                 {
                                    'ori_step_caption_1': ['new_caption_1', 'new_caption_2'...],
                                    'ori_step_caption_2': ['new_caption_1', 'new_caption_2'...],
                                 }
        sample_size:             video frame size
        sample_n_frames:         number of frames used to randomly or uniformly select memories
        preprocess:              frame preprocessing strategy, resize or center_crop_resize
        valid_cam:               list of cam names 
        chunk:                   number of video frames to predict
        action_chunk:            number of actions to predict, action_chunk should be an integer multiple of chunk.
        n_previous:              number of memory frames
        previous_pick_mode:      how to select memories
        random_crop:             randomly crop images
        dataset_info_cache_path: path to save dataset meta information cache
        action_type:             action space to use in this dataset
                                    'absolute': norm(act_t)
                                    'delta':    norm(act_t - act_{t-1})
                                    'relative': norm(act_t) - norm(state)
        action_space:            joint or eef, which is used to determinate the statistics values only in this dataset
        ignore_seek:             if True, load the first furture frame only
        use_unified_prompt:      if set all prompt the same
        unified_prompt:          unified prompt
        fix_epiidx:              used in validation stage only, set episode index to fix_epiidx
        fix_sidx:                used in validation stage only, set start index to fix_sidx
        fix_mem_idx:             used in validation stage only, set memory indexes to fix_mem_idx
        """
        
        zero_rank_print(f"loading annotations...")

        assert(action_type in ["delta", "absolute", "relative"])
        self.action_type = action_type
        assert(action_space in ["eef", "joint"])
        self.action_space = action_space


        self.task_to_indices = defaultdict(list)
        self.action_key = action_key
        self.state_key = state_key

        self.random_crop = random_crop
        
        if not isinstance(valid_cam, (list, tuple)):
            valid_cam = [valid_cam, ]
        self.valid_cam = valid_cam
        if len(data_roots) == 1 and len(domains) > 1:
            data_roots = data_roots * len(domains)
        self.data_roots = data_roots
        self.dataset = []
        self.bench_name = []

        self.index_to_task = {}
        
        if dataset_info_cache_path is not None and os.path.exists(dataset_info_cache_path):
            zero_rank_print(f"Load Cache Dataset Information from {dataset_info_cache_path}")
            with open(dataset_info_cache_path, "r") as f:
                self.dataset = json.load(f)
        else:
            # construct the dataset_info
            cnt = 0
            for _data_root, _domain_name in zip(self.data_roots, domains):
                

                print(f"Loading {_domain_name} data from {_data_root}")
                
                # into the meta folder
                meta_folder = os.path.join(_data_root, _domain_name, "meta")
                data_folder = os.path.join(_data_root, _domain_name, "data")
                video_folder = os.path.join(_data_root, _domain_name, "videos_small")
                


                with open(os.path.join(meta_folder, "info.json"), "r") as f:
                    metainfo = json.load(f)
                    total_chunks = metainfo["total_chunks"]
                    chunks_size = metainfo["chunks_size"]

                episodes_jsonl = os.path.join(meta_folder, "episodes.jsonl")
                epiosdes_data = load_jsonl(episodes_jsonl) # episode_index  tasks  length

                for episode_data in tqdm(epiosdes_data):

                    episode_index = episode_data['episode_index']
                    tasks = episode_data['tasks']
                    if len(tasks) > 1:
                        task = random.choice(tasks)
                    else:
                        task = tasks[0]
                    length = episode_data['length']
                    
                    episode_chunk = int(episode_index//chunks_size)


                    
                    parquet_path = os.path.join(data_folder, f"chunk-{episode_chunk:03d}", f"episode_{episode_index:06d}.parquet")
                    if not os.path.exists(parquet_path):
                        zero_rank_print(f"parquet file not found: {parquet_path}")
                        continue

                    video_path = os.path.join(video_folder, f"chunk-{episode_chunk:03d}", "{}", f"episode_{episode_index:06d}.mp4")
                    # import pdb
                    # pdb.set_trace()
                    info = [
                        video_path,
                        None, # no need for camera_info
                        parquet_path,
                        _domain_name,
                        None, task, # no task_info
                        length,
                    ]
                    self.dataset.append(info)
                    self.index_to_task[len(self.dataset)-1] = cnt

                
                self.bench_name.append(_domain_name)
                self.task_to_indices[cnt] = list(range(len(self.dataset)-len(epiosdes_data), len(self.dataset)))
                cnt += 1

        if dataset_info_cache_path is not None and not(os.path.exists(dataset_info_cache_path)):
            zero_rank_print(f"Save Cache Dataset Information to {dataset_info_cache_path}")
            with open(dataset_info_cache_path, "w") as f:
                json.dump(self.dataset, f)

        self.length = len(self.dataset)

        zero_rank_print(f"data scale: {self.length}")

        self.chunk = chunk
        if action_chunk is None:
            action_chunk = chunk
        self.action_chunk = action_chunk
        # action_chunk 54 chunk 9
        self.video_temporal_stride = self.action_chunk // self.chunk
        assert(self.chunk * self.video_temporal_stride == self.action_chunk)

        self.sample_n_frames = sample_n_frames
        
        self.sample_size = sample_size

        if preprocess == 'center_crop_resize':
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(min(sample_size)),  # the size of shape (1,) means the smaller edge will be resized to it and the img will keep the h-w ratio.
                transforms.CenterCrop(sample_size),
            ])
        if preprocess == 'resize':
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(sample_size),
            ])
        self.pixel_transforms_norm = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.preprocess = preprocess

        if n_previous > 1:
            self.n_previous = n_previous
            self.previous_pick_mode = previous_pick_mode
        else:
            self.n_previous = self.sample_n_frames - self.chunk
            self.previous_pick_mode = 'uniform'

        if task_recap_file is not None:
            with open(task_recap_file, 'r', encoding='UTF-8') as f:
                self.task_recap_map = json.load(f)
        else:
            self.task_recap_map = None

        if step_recap_file is not None:
            with open(step_recap_file, 'r', encoding='UTF-8') as f:
                self.step_recap_map = json.load(f)
        else:
            self.step_recap_map = None

        self.use_unified_prompt = use_unified_prompt

        ### validation only
        self.fix_epiidx = fix_epiidx
        self.fix_sidx = fix_sidx
        self.fix_mem_idx = fix_mem_idx

    def get_frame_indexes(self, total_frames, domain_name):
        """
        select self.n_previous memory frames and self.action_chunk prediction frmaes
        1. randomly select the end frame
        2. take frames from {end-action_chunk} to {end} as the prediction frames
        3. uniformly/randomly select memory frames from {end-self.sample_n_frames} to {end-action_chunk}
        """

        if self.fix_sidx is not None and self.fix_mem_idx is not None:
            action_indexes = list(range(self.fix_sidx, self.fix_sidx+self.action_chunk))
            frame_indexes = action_indexes[::self.video_temporal_stride]
            return self.fix_mem_idx + frame_indexes, self.fix_mem_idx + action_indexes

        is_agibot = domain_name.isdigit()
        
        if not is_agibot:  # galaxea
            chunk_end = random.randint(self.action_chunk + self.video_temporal_stride * 4, total_frames)
            video_end = np.array(list(range(chunk_end-self.action_chunk, chunk_end)))
            anchor = video_end[self.video_temporal_stride - 1]
            mem_indexes = [anchor - i * self.video_temporal_stride for i in range(4, 0, -1)]
        
            frame_indexes = list(mem_indexes) + list(video_end[self.video_temporal_stride-1::self.video_temporal_stride])

            action_indexes = list(mem_indexes) + list(video_end)
            
            act_tokens_index = [i for i in video_end[self.video_temporal_stride-1::self.video_temporal_stride]]
            
            return frame_indexes, action_indexes, act_tokens_index
        else:  # agibot 
            agibot_stride = self.video_temporal_stride * 2
            agibot_action_chunk = self.action_chunk * 2
            chunk_end = random.randint(agibot_action_chunk + agibot_stride * 4, total_frames)
            video_end = np.array(list(range(chunk_end-agibot_action_chunk, chunk_end)))
            anchor = video_end[agibot_stride - 1]
            mem_indexes = [anchor - i * agibot_stride for i in range(4, 0, -1)]
        
            frame_indexes = list(mem_indexes) + list(video_end[agibot_stride-1::agibot_stride])

            action_indexes = list(mem_indexes) + list(video_end)
            
            act_tokens_index = [i for i in video_end[agibot_stride-1::agibot_stride]]
            
            return frame_indexes, action_indexes, act_tokens_index

    def get_task_id(self, index):
        return self.index_to_task[index]

    def seek_mp4(self, video_path, cam_name_list, slices):
        """
        seek video frames according to the input slices;
        output video shape: (c,v,t,h,w)
        """
        video_list = []
        for cam_name in cam_name_list:
            try:
                video_reader = VideoFileClip(video_path.format(cam_name))
                fps = video_reader.fps
                video = []

                for idx in slices:
                    try:
                        frame = video_reader.get_frame(float(idx) / fps)
                        video.append(frame)
                    except Exception as e:
                        print(f"[Error] Failed in get_frame {cam_name}: {e}")
                        video = None
                        break
            except Exception as e:
                print(f"[Error] Failed to open video {cam_name}: {e}")
                video = None
            finally:
                if 'video_reader' in locals():
                    video_reader.close()

            if video is None:
                return None  
            else:
                video = torch.from_numpy(np.stack(video)).permute(3, 0, 1, 2).contiguous()
                video = video.float() / 255.
                video_list.append(video)

        sizes = [(v.shape[-2], v.shape[-1]) for v in video_list]
        all_same_size = all(s == sizes[0] for s in sizes)

        if not all_same_size:
            
            min_h = min(h for h, _ in sizes)
            min_w = min(w for _, w in sizes)
            sample_size = (min_h, min_w)
            resize_transform = transforms.Resize(sample_size)  # (H, W)
            video_list = [resize_transform(v) for v in video_list]
        else:
            
            pass

        video_list = torch.stack(video_list, dim=1)
        return video_list


    def normalize_video(self, video, specific_transforms_norm):
        """
        input video should have shape (c,v,t,h,w)
        """
        c,v,t,h,w = video.shape
        video = specific_transforms_norm(video.permute(1,2,0,3,4).reshape(-1,c,h,w)).reshape(v,t,c,h,w).permute(2,0,1,3,4)
        return video


    def get_transform(self, ):
        sample_size = self.sample_size
        specific_transforms_resize = self.pixel_transforms_resize
        specific_transforms_norm = self.pixel_transforms_norm
        return sample_size, specific_transforms_resize, specific_transforms_norm


    def get_long_recaption(self, step_captions, task_caption):
        newcap = []
        # find = []
        for step_caption in step_captions:
            if self.step_recap_map is not None:
                recap_list = self.step_recap_map.get(step_caption,[])
                recap_list.append(step_caption)
                step_caption = np.random.choice(recap_list,1)
                newcap.append(str(step_caption[0]))
            else:
                newcap.append(step_caption)

        newcap = ", ".join(newcap)
        newcap = newcap.replace(" the "," ")
        if self.task_recap_map is not None:
            task_recap_list = self.task_recap_map.get(task_caption,[])
            task_recap_list.append(task_caption)
            task_newcap = np.random.choice(task_recap_list,1)
            task_newcap = str(task_newcap[0])
            fullcap = task_newcap + ": " + newcap
        else:
            task_newcap = task_caption
            fullcap = task_caption + ": " + newcap
        cap_type = random.randint(0,2)
        allcap = [fullcap, task_newcap, newcap]
        recap = allcap[cap_type]
        return recap

    @staticmethod
    def ensure_array(x):
        if isinstance(x, np.ndarray):
            return x
        else:
            return np.array([x], dtype=float)

    def get_batch(self, idx):
        
        video_path = self.dataset[idx][0]
        parquet_path = self.dataset[idx][2]
        domain_name = self.dataset[idx][3]
        caption = self.dataset[idx][5]
        total_frames = self.dataset[idx][6]
        
        
        sample_size, specific_transforms_resize, specific_transforms_norm = self.get_transform()
        vid_indexes, indexes, act_tokens_index = self.get_frame_indexes(total_frames, domain_name)
        
        data = pd.read_parquet(parquet_path)

        try:
            if 'action' in data.keys():
                action = np.stack([data['action'][i] for i in range(data['action'].shape[0])])
                if action.shape[1] == 14:
                    return None, None, None
            else:
                cols = ['action.left_arm', 'action.left_gripper', 'action.right_arm', 'action.right_gripper']
                action = np.stack([
                    np.concatenate([self.ensure_array(data.at[i, col]) for col in cols])
                    for i in range(len(data))
                ])
                if action.shape[1] == 16:
                    return None, None, None
                # print(action.shape)
                
        except:
            raise ValueError("We currently only support action and state data with the shape of T*C!")

        action_tokens_need = action[act_tokens_index].astype(np.float32)
        action_tokens_need = torch.FloatTensor(action_tokens_need)

        
        min_val_1 = NORM_SET["MIN_VAL_PRETRAIN_1"]
        max_val_1 = NORM_SET["MAX_VAL_PRETRAIN_1"] 
        
        min_val_2 = NORM_SET["MIN_VAL_PRETRAIN_2"]
        max_val_2 = NORM_SET["MAX_VAL_PRETRAIN_2"]

        if action_tokens_need.shape[1] == 14:
            min_val = min_val_1
            max_val = max_val_1
        elif action_tokens_need.shape[1] == 16:
            min_val = min_val_2
            max_val = max_val_2
        else:
            raise ValueError("We currently only support action data with the shape of 14 or 16!")
        
        action_tokens_need = (action_tokens_need - torch.FloatTensor(min_val)) / (torch.FloatTensor(max_val) - torch.FloatTensor(min_val))
        action_tokens_need = action_tokens_need * 2.0 - 1.0

        pad_size = 30 - action_tokens_need.shape[1]
        action_tokens_need = torch.nn.functional.pad(action_tokens_need, (0, pad_size), mode='constant', value=0.0)
        

        videos = self.seek_mp4(video_path, self.valid_cam, vid_indexes)

        if videos is None:
            print("seek_mp4 failed, return None")
            return videos, caption, action_tokens_need

        videos = self.normalize_video(videos, specific_transforms_norm)

        return videos, caption, action_tokens_need


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        
        max_retries = 3 

        if self.fix_epiidx is not None:
            try:
                video, caption, action_tokens = self.get_batch(self.fix_epiidx)
                if video is None:
                    return {} 
                return dict(
                    video=video,
                    caption=caption,
                    action_tokens=action_tokens
                )
            except Exception as e:
                traceback.print_exc()
                print(f"[Error] Failed to load fixed episode {self.fix_epiidx}")
                return {}

        for attempt in range(max_retries):
            try:
                video, caption, action_tokens = self.get_batch(idx)

                if video is None:
                    print(f"[Warning] Video is None for data item {idx}, attempt {attempt+1}/{max_retries}")
                    raise ValueError("Video is None") 

                sample = dict(
                    video=video,
                    caption=caption,
                    action_tokens=action_tokens
                )
                return sample

            except (FileNotFoundError, IOError, ValueError, TypeError, IndexError, KeyError) as e:
                traceback.print_exc()
                print(f"[Warning] Error loading data item {idx}, attempt {attempt+1}/{max_retries}: {e}")

            except Exception as e:
                traceback.print_exc()
                print(f"[Warning] Unexpected error loading data item {idx}, attempt {attempt+1}/{max_retries}: {e}")

        print(f"[Error] Skipping data item {idx} after {max_retries} attempts")
        return {}
