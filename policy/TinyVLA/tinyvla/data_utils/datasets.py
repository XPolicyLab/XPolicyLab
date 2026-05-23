import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2
from time import time
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

import IPython
from data_utils.processor import preprocess, preprocess_multimodal
import copy
e = IPython.embed

def flatten_list(l):
    return [item for sublist in l for item in sublist]

class EpisodicDataset(torch.utils.data.Dataset):
    """
    It supports loading multi-camera images, actions, and robot states for each time step in an episode,
    with optional augmentations and normalization. The data is loaded from HDF5 files, and padding is applied
    to ensure consistent episode lengths.
    """
    def __init__(self, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len, chunk_size, policy_class, llava_pythia_process=None, imsize=480):
        """
        Initializes the EpisodicDataset class with the given parameters.
        
        Args:
            dataset_path_list (list of str): A list of file paths to the dataset files (e.g., HDF5 files), 
                                              each corresponding to a different episode.
            camera_names (list of str): A list of camera names (e.g., ['top', 'left', 'right']) indicating the 
                                        cameras whose images are collected for each episode.
            norm_stats (dict): A dictionary containing normalization statistics, including:
                               - "action_mean" and "action_std" for normalizing actions,
                               - "qpos_mean" and "qpos_std" for normalizing robot states (qpos).
            episode_ids (list of int): A list of episode IDs that correspond to each entry in `dataset_path_list`. 
                                       Each episode ID helps in locating the dataset file for that specific episode.
            episode_len (list of int): A list of integers representing the lengths of each episode (number of time steps).
                                       Used for calculating cumulative lengths and indexing.
            chunk_size (int): The number of time steps (or data points) to return in each sample when the dataset is accessed.
            policy_class (str): A string indicating the type of policy being used (e.g., "ACT", "diffusion").
            llava_pythia_process (optional): An optional process or model for further processing the data after it is loaded.
                                              This can be set to `None` if no additional processing is required.
            imsize (int, default=480): The image size to which the images should be resized. Default is 480 pixels.
        """
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class
        self.llava_pythia_process = llava_pythia_process
        self.imsize = imsize
        if self.imsize == 320:
            print("########################Current Image Size is [180,320]; maybe due to the pretrain data image size###################################")
        if 'diffusion' in self.policy_class:
            self.augment_images = True
        else:
            self.augment_images = False
        self.transformations = None
        a = self.__getitem__(0) # initialize self.is_sim and self.transformations
        if len(a['image_top'].shape) == 4:
            print("%"*40)
            print("There are three views: left, right, top")
        # is_sim indicates whether the data comes from a simulation environment.
        self.is_sim = False

    def __len__(self):
        return sum(self.episode_len)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts


    def __getitem__(self, index):
        """
        Retrieves a single data sample from the dataset at the specified index.

        Args:
            index (int): The index of the data sample to retrieve.

        Returns:
            Depending on the policy_class, either a tuple of (image_data, qpos_data, action_data, is_pad) 
            or a processed sample dictionary containing image, state, action, is_pad, and raw_lang.
        """
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        # open the corresponding HDF5 file and read the necessary data attributes.
        with h5py.File(dataset_path, 'r') as root:
            try: # some legacy data does not have this attribute
                is_sim = root.attrs['sim']
            except:
                is_sim = False
            compressed = root.attrs.get('compress', False)

            raw_lang = root['language_raw'][0].decode('utf-8')

            action = root['/action'][()]
            original_action_shape = action.shape
            episode_len = original_action_shape[0]

            # qpos represents the robot's position (e.g., joint angles) at the given timestamp (start_ts),
            # qvel represents the robot's velocity (e.g., joint velocities) at the same timestamp.
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
                if self.imsize != image_dict[cam_name].shape[1]:
                    image_dict[cam_name] = cv2.resize(image_dict[cam_name], (320, 180))

            if compressed:
                for cam_name in image_dict.keys():
                    decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                    image_dict[cam_name] = np.array(decompressed_image)

            # get all actions after and including start_ts
            if is_sim:
                action = action[start_ts:]
                action_len = episode_len - start_ts
            else:
                action = action[max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned
        # pad actions to ensure consistent episode lengths.
        padded_action = np.zeros((self.max_episode_len, original_action_shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.max_episode_len)
        is_pad[action_len:] = 1

        padded_action = padded_action[:self.chunk_size]
        is_pad = is_pad[:self.chunk_size]

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # convert the image data from BGR to RGB format 
        if 'top' in self.camera_names:
            image_data = torch.stack([torch.from_numpy(cv2.cvtColor(img.numpy(), cv2.COLOR_BGR2RGB)) for img in image_data], dim=0)

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # augmentation
        if self.transformations is None:
            print('Initializing transformations')
            original_size = image_data.shape[2:]
            ratio = 0.95
            self.transformations = [
                transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                transforms.Resize(original_size, antialias=True),
                transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5) #, hue=0.08)
            ]

        if self.augment_images:
            for transform in self.transformations:
                image_data = transform(image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0

        if 'diffusion' in self.policy_class: # for diffusion
            # normalize to [-1, 1]
            action_data = ((action_data - self.norm_stats["action_min"]) / (self.norm_stats["action_max"] - self.norm_stats["action_min"])) * 2 - 1
        else: # for act
            # normalize to mean 0 std 1
            action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        if self.policy_class == 'ACT':
            return image_data, qpos_data, action_data, is_pad
        sample = {
            'image': image_data,
            'state': qpos_data,
            'action': action_data,
            'is_pad': is_pad,
            'raw_lang': raw_lang
        }
        assert raw_lang is not None, ""
        return self.llava_pythia_process.forward_process(sample)
        # print(image_data.dtype, qpos_data.dtype, action_data.dtype, is_pad.dtype)


class LlavaPythiaProcess:
    def __init__(
            self,
            data_args=None,
            tokenizer=None,
    ):
        super().__init__()

        self.data_args = data_args
        self.processor = self.data_args.image_processor
        self.tokenizer = tokenizer
        # self.language = language

    def parse_image(self, image_file):
        # image_file = self.list_data_dict[i]['image']

        image = image_file
        if isinstance(image, torch.Tensor):
            image = image.permute(0, 2, 3, 1).numpy()
        if self.data_args.image_aspect_ratio == 'pad':
            def expand2square_batch_numpy(pil_imgs, background_color):
                batch_size, height, width, channels = pil_imgs.shape
                max_dim = max(height, width)
                expanded_imgs = np.full((batch_size, max_dim, max_dim, channels), background_color, dtype=np.float32)

                if height == width:
                    expanded_imgs[:, :height, :width] = pil_imgs
                elif height > width:
                    offset = (max_dim - width) // 2
                    expanded_imgs[:, :height, offset:offset + width] = pil_imgs
                else:
                    offset = (max_dim - height) // 2
                    expanded_imgs[:, offset:offset + height, :width] = pil_imgs

                return expanded_imgs

            image = expand2square_batch_numpy(image, tuple(x for x in self.processor.image_mean))
            image = self.processor.preprocess(image, return_tensors='pt', do_normalize=True, do_rescale=False,
                                              do_center_crop=False)['pixel_values']   # B C H W
        else:
            image = self.processor.preprocess(image, return_tensors='pt', do_normalize=True, do_rescale=False,
                                              do_center_crop=False)['pixel_values']
        return image

    def forward_process(self, sample):
        """
        Processes a sample from the dataset to prepare it for model input.

        Args:
            sample (dict): A dictionary containing the sample data. It includes keys like 'image', 'state', 'action', and 'is_pad'.

        Returns:
            dict: A dictionary containing processed data ready for model input. It includes:
                - 'input_ids': Tokenized input IDs for the language model.
                - 'labels': Corresponding labels for the input IDs.
                - 'image': The first image from the sample, processed.
                - 'image_r': The second image from the sample, processed.
                - 'image_top': The third image from the sample, processed (if available).
                - 'state': The state data from the sample.
                - 'action': The action data from the sample.
                - 'is_pad': Padding information for the sample.
        """
        sources = self.datastruct_droid2llava(sample)
        image = self.parse_image(sample['image'])

        if not isinstance(sources, list):
            sources = [sources]
        sources = preprocess_multimodal(
            copy.deepcopy([e["conversations"] for e in sources]),
            self.data_args)

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)

        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         labels=data_dict["labels"][0])

        images_all = torch.chunk(image, image.shape[0], dim=0)
        data_dict['image'] = images_all[0]
        data_dict['image_r'] = images_all[1]
        if image.shape[0] == 3:

            data_dict['image_top'] = images_all[2]
        data_dict['state'] = sample['state']
        data_dict['action'] = sample['action']
        data_dict['is_pad'] = sample['is_pad']
        return data_dict

    def datastruct_droid2llava(self, sample):
        sources = {
            'id': "",
            'image': None,
            'state': [],
            'action': [],
            "conversations": [{"from": "human", "value": "<image>\n"}, {"from": "gpt", "value": " "}]
        }
        sources['action'] = sample['action']
        sources['state'] = sample['state']
        # sources['image'] = sample['obs']['camera/image/varied_camera_1_left_image']
        sources["conversations"][0]["value"] += sample['raw_lang']
        # print(sample['obs']['raw_language'].decode('utf-8'))
        return sources

def get_norm_stats(dataset_path_list):
    """
    Computes normalization statistics for action and qpos data from a list of dataset paths. Prepare to standardize the action data and robot position data (qpos data).

    Args:
        dataset_path_list (list of str): A list of file paths to the dataset files (e.g., HDF5 files).

    Returns:
        tuple: A tuple containing:
            - stats (dict): A dictionary with the following keys:
                - "action_mean": The mean of the action data across all episodes.
                - "action_std": The standard deviation of the action data, clipped to a minimum of 0.01.
                - "action_min": The minimum value of the action data, adjusted by a small epsilon.
                - "action_max": The maximum value of the action data, adjusted by a small epsilon.
                - "qpos_mean": The mean of the qpos data across all episodes.
                - "qpos_std": The standard deviation of the qpos data, clipped to a minimum of 0.01.
                - "example_qpos": An example qpos array from the last processed dataset.
            - all_episode_len (list of int): A list of integers representing the lengths of each episode.

    Raises:
        Exception: If there is an error loading a dataset file, the function will print an error message and terminate the program.
    """
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []

    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                qvel = root['/observations/qvel'][()]
                action = root['/action'][()]
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            print(e)
            quit()
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(qpos))
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
             "action_min": action_min.numpy() - eps,"action_max": action_max.numpy() + eps,
             "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
             "example_qpos": qpos}

    return stats, all_episode_len

def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files

def BatchSampler(batch_size, episode_len_l, sample_weights):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch

def load_data(dataset_dir_l, name_filter, camera_names, batch_size_train, batch_size_val, chunk_size, config, skip_mirrored_data=False, policy_class=None, stats_dir_l=None, sample_weights=None, train_ratio=0.99, return_dataset=False, llava_pythia_process=None):
    """
    Loads and prepares datasets for training and evaluation.

    Args:
        dataset_dir_l (list or str): A list of directories or a single directory containing dataset files.
        name_filter (function): A function to filter dataset files based on their names.
        camera_names (list of str): A list of camera names to be used for data collection.
        batch_size_train (int): The batch size for training.
        batch_size_val (int): The batch size for validation.
        chunk_size (int): The number of time steps to return in each sample.
        config (dict): Configuration dictionary containing various settings, including image size.
        skip_mirrored_data (bool, optional): Whether to skip mirrored data files. Defaults to False.
        policy_class (str, optional): The type of policy being used (e.g., "ACT", "diffusion"). Defaults to None.
        stats_dir_l (list or str, optional): Directories for computing normalization statistics. Defaults to None.
        sample_weights (list, optional): Weights for sampling episodes. Defaults to None.
        train_ratio (float, optional): The ratio of data to be used for training. Defaults to 0.99.
        return_dataset (bool, optional): Whether to return the datasets and parameters. Defaults to False.
        llava_pythia_process (object, optional): An optional process for further data processing. Defaults to None.

    Returns:
        tuple: If `return_dataset` is True, returns a tuple containing:
            - train_dataset (EpisodicDataset): The training dataset.
            - val_dataset (EpisodicDataset): The validation dataset.
            - norm_stats (dict): Normalization statistics.
            - sampler_params (dict): Parameters for data sampling.
    """
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    # obtain train test split on dataset_dir_l[0]
    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    train_episode_ids_0 = shuffled_episode_ids_0[:int(train_ratio * num_episodes_0)]
    val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0):]
    train_episode_ids_l = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for idx, num_episodes in enumerate(num_episodes_l[1:])]
    val_episode_ids_l = [val_episode_ids_0]

    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    print(f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')

    _, all_episode_len = get_norm_stats(dataset_path_list)
    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]
    val_episode_len_l = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l]
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]
    norm_stats, _ = get_norm_stats(flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data) for stats_dir in stats_dir_l]))
    print(f'Norm stats from: {stats_dir_l}')
    print(f'train_episode_len_l: {train_episode_len_l}')

    train_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, train_episode_ids, train_episode_len, chunk_size, policy_class, llava_pythia_process=llava_pythia_process, imsize=config['training_args'].pretrain_image_size)
    val_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, val_episode_ids, val_episode_len, chunk_size, policy_class, llava_pythia_process=llava_pythia_process, imsize=config['training_args'].pretrain_image_size)

    sampler_params = {
        'train': {"batch_size": batch_size_train, 'episode_len_l': train_episode_len_l, 'sample_weights':sample_weights},
        'eval': {"batch_size": batch_size_val, 'episode_len_l': val_episode_len_l, 'sample_weights': None}
    }

    if return_dataset:
        return train_dataset, val_dataset, norm_stats, sampler_params


def calibrate_linear_vel(base_action, c=None):
    if c is None:
        c = 0.0 # 0.19
    v = base_action[..., 0]
    w = base_action[..., 1]
    base_action = base_action.copy()
    base_action[..., 0] = v - c * w
    return base_action

def smooth_base_action(base_action):
    return np.stack([
        np.convolve(base_action[:, i], np.ones(5)/5, mode='same') for i in range(base_action.shape[1])
    ], axis=-1).astype(np.float32)

def preprocess_base_action(base_action):
    # base_action = calibrate_linear_vel(base_action)
    base_action = smooth_base_action(base_action)

    return base_action

def postprocess_base_action(base_action):
    linear_vel, angular_vel = base_action
    linear_vel *= 1.0
    angular_vel *= 1.0
    # angular_vel = 0
    # if np.abs(linear_vel) < 0.05:
    #     linear_vel = 0
    return np.array([linear_vel, angular_vel])

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
