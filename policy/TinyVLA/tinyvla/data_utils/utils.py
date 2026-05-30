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
    A custom PyTorch Dataset class for episodic data.

    Attributes:
        dataset_path_list (list): List of paths to the dataset files.
        camera_names (list): List of camera names used in the dataset.
        norm_stats (dict): Normalization statistics for actions and states.
        episode_ids (list): List of episode identifiers.
        episode_len (list): List of lengths of each episode.
        chunk_size (int): The size of data chunks to be processed.
        policy_class (str): The class of policy used, affects data processing.
        llava_pythia_process (object): Optional processing object for additional data handling.
        imsize (int): Image size for processing, default is 480.
        augment_images (bool): Flag to determine if image augmentation is applied.
        transformations (list): List of transformations for image augmentation.
        cumulative_len (numpy.ndarray): Cumulative sum of episode lengths.
        max_episode_len (int): Maximum length of an episode.
        is_sim (bool): Flag indicating if the data is from a simulation.
    """
    def __init__(self, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len, chunk_size, policy_class, llava_pythia_process=None, imsize=480):
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
            print("########################Current Image Size is [180,320]###################################")
        if 'diffusion' in self.policy_class:
            self.augment_images = True
        else:
            self.augment_images = False
        self.transformations = None
        a = self.__getitem__(0) # initialize self.is_sim and self.transformations
        if len(a['image_top'].shape) == 4:
            print("%"*40)
            print("There are three views: left, right, top")
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
        Retrieves a data sample for a given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the processed data sample.
        """
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]

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

        # self.is_sim = is_sim
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

        if 'diffusion' in self.policy_class:
            # normalize to [-1, 1]
            action_data = ((action_data - self.norm_stats["action_min"]) / (self.norm_stats["action_max"] - self.norm_stats["action_min"])) * 2 - 1
        else:
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
            language=None
    ):
        """
        Initializes the LlavaPythiaProcess class.

        Args:
            data_args: Arguments related to data processing, expected to have an image_processor attribute.
            tokenizer: Tokenizer object for processing text data.
            language: Optional language parameter, currently not used.
        """
        super().__init__()

        self.data_args = data_args
        self.processor = self.data_args.image_processor
        self.tokenizer = tokenizer
        # self.language = language

    def parse_image(self, image_file):
        """
        Parses and preprocesses an image file.

        Args:
            image_file: The image file to be processed, can be a torch.Tensor.

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
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
        Processes a sample to prepare it for model input.

        Args:
            sample: A dictionary containing the sample data.

        Returns:
            dict: A dictionary containing processed data ready for model input.
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
    Computes normalization statistics for qpos and action data from a list of dataset paths.

    Args:
        dataset_path_list (list): A list of paths to the dataset files.

    Returns:
        tuple: A tuple containing:
            - stats (dict): A dictionary with normalization statistics including:
                - "action_mean": Mean of the action data.
                - "action_std": Standard deviation of the action data, clipped to a minimum of 0.01.
                - "action_min": Minimum value of the action data, slightly adjusted by a small epsilon.
                - "action_max": Maximum value of the action data, slightly adjusted by a small epsilon.
                - "qpos_mean": Mean of the qpos data.
                - "qpos_std": Standard deviation of the qpos data, clipped to a minimum of 0.01.
                - "example_qpos": An example qpos array from the last processed dataset.
            - all_episode_len (list): A list of episode lengths corresponding to each dataset.

    Raises:
        Exception: If there is an error loading a dataset file, it prints the error and exits the program.
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
    Loads and prepares datasets for training and validation.

    Args:
        dataset_dir_l (str or list): Directory or list of directories containing dataset files.
        name_filter (function): A function to filter dataset file names.
        camera_names (list): List of camera names used in the dataset.
        batch_size_train (int): Batch size for training data.
        batch_size_val (int): Batch size for validation data.
        chunk_size (int): Size of data chunks to be processed.
        config (dict): Configuration dictionary containing training arguments.
        skip_mirrored_data (bool, optional): Whether to skip mirrored data files. Defaults to False.
        policy_class (str, optional): Class of policy used, affects data processing. Defaults to None.
        stats_dir_l (str or list, optional): Directory or list of directories for normalization statistics. Defaults to None.
        sample_weights (list, optional): Weights for sampling episodes. Defaults to None.
        train_ratio (float, optional): Ratio of data used for training. Defaults to 0.99.
        return_dataset (bool, optional): Whether to return the dataset objects. Defaults to False.
        llava_pythia_process (object, optional): Optional processing object for additional data handling. Defaults to None.

    Returns:
        tuple: If return_dataset is True, returns a tuple containing:
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
