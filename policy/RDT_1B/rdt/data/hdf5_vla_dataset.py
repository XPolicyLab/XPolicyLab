import os
import fnmatch

import h5py
import yaml
import cv2
import numpy as np

from configs.state_vec import STATE_VEC_IDX_MAPPING


class HDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """
    def __init__(self, use_precomp_lang_embed=False) -> None:
        # Each HDF5 file contains one episode.
        self.HDF5_DIR = os.environ.get("RDT_HDF5_DIR")
        if not self.HDF5_DIR:
            raise ValueError("RDT_HDF5_DIR must be set when loading HDF5 training data.")
        self.DATASET_NAME = os.environ.get("RDT_DATASET_NAME", "robodojo_aloha_hdf5")
        self.lang_embed_dir = os.environ.get("RDT_LANG_EMBED_DIR")
        self.use_precomp_lang_embed = use_precomp_lang_embed
        
        self.file_paths = []
        for root, _, files in os.walk(self.HDF5_DIR):
            for filename in fnmatch.filter(files, '*.hdf5'):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)
        if not self.file_paths:
            raise FileNotFoundError(f"No .hdf5 files found under {self.HDF5_DIR}")
                
        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
    
        # Get each episode's len
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res['state'].shape[0] if valid else 0
            episode_lens.append(_len)
        total_len = np.sum(episode_lens)
        if total_len <= 0:
            raise RuntimeError(f"No valid HDF5 episodes found under {self.HDF5_DIR}")
        self.episode_sample_weights = np.array(episode_lens) / total_len
    
    def __len__(self):
        return len(self.file_paths)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = self.parse_hdf5_file(file_path) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))
    
    @staticmethod
    def _is_robodojo_format(h5_file):
        return "state" in h5_file and "action" in h5_file and "vision" in h5_file

    def _read_bimanual_qpos(self, h5_file, group_name):
        group = h5_file[group_name]
        left = np.concatenate(
            [group["left_arm_joint_states"][:], group["left_ee_joint_states"][:]],
            axis=-1,
        )
        right = np.concatenate(
            [group["right_arm_joint_states"][:], group["right_ee_joint_states"][:]],
            axis=-1,
        )
        return np.concatenate([left, right], axis=-1)

    def _parse_robodojo_images(self, h5_file, step_id, first_idx):
        camera_map = {
            "cam_high": "cam_head",
            "cam_left_wrist": "cam_left_wrist",
            "cam_right_wrist": "cam_right_wrist",
        }
        valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
        mask = np.array(
            [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
        )
        parsed = {}
        for output_key, source_key in camera_map.items():
            imgs = []
            if source_key in h5_file["vision"]:
                colors = h5_file["vision"][source_key]["colors"]
                for i in range(max(step_id - self.IMG_HISORY_SIZE + 1, 0), step_id + 1):
                    imgs.append(self._decode_image(colors[i]))
            if imgs:
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    imgs = np.concatenate(
                        [
                            np.tile(imgs[:1], (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1, 1)),
                            imgs,
                        ],
                        axis=0,
                    )
            else:
                imgs = np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0), dtype=np.uint8)
            parsed[output_key] = imgs
            parsed[f"{output_key}_mask"] = mask.copy()
        return parsed

    def parse_hdf5_file(self, file_path):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },                           
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            if self._is_robodojo_format(f):
                qpos = self._read_bimanual_qpos(f, "state")
            else:
                qpos = f['observations']['qpos'][:]
            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            if num_steps < 128:
                return False, None
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")
            
            # We randomly sample a timestep
            step_id = np.random.randint(first_idx-1, num_steps)
            
            # One RoboDojo task/env group shares a single instruction embedding.
            instruction = self._get_instruction(file_path, f)
            
            # Assemble the meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }
            
            if self._is_robodojo_format(f):
                target_qpos = self._read_bimanual_qpos(f, "action")[step_id:step_id+self.CHUNK_SIZE]
            else:
                target_qpos = f['action'][step_id:step_id+self.CHUNK_SIZE]
            
            # Parse the state and action
            state = qpos[step_id:step_id+1]
            state_std = np.std(qpos, axis=0)
            state_mean = np.mean(qpos, axis=0)
            state_norm = np.sqrt(np.mean(qpos**2, axis=0))
            actions = target_qpos
            if actions.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                actions = np.concatenate([
                    actions,
                    np.tile(actions[-1:], (self.CHUNK_SIZE-actions.shape[0], 1))
                ], axis=0)
            
            # Fill the state/action into the unified vector
            left_arm_dim, right_arm_dim = self._get_arm_dims(f, qpos.shape[-1])
            def fill_in_state(values):
                return self._fill_in_bimanual_state(values, left_arm_dim, right_arm_dim)
            state = fill_in_state(state)
            state_indicator = fill_in_state(np.ones_like(state_std))
            state_std = fill_in_state(state_std)
            state_mean = fill_in_state(state_mean)
            state_norm = fill_in_state(state_norm)
            # If action's format is different from state's,
            # you may implement fill_in_action()
            actions = fill_in_state(actions)
            
            if self._is_robodojo_format(f):
                image_data = self._parse_robodojo_images(f, step_id, first_idx)
                cam_high = image_data["cam_high"]
                cam_high_mask = image_data["cam_high_mask"]
                cam_left_wrist = image_data["cam_left_wrist"]
                cam_left_wrist_mask = image_data["cam_left_wrist_mask"]
                cam_right_wrist = image_data["cam_right_wrist"]
                cam_right_wrist_mask = image_data["cam_right_wrist_mask"]
            else:
                def parse_img(key):
                    if key not in f['observations']['images']:
                        return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0), dtype=np.uint8)
                    imgs = []
                    for i in range(max(step_id-self.IMG_HISORY_SIZE+1, 0), step_id+1):
                        img = f['observations']['images'][key][i]
                        imgs.append(self._decode_image(img))
                    imgs = np.stack(imgs)
                    if imgs.shape[0] < self.IMG_HISORY_SIZE:
                        imgs = np.concatenate([
                            np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                            imgs
                        ], axis=0)
                    return imgs
                cam_high = parse_img('cam_high')
                valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
                cam_high_mask = np.array(
                    [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
                )
                cam_left_wrist = parse_img('cam_left_wrist')
                cam_left_wrist_mask = cam_high_mask.copy()
                cam_right_wrist = parse_img('cam_right_wrist')
                cam_right_wrist_mask = cam_high_mask.copy()
            
            # Return the resulting sample
            # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
            # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
            # if the left-wrist camera is unavailable on your robot
            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high,
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask
            }

    def parse_hdf5_file_state_only(self, file_path):
        """[Modify] Parse a hdf5 file to generate a state trajectory.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            if self._is_robodojo_format(f):
                qpos = self._read_bimanual_qpos(f, "state")
            else:
                qpos = f['observations']['qpos'][:]
            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            if num_steps < 128:
                return False, None
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")
            
            if self._is_robodojo_format(f):
                target_qpos = self._read_bimanual_qpos(f, "action")
            else:
                target_qpos = f['action'][:]
            
            # Parse the state and action
            state = qpos[first_idx-1:]
            action = target_qpos[first_idx-1:]
            
            # Fill the state/action into the unified vector
            left_arm_dim, right_arm_dim = self._get_arm_dims(f, qpos.shape[-1])
            def fill_in_state(values):
                return self._fill_in_bimanual_state(values, left_arm_dim, right_arm_dim)
            state = fill_in_state(state)
            action = fill_in_state(action)
            
            # Return the resulting sample
            return True, {
                "state": state,
                "action": action
            }

    def _lang_embed_path(self, file_path):
        if not self.lang_embed_dir:
            return None
        rel = os.path.relpath(file_path, self.HDF5_DIR)
        parts = rel.split(os.sep)
        if len(parts) >= 3 and parts[-2] == "data":
            task_env = os.path.join(parts[0], parts[1])
        elif len(parts) >= 2:
            task_env = parts[0]
        else:
            return None
        dataset_key = os.path.basename(os.path.normpath(self.HDF5_DIR))
        return os.path.join(self.lang_embed_dir, dataset_key, task_env, "lang_embed.pt")

    def _get_instruction(self, file_path, h5_file):
        embed_path = self._lang_embed_path(file_path)
        if self.use_precomp_lang_embed and embed_path and os.path.exists(embed_path):
            return embed_path
        if "instruction" in h5_file:
            instruction = h5_file["instruction"][()]
        else:
            instruction = h5_file.attrs.get("language_instruction", "")
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")
        if not instruction:
            raise ValueError(f"No instruction or lang_embed.pt found for {file_path}")
        return str(instruction)

    def _get_arm_dims(self, h5_file, state_width):
        if self._is_robodojo_format(h5_file):
            left_arm_dim = h5_file["state"]["left_arm_joint_states"].shape[-1] + \
                h5_file["state"]["left_ee_joint_states"].shape[-1]
            right_arm_dim = h5_file["state"]["right_arm_joint_states"].shape[-1] + \
                h5_file["state"]["right_ee_joint_states"].shape[-1]
            return left_arm_dim, right_arm_dim
        obs = h5_file["observations"]
        if "left_arm_dim" in obs and "right_arm_dim" in obs:
            return int(obs["left_arm_dim"][()]), int(obs["right_arm_dim"][()])
        left_arm_dim = state_width // 2
        return left_arm_dim, state_width - left_arm_dim

    def _fill_in_bimanual_state(self, values, left_arm_dim, right_arm_dim):
        values = np.asarray(values, dtype=np.float32)
        expected_dim = left_arm_dim + right_arm_dim
        if values.shape[-1] != expected_dim:
            raise ValueError(f"Expected state dim {expected_dim}, got {values.shape[-1]}")

        uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,), dtype=np.float32)
        left_values = values[..., :left_arm_dim]
        right_values = values[..., left_arm_dim:]

        self._fill_one_arm(uni_vec, left_values, "left")
        self._fill_one_arm(uni_vec, right_values, "right")
        return uni_vec

    def _fill_one_arm(self, uni_vec, arm_values, side):
        joint_dim = max(arm_values.shape[-1] - 1, 0)
        for i in range(min(joint_dim, 10)):
            uni_vec[..., STATE_VEC_IDX_MAPPING[f"{side}_arm_joint_{i}_pos"]] = arm_values[..., i]
        if arm_values.shape[-1] > 0:
            uni_vec[..., STATE_VEC_IDX_MAPPING[f"{side}_gripper_open"]] = arm_values[..., -1]

    @staticmethod
    def _decode_image(img):
        if isinstance(img, np.ndarray) and img.ndim == 3:
            return img.astype(np.uint8, copy=False)
        decoded = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"Failed to decode image with shape {getattr(img, 'shape', None)}")
        return decoded

if __name__ == "__main__":
    ds = HDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)
