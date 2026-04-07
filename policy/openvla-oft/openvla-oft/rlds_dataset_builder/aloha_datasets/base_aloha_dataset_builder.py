from functools import partial
from typing import Any, Dict, Iterator, Tuple

import glob
import os
import sys

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from aloha_datasets.conversion_utils import MultiThreadedDatasetBuilder


DEFAULT_IMAGE_KEYS = {
    "image": "/observations/images/cam_high",
    "left_wrist_image": "/observations/images/cam_left_wrist",
    "right_wrist_image": "/observations/images/cam_right_wrist",
}


def dataset_name_to_env_var(dataset_name: str) -> str:
    normalized = []
    for ch in dataset_name.upper():
        normalized.append(ch if ch.isalnum() else "_")
    return f'{"".join(normalized)}_PREPROCESSED_DIR'


def _read_instruction(root: h5py.File) -> str:
    instruction = root.attrs.get("language_instruction", "")
    if isinstance(instruction, bytes):
        return instruction.decode("utf-8")
    return str(instruction)


def _generate_aloha_examples(
    paths,
    image_keys: Dict[str, str],
    state_key: str,
    action_key: str,
) -> Iterator[Tuple[str, Any]]:
    def _parse_example(episode_path):
        with h5py.File(episode_path, "r") as root:
            actions = root[action_key][()]
            states = root[state_key][()]
            images = {name: root[hdf5_key][()] for name, hdf5_key in image_keys.items()}
            command = _read_instruction(root)

        episode = []
        for i in range(actions.shape[0]):
            observation = {
                "state": np.asarray(states[i], np.float32),
            }
            for obs_key, values in images.items():
                observation[obs_key] = values[i]

            episode.append(
                {
                    "observation": observation,
                    "action": np.asarray(actions[i], dtype=np.float32),
                    "discount": 1.0,
                    "reward": float(i == (actions.shape[0] - 1)),
                    "is_first": i == 0,
                    "is_last": i == (actions.shape[0] - 1),
                    "is_terminal": i == (actions.shape[0] - 1),
                    "language_instruction": command,
                }
            )

        sample = {
            "steps": episode,
            "episode_metadata": {
                "file_path": episode_path,
                "language_instruction": command,
            },
        }
        return episode_path, sample

    for sample in paths:
        yield _parse_example(sample)


class BaseAlohaDatasetBuilder(MultiThreadedDatasetBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    DATASET_NAME = "aloha_dataset"
    PREPROCESSED_DIR = None
    PREPROCESSED_ENV_VAR = None
    IMAGE_KEYS = DEFAULT_IMAGE_KEYS
    STATE_KEY = "/observations/qpos"
    ACTION_KEY = "/action"
    IMAGE_SHAPE = (256, 256, 3)
    STATE_DIM = 14
    ACTION_DIM = 14
    N_WORKERS = 8
    MAX_PATHS_IN_MEMORY = 64
    PARSE_FCN = None

    @classmethod
    def resolve_preprocessed_dir(cls) -> str:
        env_candidates = []
        if cls.PREPROCESSED_ENV_VAR:
            env_candidates.append(cls.PREPROCESSED_ENV_VAR)
        env_candidates.append(dataset_name_to_env_var(cls.DATASET_NAME))
        env_candidates.append("ALOHA_PREPROCESSED_DIR")

        for env_name in env_candidates:
            env_value = os.environ.get(env_name)
            if env_value:
                return env_value

        if cls.PREPROCESSED_DIR is not None:
            return cls.PREPROCESSED_DIR

        builder_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(builder_dir))
        return os.path.join(repo_root, "data", "aloha_preprocessed", cls.DATASET_NAME)

    def get_parse_fcn(self):
        return partial(
            _generate_aloha_examples,
            image_keys=type(self).IMAGE_KEYS,
            state_key=type(self).STATE_KEY,
            action_key=type(self).ACTION_KEY,
        )

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=type(self).IMAGE_SHAPE,
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Main camera RGB observation.",
                                    ),
                                    "left_wrist_image": tfds.features.Image(
                                        shape=type(self).IMAGE_SHAPE,
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Left wrist camera RGB observation.",
                                    ),
                                    "right_wrist_image": tfds.features.Image(
                                        shape=type(self).IMAGE_SHAPE,
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Right wrist camera RGB observation.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(type(self).STATE_DIM,),
                                        dtype=np.float32,
                                        doc="Robot joint state.",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(type(self).ACTION_DIM,),
                                dtype=np.float32,
                                doc="Robot action.",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on first step of the episode.",
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode.",
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on final terminal demo step.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language instruction.",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original preprocessed HDF5 file.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Episode-level language instruction.",
                            ),
                        }
                    ),
                }
            )
        )

    def _split_paths(self):
        dataset_root = type(self).resolve_preprocessed_dir()
        split_paths = {
            "train": sorted(glob.glob(os.path.join(dataset_root, "train", "*.hdf5"))),
        }
        val_paths = sorted(glob.glob(os.path.join(dataset_root, "val", "*.hdf5")))
        if val_paths:
            split_paths["val"] = val_paths
        return split_paths


def make_aloha_builder_class(
    dataset_name: str,
    preprocessed_dir: str = None,
    state_dim: int = 14,
    action_dim: int = 14,
):
    attrs = {
        "DATASET_NAME": dataset_name,
        "PREPROCESSED_DIR": preprocessed_dir,
        "PREPROCESSED_ENV_VAR": dataset_name_to_env_var(dataset_name),
        "STATE_DIM": state_dim,
        "ACTION_DIM": action_dim,
        "__module__": __name__,
    }
    return type(dataset_name, (BaseAlohaDatasetBuilder,), attrs)
