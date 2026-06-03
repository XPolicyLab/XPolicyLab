from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import h5py
import numpy as np
from mcap.reader import make_reader
from mcap_ros1.decoder import DecoderFactory as DecoderFactory
from mcap_ros2.decoder import DecoderFactory as DecoderFactory2
import cv2

def load_hdf5_dataset(
    episode_path: str | Path,
) -> dict:
    """Load hdf5 dataset and return a dict with observations and actions"""

    with h5py.File(episode_path) as f:
        state_images_cam_high = np.array(f["observations/images/cam_high"])
        state_images_cam_left_wrist = np.array(f["observations/images/cam_left_wrist"])
        state_images_cam_right_wrist = np.array(f["observations/images/cam_right_wrist"])
        state_qpos = np.array(f["observations/qpos"])

    assert (
        state_images_cam_high.shape[0]
        == state_images_cam_left_wrist.shape[0]
        == state_images_cam_right_wrist.shape[0]
        == state_qpos.shape[0]
    )

    epi_len = state_images_cam_high.shape[0]
    episode = {
        "observation.state": state_qpos.reshape((epi_len, -1)),
        "observation.images.top_head": state_images_cam_high,
        "observation.images.hand_left": state_images_cam_left_wrist,
        "observation.images.hand_right": state_images_cam_right_wrist,
        "action": state_qpos.reshape((epi_len, -1)),
        "epi_len": epi_len
    }
    return episode

def lazy_load_hdf5_dataset(
    episode_path: str | Path,
) -> Tuple[Dict, h5py.File]:
    """Load hdf5 dataset and return a dict with observations and actions"""
    f = h5py.File(episode_path, 'r')

    state_images_cam_high = f["observations/images/cam_high"]
    state_images_cam_left_wrist = f["observations/images/cam_left_wrist"]
    state_images_cam_right_wrist = f["observations/images/cam_right_wrist"]
    state_qpos = np.array(f["observations/qpos"])

    epi_len = state_qpos.shape[0]
    episode = {
        "observation.state": state_qpos.reshape((epi_len, -1)),
        "observation.images.top_head": state_images_cam_high,
        "observation.images.hand_left": state_images_cam_left_wrist,
        "observation.images.hand_right": state_images_cam_right_wrist,
        "action": state_qpos.reshape((epi_len, -1)),
        "epi_len": epi_len
    }
    return episode, f

def lazy_load_hdf5_dataset_noimg(
    episode_path: str | Path,
) -> Tuple[Dict, h5py.File]:
    """Load hdf5 dataset and return a dict with observations and actions"""
    f = h5py.File(episode_path, 'r')

    state_qpos = np.array(f["observations/qpos"])

    epi_len = state_qpos.shape[0]
    episode = {
        "observation.state": state_qpos.reshape((epi_len, -1)),
        "observation.images.top_head": None,
        "observation.images.hand_left": None,
        "observation.images.hand_right": None,
        "action": state_qpos.reshape((epi_len, -1)),
        "epi_len": epi_len
    }
    return episode, f

def load_mcap_dataset(
    episode_path: str | Path,
) -> dict:
    """Load mcap dataset and return a dict with observations and actions"""
    decoder_factory = DecoderFactory()
    topics: Dict[str, List[Any]] = defaultdict(list)
    with open(episode_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            topic = channel.topic
            message_type = schema.name
            decoded_msg = decoder_factory.decoder_for("ros1", schema)(message.data)
            if message_type.endswith("Image"):
                image_array = np.frombuffer(decoded_msg.data, dtype=np.uint8).reshape((480, 640, 3))
                topics[topic].append(image_array)
            else:
                # Decode ROS JointState message
                topics[topic].append(decoded_msg.position)

    state_images_cam_high = np.stack(topics["/camera/cam_high/image_raw"])
    state_images_cam_left_wrist = np.stack(topics["/camera/cam_left_wrist/image_raw"])
    state_images_cam_right_wrist = np.stack(topics["/camera/cam_right_wrist/image_raw"])
    state_qpos = np.array(topics["/robot/puppet/joint_states"], dtype=np.float32)
    state_qpos_master = np.array(topics["/robot/master/joint_states"], dtype=np.float32)
    state_qpos[:, 6] = state_qpos_master[:, 6]  # sync gripper state
    state_qpos[:, 13] = state_qpos_master[:, 13]  # sync gripper state
    del state_qpos_master

    assert (
        state_images_cam_high.shape[0]
        == state_images_cam_left_wrist.shape[0]
        == state_images_cam_right_wrist.shape[0]
        == state_qpos.shape[0]
    )

    epi_len = state_images_cam_high.shape[0]
    episode = {
        "observation.state": state_qpos,
        "observation.images.top_head": state_images_cam_high,
        "observation.images.hand_left": state_images_cam_left_wrist,
        "observation.images.hand_right": state_images_cam_right_wrist,
        "action": state_qpos.copy(),
        "epi_len": epi_len
    }
    return episode
    # return state_images_cam_high, state_images_cam_left_wrist, state_images_cam_right_wrist, state_qpos_puppet, state_qpos_master


def load_mcap_dataset2(
    episode_path: str | Path,
) -> dict:
    """Load mcap dataset and return a dict with observations and actions, using ROS2"""
    decoder_factory = DecoderFactory2()
    topics: Dict[str, List[Any]] = defaultdict(list)
    with open(episode_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            topic = channel.topic
            message_type = schema.name
            decoded_msg = decoder_factory.decoder_for("cdr", schema)(message.data)
            if message_type.endswith("Image"):
                if topic.endswith("/color/image_raw") and 'fisheye' not in topic:
                    decoded_msg = decoder_factory.decoder_for("cdr", schema)(message.data)
                    np_arr = np.frombuffer(decoded_msg.data, dtype=np.uint8)
                    np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    # to rgb
                    np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                    # 720 * 1280 crop to 720 * 960, from center, and the resize to 480 * 640
                    np_img = np_img[:, 160:1120, :]
                    np_img = cv2.resize(np_img, (640, 480))
                    topics[topic].append(np_img)
            elif message_type.endswith("JointState"):
                decoded_msg = decoder_factory.decoder_for("cdr", schema)(message.data)
                topics[topic].append(decoded_msg.position)
            else:
                continue

    state_images_cam_high = np.stack(topics["/camera_f/color/image_raw"])
    state_images_cam_left_wrist = np.stack(topics["/camera_l/color/image_raw"])
    state_images_cam_right_wrist = np.stack(topics["/camera_r/color/image_raw"])
    puppet_qpos_left = np.array(topics["/puppet/joint_left"], dtype=np.float32)
    puppet_qpos_right = np.array(topics["/puppet/joint_right"], dtype=np.float32)
    master_qpos_left = np.array(topics["/master/joint_left"], dtype=np.float32)
    master_qpos_right = np.array(topics["/master/joint_right"], dtype=np.float32)
    max_len = min(state_images_cam_high.shape[0], state_images_cam_left_wrist.shape[0], state_images_cam_right_wrist.shape[0])
    state_images_cam_high = state_images_cam_high[:max_len]
    state_images_cam_left_wrist = state_images_cam_left_wrist[:max_len]
    state_images_cam_right_wrist = state_images_cam_right_wrist[:max_len]
    # cam at 30hz, joint at 200hz, so we need to downsample joint to 30hz
    puppet_qpos_left = puppet_qpos_left[np.linspace(0, puppet_qpos_left.shape[0]-1, max_len, dtype=np.int32)]
    puppet_qpos_right = puppet_qpos_right[np.linspace(0, puppet_qpos_right.shape[0]-1, max_len, dtype=np.int32)]
    master_qpos_left = master_qpos_left[np.linspace(0, master_qpos_left.shape[0]-1, max_len, dtype=np.int32)]
    master_qpos_right = master_qpos_right[np.linspace(0, master_qpos_right.shape[0]-1, max_len, dtype=np.int32)]

    # return state_images_cam_high, state_images_cam_left_wrist, state_images_cam_right_wrist, puppet_qpos_left, puppet_qpos_right, master_qpos_left, master_qpos_right

    state_qpos = np.concatenate([puppet_qpos_left, puppet_qpos_right], axis=-1)
    state_qpos_master = np.concatenate([master_qpos_left, master_qpos_right], axis=-1)
    state_qpos[:, 6] = state_qpos_master[:, 6]  # sync gripper state
    state_qpos[:, 13] = state_qpos_master[:, 13]  # sync gripper state
    del state_qpos_master

    epi_len = state_images_cam_high.shape[0]
    episode = {
        "observation.state": state_qpos,
        "observation.images.top_head": state_images_cam_high,
        "observation.images.hand_left": state_images_cam_left_wrist,
        "observation.images.hand_right": state_images_cam_right_wrist,
        "action": state_qpos.copy(),
        "epi_len": epi_len
    }
    return episode