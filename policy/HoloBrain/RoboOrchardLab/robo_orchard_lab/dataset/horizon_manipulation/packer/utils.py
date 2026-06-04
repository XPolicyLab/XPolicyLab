# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import io
import logging
from typing import ClassVar, Dict, List

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, Field
from robo_orchard_core.datatypes.tf_graph import BatchFrameTransformGraph
from robo_orchard_core.utils.logging import LoggerManager
from robo_orchard_core.utils.math.transform import Transform2D_M

from robo_orchard_lab.dataset.datatypes import (
    BatchCameraDataEncoded,
    BatchFrameTransform,
    BatchImageData,
    BatchJointsState,
    Distortion,
    ImageMode,
)
from robo_orchard_lab.dataset.experimental.mcap.batch_split import (
    MakeIterMsgArgs,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_decoder import (
    McapDecoderContext,
)
from robo_orchard_lab.dataset.experimental.mcap.reader import McapReader

# Setup Logger
logger = LoggerManager().get_child(__name__)
logger.setLevel(logging.INFO)


class ParseConfig(BaseModel):
    pass


class McapParseConfig(ParseConfig):
    _DEFAULT_CAMERAS: ClassVar[List[str]] = ["middle", "left", "right"]

    CAMERAS: List[str] = Field(
        default=_DEFAULT_CAMERAS,  # 引用 ClassVar
        description="List of camera names",
    )

    # Topic configurations
    SLAVE_LEFT_JOINT: str = "/observation/robot_state/left/joint"
    SLAVE_RIGHT_JOINT: str = "/observation/robot_state/right/joint"
    MASTER_LEFT_JOINT: str = "/observation/robot_state/left_master/joint"
    MASTER_RIGHT_JOINT: str = "/observation/robot_state/right_master/joint"

    TF_STATIC: str = Field(
        default="/tf_static",
        description="Static TF topic",
    )

    COLOR_IMAGE_TOPICS: List[str] = Field(
        default_factory=lambda: [
            f"/observation/cameras/{cam}/color_image/image_raw"
            for cam in McapParseConfig._DEFAULT_CAMERAS
        ],
        description="List of color image topics for all cameras",
    )

    DEPTH_IMAGE_TOPICS: List[str] = Field(
        default_factory=lambda: [
            f"/observation/cameras/{cam}/depth_image/image_raw"
            for cam in McapParseConfig._DEFAULT_CAMERAS
        ],
        description="List of depth image topics for all cameras",
    )

    COLOR_INFO_TOPICS: List[str] = Field(
        default_factory=lambda: [
            f"/observation/cameras/{cam}/color_image/camera_info"
            for cam in McapParseConfig._DEFAULT_CAMERAS
        ],
        description="List of color info topics for all cameras",
    )

    DEPTH_INFO_TOPICS: List[str] = Field(
        default_factory=lambda: [
            f"/observation/cameras/{cam}/depth_image/camera_info"
            for cam in McapParseConfig._DEFAULT_CAMERAS
        ],
        description="List of depth info topics for all cameras",
    )


class PackConfig(BaseModel):
    SYNC_CAMERA: str = Field(
        default="/observation/cameras/middle/color_image/image_raw",
        description="Camera to use as the base timeline for synchronization",
    )

    IMAGE_SCALE: float = Field(
        default=1,
        description="Scale factor for downsampling images (0 < scale <= 1)",
        gt=0,
        le=1,
    )

    STATIC_THRESHOLD: float = Field(
        default=1e-3,
        description="Threshold for detecting static frames based on joint movement",  # noqa: E501
        ge=0,
    )
    HEAD_TIME_TO_FILTER: float | None = Field(
        default=None,
        description="Time in seconds to filter from the start of the episode",
        ge=0,
    )
    TAIL_TIME_TO_FILTER: float | None = Field(
        default=None,
        description="Time in seconds to filter from the end of the episode",
        ge=0,
    )
    PARSE_CONFIG: ParseConfig = Field(
        default=McapParseConfig(),
        description="Configuration for parsing MCAP files",
    )
    EXTRINSIC_OVERRIDES: Dict[str, BatchFrameTransform] | None = Field(
        default=None,
        description=(
            "Mapping from camera topic to replacement extrinsic transform "
            "in BatchFrameTransform format"
        ),
    )


def _scale_camera_streams(
    image_data: List[Dict[str, BatchCameraDataEncoded]],
    image_scale_ratio: float,
):
    if image_scale_ratio == 1.0 or len(image_data) == 0:
        return

    def color_decoder(image_bytes: bytes, format: str) -> BatchImageData:
        pil_img = Image.open(io.BytesIO(image_bytes))
        img_tensor = torch.from_numpy(np.array(pil_img)).unsqueeze(0)
        return BatchImageData(sensor_data=img_tensor, pix_fmt=ImageMode.RGB)

    def depth_decoder(image_bytes: bytes, format: str) -> BatchImageData:
        pil_img = Image.open(io.BytesIO(image_bytes))
        img_tensor = (
            torch.from_numpy(np.array(pil_img)).unsqueeze(0).unsqueeze(-1)
        )
        return BatchImageData(sensor_data=img_tensor, pix_fmt=ImageMode.I16)

    def color_encoder(color_msg: BatchImageData) -> list[bytes]:
        sensor_data = color_msg.sensor_data
        b, h, w, c = sensor_data.shape

        image_byte_list = []
        for i in range(b):
            pil_img = Image.fromarray(sensor_data[i].numpy().astype(np.uint8))
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=95)
            image_byte_list.append(buf.getvalue())
        return image_byte_list

    def depth_encoder(depth_msg: BatchImageData) -> list[bytes]:
        sensor_data = depth_msg.sensor_data
        b, h, w, c = sensor_data.shape

        image_byte_list = []
        for i in range(b):
            pil_img = Image.fromarray(
                sensor_data[i].numpy().astype(np.uint16)[:, :, 0]
            )
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            image_byte_list.append(buf.getvalue())
        return image_byte_list

    for msg_dict in image_data:
        for topic, msg_data in msg_dict.items():
            if "depth" in topic:
                decoder = depth_decoder
                encoder = depth_encoder
            else:
                decoder = color_decoder
                encoder = color_encoder
            # Decode Image Bytes to BatchImageData
            decoded_msg = msg_data.decode(decoder=decoder)

            # Resize Image using scale ratio
            height, width = msg_data.image_shape
            target_hw = (
                int(height * image_scale_ratio),
                int(width * image_scale_ratio),
            )

            scale_matrix = torch.eye(3)
            scale_matrix[0, 0] = image_scale_ratio
            scale_matrix[1, 1] = image_scale_ratio
            scale_transform = Transform2D_M(matrix=scale_matrix)
            scaled_decoded_msg = decoded_msg.apply_transform2d(
                transform=scale_transform, target_hw=target_hw
            )

            # Encode to Image Bytes
            resized_image_bytes = encoder(scaled_decoded_msg)

            # update msg_data
            msg_data.sensor_data = resized_image_bytes
            msg_data.image_shape = target_hw
            msg_data.intrinsic_matrices = scaled_decoded_msg.intrinsic_matrices


def override_extrinsics(
    batch_tf_list: List[BatchFrameTransform],
    camera_topics: Dict[str, BatchCameraDataEncoded],
    extrinsic_overrides: Dict[str, BatchFrameTransform] | None,
) -> List[BatchFrameTransform]:
    if not extrinsic_overrides:
        return batch_tf_list

    extrinsic_map = {}
    for topic, extrinsic in extrinsic_overrides.items():
        if topic not in camera_topics:
            raise KeyError(
                f"Unknown camera topic for extrinsic override: {topic}"
            )

        original_camera = camera_topics[topic]
        # Reuse the parsed camera frame_id so topic-based overrides still
        # flow through the existing TF lookup and export logic.
        extrinsic_map[
            (extrinsic.parent_frame_id, original_camera.frame_id)
        ] = BatchFrameTransform(
            parent_frame_id=extrinsic.parent_frame_id,
            child_frame_id=original_camera.frame_id,
            xyz=extrinsic.xyz,
            quat=extrinsic.quat,
            timestamps=extrinsic.timestamps,
        )

    updated_batch_tf_list = [
        tf
        for tf in batch_tf_list
        if (tf.parent_frame_id, tf.child_frame_id) not in extrinsic_map
    ]
    updated_batch_tf_list.extend(extrinsic_map.values())
    return updated_batch_tf_list


def apply_camera_calibration_overrides(
    *,
    batch_tf_list: List[BatchFrameTransform],
    image_data: List[Dict[str, BatchCameraDataEncoded]] | None = None,
    image_scale_ratio: float = 1.0,
    extrinsic_overrides: Dict[str, BatchFrameTransform] | None = None,
) -> List[BatchFrameTransform]:
    if image_data is not None:
        _scale_camera_streams(
            image_data=image_data, image_scale_ratio=image_scale_ratio
        )

    camera_topics = {}
    if image_data is not None:
        # Flatten the per-stream dictionaries into one topic index for
        # override lookup.
        for msg_dict in image_data:
            camera_topics.update(msg_dict)

    return override_extrinsics(
        batch_tf_list=batch_tf_list,
        camera_topics=camera_topics,
        extrinsic_overrides=extrinsic_overrides,
    )


def scale_images_and_update_intrinsics(
    image_data: List[Dict[str, BatchCameraDataEncoded]],
    image_scale_ratio: float,
):
    _scale_camera_streams(
        image_data=image_data,
        image_scale_ratio=image_scale_ratio,
    )


def update_camera_poses_from_tf_graph(
    *,
    tf_graph: BatchFrameTransformGraph,
    camera_dict: Dict[str, BatchCameraDataEncoded],
    num_steps: int,
    base_frame_id: str = "world",
):
    for topic, camera_data in camera_dict.items():
        try:
            # Poses are derived from the final TF graph after all overrides
            # and auxiliary transforms have been added.
            pose = tf_graph.get_tf(base_frame_id, camera_data.frame_id)
        except Exception as exc:
            logger.warning(
                "Skip pose update for %s (%s -> %s): %s",
                topic,
                base_frame_id,
                camera_data.frame_id,
                exc,
            )
            continue
        if pose.batch_size == 1 and num_steps > 1:
            pose = pose.repeat(num_steps)
        camera_data.pose = pose


def filter_static_frames(
    data: List[Dict[str, BatchCameraDataEncoded | BatchJointsState]],
    joint_positions: torch.Tensor,
    base_time: List[int],
    static_threshold: float,
    head_time_to_filter: float | None,
    tail_time_to_filter: float | None,
):
    # Filtering
    num_steps = joint_positions.shape[0]
    static_mask = np.ones(num_steps, dtype=bool)
    if static_threshold > 0:
        static_mask[1:] = np.any(
            np.abs(np.diff(joint_positions, axis=0)) > static_threshold,
            axis=1,
        )

    base_time = np.array(base_time)
    time_mask = np.ones(num_steps, dtype=bool)
    if head_time_to_filter is not None:
        time_mask[(base_time - base_time[0]) / 1e9 < head_time_to_filter] = (
            False
        )
    if tail_time_to_filter is not None:
        time_mask[(base_time[-1] - base_time) / 1e9 < tail_time_to_filter] = (
            False
        )

    if head_time_to_filter is None and tail_time_to_filter is None:
        retained_index = static_mask
    else:
        retained_index = static_mask | time_mask

    logger.info(
        f"Filtering: {num_steps} steps -> {retained_index.sum()} steps"
    )

    for msg_dict in data:
        for _, msg_data in msg_dict.items():
            msg_time = np.array(msg_data.timestamps)
            filtered_msg_time = msg_time[retained_index]

            if isinstance(msg_data, BatchJointsState):
                msg_data.position = msg_data.position[retained_index]
                msg_data.velocity = msg_data.velocity[retained_index]
                msg_data.effort = msg_data.effort[retained_index]
                msg_data.timestamps = filtered_msg_time.tolist()

            elif isinstance(msg_data, BatchCameraDataEncoded):
                msg_data.intrinsic_matrices = msg_data.intrinsic_matrices[
                    retained_index
                ]
                msg_data.sensor_data = np.array(msg_data.sensor_data)[
                    retained_index
                ].tolist()

                if msg_data.pose is not None:
                    msg_data.pose = BatchFrameTransform(
                        parent_frame_id=msg_data.pose.parent_frame_id,
                        child_frame_id=msg_data.pose.child_frame_id,
                        xyz=msg_data.pose.xyz[retained_index],
                        quat=msg_data.pose.quat[retained_index],
                    )

                msg_data.timestamps = filtered_msg_time.tolist()


def time_sync(
    data: list[Dict[str, BatchCameraDataEncoded | BatchJointsState]],
    base_time: list[int],
):
    """Synchronizes all data streams to a base timeline.

    Args:
        data (Tuple[Dict[str, BatchCameraDataEncoded | BatchJointsState]]):
            A tuple containing dictionaries of image and joint data.
        base_time (list[int]): The base timestamps to which the source data, in
            nanoseconds(1e-9 seconds).

    """
    assert len(base_time) != 0

    base_time = np.array(base_time)
    for msg_dict in data:
        for topic, msg_data in msg_dict.items():
            msg_time = np.array(msg_data.timestamps)
            time_diff = np.abs(base_time[:, None] - msg_time) / 1e9
            logger.info(
                f"{topic:<50} - "
                + f"max time diff: {time_diff.min(axis=-1).max():.4f}, "
                + f"mean time diff: {time_diff.min(axis=-1).mean():.4f}"
            )

            index = np.argmin(time_diff, axis=1)
            synced_msg_time = msg_time[index]

            if isinstance(msg_data, BatchJointsState):
                msg_data.position = msg_data.position[index]
                msg_data.velocity = msg_data.velocity[index]
                msg_data.effort = msg_data.effort[index]
                msg_data.timestamps = synced_msg_time.tolist()

            elif isinstance(msg_data, BatchCameraDataEncoded):
                msg_data.intrinsic_matrices = msg_data.intrinsic_matrices[
                    index
                ]
                msg_data.sensor_data = [msg_data.sensor_data[i] for i in index]
                msg_data.timestamps = synced_msg_time.tolist()


def parse_mcap(parse_config: McapParseConfig, mcap_path: str):
    """Loads, synchronizes, and filters all data from the MCAP file.

    This is the core data processing method. It performs the following
    steps:
    1.  Finds and opens the MCAP file for the episode.
    2.  Reads all relevant ROS messages (images, joints, TF, etc.).
    """

    def format_time(ts_list):
        """Converts ROS-style timestamps to nanoseconds.

        ROS timestamps are often represented as a tuple or array of two
        integers: seconds and nanoseconds. This function combines them into a
        single int representing the total nanoseconds.

        Args:
            ts_list (list): A list of timestamps, where each element has
                `sec` and `nanosec` attributes.

        Returns:
            list: A list of timestamps in nanoseconds.
        """

        ns_list = []
        for ts in ts_list:
            ns_list.append(int(ts.sec * 1e9) + int(ts.nanosec))
        return ns_list

    joint_topics = [
        parse_config.SLAVE_LEFT_JOINT,
        parse_config.SLAVE_RIGHT_JOINT,
        parse_config.MASTER_LEFT_JOINT,
        parse_config.MASTER_RIGHT_JOINT,
    ]
    all_topics = (
        parse_config.COLOR_IMAGE_TOPICS
        + parse_config.DEPTH_IMAGE_TOPICS
        + parse_config.COLOR_INFO_TOPICS
        + parse_config.DEPTH_INFO_TOPICS
        + joint_topics
        + [parse_config.TF_STATIC]
    )

    tf_list = []
    images, depths, joints, image_infos, depth_infos = {}, {}, {}, {}, {}
    for topic in parse_config.COLOR_IMAGE_TOPICS:
        images[topic] = []
        image_infos[topic] = []
    for topic in parse_config.DEPTH_IMAGE_TOPICS:
        depths[topic] = []
        depth_infos[topic] = []
    for topic in joint_topics:
        joints[topic] = []

    with open(mcap_path, "rb") as f:
        reader = McapReader.make_reader(f)
        for msg_tuple in reader.iter_decoded_messages(
            decoder_ctx=McapDecoderContext(),
            iter_config=MakeIterMsgArgs(topics=all_topics),
        ):
            msg, topic = msg_tuple.decoded_message, msg_tuple.channel.topic
            if topic in images:
                images[topic].append(msg)
            elif topic in depths:
                depths[topic].append(msg)
            elif topic in joints:
                joints[topic].append(msg)
            elif topic in parse_config.COLOR_INFO_TOPICS:
                idx = parse_config.COLOR_INFO_TOPICS.index(topic)
                img_topic = parse_config.COLOR_IMAGE_TOPICS[idx]
                image_infos[img_topic].append(msg)
            elif topic in parse_config.DEPTH_INFO_TOPICS:
                idx = parse_config.DEPTH_INFO_TOPICS.index(topic)
                dpt_topic = parse_config.DEPTH_IMAGE_TOPICS[idx]
                depth_infos[dpt_topic].append(msg)
            elif topic == parse_config.TF_STATIC:
                for tf in msg.transforms:
                    tf_list.append(tf)

    assert image_infos.keys() == images.keys()
    assert depth_infos.keys() == depths.keys()

    # Transfer TF to BatchFrameTransform
    batch_tf_list = []
    tf_pair_set = set()
    for tf in tf_list:
        if (tf.header.frame_id, tf.child_frame_id) in tf_pair_set:
            continue
        tf_pair_set.add((tf.header.frame_id, tf.child_frame_id))
        xyz = [
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        ]
        quat = [
            tf.transform.rotation.w,
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
        ]
        batch_tf_msg = BatchFrameTransform(
            parent_frame_id=tf.header.frame_id,
            child_frame_id=tf.child_frame_id,
            xyz=torch.tensor(xyz),
            quat=torch.tensor(quat),
        )
        batch_tf_list.append(batch_tf_msg)

    # Transfer Joint to BatchJointsState
    batch_joint_dict = {}
    for topic in joints:
        frame_num = len(joints[topic])
        pos = np.array(
            [joints[topic][idx].position for idx in range(frame_num)]
        )
        vel = np.array(
            [joints[topic][idx].velocity for idx in range(frame_num)]
        )
        eff = np.array([joints[topic][idx].effort for idx in range(frame_num)])

        joint_ts_ns = [
            joints[topic][idx].header.stamp for idx in range(frame_num)
        ]
        joint_ts_ns = format_time(joint_ts_ns)
        joint_names = [f"{topic}/{name}" for name in joints[topic][0].name]

        # agilex piper hardcode: if only has 6 dimensions, pad it with zeros
        if vel.shape[1] == 6:
            new_vel = np.zeros_like(pos)
            new_vel[:, :6] = vel
            vel = new_vel

        batch_joint_msg = BatchJointsState(
            position=torch.from_numpy(pos).to(dtype=torch.float32),
            velocity=torch.from_numpy(vel).to(dtype=torch.float32),
            effort=torch.from_numpy(eff).to(dtype=torch.float32),
            names=joint_names,
            timestamps=joint_ts_ns,
        )
        batch_joint_dict[topic] = batch_joint_msg

    # Transfer Image to BatchCameraDataEncoded
    batch_image_dict = {}
    for topic in images:
        frame_num = len(images[topic])
        if len(images[topic]) == len(image_infos[topic]):
            intrinsic = [image_infos[topic][idx].p for idx in range(frame_num)]
            intrinsic = np.array(intrinsic).reshape(frame_num, 3, 4)
            intrinsic_matrix = torch.from_numpy(intrinsic[..., :3, :3]).to(
                dtype=torch.float32
            )
        else:
            intrinsic = image_infos[topic][0].p
            intrinsic = np.array(intrinsic).reshape(3, 4)
            intrinsic_matrix = (
                torch.from_numpy(intrinsic[..., :3, :3])
                .repeat(frame_num, 1, 1)
                .to(dtype=torch.float32)
            )

        frame_id = images[topic][0].header.frame_id
        image_hw = (image_infos[topic][0].height, image_infos[topic][0].width)
        distortion = Distortion(
            model="plumb_bob", coefficients=torch.zeros(5, dtype=torch.float32)
        )
        sensor_data = [images[topic][idx].data for idx in range(frame_num)]
        image_ts_ns = [
            images[topic][idx].header.stamp for idx in range(frame_num)
        ]
        image_ts_ns = format_time(image_ts_ns)

        batch_image_msg = BatchCameraDataEncoded(
            topic=topic,
            frame_id=frame_id,
            image_shape=image_hw,
            intrinsic_matrices=intrinsic_matrix,
            distortion=distortion,
            sensor_data=sensor_data,
            format="jpeg",
            timestamps=image_ts_ns,
        )
        batch_image_dict[topic] = batch_image_msg

    batch_depth_dict = {}
    for topic in depths:
        frame_num = len(depths[topic])

        if len(depths[topic]) == len(depth_infos[topic]):
            intrinsic = [depth_infos[topic][idx].p for idx in range(frame_num)]
            intrinsic = np.array(intrinsic).reshape(frame_num, 3, 4)
            intrinsic_matrix = torch.from_numpy(intrinsic[..., :3, :3]).to(
                dtype=torch.float32
            )
        else:
            intrinsic = depth_infos[topic][0].p
            intrinsic = np.array(intrinsic).reshape(3, 4)
            intrinsic_matrix = (
                torch.from_numpy(intrinsic[..., :3, :3])
                .repeat(frame_num, 1, 1)
                .to(dtype=torch.float32)
            )

        image_hw = (depth_infos[topic][0].height, depth_infos[topic][0].width)
        frame_id = depths[topic][0].header.frame_id
        distortion = Distortion(
            model="plumb_bob", coefficients=torch.zeros(5, dtype=torch.float32)
        )
        sensor_data = [depths[topic][idx].data for idx in range(frame_num)]
        image_ts_ns = [
            depths[topic][idx].header.stamp for idx in range(frame_num)
        ]
        image_ts_ns = format_time(image_ts_ns)

        batch_depth_msg = BatchCameraDataEncoded(
            topic=topic,
            frame_id=frame_id,
            image_shape=image_hw,
            intrinsic_matrices=intrinsic_matrix,
            distortion=distortion,
            sensor_data=sensor_data,
            format="png",
            timestamps=image_ts_ns,
        )
        batch_depth_dict[topic] = batch_depth_msg

    return batch_tf_list, batch_joint_dict, batch_image_dict, batch_depth_dict


def get_index_tf(data: BatchFrameTransform, index):
    pose = BatchFrameTransform(
        parent_frame_id=data.parent_frame_id,
        child_frame_id=data.child_frame_id,
        xyz=data.xyz[index : index + 1],
        quat=data.quat[index : index + 1],
        timestamps=data.timestamps[index : index + 1]
        if data.timestamps is not None
        else None,
    )
    return pose


def get_index_camera(data: BatchCameraDataEncoded, index):
    ret_dict = {}
    for key in [
        "topic",
        "frame_id",
        "image_shape",
        "distortion",
        "format",
    ]:
        ret_dict[key] = getattr(data, key)

    ret_dict["sensor_data"] = [data.sensor_data[index]]
    ret_dict["timestamps"] = [data.timestamps[index]]
    ret_dict["intrinsic_matrices"] = data.intrinsic_matrices[index : index + 1]

    if data.pose is not None:
        pose = get_index_tf(data.pose, index)
        ret_dict["pose"] = pose

    ret = BatchCameraDataEncoded(**ret_dict)
    return ret
