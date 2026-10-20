import numpy as np
import os
import cv2
from XPolicyLab.utils.load_file import load_json, load_yaml

def _validate_config(action_type: str, robot_action_dim_info: dict, source_type: str):
    """
    Validate configuration and return normalized values.

    Args:
        action_type: 'joint' or 'ee'.
        robot_action_dim_info: Dict with keys:
            - 'arm_dim': list[int]
            - 'ee_dim': list[int]
        source_type: 'obs' or 'dataset'.

    Returns:
        arm_dims, ee_dims, num_arms. For action_type='ee', arm_dims holds the
        7-value pose width per arm instead of the joint count.
    """
    if action_type not in {"joint", "ee"}:
        raise ValueError(
            f"Unsupported action_type: {action_type!r}. "
            "Supported values are 'joint' and 'ee'."
        )

    if source_type not in {"obs", "dataset"}:
        raise ValueError(
            f"Unsupported source_type: {source_type!r}. "
            "Supported values are 'obs' and 'dataset'."
        )

    if "arm_dim" not in robot_action_dim_info or "ee_dim" not in robot_action_dim_info:
        raise KeyError("robot_action_dim_info must contain both 'arm_dim' and 'ee_dim'.")

    arm_dims = robot_action_dim_info["arm_dim"]
    ee_dims = robot_action_dim_info["ee_dim"]

    if not isinstance(arm_dims, (list, tuple)) or not isinstance(ee_dims, (list, tuple)):
        raise TypeError("'arm_dim' and 'ee_dim' must be list or tuple.")

    if len(arm_dims) != len(ee_dims):
        raise ValueError(
            f"'arm_dim' and 'ee_dim' must have the same length, "
            f"got {len(arm_dims)} and {len(ee_dims)}."
        )

    if len(arm_dims) not in {1, 2}:
        raise ValueError(
            f"Only single-arm or dual-arm robots are supported, got {len(arm_dims)} arms."
        )

    if any(d <= 0 for d in arm_dims) or any(d <= 0 for d in ee_dims):
        raise ValueError("All dimensions in 'arm_dim' and 'ee_dim' must be positive.")

    # Cartesian poses are [x, y, z, qw, qx, qy, qz], independent of joint DOF.
    if action_type == "ee":
        arm_dims = [7] * len(arm_dims)

    return list(arm_dims), list(ee_dims), len(arm_dims)


def _get_state_keys(action_type: str, num_arms: int, source_type: str):
    """
    Return arm keys and ee keys for the current state schema.

    source_type='obs' uses singular keys:
        single-arm:
            action_type='joint' -> ['joint_state'], ['ee_joint_state']
            action_type='ee'    -> ['ee_pose'], ['ee_joint_state']
        dual-arm:
            action_type='joint' -> ['left_arm_joint_state', 'right_arm_joint_state'],
                                   ['left_ee_joint_state', 'right_ee_joint_state']
            action_type='ee'    -> ['left_ee_pose', 'right_ee_pose'],
                                   ['left_ee_joint_state', 'right_ee_joint_state']

    source_type='dataset' uses plural keys:
        single-arm:
            action_type='joint' -> ['joint_states'], ['ee_joint_states']
            action_type='ee'    -> ['ee_poses'], ['ee_joint_states']
        dual-arm:
            action_type='joint' -> ['left_arm_joint_states', 'right_arm_joint_states'],
                                   ['left_ee_joint_states', 'right_ee_joint_states']
            action_type='ee'    -> ['left_ee_poses', 'right_ee_poses'],
                                   ['left_ee_joint_states', 'right_ee_joint_states']
    """
    suffix = "" if source_type == "obs" else "s"

    if num_arms == 1:
        arm_keys = [f"joint_state{suffix}"] if action_type == "joint" else [f"ee_pose{suffix}"]
        ee_keys = [f"ee_joint_state{suffix}"]
    else:
        if action_type == "joint":
            arm_keys = [
                f"left_arm_joint_state{suffix}",
                f"right_arm_joint_state{suffix}",
            ]
        else:
            arm_keys = [
                f"left_ee_pose{suffix}",
                f"right_ee_pose{suffix}",
            ]

        ee_keys = [
            f"left_ee_joint_state{suffix}",
            f"right_ee_joint_state{suffix}",
        ]

    return arm_keys, ee_keys


def _ensure_valid_state_array(name: str, value, expected_last_dim: int) -> np.ndarray:
    """Convert value to np.ndarray and validate only the last dimension."""
    arr = np.asarray(value)

    if arr.shape[-1] != expected_last_dim:
        raise ValueError(
            f"State field '{name}' last dim mismatch: "
            f"expected {expected_last_dim}, got {arr.shape[-1]}."
        )

    return arr

def pack_robot_state(
    obs: dict,
    action_type: str,
    robot_action_dim_info: dict,
    source_type: str = "obs",
    state_type: str = "state",
) -> np.ndarray:
    """
    Pack robot state from obs['state'] into one vector along the last dimension.

    Packing order:
        single-arm:
            [arm_0, ee_0]
        dual-arm:
            [arm_0, ee_0, arm_1, ee_1]
    """
    if state_type not in obs:
        raise KeyError(f"Input obs must contain a '{state_type}' field.")

    state_dict = obs[state_type]

    arm_dims, ee_dims, num_arms = _validate_config(action_type, robot_action_dim_info, source_type)
    arm_keys, ee_keys = _get_state_keys(action_type, num_arms, source_type)

    parts = []
    expected_prefix_shape = None

    for i, (arm_key, ee_key, arm_dim, ee_dim) in enumerate(
        zip(arm_keys, ee_keys, arm_dims, ee_dims)
    ):
        if arm_key not in state_dict:
            raise KeyError(f"Missing key '{arm_key}' in obs['state'] for arm {i}.")
        if ee_key not in state_dict:
            raise KeyError(f"Missing key '{ee_key}' in obs['state'] for arm {i}.")

        arm_value = _ensure_valid_state_array(arm_key, state_dict[arm_key], arm_dim)
        ee_value = _ensure_valid_state_array(ee_key, state_dict[ee_key], ee_dim)

        if arm_value.shape[:-1] != ee_value.shape[:-1]:
            raise ValueError(
                f"'{arm_key}' and '{ee_key}' must share the same prefix shape, "
                f"got {arm_value.shape[:-1]} and {ee_value.shape[:-1]}."
            )

        if expected_prefix_shape is None:
            expected_prefix_shape = arm_value.shape[:-1]
        elif arm_value.shape[:-1] != expected_prefix_shape:
            raise ValueError(
                "All state fields must share the same prefix shape. "
                f"Expected {expected_prefix_shape}, got {arm_value.shape[:-1]} for '{arm_key}'."
            )

        parts.append(np.concatenate([arm_value, ee_value], axis=-1))

    return np.concatenate(parts, axis=-1)


def unpack_robot_state(
    packed_state,
    action_type: str,
    robot_action_dim_info: dict,
    source_type: str = "obs",
):
    """
    Unpack packed robot state.

    Rules:
        - source_type='obs':
            * ndim must be <= 2
            * if ndim == 1: return dict
            * if ndim == 2: return list[dict]
        - source_type='dataset':
            * return dict of arrays directly

    Unpacking order:
        single-arm:
            [arm_0, ee_0]
        dual-arm:
            [arm_0, ee_0, arm_1, ee_1]
    """
    arm_dims, ee_dims, num_arms = _validate_config(action_type, robot_action_dim_info, source_type)
    arm_keys, ee_keys = _get_state_keys(action_type, num_arms, source_type)

    packed = np.asarray(packed_state)
    expected_dim = sum(arm_dims) + sum(ee_dims)

    if packed.shape[-1] != expected_dim:
        raise ValueError(
            f"packed_state last dim mismatch: expected {expected_dim}, got {packed.shape[-1]}."
        )

    if source_type == "obs":
        assert packed.ndim <= 2, (
            f"When source_type='obs', packed_state.ndim must be <= 2, got {packed.ndim}."
        )

        def _unpack_single_action(single_action: np.ndarray) -> dict:
            result = {}
            offset = 0

            for arm_key, ee_key, arm_dim, ee_dim in zip(
                arm_keys, ee_keys, arm_dims, ee_dims
            ):
                result[arm_key] = single_action[offset : offset + arm_dim]
                offset += arm_dim

                result[ee_key] = single_action[offset : offset + ee_dim]
                offset += ee_dim

            return result

        if packed.ndim == 1:
            return _unpack_single_action(packed)

        return [_unpack_single_action(single_action) for single_action in packed]

    result = {}
    offset = 0

    for arm_key, ee_key, arm_dim, ee_dim in zip(arm_keys, ee_keys, arm_dims, ee_dims):
        result[arm_key] = packed[..., offset : offset + arm_dim]
        offset += arm_dim

        result[ee_key] = packed[..., offset : offset + ee_dim]
        offset += ee_dim

    return result

def get_robot_action_dim_info(env_cfg_type):
    env_cfg = load_yaml(os.path.join(os.path.dirname(__file__), "../../env_cfg", f"{env_cfg_type}.yml"))
    robot_name = env_cfg['config']['robot']
    robot_action_dim_info = load_json(os.path.join(os.path.dirname(__file__), "../../env_cfg/robot", "_robot_info.json"))[robot_name]

    return robot_action_dim_info

def get_batch_size(env_cfg_type):
    env_cfg = load_yaml(os.path.join(os.path.dirname(__file__), "../../env_cfg", f"{env_cfg_type}.yml"))
    sim_cfg = env_cfg['config']['sim']
    sim_info = load_yaml(os.path.join(os.path.dirname(__file__), "../../env_cfg/sim", f"{sim_cfg}.yml"))

    return sim_info['scene']['num_envs']

def get_action_dim(env_cfg_type):
    env_cfg = load_yaml(os.path.join(os.path.dirname(__file__), "../../env_cfg", f"{env_cfg_type}.yml"))
    robot_name = env_cfg['config']['robot']
    robot_action_dim_info = load_json(os.path.join(os.path.dirname(__file__), "../../env_cfg/robot", "_robot_info.json"))[robot_name]
    return sum(robot_action_dim_info["arm_dim"]) + sum(robot_action_dim_info["ee_dim"])

# Two image bit stream formats exist in XPolicyLab data, and both decode to RGB.
#
#   legacy    A JPEG written by handing an RGB array straight to cv2.imencode,
#             which reads its input as BGR. The stored bytes are therefore
#             channel-reversed with respect to the JPEG standard: cv2.imdecode
#             reverses them a second time and returns the original RGB, but any
#             conforming decoder (PIL, ffmpeg, a browser) shows red and blue
#             swapped. All data collected before the marker existed is this.
#   standard  A conforming RGB JPEG written by encode_image_bit, stamped with a
#             COM segment holding _RGB_MARKER_PAYLOAD. cv2.imdecode returns BGR
#             for it, so decoding needs exactly one swap.
#
# The marker lives inside the buffer rather than in an HDF5 attribute because
# decode_image_bit is only ever handed one buffer and has no file context to
# consult. COM is a standard segment carrying its own length, so every decoder
# skips it; PIL exposes it as Image.open(...).info["comment"], which lets a
# non-OpenCV reader tell the two formats apart as well.
#
# COM is a JPEG segment, so the two formats are only distinguishable for JPEG,
# and a buffer in any other container is necessarily read as legacy. That is
# the right reading for stored data — every non-JPEG buffer in the corpus came
# from a legacy producer calling cv2.imencode — but it does mean a conforming
# non-JPEG buffer would decode channel-reversed, which is one more reason
# encode_image_bit writes JPEG and nothing else.
_RGB_MARKER_PAYLOAD = b"XPL-RGB1"

_JPEG_PAD = 0xFF
_JPEG_STUFF = 0x00
_JPEG_TEM = 0x01
_JPEG_RST_FIRST = 0xD0
_JPEG_RST_LAST = 0xD7
_JPEG_SOI = 0xD8
_JPEG_EOI = 0xD9
_JPEG_SOS = 0xDA
_JPEG_APP0 = 0xE0
_JPEG_COM = 0xFE

_COM_SEGMENT_LENGTH_BYTES = 2
_RGB_MARKER_SEGMENT_LENGTH = len(_RGB_MARKER_PAYLOAD) + _COM_SEGMENT_LENGTH_BYTES


def _normalize_image_buffer(image_bit):
    """Reduce any container holding one encoded buffer to bytes or a contiguous
    uint8 array, so the decode and encode paths agree on what they are looking
    at."""
    if isinstance(image_bit, np.ndarray) and image_bit.dtype.kind in {"S", "U"}:
        image_bit = image_bit.item() if image_bit.ndim == 0 else image_bit.tobytes()

    if isinstance(image_bit, str):
        image_bit = image_bit.encode("utf-8")
    elif isinstance(image_bit, memoryview):
        image_bit = image_bit.tobytes()

    if isinstance(image_bit, (bytes, bytearray)):
        # Fixed-width HDF5 byte columns pad the tail with NUL. Stripping is safe
        # for what is actually stored: a JPEG ends with FF D9 and a PNG with the
        # fixed IEND CRC, so neither can end in NUL. A RIFF container such as
        # WebP pads itself to an even length with NUL and would be damaged.
        image_bit = image_bit.rstrip(b"\0")
    elif isinstance(image_bit, np.ndarray):
        image_bit = np.ascontiguousarray(image_bit)

    return image_bit


def _byte_view(image_bit) -> memoryview:
    """A flat, byte-addressable view of one normalized buffer, without copying."""
    if isinstance(image_bit, (bytes, bytearray, memoryview)):
        return memoryview(image_bit).cast("B")

    array = np.ascontiguousarray(image_bit)

    if array.dtype != np.uint8:
        return memoryview(array.tobytes())

    return memoryview(array.reshape(-1))


def _has_rgb_marker(image_bit) -> bool:
    """
    Whether a buffer is a standard RGB JPEG stamped by `encode_image_bit`.

    The header segments are walked rather than probing a fixed offset, so the
    marker is still found if another tool inserted a segment of its own ahead of
    it. Walking stops at SOS, which bounds the work to the header and keeps the
    entropy-coded payload — the bulk of a frame — untouched.
    """
    view = _byte_view(_normalize_image_buffer(image_bit))
    size = len(view)

    if size < 4 or view[0] != _JPEG_PAD or view[1] != _JPEG_SOI:
        return False

    offset = 2

    while offset + 4 <= size:
        if view[offset] != _JPEG_PAD:
            return False

        marker = view[offset + 1]

        if marker == _JPEG_PAD:  # fill byte ahead of the real marker
            offset += 1
            continue

        if marker == _JPEG_STUFF:  # only valid inside entropy-coded data
            return False

        if marker in (_JPEG_TEM, _JPEG_SOI, _JPEG_EOI) or (
            _JPEG_RST_FIRST <= marker <= _JPEG_RST_LAST
        ):
            offset += 2
            continue

        if marker == _JPEG_SOS:  # entropy-coded data starts; no header left
            return False

        length = (view[offset + 2] << 8) | view[offset + 3]

        if length < _COM_SEGMENT_LENGTH_BYTES:
            return False

        if marker == _JPEG_COM and length == _RGB_MARKER_SEGMENT_LENGTH:
            payload = view[offset + 4 : offset + 2 + length]
            if bytes(payload) == _RGB_MARKER_PAYLOAD:
                return True

        offset += 2 + length

    return False


def _insert_rgb_marker(jpeg_bytes: bytes) -> bytes:
    """
    Stamp a JPEG COM segment marking the stream as standard RGB.

    The segment goes after the JFIF APP0 rather than straight after SOI: JFIF
    requires APP0 to follow SOI immediately, and a hand-written parser that
    relies on that layout would otherwise trip over the marker.
    """
    segment = (
        bytes([_JPEG_PAD, _JPEG_COM])
        + _RGB_MARKER_SEGMENT_LENGTH.to_bytes(2, "big")
        + _RGB_MARKER_PAYLOAD
    )

    offset = 2

    if len(jpeg_bytes) >= 6 and jpeg_bytes[2:4] == bytes([_JPEG_PAD, _JPEG_APP0]):
        app0_end = 4 + int.from_bytes(jpeg_bytes[4:6], "big")
        if app0_end <= len(jpeg_bytes):
            offset = app0_end

    return jpeg_bytes[:offset] + segment + jpeg_bytes[offset:]


def _decode_single_image_bit(image_bit):
    """Decode one encoded image buffer into an HWC uint8 RGB array."""
    image_bit = _normalize_image_buffer(image_bit)

    image = cv2.imdecode(np.frombuffer(image_bit, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(
            f"Failed to decode image bits (type={type(image_bit).__name__}, "
            f"size={getattr(image_bit, 'size', len(image_bit) if hasattr(image_bit, '__len__') else '?')})."
        )

    # cv2.imdecode returns BGR for a conforming JPEG. A marked buffer is one, so
    # it needs the swap; a legacy buffer stores reversed channels that imdecode
    # has already reversed back, so swapping it is what would break the order.
    # Either way the caller gets RGB, which is why no caller ever swaps.
    if _has_rgb_marker(image_bit):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def _decode_image_bit_sequence(image_bits):
    frames = []

    for index, image_bit in enumerate(image_bits):
        try:
            frames.append(decode_image_bit(image_bit))
        except ValueError as exc:
            raise ValueError(f"Frame {index}: {exc}") from exc

    if not frames:
        return np.zeros((0,), dtype=np.uint8)

    return np.stack(frames, axis=0)


def decode_image_bit(image_bits):
    """
    Decode encoded image bit stream(s) into uint8 RGB image array(s).

    The output is RGB for both stored byte formats — the legacy
    channel-reversed streams and the standard RGB JPEGs written by
    `encode_image_bit` — because this function reads the marker that tells them
    apart and swaps only where a swap is owed. Never add a COLOR_BGR2RGB after
    this function to "correct" the output: the decoded pixels are
    indistinguishable to the eye — only the marker in the encoded buffer tells
    the formats apart — so a caller-side swap is right on at most one of them
    and silently wrong on the other. This
    is also why hand-rolled decoding is unsupported, PIL included: PIL reads the
    standard format correctly and the legacy format reversed.

    Deliberately converting the RGB result to BGR is a different thing and is
    allowed where a checkpoint was trained on BGR data; that must be an opt-in
    documented in the adapter (see Dexora_1B's `input_color_order`), never a
    silent fix applied at the decode site.

    The marker is a JPEG COM segment, so the guarantee above covers the JPEG
    buffers this corpus stores. A buffer in any other container has no marker to
    read and is treated as legacy, which is correct for the legacy producers but
    means a conforming non-JPEG buffer decodes channel-reversed. Store JPEG,
    which is all `encode_image_bit` writes.

    Values that are already decoded are returned unchanged, so this function is
    safe to call on an observation or trajectory field without knowing whether
    the producer encoded it.

    Dispatch is on dtype first, then ndim:
        - bytes / bytearray / memoryview / str  -> one encoded buffer
        - ndarray of dtype kind 'S', 'U', 'O'   -> sequence of encoded buffers,
                                                   or one buffer when 0-d
        - uint8 ndarray, ndim == 1              -> one encoded buffer
        - uint8 ndarray, ndim == 2              -> (T, N) stack of encoded buffers
        - uint8 ndarray, ndim >= 3              -> already decoded, returned as is
        - ndarray of any other dtype            -> already decoded, returned as is
        - list / tuple                          -> element-wise, stacked on axis 0

    Grayscale (H, W) uint8 images are not supported: a 2-D uint8 array is always
    read as a stack of encoded buffers.

    Raises:
        ValueError: if any buffer fails to decode.
    """
    if isinstance(image_bits, (bytes, bytearray, memoryview, str)):
        return _decode_single_image_bit(image_bits)

    if isinstance(image_bits, np.ndarray):
        if image_bits.dtype.kind in {"S", "U", "O"}:
            if image_bits.ndim == 0:
                return _decode_single_image_bit(image_bits.item())
            return _decode_image_bit_sequence(image_bits)

        if image_bits.dtype == np.uint8:
            if image_bits.ndim == 1:
                return _decode_single_image_bit(image_bits)
            if image_bits.ndim == 2:
                return _decode_image_bit_sequence(image_bits)

        return image_bits

    if isinstance(image_bits, (list, tuple)):
        return _decode_image_bit_sequence(image_bits)

    return _decode_single_image_bit(image_bits)


# Color fields of one camera in an observation. 'depth' is deliberately absent:
# depth is stored as 16-bit or float data that cv2.IMREAD_COLOR would destroy.
OBS_IMAGE_KEYS = ("color", "colors", "rgb", "image")

_DECODABLE_TYPES = (bytes, bytearray, memoryview, np.ndarray, list, tuple)


def _decode_obs_image(value, camera_name, image_key):
    if value is None:
        return value

    try:
        return decode_image_bit(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Failed to decode obs['vision']['{camera_name}']['{image_key}']: {exc}"
        ) from exc


def decode_obs_images(obs):
    """
    Decode the encoded color streams of a runtime observation, in place.

    The policy server calls this before handing an observation to the model, so
    `update_obs` / `update_obs_batch` always receive plain RGB image arrays and
    no adapter has to decode anything. Values that are already decoded pass
    through untouched, and only the color fields listed in OBS_IMAGE_KEYS are
    visited — depth maps, intrinsics, extrinsics and shapes are left alone.

    What the model receives is RGB, for either stored byte format, because
    `decode_image_bit` already resolved the difference. `model.py` must not swap
    channels on top of that.

    Args:
        obs: One observation dict, or a list/tuple of them for batched eval.
             Anything else is returned unchanged.

    Returns:
        The same object, with encoded color fields replaced by decoded arrays.

    Raises:
        ValueError: if a color field fails to decode, naming the camera.
    """
    if isinstance(obs, (list, tuple)):
        for single_obs in obs:
            decode_obs_images(single_obs)
        return obs

    if not isinstance(obs, dict):
        return obs

    vision = obs.get("vision")
    if not isinstance(vision, dict):
        return obs

    for camera_name, camera in vision.items():
        if isinstance(camera, dict):
            for image_key in OBS_IMAGE_KEYS:
                if image_key in camera:
                    camera[image_key] = _decode_obs_image(
                        camera[image_key], camera_name, image_key
                    )
        elif isinstance(camera, _DECODABLE_TYPES):
            # Layout where vision/<camera> is the image itself.
            vision[camera_name] = _decode_obs_image(camera, camera_name, "color")

    return obs

def _encode_single_image_bit(image, quality=None):
    """Encode one uint8 RGB image into a marked standard JPEG buffer."""
    array = np.asarray(image)

    if array.dtype != np.uint8:
        raise TypeError(
            f"Expected a uint8 RGB image, got dtype {array.dtype}. Scale and cast "
            "before encoding — a float array is ambiguous between [0, 1] and "
            "[0, 255], and guessing is how channel and range bugs start."
        )

    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            f"Expected one (H, W, 3) RGB image, got shape {array.shape}. "
            "Grayscale is not supported, and an RGBA frame must be sliced to "
            "[..., :3] first."
        )

    params = [] if quality is None else [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    bgr = cv2.cvtColor(np.ascontiguousarray(array), cv2.COLOR_RGB2BGR)
    success, encoded_image = cv2.imencode(".jpg", bgr, params)

    if not success:
        raise ValueError(f"Failed to JPEG-encode an RGB frame of shape {array.shape}.")

    return _insert_rgb_marker(encoded_image.tobytes())


def _passthrough_encoded_buffer(image_bit):
    """
    Return an already-encoded buffer unchanged, refusing legacy bytes.

    Relabelling a legacy buffer as standard RGB is the one way this scheme can
    silently reverse channels, so an unmarked buffer is an error rather than a
    no-op.
    """
    image_bit = _normalize_image_buffer(image_bit)

    if not _has_rgb_marker(image_bit):
        raise ValueError(
            "Refusing to pass through an unmarked image buffer: it is in the "
            "legacy channel-reversed format, and stamping it as standard RGB "
            "would reverse its channels. Decode it with decode_image_bit and "
            "encode the resulting array if a lossy re-encode is intended."
        )

    if isinstance(image_bit, (bytes, bytearray)):
        return bytes(image_bit)

    # Strip the same NUL padding _normalize_image_buffer takes off the bytes
    # form, so a buffer read out of a fixed-width HDF5 column does not carry its
    # padding into whatever column it is stored in next.
    return np.asarray(image_bit).reshape(-1).tobytes().rstrip(b"\0")


def _encode_image_bit_sequence(images, quality):
    buffers = []

    for index, image in enumerate(images):
        try:
            buffers.append(encode_image_bit(image, quality=quality))
        except (ValueError, TypeError) as exc:
            raise type(exc)(f"Frame {index}: {exc}") from exc

    return buffers


def encode_image_bit(images, quality=None):
    """
    Encode uint8 RGB image(s) into the image bit stream(s) that
    `decode_image_bit` reads back, and mark them as standard RGB.

    This is the inverse of `decode_image_bit` and the only supported encoder.
    The buffers it produces are conforming RGB JPEGs, so PIL, ffmpeg, a dataset
    viewer or a browser all show the right colours, and they carry a JPEG COM
    marker so `decode_image_bit` knows not to treat them as legacy
    channel-reversed data. Encoding by hand with `cv2.imencode` skips the
    marker, so the buffer can only be read as legacy: channel-reversed on
    decode if the frame was converted to BGR before encoding, and even when
    fed RGB it mints more legacy data that every conforming viewer shows
    reversed.

    Dispatch mirrors `decode_image_bit`, on dtype first, then ndim:
        - uint8 ndarray, ndim == 3      -> one (H, W, 3) RGB image  -> bytes
        - uint8 ndarray, ndim == 4      -> (T, H, W, 3) stack       -> list[bytes]
        - list / tuple                  -> element-wise             -> list[bytes]
        - bytes / bytearray / memoryview / str, uint8 ndarray with ndim == 1,
          and ndarray of dtype kind 'S', 'U', 'O'
                                        -> already encoded, returned as bytes
        - uint8 ndarray, ndim == 2      -> stack of encoded buffers -> list[bytes]

    Grayscale is not supported, matching `decode_image_bit`: a 2-D uint8 array
    is always read as a stack of already-encoded buffers, never as one image.

    Args:
        images: One RGB image, a sequence of them, or already-encoded buffers.
        quality: Optional JPEG quality; OpenCV's default is used when omitted.

    Returns:
        bytes for a single image, list[bytes] for a sequence.

    Raises:
        TypeError: if image data is not uint8.
        ValueError: if a frame has an unsupported shape, fails to encode, or is
            an already-encoded buffer in the legacy format.
    """
    if isinstance(images, (bytes, bytearray, memoryview, str)):
        return _passthrough_encoded_buffer(images)

    if isinstance(images, np.ndarray):
        if images.dtype.kind in {"S", "U", "O"}:
            if images.ndim == 0:
                return _passthrough_encoded_buffer(images.item())
            return _encode_image_bit_sequence(images, quality)

        if images.dtype == np.uint8:
            if images.ndim == 0:
                raise ValueError(
                    "Expected an RGB image or a sequence of them, got a 0-d uint8 "
                    "array."
                )
            if images.ndim == 1:
                return _passthrough_encoded_buffer(images)
            if images.ndim == 3:
                return _encode_single_image_bit(images, quality)
            if images.ndim == 2:
                # Ambiguous shape, so say which reading was attempted: the most
                # likely way to land here is passing a grayscale frame.
                try:
                    return _encode_image_bit_sequence(images, quality)
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        "A 2-D uint8 array is read as a stack of already-encoded "
                        "buffers, never as a grayscale image, and this one is not "
                        f"a valid stack: {exc}"
                    ) from exc
            return _encode_image_bit_sequence(images, quality)

        raise TypeError(
            f"Expected uint8 image data, got an array of dtype {images.dtype}. "
            "Scale and cast before encoding."
        )

    if isinstance(images, (list, tuple)):
        return _encode_image_bit_sequence(images, quality)

    return _encode_single_image_bit(images, quality)


def images_encoding(imgs):
    """
    JPEG-encode RGB frames for storage in the XPolicyLab layout and report the
    width of the fixed-size HDF5 byte column they fit into, which pads the
    shorter buffers with NUL on write.

    Feed this RGB, never BGR and never a channel-swapped array. Encoding runs
    through `encode_image_bit`, so the buffers are conforming RGB JPEGs marked
    for `decode_image_bit`.
    """
    encode_data = encode_image_bit(imgs)

    if isinstance(encode_data, bytes):
        encode_data = [encode_data]

    max_len = max((len(jpeg_data) for jpeg_data in encode_data), default=0)

    return encode_data, max_len