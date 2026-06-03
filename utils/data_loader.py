# 加载数据, 用来数据转化
import h5py, json, cv2, argparse
import numpy as np

def load_xspark_v1(hdf5_path, decode_images=True):
    def decode_image(img_bytes):
        try:
            if isinstance(img_bytes, (bytes, np.bytes_)):
                jpeg_bytes = img_bytes.rstrip(b"\0")
            elif isinstance(img_bytes, np.ndarray) and img_bytes.dtype.kind in ['S', 'U']:
                jpeg_bytes = img_bytes.item().rstrip(b"\0")
            else:
                return img_bytes
            
            nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            return img_bytes

    def h5_to_dict(obj):
        d = {}
        for key, item in obj.items():
            if isinstance(item, h5py.Group):
                d[key] = h5_to_dict(item)
            elif isinstance(item, h5py.Dataset):
                val = item[()]
                
                if key == "colors" and isinstance(val, np.ndarray):
                    decoded_frames = []
                    for frame in val:
                        decoded_frames.append(frame if not decode_images else decode_image(frame))
                    d[key] = np.array(decoded_frames)
                    continue

                if isinstance(val, (bytes, np.bytes_)):
                    try:
                        decoded_str = val.decode("utf-8")
                        try:
                            d[key] = json.loads(decoded_str)
                        except json.JSONDecodeError:
                            d[key] = decoded_str
                    except Exception:
                        d[key] = val
                elif isinstance(val, np.ndarray) and val.dtype.kind in ["S", "U", "O"]:
                    try:
                        val_item = val.item() if val.size == 1 else val
                        if isinstance(val_item, (bytes, np.bytes_)):
                            decoded_str = val_item.decode("utf-8")
                        elif isinstance(val_item, str):
                            decoded_str = val_item
                        else:
                            d[key] = val
                            continue
                        try:
                            d[key] = json.loads(decoded_str)
                        except json.JSONDecodeError:
                            d[key] = decoded_str
                    except Exception:
                        d[key] = val
                else:
                    d[key] = val
        return d

    with h5py.File(hdf5_path, "r") as f:
        return h5_to_dict(f)

def load(data_path, data_type="xspark", data_version="v1.0"):
    # RoboDojo sim_cloud HDF5 uses the same xspark v1.0 layout.
    if data_type in {"xspark", "RoboDojo"}:
        if data_version == "v1.0":
            return load_xspark_v1(data_path, decode_images=True)
        raise NotImplementedError(f"{data_version} is not valid in {data_type} .")
    raise NotImplementedError(f"{data_type} is not valid data type. ")