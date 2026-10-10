
from PIL import Image
import io
import os
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

def image_to_byte_array(image) -> bytes:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def create_data(lst, skip_keys):
  # print(lst)
  # for d in tqdm.tqdm(lst):
  for d in lst:
    if "rgb_obs" in d:
      if not isinstance(d["rgb_obs"], Image.Image):
        d["rgb_obs"] = Image.fromarray(d["rgb_obs"]).convert('RGB')
      d["rgb_obs"] = image_to_byte_array(d["rgb_obs"])
    for key in d.keys():
      # if key == "rgb_obs" or key == "language_label" or key == "frame_count" or key == "seq_name":
      if key in skip_keys:
        continue
      d[key] = d[key].reshape(-1)
  return lst

def save_data_to_parquet(
  all_data, idx, save_path, chunk_id, dataset_prefix,
  skip_keys
):
    all_data = create_data(all_data, skip_keys)
    save_path = os.path.join(
      save_path, 
      "{}_{}_{:04d}".format(dataset_prefix, chunk_id, idx) + ".parquet"
    )

    df = pd.DataFrame(all_data)
    df.to_parquet(save_path, engine='pyarrow')


def parquet_to_dataset_generator(file_paths):
    index = 0
    for file_path in tqdm(file_paths):
        df = pd.read_parquet(file_path)
        dataset = Dataset.from_pandas(df)
        index = index + 1
        yield dataset