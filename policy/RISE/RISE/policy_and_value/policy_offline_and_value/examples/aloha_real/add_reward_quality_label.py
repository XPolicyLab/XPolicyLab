import os
import random
from tqdm import tqdm
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import json

# The original function for finding files is clear and remains largely the same.
def get_all_parquet_files(data_root: str) -> list[str]:
    """Recursively finds all .parquet files within the given data_root."""
    parquet_files = []
    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    return parquet_files

def add_is_suboptimal_column(data_root: str, output_root: str):
    """
    Iterates over all Parquet files, copies them to the output_root, 
    and adds an 'is_suboptimal' column initialized to False (0).
    
    The function uses os.rename for atomic replacement to ensure file integrity.
    """
    parquet_files = get_all_parquet_files(data_root)
    os.makedirs(output_root, exist_ok=True)

    for file_path in tqdm(parquet_files, desc="Processing parquet files"):
        relative_path = os.path.relpath(file_path, data_root)
        output_file_path = os.path.join(output_root, relative_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        try:
            # 1. Load the original parquet table
            parquet_table = pq.read_table(file_path)
            
            # 2. Define the new column data (all False/0 initially)
            num_rows = parquet_table.num_rows
            
            new_column_array = pa.array(np.ones(num_rows))   # * Expert
            
            
            # new_column_array = pa.array(np.ones(num_rows) * -1.)   # * Negative
            
            
            # 3. Add the new column
            # Note: We append the column to the original table *in memory*
            new_table = parquet_table.append_column(
                'action_advantage',
                new_column_array
            )

            # 4. Write the new table to a temporary file for safety
            temp_output_path = output_file_path + ".tmp"
            pq.write_table(new_table, temp_output_path)
            
            # 5. Atomically replace the original file with the new one.
            # This ensures that if the process fails mid-write, the original file is preserved.
            os.replace(temp_output_path, output_file_path)
            
        except Exception as e:
            tqdm.write(f"Error processing file {file_path}: {e}")

def deal_mata(data_root: str, output_root: str):
    
    
    os.system(f"cp -r {os.path.join(data_root, 'meta')} {output_root}")
    
    # * append advantage into meta/info.json
    with open(os.path.join(output_root, 'meta', 'info.json'), 'r') as f:
        meta_info = json.load(f)
        meta_info['features']['action_advantage'] = {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": None
        }
    # * dump back to file
    
    # os.makedirs(os.path.join(output_root, 'meta'), exist_ok=True)
    with open(os.path.join(output_root, 'meta', 'info.json'), 'w') as f:
        json.dump(meta_info, f, indent=4)


def soft_link_video(data_root: str, output_root: str):
    video_dir = os.path.join(data_root, "videos")
    
    # remove videos suffix if exists
    if output_root.endswith('/videos'):
        output_root = output_root.replace('/videos', '')
    
    try:
        os.system(f"ln -s {video_dir} {output_root}")
    except:
        pass


data_roots = [
    "path/to/your/dataset",
]

if not isinstance(data_roots, list):
    data_roots = [data_roots]

for data_root in data_roots:

    output_dir = data_root + "_w_adv_expert"

    # Execute the improved function
    add_is_suboptimal_column(data_root, output_dir)
    deal_mata(data_root, output_dir)
    soft_link_video(data_root, output_dir)
    print(f"✅ Finished processing. New files are saved to: {output_dir}")