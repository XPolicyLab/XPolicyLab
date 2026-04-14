# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import os
import subprocess
import logging
import glob
import pyarrow.fs as pf
import torch.distributed as dist
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_parquet_data_paths(data_dir_list, num_sampled_data_paths, rank=0, world_size=1):
    num_data_dirs = len(data_dir_list)
    if world_size > 1:
        chunk_size = (num_data_dirs + world_size - 1) // world_size
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, num_data_dirs)
        local_data_dir_list = data_dir_list[start_idx:end_idx]
        local_num_sampled_data_paths = num_sampled_data_paths[start_idx:end_idx]
    else:
        local_data_dir_list = data_dir_list
        local_num_sampled_data_paths = num_sampled_data_paths

    local_data_paths = []
    for data_dir, num_data_path in zip(local_data_dir_list, local_num_sampled_data_paths):
        if data_dir.startswith("hdfs://"):
            files = hdfs_ls_cmd(data_dir)
            data_paths_per_dir = [
                file for file in files if file.endswith(".parquet")
            ]
        elif any(n in data_dir for n in ["openx"]):
            """lerobot_dataset
            dataset_name/data/chunk-xxx/episode-xxx
            """
            data_paths_per_dir = glob.glob(f"{data_dir}/data/chunk-**/episode_*.parquet")
        else:
            files = os.listdir(data_dir)
            data_paths_per_dir = [
                os.path.join(data_dir, name)
                for name in files
                if name.endswith(".parquet")
            ]
        repeat = num_data_path // len(data_paths_per_dir)
        data_paths_per_dir = data_paths_per_dir * (repeat + 1)
        local_data_paths.extend(data_paths_per_dir[:num_data_path])

    if world_size > 1:
        gather_list = [None] * world_size
        dist.all_gather_object(gather_list, local_data_paths)

        combined_chunks = []
        for chunk_list in gather_list:
            if chunk_list is not None:
                combined_chunks.extend(chunk_list)
    else:
        combined_chunks = local_data_paths

    return combined_chunks


# NOTE: cumtomize this function for your cluster
def get_hdfs_host():
    return "hdfs://xxx"


# NOTE: cumtomize this function for your cluster
def get_hdfs_block_size():
    return 134217728


# NOTE: cumtomize this function for your cluster
def get_hdfs_extra_conf():
    return None


def init_arrow_pf_fs(parquet_file_path):
    if parquet_file_path.startswith("hdfs://"):
        fs = pf.HadoopFileSystem(
            host=get_hdfs_host(),
            port=0,
            buffer_size=get_hdfs_block_size(),
            extra_conf=get_hdfs_extra_conf(),
        )
    else:
        fs = pf.LocalFileSystem()
    return fs


def hdfs_ls_cmd(dir):
    result = subprocess.run(["hdfs", "dfs", "ls", dir], capture_output=True, text=True).stdout
    return ['hdfs://' + i.split('hdfs://')[-1].strip() for i in result.split('\n') if 'hdfs://' in i]


def calculate_dataset_statistics(parquet_paths: List[Path]) -> Dict[str, Any]:
    """
    Calculate statistics for all numeric columns (including vectors and scalars) in a dataset.
    Ensures all output statistics are in list format.
    """
    if not parquet_paths:
        return {}

    # --- 1. Read all Parquet files into a single large DataFrame ---
    all_data_list = []
    print(f"Calculating statistics for {len(parquet_paths)} parquet files...")
    for parquet_path in tqdm(sorted(list(parquet_paths)), desc="Collecting parquet files"):
        try:
            df = pd.read_parquet(parquet_path)
            all_data_list.append(df)
        except Exception as e:
            print(f"Could not read {parquet_path}: {e}")
            continue

    if not all_data_list:
        return {}
    all_data = pd.concat(all_data_list, axis=0, ignore_index=True)

    dataset_statistics = {}

    # --- 2. Iterate through all columns with smart handling ---
    for column_name in all_data.columns:
        # Optional: skip metadata columns that don't need statistics
        if column_name in ['episode_index', 'index']:
             print(f"Skipping metadata column: {column_name}")
             continue

        print(f"Computing statistics for {column_name}...")

        # Get the first non-null data item to determine type
        first_valid_item = all_data[column_name].dropna().iloc[0]

        np_data = None
        # --- Core logic: correctly handle vectors and scalars ---
        if isinstance(first_valid_item, (np.ndarray, list)):
            # If it's a vector/list, safely stack them
            # dropna() ensures we don't try to stack NaN values
            valid_series = all_data[column_name].dropna()
            np_data = np.stack(valid_series.to_numpy())
        elif np.isscalar(first_valid_item):
            # If it's a scalar (int, float)
            valid_series = all_data[column_name].dropna()
            # Convert to (N, 1) 2D array for unified processing
            np_data = valid_series.to_numpy(dtype=np.float32).reshape(-1, 1)
        else:
            print(f"Skipping column {column_name} of unknown type: {type(first_valid_item)}")
            continue

        # Ensure data is floating point type for computation
        try:
            if not np.issubdtype(np_data.dtype, np.floating):
                np_data = np_data.astype(np.float32)
        except Exception:
            # If conversion fails, it's non-standard data, skip it
            continue

        # --- 3. Calculate statistics and ensure list format ---
        # np.mean(..., axis=0) returns a 1D array on 2D arrays, which is what we want
        mean_val = np.mean(np_data, axis=0).tolist()
        std_val = np.std(np_data, axis=0).tolist()
        min_val = np.min(np_data, axis=0).tolist()
        max_val = np.max(np_data, axis=0).tolist()
        q01_val = np.quantile(np_data, 0.01, axis=0).tolist()
        q99_val = np.quantile(np_data, 0.99, axis=0).tolist()

        dataset_statistics[column_name] = {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "q01": q01_val,
            "q99": q99_val,
        }
            
    return dataset_statistics