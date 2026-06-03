# import importlib
# from pathlib import Path

# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# import groot.vla.common.utils as U
# from groot.vla.data.schema import (
#     EmbodimentTag,
#     LeRobotModalityMetadata,
#     LeRobotStateActionMetadata,
#     DatasetMetadata,
# )

# from .macro import (
#     FULL_SET_NAME,
#     LE_ROBOT_EMBODIMENT_FILENAME,
#     LE_ROBOT_FEATURES_FILENAME,
#     LE_ROBOT_METADATA_FILENAME,
#     LE_ROBOT_MODALITY_FILENAME,
#     LE_ROBOT_STATISTICS_FILENAME,
# )
# from .registry import EMBODIMENT_TAGS_TO_DATASET_PATHS

# METADATA_DIR = Path(importlib.import_module("groot.vla.data").__file__).parent / "metadata"  # type: ignore


# def calculate_dataset_statistics(
#     parquet_paths: list[Path], features: list[str] | None = None
# ) -> dict:
#     """Calculate the dataset statistics of all columns for a list of parquet files."""
#     # Dataset statistics
#     all_low_dim_data_list = []
#     # Collect all the data
#     for parquet_path in tqdm(
#         sorted(list(parquet_paths)),
#         desc="Collecting all parquet files...",
#     ):
#         # Load the parquet file
#         parquet_data = pd.read_parquet(parquet_path)
#         parquet_data = parquet_data
#         all_low_dim_data_list.append(parquet_data)
#     all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
#     # Compute dataset statistics
#     num_steps = len(all_low_dim_data.index)
#     dataset_statistics: dict = {
#         "num_trajectories": len(all_low_dim_data_list),
#         "total_trajectory_length": num_steps,
#     }
#     if features is None:
#         features = list(all_low_dim_data.columns)
#     for le_modality in features:
#         print(f"Computing statistics for {le_modality}...")
#         np_data = np.vstack(
#             [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]  # type: ignore
#         )
#         dataset_statistics[le_modality] = {
#             "mean": np.mean(np_data, axis=0).tolist(),
#             "std": np.std(np_data, axis=0).tolist(),
#             "min": np.min(np_data, axis=0).tolist(),
#             "max": np.max(np_data, axis=0).tolist(),
#             "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
#             "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
#         }
#     return dataset_statistics


# def get_metadata(
#     embodiment_tag: EmbodimentTag,
#     metadata_version: str,
#     regenerate_stats: bool = False,
#     regenerate_metadata: bool = False,
# ) -> DatasetMetadata:
#     """Get the metadata corresponding to the given embodiment tag and metadata version."""
#     metadata_dir = METADATA_DIR / embodiment_tag.value / metadata_version
#     metadata_path = metadata_dir / LE_ROBOT_METADATA_FILENAME
#     if metadata_path.exists() and not regenerate_metadata:
#         metadata = DatasetMetadata.model_validate_json(metadata_path.read_text())
#         return metadata


# def get_metadata(
#     embodiment_tag: EmbodimentTag,
#     metadata_version: str,
#     regenerate_stats: bool = False,
#     regenerate_metadata: bool = False,
# ) -> TrainableDatasetMetadata_V1_2:
#     """Get the metadata corresponding to the given embodiment tag and metadata version.

#     Args:
#         embodiment_tag: The embodiment tag to load the metadata for.
#         metadata_version: The version of the metadata to load.
#         generate_metadata: Whether to generate the metadata if it does not exist.
#     """
#     metadata_dir = METADATA_DIR / embodiment_tag.value / metadata_version
#     metadata_path = metadata_dir / LE_ROBOT_METADATA_FILENAME
#     if metadata_path.exists() and not regenerate_metadata:
#         metadata = TrainableDatasetMetadata_V1_2.model_validate_json(metadata_path.read_text())
#         return metadata

#     assert (
#         embodiment_tag in EMBODIMENT_TAGS_TO_DATASET_PATHS
#     ), f"Embodiment tag {embodiment_tag} not found in dataset registry. Available tags: {EMBODIMENT_TAGS_TO_DATASET_PATHS.keys()}"

#     dataset_paths = EMBODIMENT_TAGS_TO_DATASET_PATHS[embodiment_tag]
#     # Load supporting metadata
#     le_modality_meta_path = metadata_dir / LE_ROBOT_MODALITY_FILENAME
#     le_features_path = metadata_dir / LE_ROBOT_FEATURES_FILENAME
#     embodiment_meta_path = metadata_dir / LE_ROBOT_EMBODIMENT_FILENAME
#     le_modality_meta = LeRobotModalityMetadata.model_validate_json(
#         le_modality_meta_path.read_text()
#     )
#     le_features = U.load_json(le_features_path)
#     embodiment_meta = U.load_json(embodiment_meta_path)
#     # Load stats
#     if regenerate_stats:
#         le_statistics = None
#     else:
#         le_statistics_path = metadata_dir / LE_ROBOT_STATISTICS_FILENAME
#         le_statistics = U.load_json(le_statistics_path)
#     # Generate metadata
#     metadata, le_statistics = generate_metadata(
#         embodiment_tag=embodiment_tag,
#         dataset_paths=dataset_paths,
#         le_modality_meta=le_modality_meta,
#         le_features=le_features,
#         embodiment_meta=embodiment_meta,
#         le_statistics=le_statistics,
#     )

#     # Save metadata
#     print(f"Generated metadata at {metadata_path}")
#     metadata_path.write_text(metadata.model_dump_json(indent=4))
#     # Save stats
#     if regenerate_stats:
#         le_statistics_path = metadata_dir / LE_ROBOT_STATISTICS_FILENAME
#         U.dump_json(le_statistics, le_statistics_path, indent=4)

#     return metadata


# def generate_metadata(
#     embodiment_tag: EmbodimentTag,
#     dataset_paths: list[Path],
#     le_modality_meta: LeRobotModalityMetadata,
#     le_features: dict,
#     embodiment_meta: dict,
#     le_statistics: dict | None = None,
# ):
#     dataset_name = f"{embodiment_tag.value}:{FULL_SET_NAME}"

#     # Generate our custom modality metadata
#     our_modality_meta: dict[str, dict] = {}
#     for modality in ["state", "action"]:
#         our_modality_meta[modality] = {}
#         le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(
#             le_modality_meta, modality
#         )
#         for subkey in le_state_action_meta:
#             state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
#             if np.issubdtype(state_action_dtype, np.floating):
#                 continuous = True
#             else:
#                 continuous = False
#             our_modality_meta[modality][subkey] = {
#                 "absolute": le_state_action_meta[subkey].absolute,
#                 "rotation_type": le_state_action_meta[subkey].rotation_type,
#                 "shape": [le_state_action_meta[subkey].end - le_state_action_meta[subkey].start],
#                 "continuous": continuous,
#             }

#     # Add video modalities
#     our_modality_meta["video"] = {}
#     for new_key in le_modality_meta.video:
#         original_key = le_modality_meta.video[new_key].original_key
#         le_video_meta = le_features[original_key]
#         height = le_video_meta["shape"][le_video_meta["names"].index("height")]
#         width = le_video_meta["shape"][le_video_meta["names"].index("width")]
#         channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
#         if "info" in le_video_meta:
#             fps = le_video_meta["info"]["video.fps"]
#         elif "video_info" in le_video_meta:
#             fps = le_video_meta["video_info"]["video.fps"]
#         else:
#             raise ValueError(
#                 f"Video modality {new_key} does not contain video_info or info: {le_video_meta.keys()}"
#             )
#         our_modality_meta["video"][new_key] = {
#             "resolution": [width, height],
#             "channels": channels,
#             "fps": fps,
#         }

#     # Add annotation metadata
#     our_modality_meta["annotation"] = {}
#     if le_modality_meta.annotation is not None:
#         for annotation_key in le_modality_meta.annotation:
#             key_split = annotation_key.split(".")
#             annotation_source = key_split[0]
#             annotation_type = ".".join(key_split[1:])
#             if annotation_source not in our_modality_meta["annotation"]:
#                 our_modality_meta["annotation"][annotation_source] = []
#             our_modality_meta["annotation"][annotation_source].append(annotation_type)

#     lowdim_features = []
#     for feature in le_features:
#         if "float" in le_features[feature]["dtype"]:
#             lowdim_features.append(feature)

#     # Dataset statistics
#     if le_statistics is None:
#         print(f"Calculating dataset statistics for {dataset_name}")
#         # Get all parquet files in the dataset paths
#         parquet_files = []
#         for dataset_path in dataset_paths:
#             parquet_files.extend(list(dataset_path.glob("data/*/*.parquet")))
#         le_statistics = calculate_dataset_statistics(parquet_files, lowdim_features)
#     for le_modality in le_statistics:
#         if not isinstance(le_statistics[le_modality], dict):
#             continue
#         for stat in le_statistics[le_modality]:
#             le_statistics[le_modality][stat] = np.asarray(le_statistics[le_modality][stat])

#     # Split statistics keys to our format
#     dataset_statistics = {
#         "num_trajectories": le_statistics["num_trajectories"],
#         "total_trajectory_length": le_statistics["total_trajectory_length"],
#     }
#     for our_modality in ["state", "action"]:
#         dataset_statistics[our_modality] = {}
#         for subkey in our_modality_meta[our_modality]:
#             dataset_statistics[our_modality][subkey] = {}
#             state_action_meta = le_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
#             assert isinstance(state_action_meta, LeRobotStateActionMetadata)
#             le_modality = state_action_meta.original_key
#             for stat in le_statistics[le_modality]:
#                 indices = np.arange(
#                     state_action_meta.start,
#                     state_action_meta.end,
#                 )
#                 dataset_statistics[our_modality][subkey][stat] = le_statistics[le_modality][stat][
#                     indices
#                 ].tolist()

#     # Full dataset metadata
#     metadata = TrainableDatasetMetadata_V1_2(
#         dataset_name=dataset_name,
#         dataset_statistics=dataset_statistics,  # type: ignore
#         modalities=our_modality_meta,  # type: ignore
#         embodiment=embodiment_meta,  # type: ignore
#     )

#     # Convert stats from numpy to list
#     for le_modality in le_statistics:
#         if not isinstance(le_statistics[le_modality], dict):
#             continue
#         for stat in le_statistics[le_modality]:
#             le_statistics[le_modality][stat] = le_statistics[le_modality][stat].tolist()

#     return metadata, le_statistics
