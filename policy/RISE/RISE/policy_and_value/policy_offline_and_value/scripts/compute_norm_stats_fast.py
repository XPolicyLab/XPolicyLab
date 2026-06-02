"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""
import dataclasses
import os
import pathlib
import sys

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
OFFLINE_DIR = CURRENT_DIR.parent
SRC_DIR = OFFLINE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import tqdm
import tyro

import openpi_value.models.model as _model
import openpi_value.shared.normalize as normalize
import openpi_value.training.config as _config
import openpi_value.training.data_loader as _data_loader
import openpi_value.transforms as transforms

# * python scripts/compute_norm_stats_fast.py --config-name Compute_norm
# * Remember to modify the Compute_norm beforehand

@dataclasses.dataclass(frozen=True)
class FakeInputs(transforms.DataTransformFn):
    """Inputs for the CustomAgilex policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0
    
    # if set all state to zeros
    mask_state: bool = False

    # if convert to eef position
    convert_to_eef_position: bool = False



    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST
        mask_padding = self.model_type == _model.ModelType.PI0

        # Pad the proprioceptive input to the action dimension of the model
        state = transforms.pad_to_dim(data["state"], self.action_dim)
        # Ensure state has correct shape [batch_size, state_dim]
        state = state.squeeze()


        # * We need to mask out extremely large values.
        state = np.where(state > np.pi, 0, state)
        state = np.where(state < -np.pi, 0, state)

        # Prepare inputs dictionary
        masked_state = np.zeros_like(state) if self.mask_state else state
        inputs = {
            "state": masked_state,
        }

        # Add actions if present
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            
            # * We need to mask out extremely large values.
            actions = np.where(actions > np.pi, 0, actions)
            actions = np.where(actions < -np.pi, 0, actions)
            
            if mask_padding:
                # Create action mask for padding
                action_mask = np.ones_like(actions, dtype=bool)
                action_mask[:, self.action_dim:] = False
                inputs["action_mask"] = action_mask

            inputs["actions"] = actions.squeeze()

        # Add prompt if present
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class CustomAgilexOutputs(transforms.DataTransformFn):
    """Outputs for the CustomAgilex policy."""

    def __call__(self, data: dict) -> dict:
        # Return the first 14 dimensions of actions (13 joints + 1 gripper)
        return {"actions": np.asarray(data["actions"][:, :14])} 


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    
    dataset = _data_loader.create_torch_dataset_naive(data_config, action_horizon, model_config)   # * Using create_torch_dataset_naive

    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    xpolicylab_dataset = None
    if config_name == "Compute_norm":
        dataset = os.environ.get("RISE_XPOLICYLAB_DATASET")
        if not dataset:
            raise SystemExit(
                "RISE_XPOLICYLAB_DATASET is not visible to Python. "
                "Run from XPolicyLab/policy/RISE with `bash process_data.sh ...`, "
                "or export it before running this script manually."
            )
        xpolicylab_dataset = pathlib.Path(dataset).expanduser().resolve()
        info_path = xpolicylab_dataset / "meta" / "info.json"
        if not info_path.is_file():
            raise SystemExit(f"Converted LeRobot dataset is missing metadata: {info_path}")
        print(f"RISE_XPOLICYLAB_DATASET={xpolicylab_dataset}")

    config = _config.get_config(config_name)
    print(f"openpi_value.training.config={_config.__file__}")
    data_config: _config.LerobotCustomAgilexDataConfig = config.data.create(config.assets_dirs, config.model)
    if xpolicylab_dataset is not None:
        data_config = dataclasses.replace(
            data_config,
            repo_id=[str(xpolicylab_dataset)],
            asset_id=xpolicylab_dataset.name,
        )
    print(f"repo_id={data_config.repo_id}")
    print(f"asset_id={data_config.asset_id}")
    new_data_transforms = transforms.Group(
                inputs=[
                    FakeInputs(
                        action_dim=config.model.action_dim,
                        model_type=config.model.model_type,
                        mask_state=False,
                        convert_to_eef_position=False,
                    )
                ],
                outputs=[CustomAgilexOutputs()],
            )
    data_config = dataclasses.replace(
        data_config,
        repack_transforms=transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        ),
        data_transforms=new_data_transforms,
    )

    assets_dir = pathlib.Path(config.data.assets.assets_dir) if config.data.assets.assets_dir else config.assets_dirs
    if data_config.asset_id is not None:
        output_path = assets_dir / data_config.asset_id
    else:
        repo_id = data_config.repo_id
        if isinstance(repo_id, list):
            if len(repo_id) != 1:
                raise ValueError("Need to specify assets.asset_id when using multiple datasets")
            repo_id = repo_id[0]
        output_path = assets_dir / pathlib.Path(repo_id).name

    print("Output_path:", output_path)

    
    data_loader, num_batches = create_torch_dataloader(
        data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:

            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
