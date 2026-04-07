# `aloha_datasets`

Generic RLDS conversion utilities for ALOHA-style datasets that have already
been converted into the OpenVLA-OFT preprocessed layout.

Expected input layout:

```text
/path/to/preprocessed_aloha_dataset/
  train/
    episode_0.hdf5
    ...
  val/
    episode_0.hdf5
    ...
```

The generic builder runner is:

`aloha_datasets/build_aloha_dataset.py`

Example:

```bash
conda activate openvla
cd /mnt/pfs/pg4hw0/niantian/openvla-oft
python rlds_dataset_builder/aloha_datasets/build_aloha_dataset.py \
  --dataset_name aloha_put_back_block_200_demos \
  --preprocessed_dir /mnt/pfs/pg4hw0/niantian/openvla-oft/data/aloha_preprocessed/aloha_put_back_block_200_demos \
  --tfds_data_dir /mnt/pfs/pg4hw0/niantian/tensorflow_datasets \
  --overwrite
```

If you still want a dataset-specific builder file for bookkeeping or IDE
navigation, keep a thin wrapper such as
`aloha_put_back_block_200_demos_dataset_builder.py` that subclasses the generic
base builder.
