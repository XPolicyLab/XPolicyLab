import os
import sys
from omegaconf import OmegaConf

# Set environment variables
os.environ["ABOT_SKIP_DEFAULT_MIXTURES"] = "1"
os.environ["ABOT_DATASETS_ROOT"] = "/vepfs-cnbje63de6fae220/xspark_shared/lerobot"
os.environ["ABOT_SIM_STACK_BOWLS_REPO"] = "sim_stack_bowls_video"

try:
    from ABot.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
    print(f"DATASET_NAMED_MIXTURES contains sim_stack_bowls: {'sim_stack_bowls' in DATASET_NAMED_MIXTURES}")
    if 'sim_stack_bowls' not in DATASET_NAMED_MIXTURES:
         print(f"Available mixtures: {list(DATASET_NAMED_MIXTURES.keys())}")

    from ABot.dataloader.lerobot_datasets import get_vla_dataset
    
    data_cfg = OmegaConf.create({
        "data_root_dir": "/vepfs-cnbje63de6fae220/xspark_shared/lerobot",
        "data_mix": "sim_stack_bowls",
        "delete_pause_frame": False
    })
    
    print("Attempting to load dataset...")
    dataset = get_vla_dataset(data_cfg=data_cfg)
    
    print(f"Dataset length: {len(dataset)}")
    if len(dataset) > 0:
        first_sample = dataset[0]
        print(f"First sample keys: {list(first_sample.keys())}")
    
    print("Success: True")

except Exception as e:
    print(f"Success: False")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
