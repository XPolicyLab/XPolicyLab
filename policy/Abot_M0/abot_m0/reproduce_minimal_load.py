import os
import sys
from omegaconf import OmegaConf

# Set environment variables
os.environ["ABOT_SKIP_DEFAULT_MIXTURES"] = "1"
os.environ["ABOT_DATASETS_ROOT"] = "/vepfs-cnbje63de6fae220/xspark_shared/lerobot"
os.environ["ABOT_SIM_STACK_BOWLS_REPO"] = "sim_stack_bowls_video"

# Add current directory to path to find ABot
sys.path.append("/vepfs-cnbje63de6fae220/niantian/ABot-Manipulation")

try:
    from ABot.dataloader.lerobot_datasets import get_vla_dataset
    
    # Construct minimal OmegaConf data_cfg
    data_cfg = OmegaConf.create({
        "video_backend": "torchvision_av",
        "image_size": [224, 224],
        "include_state": False,
        "dataset_name": "sim_stack_bowls_video",
        "fps": 1,
        "data_root_dir": "/vepfs-cnbje63de6fae220/xspark_shared/lerobot", # Explicitly add this based on error
    })
    
    print("Attempting to call get_vla_dataset...")
    dataset = get_vla_dataset(data_cfg=data_cfg)
    
    print(f"Dataset loaded. Length: {len(dataset)}")
    
    sample = dataset[0]
    print("Sample keys:", list(sample.keys()))
    print("Verification SUCCESS")

except Exception as e:
    print(f"Verification FAILED")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
