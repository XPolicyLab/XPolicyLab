"""Auto-generated Dexbotic data source for RoboDojo-cotrain-arx_x5-3500-ee."""

from dexbotic.data.data_source.register import register_dataset

ROBODOJO_DATASET = {
    "RoboDojo-cotrain-arx_x5-3500-ee": {
        "data_path_prefix": "/vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/Dexbotic_DM0/data/RoboDojo-cotrain-arx_x5-3500-ee/video",
        "annotations": "/vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/Dexbotic_DM0/data/RoboDojo-cotrain-arx_x5-3500-ee",
        "frequency": 1,
    },
}

meta_data = {
    "non_delta_mask": [6, 20],
    "periodic_mask": None,
    "periodic_range": None,
}

register_dataset(ROBODOJO_DATASET, meta_data=meta_data, prefix="robodojo")
