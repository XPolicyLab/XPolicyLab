from dexbotic.data.data_source.register import register_dataset

UNINAVID_DATASET = {
    "objnav": {
        "data_path_prefix": "./data/objnav/video",
        "annotations": "./data/objnav",
        "frequency": 1,
    },
}

meta_data = {}

register_dataset(UNINAVID_DATASET, meta_data=meta_data, prefix="uninavid")
