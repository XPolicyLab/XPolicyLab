import math

from dexbotic.data.data_source.register import register_dataset


CALVIN_DATASET = {
    "ABC": {
        "data_path_prefix": "./data/calvin/task_ABC_D/video",
        "annotations": './data/calvin/task_ABC_D',
        "frequency": 1,
    },
    
}

meta_data = {
    'non_delta_mask': [6],
    'periodic_mask': [3, 4, 5],
    'periodic_range': 2 * math.pi,
}


register_dataset(CALVIN_DATASET, meta_data=meta_data, prefix='calvin')
