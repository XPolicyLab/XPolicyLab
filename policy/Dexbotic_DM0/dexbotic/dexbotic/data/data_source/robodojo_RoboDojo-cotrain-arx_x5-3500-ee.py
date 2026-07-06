"""Auto-generated Dexbotic data source for RoboDojo-cotrain-arx_x5-3500-ee."""

from pathlib import Path

from dexbotic.data.data_source.register import register_dataset

# .../policy/Dexbotic_DM0/data, resolved relative to this file inside the adapter
_DATA_ROOT = Path(__file__).resolve().parents[4] / "data"

ROBODOJO_DATASET = {
    "RoboDojo-cotrain-arx_x5-3500-ee": {
        "data_path_prefix": str(_DATA_ROOT / "RoboDojo-cotrain-arx_x5-3500-ee" / "video"),
        "annotations": str(_DATA_ROOT / "RoboDojo-cotrain-arx_x5-3500-ee"),
        "frequency": 1,
    },
}

meta_data = {
    "non_delta_mask": [6, 20],
    "periodic_mask": None,
    "periodic_range": None,
}

register_dataset(ROBODOJO_DATASET, meta_data=meta_data, prefix="robodojo")
