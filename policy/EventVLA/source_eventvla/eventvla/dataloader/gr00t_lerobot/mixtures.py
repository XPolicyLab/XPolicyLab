"""Dataset mixture definitions for EventVLA."""

from typing import Dict, List, Tuple


DATASET_NAMED_MIXTURES: Dict[str, List[Tuple[str, float, str]]] = {
    "robotwin_mem": [
        ("cover_blocks_hard", 1.0, "robotwin_mem"),
        ("put_back_block_hard", 1.0, "robotwin_mem"),
        ("rearrange_blocks_hard", 1.0, "robotwin_mem"),
        ("observe_and_pickup_hard", 1.0, "robotwin_mem"),
        ("find_seal_and_seal_stamp", 1.0, "robotwin_mem"),
        ("observe_and_pickup_object", 1.0, "robotwin_mem"),
        ("reproduct_route", 1.0, "robotwin_mem"),
        ("press_button_keyframe", 1.0, "robotwin_mem"),
    ],
    "robodojo":[
        ("RoboDojo_lerobot_v21_video", 1.0, "robotwin_mem"),
    ],
}
