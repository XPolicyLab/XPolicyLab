from dexbotic.data.data_source.register import register_dataset


ROBOTWIN2_DATASET = {
    "adjust_bottle": {
        "data_path_prefix": "./data/robotwin/video",
        "annotations": './data/robotwin/jsonl/adjust_bottle',
        "frequency": 1,
    },
    "beat_block_hammer": {
        "data_path_prefix": "./data/robotwin/video",
        "annotations": './data/robotwin/jsonl/beat_block_hammer',
        "frequency": 1,
    },
    "blocks_ranking_rgb": {
        "data_path_prefix": "./data/robotwin/video",
        "annotations": './data/robotwin/jsonl/blocks_ranking_rgb',
        "frequency": 1,
    },
    "blocks_ranking_size": {
        "data_path_prefix": "./data/robotwin/video",
        "annotations": './data/robotwin/jsonl/blocks_ranking_size',
        "frequency": 1,
    },
    "click_alarmclock": {
        "data_path_prefix": "./data/robotwin/video",
        "annotations": './data/robotwin/jsonl/click_alarmclock',
        "frequency": 1,
    },
    "click_bell": {
        "data_path_prefix": "./data/robotwin/video",
        "annotations": './data/robotwin/jsonl/click_bell',
        "frequency": 1,
    },
    "dump_bin_bigbin": {
        "data_path_prefix": "./data/robotwin/video",
        "annotations": './data/robotwin/jsonl/dump_bin_bigbin',
        "frequency": 1,
    },
    "grab_roller": {
        "data_path_prefix": "./data/robotwin/video",
        "annotations": './data/robotwin/jsonl/grab_roller',
        "frequency": 1,
    },
    "handover_block": {
        "data_path_prefix": "./data/robotwin/video",
        "annotations": './data/robotwin/jsonl/handover_block',
        "frequency": 1,
    },
    "handover_mic": {
        "data_path_prefix": "./data/robotwin/video",
        "annotations": './data/robotwin/jsonl/handover_mic',
        "frequency": 1,
    },
    "hanging_mug": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/hanging_mug',
        "frequency": 1,
    },
    "lift_pot": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/lift_pot',
        "frequency": 1,
    },
    "move_can_pot": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/move_can_pot',
        "frequency": 1,
    },
    "move_pillbottle_pad": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/move_pillbottle_pad',
        "frequency": 1,
    },
    "move_playingcard_away": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/move_playingcard_away',
        "frequency": 1,
    },
    "move_stapler_pad": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/move_stapler_pad',
        "frequency": 1,
    },
    "open_laptop": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/open_laptop',
        "frequency": 1,
    },
    "open_microwave": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/open_microwave',
        "frequency": 1,
    },
    "pick_diverse_bottles": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/pick_diverse_bottles',
        "frequency": 1,
    },
    "pick_dual_bottles": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/pick_dual_bottles',
        "frequency": 1,
    },
    "place_a2b_left": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/place_a2b_left',
        "frequency": 1,
    },
    "place_a2b_right": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/place_a2b_right',
        "frequency": 1,
    },
    "place_bread_basket": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/place_bread_basket',
        "frequency": 1,
    },
    "place_bread_skillet": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/place_bread_skillet',
        "frequency": 1,
    },
    "place_burger_fries": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/place_burger_fries',
        "frequency": 1,
    },
    "place_can_basket": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/place_can_basket',
        "frequency": 1,
    },
    "place_cans_plasticbox": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/place_cans_plasticbox',
        "frequency": 1,
    },
    "place_container_plate": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/place_container_plate',
        "frequency": 1,
    },
    "place_dual_shoes": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/place_dual_shoes',
        "frequency": 1,
    },
    "place_empty_cup": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/place_empty_cup',
        "frequency": 1,
    },
    "place_fan": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/place_fan',
        "frequency": 1, 
    },
    "place_mouse_pad": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/place_mouse_pad',
        "frequency": 1,
    },
    "place_object_basket": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/place_object_basket',
        "frequency": 1,
    },
    "place_object_scale": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/place_object_scale',
        "frequency": 1,
    },
    "place_object_stand": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/place_object_stand',
        "frequency": 1,
    },
    "place_phone_stand": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/place_phone_stand',
        "frequency": 1,
    },
    "place_shoe": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/place_shoe',
        "frequency": 1,
    },
    "press_stapler": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/press_stapler',
        "frequency": 1,
    },
    "put_bottles_dustbin": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/put_bottles_dustbin',
        "frequency": 1,
    },
    "put_object_cabinet": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/put_object_cabinet',
        "frequency": 1,
    },
    "rotate_qrcode": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/rotate_qrcode',
        "frequency": 1,
    },
    "scan_object": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/scan_object',
        "frequency": 1,
    },
    "shake_bottle": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/shake_bottle',
        "frequency": 1,
    },
    "shake_bottle_horizontally": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/shake_bottle_horizontally',
        "frequency": 1,
    },
    "stack_blocks_three": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/stack_blocks_three',
        "frequency": 1,
    },
    "stack_blocks_two": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/stack_blocks_two',
        "frequency": 1,
    },
    "stack_bowls_three": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/stack_bowls_three',
        "frequency": 1,
    },
    "stack_bowls_two": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/stack_bowls_two',
        "frequency": 1,
    },
    "stamp_seal": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/stamp_seal',
        "frequency": 1,
    },
    "turn_switch": {
        "data_path_prefix": "./data/robotwin/video/",
        "annotations": './data/robotwin/jsonl/turn_switch',
        "frequency": 1,
    },
}


meta_data = {
    'non_delta_mask': [6, 13],
    'periodic_mask': None,
    'periodic_range': None
}

register_dataset(ROBOTWIN2_DATASET, meta_data=meta_data, prefix='robotwin2')
