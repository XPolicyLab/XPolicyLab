
ROBO_CHALLENGE_TASKS = {
    "arrange_flowers": {
        "prompt": "insert the three flowers on the table into the vase one by one",
        "robot_type": "arx5",
    },
    "arrange_fruits_in_basket": {
        "prompt": "Place the four fruits into the nearby basket one by one",
        "robot_type": "ur5",
    },
    "arrange_paper_cups": {
        "prompt": "stack the four paper cups on top of the paper cup closest to the shelf one by one and place the stacked cups on the shelf",
        "robot_type": "arx5",
    },
    "clean_dining_table": {
        "prompt": "place all the trash into the green trash bin, and put the dishes into the transparent basket",
        "robot_type": "aloha",
    },
    "fold_dishcloth": {
        "prompt": "fold the dishcloth in half twice, then place it in the position slightly to the front and left",
        "robot_type": "arx5",
    },
    "hang_toothbrush_cup": {
        "prompt": "hang the orange toothbrush cup on the cup holder",
        "robot_type": "ur5",
    },
    "make_vegetarian_sandwich": {
        "prompt": "make a vegetable sandwich",
        "robot_type": "aloha",
    },
    "move_objects_into_box": {
        "prompt": "place all the clutter on the desk into the white box",
        "robot_type": "franka",
    },
    "open_the_drawer": {
        "prompt": "open the drawer",
        "robot_type": "arx5",
    },
    "place_shoes_on_rack": {
        "prompt": "place these shoes on the shoe rack",
        "robot_type": "arx5",
    },
    "plug_in_network_cable": {
        "prompt": "Insert the RJ45 connector of the Ethernet cable into the host.",
        "robot_type": "aloha",
    },
    "pour_fries_into_plate": {
        "prompt": "open the box lid and pour the chips from the box onto the plate.",
        "robot_type": "aloha",
    },
    "press_three_buttons": {
        "prompt": "press the pink, blue, and green buttons in sequence",
        "robot_type": "franka",
    },
    "put_cup_on_coaster": {
        "prompt": "place the cup on the coaster",
        "robot_type": "arx5",
    },
    "put_opener_in_drawer": {
        "prompt": "place the can opener into the right-hand drawer",
        "robot_type": "aloha",
    },
    "put_pen_into_pencil_case": {
        "prompt": "place the pen on the table into the pencil case",
        "robot_type": "aloha",
    },
    "scan_QR_code": {
        "prompt": "scan the QR code on the medicine box using the scanner",
        "robot_type": "aloha",
    },
    "search_green_boxes": {
        "prompt": "search through the stack of blocks for the green blocks and place it into the yellow box",
        "robot_type": "arx5",
    },
    "set_the_plates": {
        "prompt": "place the three plates onto the plate rack one by one",
        "robot_type": "ur5",
    },
    "shred_scrap_paper": {
        "prompt": "place the white paper on the shelf into the shredder for shredding",
        "robot_type": "ur5",
    },
    "sort_books": {
        "prompt": "place the three books on the shelf into the corresponding book sections of the black bookshelf",
        "robot_type": "ur5",
    },
    "sort_electronic_products": {
        "prompt": "classify the four electronic products on the table and place them into the corresponding transparent baskets",
        "robot_type": "arx5",
    },
    "stack_bowls": {
        "prompt": "stack the two smaller bowls on top of the largest bowl one by one.",
        "robot_type": "aloha",
    },
    "stack_color_blocks": {
        "prompt": "stack the yellow block on top of the orange block",
        "robot_type": "ur5",
    },
    "stick_tape_to_box": {
        "prompt": "tear off a piece of clear tape and stick it onto the metal box",
        "robot_type": "aloha",
    },
    "sweep_the_rubbish": {
        "prompt": "sweep the trash into the dustpan using a broom",
        "robot_type": "aloha",
    },
    "turn_on_faucet": {
        "prompt": "grasp the faucet switch and turn it on",
        "robot_type": "aloha",
    },
    "turn_on_light_switch": {
        "prompt": "turn on the light switch",
        "robot_type": "arx5",
    },
    "water_potted_plant": {
        "prompt": "water the potted plant using the kettle",
        "robot_type": "arx5",
    },
    "wipe_the_table": {
        "prompt": "pull out a tissue, wipe the stains on the table clean, and then discard the tissue into the trash bin",
        "robot_type": "arx5",
    },
}


def get_task_info(task_name: str) -> dict | None:
    """Get prompt and robot_type by task_name"""
    return ROBO_CHALLENGE_TASKS.get(task_name)


def get_prompt(task_name: str) -> str | None:
    """Get the prompt corresponding to the task"""
    info = ROBO_CHALLENGE_TASKS.get(task_name)
    return info["prompt"] if info else None


def get_robot_type(task_name: str) -> str | None:
    """Get the robot arm type corresponding to the task (aloha/arx5/franka/ur5)"""
    info = ROBO_CHALLENGE_TASKS.get(task_name)
    return info["robot_type"] if info else None
