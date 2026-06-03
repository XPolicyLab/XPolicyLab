
from BeingH.dataset.datasets.vla_dataset import LeRobotIterableDataset
from BeingH.dataset.datasets.vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    'libero_posttrain': LeRobotIterableDataset,
    'robocasa_human_posttrain': LeRobotIterableDataset,
    'uni_posttrain': LeRobotIterableDataset,
    'robotwin_posttrain': LeRobotIterableDataset,
    'eef_robotwin': LeRobotIterableDataset,
    'robodojo_posttrain': LeRobotIterableDataset,
}


DATASET_INFO = {
    'libero_posttrain': {
        'libero_spatial': {
            'dataset_path': "/share/dataset/beingh_posttrain/libero/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
        },
        'libero_object': {
            'dataset_path': "/share/dataset/beingh_posttrain/libero/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot",
        },
        'libero_goal': {
            'dataset_path': "/share/dataset/beingh_posttrain/libero/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
        },
        'libero_10': {
            'dataset_path': "/share/dataset/beingh_posttrain/libero/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",
        },
    },

    'robocasa_human_posttrain': {
        'single_panda_gripper.CloseDoubleDoor': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/CloseDoubleDoor",
        },
        'single_panda_gripper.CloseDrawer': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/CloseDrawer",
        },
        'single_panda_gripper.CloseSingleDoor': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/CloseSingleDoor",
        },

        'single_panda_gripper.CoffeePressButton': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/CoffeePressButton",
        },
        'single_panda_gripper.CoffeeServeMug': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/CoffeeServeMug",
        },
        'single_panda_gripper.CoffeeSetupMug': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/CoffeeSetupMug",
        },

        'single_panda_gripper.OpenDoubleDoor': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/OpenDoubleDoor",
        },
        'single_panda_gripper.OpenDrawer': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/OpenDrawer",
        },
        'single_panda_gripper.OpenSingleDoor': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/OpenSingleDoor",
        },

        'single_panda_gripper.PnPCabToCounter': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/PnPCabToCounter",
        },
        'single_panda_gripper.PnPCounterToCab': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/PnPCounterToCab",
        },
        'single_panda_gripper.PnPCounterToMicrowave': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/PnPCounterToMicrowave",
        },
        'single_panda_gripper.PnPCounterToSink': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/PnPCounterToSink",
        },
        'single_panda_gripper.PnPCounterToStove': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/PnPCounterToStove",
        },
        'single_panda_gripper.PnPMicrowaveToCounter': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/PnPMicrowaveToCounter",
        },
        'single_panda_gripper.PnPSinkToCounter': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/PnPSinkToCounter",
        },
        'single_panda_gripper.PnPStoveToCounter': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/PnPStoveToCounter",
        },

        'single_panda_gripper.TurnOffMicrowave': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/TurnOffMicrowave",
        },
        'single_panda_gripper.TurnOffSinkFaucet': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/TurnOffSinkFaucet",
        },
        'single_panda_gripper.TurnOffStove': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/TurnOffStove",
        },
        'single_panda_gripper.TurnOnMicrowave': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/TurnOnMicrowave",
        },
        'single_panda_gripper.TurnOnSinkFaucet': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/TurnOnSinkFaucet",
        },
        'single_panda_gripper.TurnOnStove': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/TurnOnStove",
        },
        'single_panda_gripper.TurnSinkSpout': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/TurnSinkSpout",
        },
    },

    'uni_posttrain': {
        # ========================================================================
        # ROBOCASA datasets
        # ========================================================================
        'single_panda_gripper.CloseDoubleDoor': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/CloseDoubleDoor",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.CloseDoubleDoor',
        },

        'single_panda_gripper.CloseDrawer': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/CloseDrawer",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.CloseDrawer',
        },

        'single_panda_gripper.CloseSingleDoor': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/CloseSingleDoor",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.CloseSingleDoor',
        },

        'single_panda_gripper.CoffeePressButton': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/CoffeePressButton",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.CoffeePressButton',
        },

        'single_panda_gripper.CoffeeServeMug': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/CoffeeServeMug",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.CoffeeServeMug',
        },

        'single_panda_gripper.CoffeeSetupMug': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/CoffeeSetupMug",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.CoffeeSetupMug',
        },

        'single_panda_gripper.OpenDoubleDoor': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/OpenDoubleDoor",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.OpenDoubleDoor',
        },

        'single_panda_gripper.OpenDrawer': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/OpenDrawer",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.OpenDrawer',
        },

        'single_panda_gripper.OpenSingleDoor': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/OpenSingleDoor",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.OpenSingleDoor',
        },

        'single_panda_gripper.PnPCabToCounter': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/PnPCabToCounter",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPCabToCounter',
        },

        'single_panda_gripper.PnPCounterToCab': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/PnPCounterToCab",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPCounterToCab',
        },

        'single_panda_gripper.PnPCounterToMicrowave': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/PnPCounterToMicrowave",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPCounterToMicrowave',
        },

        'single_panda_gripper.PnPCounterToSink': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/PnPCounterToSink",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPCounterToSink',
        },

        'single_panda_gripper.PnPCounterToStove': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/PnPCounterToStove",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPCounterToStove',
        },

        'single_panda_gripper.PnPMicrowaveToCounter': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/PnPMicrowaveToCounter",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPMicrowaveToCounter',
        },

        'single_panda_gripper.PnPSinkToCounter': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/PnPSinkToCounter",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPSinkToCounter',
        },

        'single_panda_gripper.PnPStoveToCounter': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/PnPStoveToCounter",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPStoveToCounter',
        },

        'single_panda_gripper.TurnOffMicrowave': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/TurnOffMicrowave",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.TurnOffMicrowave',
        },

        'single_panda_gripper.TurnOffSinkFaucet': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/TurnOffSinkFaucet",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.TurnOffSinkFaucet',
        },

        'single_panda_gripper.TurnOffStove': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/TurnOffStove",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.TurnOffStove',
        },

        'single_panda_gripper.TurnOnMicrowave': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/TurnOnMicrowave",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.TurnOnMicrowave',
        },

        'single_panda_gripper.TurnOnSinkFaucet': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/TurnOnSinkFaucet",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.TurnOnSinkFaucet',
        },

        'single_panda_gripper.TurnOnStove': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/TurnOnStove",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.TurnOnStove',
        },

        'single_panda_gripper.TurnSinkSpout': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/TurnSinkSpout",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.TurnSinkSpout',
        },
    },

    # RoboDojo LeRobot v2.1 (from XPolicyLab/scripts/transform_lerobot_v30_format.py or v21 export)
    'robodojo_posttrain': {
        'RoboDojo_sim_arx-x5_v21': {
            'dataset_path': '/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v21',
        },
    },

    # ─────────────────────────────────────────────────────────────
    # RoboTwin (aloha-agilex, qpos control)
    # Each key is "{task_name}-{setting}" and maps to a LeRobot
    # dataset directory produced by:
    #   python scripts/data/convert_robotwin_to_lerobot.py
    # ─────────────────────────────────────────────────────────────
    # 'robotwin_posttrain': {
    #     # Add converted tasks here, e.g.:
    #     # 'beat_block_hammer-demo_clean': {
    #     #     'dataset_path': '/path/to/robotwin_posttrain/beat_block_hammer-demo_clean',
    #     # },
    # },
    'robotwin_posttrain': {
        'adjust_bottle-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/adjust_bottle-aloha-agilex-demo_clean',
        },
        'adjust_bottle-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/adjust_bottle-aloha-agilex-demo_randomized',
        },
        'beat_block_hammer-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/beat_block_hammer-aloha-agilex-demo_clean',
        },
        'beat_block_hammer-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/beat_block_hammer-aloha-agilex-demo_randomized',
        },
        'blocks_ranking_rgb-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/blocks_ranking_rgb-aloha-agilex-demo_clean',
        },
        'blocks_ranking_rgb-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/blocks_ranking_rgb-aloha-agilex-demo_randomized',
        },
        'blocks_ranking_size-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/blocks_ranking_size-aloha-agilex-demo_clean',
        },
        'blocks_ranking_size-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/blocks_ranking_size-aloha-agilex-demo_randomized',
        },
        'click_alarmclock-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/click_alarmclock-aloha-agilex-demo_clean',
        },
        'click_alarmclock-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/click_alarmclock-aloha-agilex-demo_randomized',
        },
        'click_bell-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/click_bell-aloha-agilex-demo_clean',
        },
        'click_bell-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/click_bell-aloha-agilex-demo_randomized',
        },
        'dump_bin_bigbin-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/dump_bin_bigbin-aloha-agilex-demo_clean',
        },
        'dump_bin_bigbin-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/dump_bin_bigbin-aloha-agilex-demo_randomized',
        },
        'grab_roller-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/grab_roller-aloha-agilex-demo_clean',
        },
        'grab_roller-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/grab_roller-aloha-agilex-demo_randomized',
        },
        'handover_block-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/handover_block-aloha-agilex-demo_clean',
        },
        'handover_block-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/handover_block-aloha-agilex-demo_randomized',
        },
        'handover_mic-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/handover_mic-aloha-agilex-demo_clean',
        },
        'handover_mic-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/handover_mic-aloha-agilex-demo_randomized',
        },
        'hanging_mug-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/hanging_mug-aloha-agilex-demo_clean',
        },
        'hanging_mug-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/hanging_mug-aloha-agilex-demo_randomized',
        },
        'lift_pot-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/lift_pot-aloha-agilex-demo_clean',
        },
        'lift_pot-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/lift_pot-aloha-agilex-demo_randomized',
        },
        'move_can_pot-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/move_can_pot-aloha-agilex-demo_clean',
        },
        'move_can_pot-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/move_can_pot-aloha-agilex-demo_randomized',
        },
        'move_pillbottle_pad-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/move_pillbottle_pad-aloha-agilex-demo_clean',
        },
        'move_pillbottle_pad-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/move_pillbottle_pad-aloha-agilex-demo_randomized',
        },
        'move_playingcard_away-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/move_playingcard_away-aloha-agilex-demo_clean',
        },
        'move_playingcard_away-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/move_playingcard_away-aloha-agilex-demo_randomized',
        },
        'move_stapler_pad-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/move_stapler_pad-aloha-agilex-demo_clean',
        },
        'move_stapler_pad-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/move_stapler_pad-aloha-agilex-demo_randomized',
        },
        'open_laptop-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/open_laptop-aloha-agilex-demo_clean',
        },
        'open_laptop-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/open_laptop-aloha-agilex-demo_randomized',
        },
        'open_microwave-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/open_microwave-aloha-agilex-demo_clean',
        },
        'open_microwave-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/open_microwave-aloha-agilex-demo_randomized',
        },
        'pick_diverse_bottles-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/pick_diverse_bottles-aloha-agilex-demo_clean',
        },
        'pick_diverse_bottles-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/pick_diverse_bottles-aloha-agilex-demo_randomized',
        },
        'pick_dual_bottles-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/pick_dual_bottles-aloha-agilex-demo_clean',
        },
        'pick_dual_bottles-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/pick_dual_bottles-aloha-agilex-demo_randomized',
        },
        'place_a2b_left-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_a2b_left-aloha-agilex-demo_clean',
        },
        'place_a2b_left-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_a2b_left-aloha-agilex-demo_randomized',
        },
        'place_a2b_right-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_a2b_right-aloha-agilex-demo_clean',
        },
        'place_a2b_right-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_a2b_right-aloha-agilex-demo_randomized',
        },
        'place_bread_basket-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_bread_basket-aloha-agilex-demo_clean',
        },
        'place_bread_basket-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_bread_basket-aloha-agilex-demo_randomized',
        },
        'place_bread_skillet-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_bread_skillet-aloha-agilex-demo_clean',
        },
        'place_bread_skillet-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_bread_skillet-aloha-agilex-demo_randomized',
        },
        'place_burger_fries-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_burger_fries-aloha-agilex-demo_clean',
        },
        'place_burger_fries-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_burger_fries-aloha-agilex-demo_randomized',
        },
        'place_can_basket-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_can_basket-aloha-agilex-demo_clean',
        },
        'place_can_basket-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_can_basket-aloha-agilex-demo_randomized',
        },
        'place_cans_plasticbox-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_cans_plasticbox-aloha-agilex-demo_clean',
        },
        'place_cans_plasticbox-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_cans_plasticbox-aloha-agilex-demo_randomized',
        },
        'place_container_plate-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_container_plate-aloha-agilex-demo_clean',
        },
        'place_container_plate-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_container_plate-aloha-agilex-demo_randomized',
        },
        'place_dual_shoes-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_dual_shoes-aloha-agilex-demo_clean',
        },
        'place_dual_shoes-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_dual_shoes-aloha-agilex-demo_randomized',
        },
        'place_empty_cup-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_empty_cup-aloha-agilex-demo_clean',
        },
        'place_empty_cup-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_empty_cup-aloha-agilex-demo_randomized',
        },
        'place_fan-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_fan-aloha-agilex-demo_clean',
        },
        'place_fan-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_fan-aloha-agilex-demo_randomized',
        },
        'place_mouse_pad-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_mouse_pad-aloha-agilex-demo_clean',
        },
        'place_mouse_pad-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_mouse_pad-aloha-agilex-demo_randomized',
        },
        'place_object_basket-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_object_basket-aloha-agilex-demo_clean',
        },
        'place_object_basket-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_object_basket-aloha-agilex-demo_randomized',
        },
        'place_object_scale-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_object_scale-aloha-agilex-demo_clean',
        },
        'place_object_scale-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_object_scale-aloha-agilex-demo_randomized',
        },
        'place_object_stand-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_object_stand-aloha-agilex-demo_clean',
        },
        'place_object_stand-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_object_stand-aloha-agilex-demo_randomized',
        },
        'place_phone_stand-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_phone_stand-aloha-agilex-demo_clean',
        },
        'place_phone_stand-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_phone_stand-aloha-agilex-demo_randomized',
        },
        'place_shoe-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_shoe-aloha-agilex-demo_clean',
        },
        'place_shoe-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/place_shoe-aloha-agilex-demo_randomized',
        },
        'press_stapler-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/press_stapler-aloha-agilex-demo_clean',
        },
        'press_stapler-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/press_stapler-aloha-agilex-demo_randomized',
        },
        'put_bottles_dustbin-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/put_bottles_dustbin-aloha-agilex-demo_clean',
        },
        'put_bottles_dustbin-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/put_bottles_dustbin-aloha-agilex-demo_randomized',
        },
        'put_object_cabinet-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/put_object_cabinet-aloha-agilex-demo_clean',
        },
        'put_object_cabinet-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/put_object_cabinet-aloha-agilex-demo_randomized',
        },
        'rotate_qrcode-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/rotate_qrcode-aloha-agilex-demo_clean',
        },
        'rotate_qrcode-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/rotate_qrcode-aloha-agilex-demo_randomized',
        },
        'scan_object-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/scan_object-aloha-agilex-demo_clean',
        },
        'scan_object-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/scan_object-aloha-agilex-demo_randomized',
        },
        'shake_bottle-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/shake_bottle-aloha-agilex-demo_clean',
        },
        'shake_bottle-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/shake_bottle-aloha-agilex-demo_randomized',
        },
        'shake_bottle_horizontally-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/shake_bottle_horizontally-aloha-agilex-demo_clean',
        },
        'shake_bottle_horizontally-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/shake_bottle_horizontally-aloha-agilex-demo_randomized',
        },
        'stack_blocks_three-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/stack_blocks_three-aloha-agilex-demo_clean',
        },
        'stack_blocks_three-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/stack_blocks_three-aloha-agilex-demo_randomized',
        },
        'stack_blocks_two-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/stack_blocks_two-aloha-agilex-demo_clean',
        },
        'stack_blocks_two-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/stack_blocks_two-aloha-agilex-demo_randomized',
        },
        'stack_bowls_three-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/stack_bowls_three-aloha-agilex-demo_clean',
        },
        'stack_bowls_three-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/stack_bowls_three-aloha-agilex-demo_randomized',
        },
        'stack_bowls_two-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/stack_bowls_two-aloha-agilex-demo_clean',
        },
        'stack_bowls_two-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/stack_bowls_two-aloha-agilex-demo_randomized',
        },
        'stamp_seal-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/stamp_seal-aloha-agilex-demo_clean',
        },
        'stamp_seal-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/stamp_seal-aloha-agilex-demo_randomized',
        },
        'turn_switch-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/turn_switch-aloha-agilex-demo_clean',
        },
        'turn_switch-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/robotwin_posttrain/turn_switch-aloha-agilex-demo_randomized',
        },
    },
    'eef_robotwin': {
        'adjust_bottle-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/adjust_bottle-aloha-agilex-demo_clean',
        },
        'adjust_bottle-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/adjust_bottle-aloha-agilex-demo_randomized',
        },
        'beat_block_hammer-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/beat_block_hammer-aloha-agilex-demo_clean',
        },
        'beat_block_hammer-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/beat_block_hammer-aloha-agilex-demo_randomized',
        },
        'blocks_ranking_rgb-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/blocks_ranking_rgb-aloha-agilex-demo_clean',
        },
        'blocks_ranking_rgb-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/blocks_ranking_rgb-aloha-agilex-demo_randomized',
        },
        'blocks_ranking_size-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/blocks_ranking_size-aloha-agilex-demo_clean',
        },
        'blocks_ranking_size-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/blocks_ranking_size-aloha-agilex-demo_randomized',
        },
        'click_alarmclock-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/click_alarmclock-aloha-agilex-demo_clean',
        },
        'click_alarmclock-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/click_alarmclock-aloha-agilex-demo_randomized',
        },
        'click_bell-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/click_bell-aloha-agilex-demo_clean',
        },
        'click_bell-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/click_bell-aloha-agilex-demo_randomized',
        },
        'dump_bin_bigbin-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/dump_bin_bigbin-aloha-agilex-demo_clean',
        },
        'dump_bin_bigbin-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/dump_bin_bigbin-aloha-agilex-demo_randomized',
        },
        'grab_roller-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/grab_roller-aloha-agilex-demo_clean',
        },
        'grab_roller-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/grab_roller-aloha-agilex-demo_randomized',
        },
        'handover_block-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/handover_block-aloha-agilex-demo_clean',
        },
        'handover_block-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/handover_block-aloha-agilex-demo_randomized',
        },
        'handover_mic-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/handover_mic-aloha-agilex-demo_clean',
        },
        'handover_mic-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/handover_mic-aloha-agilex-demo_randomized',
        },
        'hanging_mug-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/hanging_mug-aloha-agilex-demo_clean',
        },
        'hanging_mug-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/hanging_mug-aloha-agilex-demo_randomized',
        },
        'lift_pot-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/lift_pot-aloha-agilex-demo_clean',
        },
        'lift_pot-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/lift_pot-aloha-agilex-demo_randomized',
        },
        'move_can_pot-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/move_can_pot-aloha-agilex-demo_clean',
        },
        'move_can_pot-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/move_can_pot-aloha-agilex-demo_randomized',
        },
        'move_pillbottle_pad-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/move_pillbottle_pad-aloha-agilex-demo_clean',
        },
        'move_pillbottle_pad-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/move_pillbottle_pad-aloha-agilex-demo_randomized',
        },
        'move_playingcard_away-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/move_playingcard_away-aloha-agilex-demo_clean',
        },
        'move_playingcard_away-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/move_playingcard_away-aloha-agilex-demo_randomized',
        },
        'move_stapler_pad-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/move_stapler_pad-aloha-agilex-demo_clean',
        },
        'move_stapler_pad-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/move_stapler_pad-aloha-agilex-demo_randomized',
        },
        'open_laptop-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/open_laptop-aloha-agilex-demo_clean',
        },
        'open_laptop-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/open_laptop-aloha-agilex-demo_randomized',
        },
        'open_microwave-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/open_microwave-aloha-agilex-demo_clean',
        },
        'open_microwave-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/open_microwave-aloha-agilex-demo_randomized',
        },
        'pick_diverse_bottles-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/pick_diverse_bottles-aloha-agilex-demo_clean',
        },
        'pick_diverse_bottles-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/pick_diverse_bottles-aloha-agilex-demo_randomized',
        },
        'pick_dual_bottles-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/pick_dual_bottles-aloha-agilex-demo_clean',
        },
        'pick_dual_bottles-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/pick_dual_bottles-aloha-agilex-demo_randomized',
        },
        'place_a2b_left-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_a2b_left-aloha-agilex-demo_clean',
        },
        'place_a2b_left-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_a2b_left-aloha-agilex-demo_randomized',
        },
        'place_a2b_right-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_a2b_right-aloha-agilex-demo_clean',
        },
        'place_a2b_right-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_a2b_right-aloha-agilex-demo_randomized',
        },
        'place_bread_basket-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_bread_basket-aloha-agilex-demo_clean',
        },
        'place_bread_basket-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_bread_basket-aloha-agilex-demo_randomized',
        },
        'place_bread_skillet-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_bread_skillet-aloha-agilex-demo_clean',
        },
        'place_bread_skillet-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_bread_skillet-aloha-agilex-demo_randomized',
        },
        'place_burger_fries-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_burger_fries-aloha-agilex-demo_clean',
        },
        'place_burger_fries-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_burger_fries-aloha-agilex-demo_randomized',
        },
        'place_can_basket-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_can_basket-aloha-agilex-demo_clean',
        },
        'place_can_basket-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_can_basket-aloha-agilex-demo_randomized',
        },
        'place_cans_plasticbox-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_cans_plasticbox-aloha-agilex-demo_clean',
        },
        'place_cans_plasticbox-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_cans_plasticbox-aloha-agilex-demo_randomized',
        },
        'place_container_plate-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_container_plate-aloha-agilex-demo_clean',
        },
        'place_container_plate-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_container_plate-aloha-agilex-demo_randomized',
        },
        'place_dual_shoes-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_dual_shoes-aloha-agilex-demo_clean',
        },
        'place_dual_shoes-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_dual_shoes-aloha-agilex-demo_randomized',
        },
        'place_empty_cup-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_empty_cup-aloha-agilex-demo_clean',
        },
        'place_empty_cup-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_empty_cup-aloha-agilex-demo_randomized',
        },
        'place_fan-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_fan-aloha-agilex-demo_clean',
        },
        'place_fan-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_fan-aloha-agilex-demo_randomized',
        },
        'place_mouse_pad-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_mouse_pad-aloha-agilex-demo_clean',
        },
        'place_mouse_pad-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_mouse_pad-aloha-agilex-demo_randomized',
        },
        'place_object_basket-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_object_basket-aloha-agilex-demo_clean',
        },
        'place_object_basket-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_object_basket-aloha-agilex-demo_randomized',
        },
        'place_object_scale-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_object_scale-aloha-agilex-demo_clean',
        },
        'place_object_scale-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_object_scale-aloha-agilex-demo_randomized',
        },
        'place_object_stand-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_object_stand-aloha-agilex-demo_clean',
        },
        'place_object_stand-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_object_stand-aloha-agilex-demo_randomized',
        },
        'place_phone_stand-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_phone_stand-aloha-agilex-demo_clean',
        },
        'place_phone_stand-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_phone_stand-aloha-agilex-demo_randomized',
        },
        'place_shoe-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_shoe-aloha-agilex-demo_clean',
        },
        'place_shoe-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/place_shoe-aloha-agilex-demo_randomized',
        },
        'press_stapler-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/press_stapler-aloha-agilex-demo_clean',
        },
        'press_stapler-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/press_stapler-aloha-agilex-demo_randomized',
        },
        'put_bottles_dustbin-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/put_bottles_dustbin-aloha-agilex-demo_clean',
        },
        'put_bottles_dustbin-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/put_bottles_dustbin-aloha-agilex-demo_randomized',
        },
        'put_object_cabinet-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/put_object_cabinet-aloha-agilex-demo_clean',
        },
        'put_object_cabinet-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/put_object_cabinet-aloha-agilex-demo_randomized',
        },
        'rotate_qrcode-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/rotate_qrcode-aloha-agilex-demo_clean',
        },
        'rotate_qrcode-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/rotate_qrcode-aloha-agilex-demo_randomized',
        },
        'scan_object-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/scan_object-aloha-agilex-demo_clean',
        },
        'scan_object-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/scan_object-aloha-agilex-demo_randomized',
        },
        'shake_bottle-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/shake_bottle-aloha-agilex-demo_clean',
        },
        'shake_bottle-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/shake_bottle-aloha-agilex-demo_randomized',
        },
        'shake_bottle_horizontally-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/shake_bottle_horizontally-aloha-agilex-demo_clean',
        },
        'shake_bottle_horizontally-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/shake_bottle_horizontally-aloha-agilex-demo_randomized',
        },
        'stack_blocks_three-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/stack_blocks_three-aloha-agilex-demo_clean',
        },
        'stack_blocks_three-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/stack_blocks_three-aloha-agilex-demo_randomized',
        },
        'stack_blocks_two-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/stack_blocks_two-aloha-agilex-demo_clean',
        },
        'stack_blocks_two-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/stack_blocks_two-aloha-agilex-demo_randomized',
        },
        'stack_bowls_three-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/stack_bowls_three-aloha-agilex-demo_clean',
        },
        'stack_bowls_three-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/stack_bowls_three-aloha-agilex-demo_randomized',
        },
        'stack_bowls_two-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/stack_bowls_two-aloha-agilex-demo_clean',
        },
        'stack_bowls_two-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/stack_bowls_two-aloha-agilex-demo_randomized',
        },
        'stamp_seal-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/stamp_seal-aloha-agilex-demo_clean',
        },
        'stamp_seal-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/stamp_seal-aloha-agilex-demo_randomized',
        },
        'turn_switch-aloha-agilex-demo_clean': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/turn_switch-aloha-agilex-demo_clean',
        },
        'turn_switch-aloha-agilex-demo_randomized': {
            'dataset_path': '/share/being-transfer/users/yiqing/datasets/eef_robotwin/turn_switch-aloha-agilex-demo_randomized',
        },
    },
}


def _load_xpolicylab_dataset_overrides() -> None:
    """Merge entries written by policy/Being_H05/process_data.sh (5-tuple paths)."""
    import json
    from pathlib import Path

    override_path = Path(__file__).parent / "dataset_info_xpolicylab.json"
    if not override_path.exists():
        return
    with open(override_path, encoding="utf-8") as f:
        overrides = json.load(f)
    for registry_name, datasets in overrides.items():
        DATASET_INFO.setdefault(registry_name, {}).update(datasets)


_load_xpolicylab_dataset_overrides()
