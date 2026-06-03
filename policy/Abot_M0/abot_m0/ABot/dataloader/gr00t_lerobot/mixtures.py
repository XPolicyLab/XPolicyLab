"""
mixtures.py

Defines a registry of dataset mixtures and weights for the UniACT Datasets. Each dataset is associated with
a float "sampling weight"
"""

import os
from typing import Dict, List, Tuple
from pathlib import Path
from .dataset_mixture.builder import generate_dataset_mixture, merge_mixtures_with_group_ratios
from .dataset_mixture.builder_oxeague import generate_oxe_auge_dataset_mixture

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASETS_ROOT = Path(os.environ.get("ABOT_DATASETS_ROOT", ""))


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


SKIP_DEFAULT_MIXTURES = _env_flag("ABOT_SKIP_DEFAULT_MIXTURES")

DATASET_NAMED_MIXTURES = {
    "oxe": [
        ("OXE/austin_sailor_dataset_lerobot", 0.026426, "oxe_franka_austin", {"lerobot_version": "v2.0"}),
        ("OXE/bc_z_lerobot", 0.090090, "oxe_google_robot_bc", {"lerobot_version": "v2.0"}),
        ("OXE/berkeley_autolab_ur5_lerobot", 0.014414, "oxe_ur5_berkeley", {"lerobot_version": "v2.0"}),
        ("OXE/berkeley_cable_routing_lerobot", 0.002402, "oxe_franka_berkeley", {"lerobot_version": "v2.0"}),
        ("OXE/berkeley_fanuc_manipulation_lerobot", 0.008408, "oxe_fanuc_mate_berkeley", {"lerobot_version": "v2.0"}),
        ("OXE/bridge_orig_lerobot", 0.159760, "oxe_widowx_bridge", {"lerobot_version": "v2.0"}),
        ("OXE/cmu_stretch_lerobot", 0.002402, "oxe_hello_stretch_cmu", {"lerobot_version": "v2.0"}),
        ("OXE/dlr_edan_shared_control_lerobot", 0.000601, "oxe_dlr_edan", {"lerobot_version": "v2.0"}),
        ("OXE/fractal20220817_data_lerobot", 0.152553, "oxe_google_robot_fractal", {"lerobot_version": "v2.0"}),
        ("OXE/furniture_bench_dataset_lerobot", 0.028829, "oxe_franka_furniture", {"lerobot_version": "v2.0"}),
        ("OXE/jaco_play_lerobot", 0.004804, "oxe_jaco2", {"lerobot_version": "v2.0"}),
        ("OXE/kuka_lerobot", 0.152553, "oxe_kuka_iiwa", {"lerobot_version": "v2.0"}),
        ("OXE/language_table_lerobot", 0.052853, "oxe_xarm_language", {"lerobot_version": "v2.0"}),
        ("OXE/nyu_franka_play_dataset_lerobot", 0.009609, "oxe_franka_nyu", {"lerobot_version": "v2.0"}),
        ("OXE/stanford_hydra_dataset_lerobot", 0.052853, "oxe_franka_stanford", {"lerobot_version": "v2.0"}),
        ("OXE/taco_play_lerobot", 0.036036, "oxe_franka_taco", {"lerobot_version": "v2.0"}),
        ("OXE/fmb_dataset_lerobot", 0.085255, "oxe_franka_fmb", {"lerobot_version": "v2.0"}),
        ("OXE/droid_lerobot", 0.120120, "oxe_franka_droid", {"lerobot_version": "v2.0"}),
    ],
}

if not SKIP_DEFAULT_MIXTURES:
    ######################################################## RoboCOIN ########################################################

    robocoin_mixture_name = 'robocoin'
    rel_robocoin_path = ["RoboCOIN", "RoboCOIN"]
    abs_robocoin_path = [str(DATASETS_ROOT / rel_path) for rel_path in rel_robocoin_path]
    robocoin_mapping_jsons= [str(REPO_ROOT / "ABot/dataloader/gr00t_lerobot/robot_type_map/RoboCoin.json"),
                             str(REPO_ROOT / "ABot/dataloader/gr00t_lerobot/robot_type_map/RoboCoin_Dexterous.json")]
    Robocoin_mixtures = generate_dataset_mixture(
        mixture_name=robocoin_mixture_name,
        abs_paths=abs_robocoin_path,
        rel_paths=rel_robocoin_path,
        dataset_subset_mapping_jsons=robocoin_mapping_jsons,
        intra_dataset_weight_mode=None,
        dataset_weights=None,
        lerobot_version="v2.0"
    )
    DATASET_NAMED_MIXTURES.update(Robocoin_mixtures)

    ######################################################## Galaxea ########################################################

    galaxea_mixture_name = 'galaxea'
    rel_galaxea_path = ["Galaxea"]
    abs_galaxea_path = [str(DATASETS_ROOT / rel_path) for rel_path in rel_galaxea_path]
    galaxea_mapping_jsons = [str(REPO_ROOT / "ABot/dataloader/gr00t_lerobot/robot_type_map/Galaxea.json")]
    galaxea_mixtures = generate_dataset_mixture(
        mixture_name=galaxea_mixture_name,
        abs_paths=abs_galaxea_path,
        rel_paths=rel_galaxea_path,
        dataset_subset_mapping_jsons=galaxea_mapping_jsons,
        intra_dataset_weight_mode=None,
        dataset_weights=None,
        lerobot_version="v2.0"
    )
    DATASET_NAMED_MIXTURES.update(galaxea_mixtures)

    ######################################################## AgiBotWorld ########################################################

    agibotworld_mixture_name = 'agibotworld'
    rel_agibotworld_path = ["Agibot-Beta"]
    abs_agibotworld_path = [str(DATASETS_ROOT / rel_path) for rel_path in rel_agibotworld_path]
    agibotworld_mapping_jsons = [str(REPO_ROOT / "ABot/dataloader/gr00t_lerobot/robot_type_map/AgiBotWorld.json")]
    agibotworld_mixtures = generate_dataset_mixture(
        mixture_name=agibotworld_mixture_name,
        abs_paths=abs_agibotworld_path,
        rel_paths=rel_agibotworld_path,
        dataset_subset_mapping_jsons=agibotworld_mapping_jsons,
        intra_dataset_weight_mode=None,
        dataset_weights=None,
        lerobot_version="v2.0"
    )
    DATASET_NAMED_MIXTURES.update(agibotworld_mixtures)

    ######################################################## OxeAuge ########################################################

    oxeauge_mixture_name = 'oxeauge'
    rel_oxeauge_path = "OXE-AugE"
    abs_oxeauge_path = str(DATASETS_ROOT / rel_oxeauge_path)
    oxeauge_mapping_jsons = str(REPO_ROOT / "ABot/dataloader/gr00t_lerobot/robot_type_map/OXE-Auge.json")
    group_prefixes = ["austin_buds_dataset", "austin_sailor_dataset", "iamlab_cmu_pickup_insert", "kaist_nonprehensile", "nyu_franka_play_dataset",
                      "taco_play", "toto", "utaustin_mutex", "viola", "berkeley_autolab_ur5", "bridge", "fractal20220817_data", "jaco_play", "language_table",
                      "ucsd_kitchen", "utokyo_xarm_pick_and_place"]
    oxeauge_mixtures = generate_oxe_auge_dataset_mixture(
        mixture_name=oxeauge_mixture_name,
        abs_path=abs_oxeauge_path,
        rel_path=rel_oxeauge_path,
        dataset_robot_mapping_json=oxeauge_mapping_jsons,
        intra_group_weight_mode=None,
        lerobot_version="v2.0",
        group_prefixes=group_prefixes,
        group_weights=None,
    )
    DATASET_NAMED_MIXTURES.update(oxeauge_mixtures)

    ######################################################## InternData-A1 ########################################################

    intern_mixture_name = 'interndata'
    rel_intern_path = ["InternData/articulation_tasks",
                       "InternData/long_horizon_tasks",
                       "InternData/basic_tasks",
                       "InternData/pick_and_place_tasks"]
    abs_intern_path = [str(DATASETS_ROOT / rel_path) for rel_path in rel_intern_path]
    intern_mapping_jsons = [str(REPO_ROOT / "ABot/dataloader/gr00t_lerobot/robot_type_map/InternData-A1_articulation_tasks.json"),
                            str(REPO_ROOT / "ABot/dataloader/gr00t_lerobot/robot_type_map/InternData-A1_long_horizon_tasks.json"),
                            str(REPO_ROOT / "ABot/dataloader/gr00t_lerobot/robot_type_map/InternData-A1_basic_tasks.json"),
                            str(REPO_ROOT / "ABot/dataloader/gr00t_lerobot/robot_type_map/InternData-A1_pick_and_place_tasks.json")]
    intern_weights = [0.187, 0.193, 0.352, 0.268]
    intern_mixtures = generate_dataset_mixture(
        mixture_name=intern_mixture_name,
        abs_paths=abs_intern_path,
        rel_paths=rel_intern_path,
        dataset_subset_mapping_jsons=intern_mapping_jsons,
        intra_dataset_weight_mode=None,
        dataset_weights=intern_weights,
        lerobot_version="v2.0"
    )
    DATASET_NAMED_MIXTURES.update(intern_mixtures)

    ######################################################## Libero ########################################################

    libero_mixture_name = 'libero'
    rel_libero_path = ["libero"]
    abs_libero_path = [str(DATASETS_ROOT / rel_path) for rel_path in rel_libero_path]
    libero_mapping_jsons = [str(REPO_ROOT / "ABot/dataloader/gr00t_lerobot/robot_type_map/Libero.json")]
    libero_mixtures = generate_dataset_mixture(
        mixture_name=libero_mixture_name,
        abs_paths=abs_libero_path,
        rel_paths=rel_libero_path,
        dataset_subset_mapping_jsons=libero_mapping_jsons,
        intra_dataset_weight_mode=None,
        dataset_weights=None,
        lerobot_version="v2.0"
    )
    DATASET_NAMED_MIXTURES.update(libero_mixtures)

    ######################################################## RoboTwin ########################################################

    robotwin_mixture_name = 'robotwin'
    rel_robotwin_path = ["robotwin/Clean",
                         "robotwin/Randomized"]
    abs_robotwin_path = [str(DATASETS_ROOT / rel_path) for rel_path in rel_robotwin_path]
    robotwin_mapping_jsons = [str(REPO_ROOT / "ABot/dataloader/gr00t_lerobot/robot_type_map/Robotwin_clean.json"),
                              str(REPO_ROOT / "ABot/dataloader/gr00t_lerobot/robot_type_map/Robotwin_random.json")]
    robotwin_weights = [0.5, 0.5]
    robotwin_mixtures = generate_dataset_mixture(
        mixture_name=robotwin_mixture_name,
        abs_paths=abs_robotwin_path,
        rel_paths=rel_robotwin_path,
        dataset_subset_mapping_jsons=robotwin_mapping_jsons,
        intra_dataset_weight_mode=None,
        dataset_weights=robotwin_weights,
        lerobot_version="v2.0"
    )
    DATASET_NAMED_MIXTURES.update(robotwin_mixtures)

    ######################################################## RoboCase ########################################################

    robotcase_mixture_name = 'robocase_gr1'
    rel_robotcase_path = ["robocase_gr1"]
    abs_robotcase_path = [str(DATASETS_ROOT / rel_path) for rel_path in rel_robotcase_path]
    robotcase_mapping_jsons = [str(REPO_ROOT / "ABot/dataloader/gr00t_lerobot/robot_type_map/RoboCase.json")]
    robotcase_mixtures = generate_dataset_mixture(
        mixture_name=robotcase_mixture_name,
        abs_paths=abs_robotcase_path,
        rel_paths=rel_robotcase_path,
        dataset_subset_mapping_jsons=robotcase_mapping_jsons,
        intra_dataset_weight_mode=None,
        dataset_weights=None,
        lerobot_version="v2.0"
    )
    DATASET_NAMED_MIXTURES.update(robotcase_mixtures)

    ############################################## Mixtures ########################################################
    mixture_ratios = {
        "oxe": 0.17,
        "oxeauge": 0.15,
        "robocoin": 0.19,
        "agibotworld": 0.20,
        "galaxea": 0.05,
        "interndata": 0.24,
    }
    real_world_training_mix = merge_mixtures_with_group_ratios(
        DATASET_NAMED_MIXTURES,
        mixture_ratios,
        "pretrain"
    )
    DATASET_NAMED_MIXTURES.update(real_world_training_mix)

sim_stack_bowls_repo = os.environ.get("ABOT_SIM_STACK_BOWLS_REPO")
if sim_stack_bowls_repo:
    DATASET_NAMED_MIXTURES["sim_stack_bowls"] = [
        (sim_stack_bowls_repo, 1.0, "robotwin", {"lerobot_version": "v2.0"})
    ]

single_dataset_repo = os.environ.get("ABOT_SINGLE_DATASET_REPO")
if single_dataset_repo:
    single_dataset_mix = os.environ.get("ABOT_SINGLE_DATASET_MIX", "single_dataset")
    single_dataset_robot_type = os.environ.get("ABOT_SINGLE_DATASET_ROBOT_TYPE", "robotwin")
    DATASET_NAMED_MIXTURES[single_dataset_mix] = [
        (single_dataset_repo, 1.0, single_dataset_robot_type, {"lerobot_version": "v2.0"})
    ]

print("Update finish!")