# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum


class EmbodimentTag(Enum):
    NEW_EMBODIMENT = "new_embodiment"

    # Libero
    LIBERO_FRANKA = "libero_franka"

    # RoboTwin
    ROBOTWIN = "robotwin"

    # RoboCase
    FOURIER_GR1_ARMS_WAIST = "fourier_gr1_arms_waist"


    # OXE-Uni   
    OXE_FRANKA_AUSTIN = "oxe_franka_austin"
    OXE_GOOGLE_ROBOT_BC = "oxe_google_robot_bc"
    OXE_UR5_BERKELEY = "oxe_ur5_berkeley"
    OXE_FRANKA_BERKELEY = "oxe_franka_berkeley"
    OXE_FANUC_MATE_BERKELEY = "oxe_fanuc_mate_berkeley"
    OXE_WIDOWX_BRIDGE = "oxe_widowx_bridge"
    OXE_HELLO_STRETCH_CMU = "oxe_hello_stretch_cmu"
    OXE_DLR_EDAN = "oxe_dlr_edan"
    OXE_FRANKA_FMB = "oxe_franka_fmb"
    OXE_GOOGLE_ROBOT_FRACTAL = "oxe_google_robot_fractal"
    OXE_FRANKA_FURNITURE = "oxe_franka_furniture"
    OXE_JACO2 = "oxe_jaco2"
    OXE_KUKA_IIWA = "oxe_kuka_iiwa"
    OXE_XARM_LANGUAGE = "oxe_xarm_language"
    OXE_FRANKA_NYU = "oxe_franka_nyu"
    OXE_FRANKA_STANFORD = "oxe_franka_stanford"
    OXE_FRANKA_TACO = "oxe_franka_taco"
    OXE_FRANKA_DROID = "oxe_franka_droid"

    # RoboCOIN
    RAGIBOT = "AgiBot-g1"
    RALPHA_BOT_2 = "alpha_bot_2"
    RCOBOT_MAGIC = "Cobot_Magic"
    RGALBOT_G1 = "Galbot_g1"
    RR1_LITE = "R1_Lite"
    RRMC_AIDA_L = "RMC-AIDA-L"
    RSPLIT_ALOHA = "Split_aloha"
    RTIANQIN_A2 = "Tianqin_A2"

    # RoboCoin_Dexterous
    RAIRBOT = "Airbot_MMK2"
    RUNITREE_G1 = "Unitree_G1"
    RLEJU_ROBOT = "leju_robot"
    
    # AgiBotWorld
    AGIBOT_WORLD_G1 = "AgiBotWorld-g1"

    # Galaxea
    GALAXEA_R1_LITE = "Galaxea_R1_Lite"

    # OXE-Auge
    OXE_AUGE_ORIGINAL = "oxe_auge_original"
    OXE_AUGE_GOOGLE_ROBOT = "oxe_auge_google_robot"
    OXE_AUGE_JACO = "oxe_auge_jaco"
    OXE_AUGE_KINIVA3 = "oxe_auge_kinova3"
    OXE_AUGE_KUKA_IIWA = "oxe_auge_kuka_iiwa"
    OXE_AUGE_PANDA = "oxe_auge_panda"
    OXE_AUGE_SAWYER = "oxe_auge_sawyer"
    OXE_AUGE_WIDOWX = "oxe_auge_widowX"
    OXE_AUGE_XARM7 = "oxe_auge_xarm7"
    OXE_AUGE_UR5E = "oxe_auge_ur5e"

    # InternData-A1
    INTERN_FRANKA = "intern_franka"
    INTERN_SPLIT_ALOHA = "intern_split_aloha"
    INTERN_LIFT2 = "intern_lift2"
    INTERN_GENIE1 = "intern_genie1"

# Embodiment tag string: to projector index in the Action Expert Module
EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.NEW_EMBODIMENT.value: 0,

    # Libero
    EmbodimentTag.LIBERO_FRANKA.value: 1,

    # RoboTwin
    EmbodimentTag.ROBOTWIN.value: 2,

    # RoboCase
    EmbodimentTag.FOURIER_GR1_ARMS_WAIST.value: 3,

    ############################################# Training #############################################

    # OXE-Uni
    EmbodimentTag.OXE_FRANKA_AUSTIN.value: 50,
    EmbodimentTag.OXE_GOOGLE_ROBOT_BC.value: 51,
    EmbodimentTag.OXE_UR5_BERKELEY.value: 52,
    EmbodimentTag.OXE_FRANKA_BERKELEY.value: 53,
    EmbodimentTag.OXE_FANUC_MATE_BERKELEY.value: 54,
    EmbodimentTag.OXE_WIDOWX_BRIDGE.value: 55,
    EmbodimentTag.OXE_HELLO_STRETCH_CMU.value: 56,
    EmbodimentTag.OXE_DLR_EDAN.value: 57,
    EmbodimentTag.OXE_FRANKA_FMB.value: 58,
    EmbodimentTag.OXE_GOOGLE_ROBOT_FRACTAL.value: 59,
    EmbodimentTag.OXE_FRANKA_FURNITURE.value: 60,
    EmbodimentTag.OXE_JACO2.value: 61,
    EmbodimentTag.OXE_KUKA_IIWA.value: 62,
    EmbodimentTag.OXE_XARM_LANGUAGE.value: 63,
    EmbodimentTag.OXE_FRANKA_NYU.value: 64,
    EmbodimentTag.OXE_FRANKA_STANFORD.value: 65,
    EmbodimentTag.OXE_FRANKA_TACO.value: 66,
    EmbodimentTag.OXE_FRANKA_DROID.value: 67,

    # RoboCOIN
    EmbodimentTag.RAGIBOT.value: 68,
    EmbodimentTag.RALPHA_BOT_2.value: 69,
    EmbodimentTag.RCOBOT_MAGIC.value: 70,
    EmbodimentTag.RGALBOT_G1.value: 71,
    EmbodimentTag.RR1_LITE.value: 72,
    EmbodimentTag.RRMC_AIDA_L.value: 73,
    EmbodimentTag.RSPLIT_ALOHA.value: 74,
    EmbodimentTag.RTIANQIN_A2.value: 75,

    # RoboCoin_Dexterous
    EmbodimentTag.RAIRBOT.value: 76,
    EmbodimentTag.RUNITREE_G1.value: 77,
    EmbodimentTag.RLEJU_ROBOT.value: 78,

    # AgiBotWorld
    EmbodimentTag.AGIBOT_WORLD_G1.value: 79,

    # Galaxea
    EmbodimentTag.GALAXEA_R1_LITE.value: 80,

    # OXE-Auge
    EmbodimentTag.OXE_AUGE_ORIGINAL.value: 81,
    EmbodimentTag.OXE_AUGE_GOOGLE_ROBOT.value: 82,
    EmbodimentTag.OXE_AUGE_JACO.value: 83,
    EmbodimentTag.OXE_AUGE_KINIVA3.value: 83,
    EmbodimentTag.OXE_AUGE_KUKA_IIWA.value: 84,
    EmbodimentTag.OXE_AUGE_PANDA.value: 85,
    EmbodimentTag.OXE_AUGE_SAWYER.value: 86,
    EmbodimentTag.OXE_AUGE_WIDOWX.value: 87,
    EmbodimentTag.OXE_AUGE_XARM7.value: 88,
    EmbodimentTag.OXE_AUGE_UR5E.value: 89,

    # InternData-A1
    EmbodimentTag.INTERN_FRANKA.value: 90,
    EmbodimentTag.INTERN_SPLIT_ALOHA.value: 91,
    EmbodimentTag.INTERN_LIFT2.value: 92,
    EmbodimentTag.INTERN_GENIE1.value: 93,

}

# Robot type to embodiment tag mapping
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "new_embodiment": EmbodimentTag.NEW_EMBODIMENT,

    # Libero
    "libero_franka": EmbodimentTag.LIBERO_FRANKA,

    # RoboTwin
    "robotwin": EmbodimentTag.ROBOTWIN,

    # RoboCase
    "fourier_gr1_arms_waist": EmbodimentTag.FOURIER_GR1_ARMS_WAIST,

    ############################################# Training #############################################

    # OXE-Uni
    "oxe_franka_austin": EmbodimentTag.OXE_FRANKA_AUSTIN,
    "oxe_google_robot_bc": EmbodimentTag.OXE_GOOGLE_ROBOT_BC,
    "oxe_ur5_berkeley": EmbodimentTag.OXE_UR5_BERKELEY,
    "oxe_franka_berkeley": EmbodimentTag.OXE_FRANKA_BERKELEY,
    "oxe_fanuc_mate_berkeley": EmbodimentTag.OXE_FANUC_MATE_BERKELEY,
    "oxe_widowx_bridge": EmbodimentTag.OXE_WIDOWX_BRIDGE,
    "oxe_hello_stretch_cmu": EmbodimentTag.OXE_HELLO_STRETCH_CMU,
    "oxe_dlr_edan": EmbodimentTag.OXE_DLR_EDAN,
    "oxe_franka_fmb": EmbodimentTag.OXE_FRANKA_FMB,
    "oxe_google_robot_fractal": EmbodimentTag.OXE_GOOGLE_ROBOT_FRACTAL,
    "oxe_franka_furniture": EmbodimentTag.OXE_FRANKA_FURNITURE,
    "oxe_jaco2": EmbodimentTag.OXE_JACO2,
    "oxe_kuka_iiwa": EmbodimentTag.OXE_KUKA_IIWA,
    "oxe_xarm_language": EmbodimentTag.OXE_XARM_LANGUAGE,
    "oxe_franka_nyu": EmbodimentTag.OXE_FRANKA_NYU,
    "oxe_franka_stanford": EmbodimentTag.OXE_FRANKA_STANFORD,
    "oxe_franka_taco": EmbodimentTag.OXE_FRANKA_TACO,
    "oxe_franka_droid": EmbodimentTag.OXE_FRANKA_DROID,
    
    # RoboCOIN
    "AgiBot-g1": EmbodimentTag.RAGIBOT,
    "alpha_bot_2": EmbodimentTag.RALPHA_BOT_2,
    "Cobot_Magic": EmbodimentTag.RCOBOT_MAGIC,
    "Galbot_g1": EmbodimentTag.RGALBOT_G1,
    "R1_Lite": EmbodimentTag.RR1_LITE,
    "RMC-AIDA-L": EmbodimentTag.RRMC_AIDA_L,
    "Split_aloha": EmbodimentTag.RSPLIT_ALOHA,
    "Tianqin_A2": EmbodimentTag.RTIANQIN_A2,

    # RoboCoin_Dexterous
    "Airbot_MMK2": EmbodimentTag.RAIRBOT,
    "Unitree_G1": EmbodimentTag.RUNITREE_G1,
    "leju_robot": EmbodimentTag.RLEJU_ROBOT,

    # AgiBotWorld
    "AgiBotWorld-g1": EmbodimentTag.AGIBOT_WORLD_G1,

    # Galaxea
    "Galaxea_R1_Lite": EmbodimentTag.GALAXEA_R1_LITE,
    
    # OXE-Auge
    "oxe_auge_original": EmbodimentTag.OXE_AUGE_ORIGINAL,
    "oxe_auge_google_robot": EmbodimentTag.OXE_AUGE_GOOGLE_ROBOT,
    "oxe_auge_jaco": EmbodimentTag.OXE_AUGE_JACO,
    "oxe_auge_kinova3": EmbodimentTag.OXE_AUGE_KINIVA3,
    "oxe_auge_kuka_iiwa": EmbodimentTag.OXE_AUGE_KUKA_IIWA,
    "oxe_auge_panda": EmbodimentTag.OXE_AUGE_PANDA,
    "oxe_auge_sawyer": EmbodimentTag.OXE_AUGE_SAWYER,
    "oxe_auge_widowX": EmbodimentTag.OXE_AUGE_WIDOWX,
    "oxe_auge_xarm7": EmbodimentTag.OXE_AUGE_XARM7,
    "oxe_auge_ur5e": EmbodimentTag.OXE_AUGE_UR5E,

    # InternData-A1
    "intern_franka": EmbodimentTag.INTERN_FRANKA,
    "intern_split_aloha": EmbodimentTag.INTERN_SPLIT_ALOHA,
    "intern_lift2": EmbodimentTag.INTERN_LIFT2,
    "intern_genie1": EmbodimentTag.INTERN_GENIE1,
}
