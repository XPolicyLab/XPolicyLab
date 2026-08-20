from enum import Enum

# DM0.5-Mem pools every history image to a 4x4 grid of soft visual tokens.
HISTORY_TOKENS_PER_IMAGE = 16
HISTORY_POOL_SIZE = int(HISTORY_TOKENS_PER_IMAGE**0.5)
assert HISTORY_POOL_SIZE * HISTORY_POOL_SIZE == HISTORY_TOKENS_PER_IMAGE


class ActionMode(Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class RobotStateDesc(Enum):
    JOINT = "joint"
    EEF = "eef"
    GRIPPER = "gripper"


class RobotType(Enum):
    DOS_W1 = "DOS W1"
    FRANKA = "Franka"
    ALOHA_ROBOTWIN2 = "Aloha RoboTwin2"
    SO101 = "SO101"


ROBOT_STATE_DESCS = {
    RobotType.DOS_W1: [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER],
    RobotType.ALOHA_ROBOTWIN2: [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER],
    RobotType.SO101: [RobotStateDesc.JOINT] * 5 + [RobotStateDesc.GRIPPER],
}
