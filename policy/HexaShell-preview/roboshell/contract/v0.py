"""RoboShell contract v0: names shared by robo-server and the robo client.

Frozen after M2: fields may be added, never renamed or removed.
"""

CONTRACT_VERSION = "0.0.1"

ARMS = ("left", "right")
POINT_PRESETS = ("down", "forward", "down45")
# `point` needs the world axis along which the fingers open; there is no default
OPEN_AXES = ("x", "y", "z")
ROTATE_FRAMES = ("world", "tool")
PLAN_FAIL_REASONS = ("ik_unreachable", "self_collision", "table_collision", "workspace_limit")
END_REASONS = ("auto_success", "done", "budget", "sim_time", "early_fail", "agent_exit", "infra")

MOVE_CLIP_M = 0.20
ROTATE_CLIP_DEG = 90.0
WAIT_MAX_S = 5.0
COMMAND_BUDGET = 60

# exit codes of the robo client
EXIT_OK = 0
EXIT_BAD_ARGS = 1  # budget not consumed
EXIT_EXEC_FAILED = 2  # budget consumed
EXIT_EPISODE_OVER = 3
EXIT_CLIENT_TIMEOUT = 4  # result can be fetched with `robo status`

# commands that consume budget
BUDGETED = ("move", "rotate", "point", "gripper", "home", "wait")
FREE = ("obs", "status", "done")

OBS_FILES = ("head.png", "wrist_l.png", "wrist_r.png", "state.json")
# depth is optional: metres as float32 .npy plus a grayscale preview
OBS_DEPTH_FILES = ("head_depth.npy", "wrist_l_depth.npy", "wrist_r_depth.npy", "head_depth.png", "wrist_l_depth.png", "wrist_r_depth.png")

# agent-facing HTTP routes
ROUTE_CMD = "/v1/cmd"
ROUTE_OBS = "/v1/obs/"  # + file name
ROUTE_TOOLS = "/v1/tools"  # extra commands of the running task, for the client
# operator-only routes, served on a different port
ROUTE_ADMIN_RESET = "/admin/reset"
ROUTE_ADMIN_RESULT = "/admin/result"
ROUTE_ADMIN_REPLAY = "/admin/replay"
ROUTE_ADMIN_SHUTDOWN = "/admin/shutdown"
ROUTE_ADMIN_FINISH = "/admin/finish"  # the agent stopped without `done`: judge and close the episode
