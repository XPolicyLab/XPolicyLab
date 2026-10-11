"""Measure the camera model that a policy may know a priori: intrinsics, the fixed head camera pose,
and the pose of each wrist camera relative to its arm's end link. Writes agent/camera_model.json."""
import argparse, json, os, time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
os.environ.setdefault("ROBODOJO_RUN_ID", time.strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
app = AppLauncher(args).app
import numpy as np
from roboshell.server import geometry as geo
from roboshell.server.envsetup import build_env, start_episode

env = build_env(app, "stack_bowls", 0, device_id=args.device_id)
start_episode(env, 0)
USD_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])
cm = env.camera_manager
rm = env.robot_manager
model = {"note": "camera-to-world matrices in the OpenCV convention (x right, y down, z forward); wrist entries are relative to the arm end link", "cameras": {}}
arms = {r.arm_name.split("_")[0]: r for r in rm.robot_list if r.type == "target"}
for ith in range(cm.num_cams):
    name = cm.camera_names[0][ith]
    K = np.asarray(cm.get_camera_intrinsics(ith, 0), dtype=float)
    T = np.asarray(cm.get_camera_extrinsics(ith, 0), dtype=float) @ USD_TO_OPENCV
    entry = {"intrinsics": np.round(K, 4).tolist(), "size": [640, 480]}
    if "wrist" in name:
        arm = arms["left" if "left" in name else "right"]
        ee = geo.pose_to_matrix(rm.get_real_endpose(arm, env_idx_list=[0])[0])
        entry["ee_to_camera"] = np.round(np.linalg.inv(ee) @ T, 6).tolist()
    else:
        entry["extrinsics_world"] = np.round(T, 6).tolist()
    model["cameras"][name] = entry
# move the left arm and check that ee_to_camera is really constant
arm = arms["left"]
q = rm.get_joint(arm, env_idx_list=[0])[0]
target = q + np.array([0.3, 0.4, 0.3, -0.2, 0.3, 0.5])
for _ in range(40):
    env.take_action({"left_arm_joint_state": target.tolist(), "left_ee_joint_state": [1.0],
                     "right_arm_joint_state": rm.get_joint(arms["right"], env_idx_list=[0])[0].tolist(), "right_ee_joint_state": [1.0]})
env.get_obs()
ith = cm.camera_names[0].index("cam_left_wrist")
T = np.asarray(cm.get_camera_extrinsics(ith, 0), dtype=float) @ USD_TO_OPENCV
ee = geo.pose_to_matrix(rm.get_real_endpose(arm, env_idx_list=[0])[0])
after = np.linalg.inv(ee) @ T
before = np.asarray(model["cameras"]["cam_left_wrist"]["ee_to_camera"])
model["check_after_motion"] = {"max_abs_diff": float(np.abs(after - before).max()), "joints_moved_rad": float(np.abs(target - q).max())}
json.dump(model, open(args.out, "w"), indent=1)
print(json.dumps(model)[:600], flush=True)
print("CAMERA-MODEL-DONE", flush=True)
os._exit(0)
