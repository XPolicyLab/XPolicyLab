"""Operator check of the depth channel: at the pixel where each TCP projects, the rendered depth should be
close to the TCP's z coordinate in the camera frame (the fingers are a few cm around it)."""
import json, os, subprocess, sys
import numpy as np

obs = os.environ["ROBO_OBS_DIR"]
subprocess.run([sys.executable, "roboshell/client/robo.py", "obs"], capture_output=True)
state = json.load(open(os.path.join(obs, "state.json")))
for cam in ("head", "wrist_l", "wrist_r"):
    path = os.path.join(obs, f"{cam}_depth.npy")
    if not os.path.exists(path):
        print(cam, "no depth file"); continue
    depth = np.load(path)
    valid = depth[depth > 0]
    print(f"{cam}: shape {depth.shape} dtype {depth.dtype} valid {valid.size/depth.size:.2%} range {valid.min():.3f}..{valid.max():.3f} m")
    K = np.array(state["cameras"][cam]["intrinsics"]); T = np.array(state["cameras"][cam]["extrinsics_world"])
    for arm in ("left", "right"):
        p = np.array(state[arm]["tcp_pos"] + [1.0]); pc = np.linalg.inv(T) @ p
        if pc[2] <= 0.05: continue
        u = int(round(K[0, 0] * pc[0] / pc[2] + K[0, 2])); v = int(round(K[1, 1] * pc[1] / pc[2] + K[1, 2]))
        if 0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]:
            window = depth[max(0, v-4):v+5, max(0, u-4):u+5]; window = window[window > 0]
            print(f"   {arm} tcp -> pixel ({u},{v}) camera z {pc[2]:.3f} m, rendered depth near pixel {window.min() if window.size else float('nan'):.3f}..{window.max() if window.size else float('nan'):.3f} m")
