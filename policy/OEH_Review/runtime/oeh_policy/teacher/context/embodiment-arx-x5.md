ARX X5 dual-arm embodiment of the OpenWAM-α/RoboDojo integration [proposal: integration content for the
OpenWAM-α/RoboDojo integration; not a harness requirement]. Every fact below is copied from the repository file and
line named after it.

- Arms: two, `left` and `right` (python/oeh-policy/oeh_policy/review.py:31). Action profile `dual_arm_ee16`, native
  action EE16 [l_xyz, l_quat_wxyz, l_grip, r_xyz, r_quat_wxyz, r_grip] (python/oeh-policy/oeh_policy/profiles.py:65-67;
  python/oeh-policy/examples/openwam-review.template.json:4); OpenWAM-α runs with env_cfg_type `arx_x5`, action_type
  `ee`, action_dim 14 (python/oeh-policy/examples/openwam-review.template.json:57-59).
- Frame and rotation: positions in the env-relative world frame, quaternions wxyz
  (python/oeh-policy/examples/openwam-review.template.json:20-21).
- Workspace boxes (x, y, z in m, per arm; python/oeh-policy/examples/openwam-review.template.json:22-49):
  left x [-1.212, 0.612], y [-1.362, 0.462], z [0.715, 1.677];
  right x [-0.612, 1.212], y [-1.362, 0.462], z [0.715, 1.677].
  Their source (python/oeh-policy/examples/openwam-review.template.json:50): "Boxes from pinned RoboDojo ee67a14: arm
  bases env_cfg/robot/dual_x5.yml (left [-0.3,-0.45,0.765], right [0.3,-0.45,0.765]); reach 0.912 m = sum of X5A.urdf
  joint-origin offsets base_link->link6->finger joints (all-links-straight upper bound); z from 5 cm below the table
  top (env_cfg/scene/default.yml: table z 0.74 + 0.025) to base + reach. Gross-error gate, not a policy constraint."
- Gripper: opening 1 = open, 0 = closed; the observed value is the previous command
  (python/oeh-policy/oeh_policy/profiles.py:74; python/oeh-policy/examples/openwam-review.template.json:18).
- Control rate: 25 Hz, "RoboDojo ee67a14 env_cfg/sim/sim_config.yml (dt 0.004, render_interval 10) and
  env_cfg/arx_x5.yml (collect_freq 25)" (python/oeh-eval/examples/matrix/robodojo-review-panel60.yaml:134-135).
- Chunks: OpenWAM-α proposes chunks of up to 32 rows (python/oeh-policy/examples/openwam-review.template.json:14);
  at 25 Hz a 32-row chunk spans 1.28 s (derived: 32 / 25).
- Review row limits (python/oeh-policy/oeh_policy/review.py:51-60, sources at review.py:62-66): `student` accepts up
  to 32 proposal rows (student_max_steps); `edit` and `eef` span 1 to 5 rows (correction_max_steps) with at most
  0.05 m and 0.35 rad per arm (correction_max_translation_m, correction_max_rotation_rad); `home` spans up to 32 rows
  (home_max_steps); rows of `edit`, `eef` and `home` move at most 0.02 m and 0.1 rad per arm per native step
  (row_max_translation_m, row_max_rotation_rad; review.py:12); a gripper counts as open for the return check at 0.8
  or more (home_open_threshold).
