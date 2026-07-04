# RoboDojo overlay payload

These files add RoboDojo post-training support to the public
[Hy-Embodied-0.5-VLA](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA)
repository, which does not ship them. They are **overlaid** onto a local clone
of that repo by [`../apply_robodojo_overlay.py`](../apply_robodojo_overlay.py),
which `install.sh` runs after cloning. We never push to the public repo.

The tree here mirrors the public repo layout, so each file lands at the same
relative path inside the clone:

| File | Destination in the Hy-Embodied clone |
|---|---|
| `hy_vla/data/robodojo_dataset.py` | `hy_vla/data/robodojo_dataset.py` |
| `hy_vla/config/dataset/robodojo_hdf5.yaml` | `hy_vla/config/dataset/robodojo_hdf5.yaml` |
| `scripts/compute_norm_robodojo.py` | `scripts/compute_norm_robodojo.py` |
| `scripts/train_robodojo_umi.sh` | `scripts/train_robodojo_umi.sh` |

The overlayer additionally inserts a `source == "robodojo"` branch into the
clone's `hy_vla/data/vla_dataset.py` dispatcher (idempotently).

These modules only depend on transforms that already exist upstream
(`hy_vla.utils.transform_utils.{convert_frame_robo_to_umi,
dual_arm_poses_to_relative, convert_PosQuat2PosRotationMatrix_batch}`); the
overlayer preflights for them and fails loudly if upstream drops them.
