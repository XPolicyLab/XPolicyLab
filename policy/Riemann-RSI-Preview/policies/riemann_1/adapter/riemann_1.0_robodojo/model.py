"""XPolicyLab adapter for SkyrobotWAM (skyrobot-va-pretrain).

Bridges the XPolicyLab eval protocol (``ModelTemplate``) to the
benchmark-agnostic ``SkyrobotWAMPolicy`` in
``<skyrobot_root>/skyrobot/deploy/policy.py``.

Data flow per env step:
    RoboDojo obs (cam_head / cam_left_wrist / cam_right_wrist + EE state)
        -> inverted-T combine_views (head 256x320 top, wrists 128x160 bottom)
        -> [384, 320, 3] uint8 frame  (exactly the training frame layout)
        -> SkyrobotWAMPolicy.act(image, state_16)
        -> ABSOLUTE 16-d EE action [L: xyz qw qx qy qz grip, R: same]
        -> XPolicyLab ee action dict (env solves IK).

The policy is streaming (num_latents latent history + internal frame buffer /
action queue), so ``get_action`` returns exactly ONE action per call and the
XPolicyLab per-step loop maps 1:1 onto ``policy.act``.

State layout (16-d, matches the RoboDojo_ee lerobot export the model was
trained on): [l_x, l_y, l_z, l_qw, l_qx, l_qy, l_qz, l_grip, r_...] with
grippers in [0, 1] - identical to the runtime obs left/right_ee_pose +
left/right_ee_joint_state, so assembly is pure concatenation.
"""
if '__riemann_skills__' not in globals():
    raise RuntimeError('Load the WAM adapter through rsi.wam_policy.create; process-wide import hooks are not supported')
import os
import sys
import numpy as np
from XPolicyLab.model_template import ModelTemplate
_CAM_ENV_TO_DS = {'cam_head': 'observation.images.cam_high', 'cam_left_wrist': 'observation.images.cam_left_wrist', 'cam_right_wrist': 'observation.images.cam_right_wrist'}
_STATE_DIM = 16
_QUAT_STARTS = (3, 11)
_GRIP_IDXS = (7, 15)
_resolve_model_path = __riemann_skills__.bind('policies/riemann_1/adapter/riemann_1.0_robodojo/model.py:_resolve_model_path:46', globals())
_load_train_config = __riemann_skills__.bind('policies/riemann_1/adapter/riemann_1.0_robodojo/model.py:_load_train_config:85', globals())
_verify_norm_stats = __riemann_skills__.bind('policies/riemann_1/adapter/riemann_1.0_robodojo/model.py:_verify_norm_stats:119', globals())

class Model(ModelTemplate):
    __init__ = __riemann_skills__.bind('policies/riemann_1/adapter/riemann_1.0_robodojo/model.py:Model.__init__:130', globals(), locals())
    _stitch_image = __riemann_skills__.bind('policies/riemann_1/adapter/riemann_1.0_robodojo/model.py:Model._stitch_image:258', globals(), locals())
    _state_from_obs = __riemann_skills__.bind('policies/riemann_1/adapter/riemann_1.0_robodojo/model.py:Model._state_from_obs:296', globals(), locals())
    _task_text = __riemann_skills__.bind('policies/riemann_1/adapter/riemann_1.0_robodojo/model.py:Model._task_text:317', globals(), locals())
    update_obs = __riemann_skills__.bind('policies/riemann_1/adapter/riemann_1.0_robodojo/model.py:Model.update_obs:327', globals(), locals())
    get_action = __riemann_skills__.bind('policies/riemann_1/adapter/riemann_1.0_robodojo/model.py:Model.get_action:335', globals(), locals())
    update_obs_batch = __riemann_skills__.bind('policies/riemann_1/adapter/riemann_1.0_robodojo/model.py:Model.update_obs_batch:360', globals(), locals())
    get_action_batch = __riemann_skills__.bind('policies/riemann_1/adapter/riemann_1.0_robodojo/model.py:Model.get_action_batch:367', globals(), locals())
    reset = __riemann_skills__.bind('policies/riemann_1/adapter/riemann_1.0_robodojo/model.py:Model.reset:370', globals(), locals())
