from easydict import EasyDict
from .shared_config import va_shared_cfg
import os

va_robotwin30_cfg = EasyDict(__name__='Config: VA robotwin30')
va_robotwin30_cfg.update(va_shared_cfg)

va_robotwin30_cfg.wan22_pretrained_model_name_or_path = "/mnt/pfs/pg4hw0/niantian/model_weights/lingbot-va-base"

va_robotwin30_cfg.attn_window = 72
va_robotwin30_cfg.frame_chunk_size = 2
va_robotwin30_cfg.env_type = 'none'

va_robotwin30_cfg.height = 256
va_robotwin30_cfg.width = 256
va_robotwin30_cfg.action_dim = 30
va_robotwin30_cfg.action_per_frame = 12

va_robotwin30_cfg.obs_cam_keys = [
    'observation.images.cam_high',
    'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]

va_robotwin30_cfg.guidance_scale = 5
va_robotwin30_cfg.action_guidance_scale = 1

va_robotwin30_cfg.num_inference_steps = 25
va_robotwin30_cfg.video_exec_step = -1
va_robotwin30_cfg.action_num_inference_steps = 50

va_robotwin30_cfg.snr_shift = 5.0
va_robotwin30_cfg.action_snr_shift = 1.0

va_robotwin30_cfg.used_action_channel_ids = list(range(30))
va_robotwin30_cfg.inverse_used_action_channel_ids = list(range(30))

va_robotwin30_cfg.action_norm_method = 'quantiles'
va_robotwin30_cfg.norm_stat = {
    "q01": [
    -0.35896676778793335,
    -0.3138428330421448,
    0.8636696338653564,
    0.47643640637397766,
    -0.580956757068634,
    -0.0031553704757243395,
    0.009808734059333801,
    -0.08750182390213013,
    -0.3128036856651306,
    0.8690950870513916,
    0.003154901321977377,
    -0.6978152394294739,
    -0.006679104175418615,
    -0.7119914889335632,
    -0.9411209225654602,
    -5.257390398583084e-07,
    -2.296771708643064e-05,
    -1.4580196142196655,
    -0.02832568995654583,
    -0.9029805064201355,
    0.0,
    -0.12483851611614227,
    0.0,
    -2.81171942333458e-05,
    -1.5230885744094849,
    -0.9887850284576416,
    -0.9706628322601318,
    0.0,
    0.0,
    0.0
  ],
  "q99": [
    0.0731387659907341,
    0.10332044959068298,
    1.0637645721435547,
    0.9464830160140991,
    0.007200557738542557,
    0.7033131122589111,
    0.7211053967475891,
    0.3397349715232849,
    0.1032673791050911,
    1.0750845670700073,
    0.7035625576972961,
    0.7032108306884766,
    0.5845568180084229,
    0.9647521376609802,
    0.1324525624513626,
    2.5647311210632324,
    2.373156785964966,
    1.1154379844665527,
    0.9222444295883179,
    1.056174397468567,
    0.0,
    0.9590040445327759,
    2.612731456756592,
    2.5346524715423584,
    1.1117970943450928,
    0.09506302326917648,
    0.9307727813720703,
    0.0,
    1.0,
    1.0
  ],
}

va_robotwin30_train_cfg = EasyDict(__name__='Config: VA robotwin30 train')
va_robotwin30_train_cfg.update(va_robotwin30_cfg)

va_robotwin30_train_cfg.dataset_path = "/mnt/pfs/pg4hw0/niantian/lerobot/robotwin4tasks_ee_joint_gripper"
va_robotwin30_train_cfg.empty_emb_path = os.path.join(va_robotwin30_train_cfg.dataset_path, 'empty_emb.pt')
va_robotwin30_train_cfg.enable_wandb = True
va_robotwin30_train_cfg.load_worker = 16
va_robotwin30_train_cfg.save_interval = 200
va_robotwin30_train_cfg.gc_interval = 50
va_robotwin30_train_cfg.cfg_prob = 0.1

va_robotwin30_train_cfg.learning_rate = 1e-5
va_robotwin30_train_cfg.beta1 = 0.9
va_robotwin30_train_cfg.beta2 = 0.95
va_robotwin30_train_cfg.weight_decay = 0.1
va_robotwin30_train_cfg.warmup_steps = 10
va_robotwin30_train_cfg.batch_size = 1
va_robotwin30_train_cfg.gradient_accumulation_steps = 8
va_robotwin30_train_cfg.num_steps = 5000
