from easydict import EasyDict
from pathlib import Path
from .shared_config import va_shared_cfg
import os

_POLICY_DIR = Path(__file__).resolve().parents[3]

va_robotwin30_cfg = EasyDict(__name__='Config: VA robotwin30')
va_robotwin30_cfg.update(va_shared_cfg)

va_robotwin30_cfg.infer_mode = "server"

va_robotwin30_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    "LINGBOT_VA_BASE_MODEL_PATH", ""
)

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
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.051365569829941,
    -3.5348887174584814e-14,
    1.741881830823392e-16,
    -1.5984010362625123,
    -0.6003574305772781,
    -1.6678147149085998,
    0.0,
    -0.4509398394823074,
    -2.5859282299029243e-14,
    1.234041714549172e-16,
    -1.64055180311203,
    -1.2633746123313903,
    -1.7645285475254058,
    0.0,
    0.0,
    0.0
  ],
  "q99": [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.5431358617544174,
    2.495765209197998,
    2.492974226474762,
    1.3241519677639007,
    1.2496635341644287,
    1.7392305016517635,
    1.0,
    1.0814965963363647,
    2.4167031812667843,
    2.3470158338546754,
    1.141731116771698,
    0.5208872479200363,
    1.489715996980667,
    1.0,
    1.0,
    1.0
  ],
}

va_robotwin30_train_cfg = EasyDict(__name__='Config: VA robotwin30 train')
va_robotwin30_train_cfg.update(va_robotwin30_cfg)

va_robotwin30_train_cfg.dataset_path = os.environ.get(
    "LINGBOT_VA_DATASET_PATH",
    str(_POLICY_DIR / "lingbot_va" / "lerobot" / "RoboDojo_sim_arx_x5_joint_100"),
)
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