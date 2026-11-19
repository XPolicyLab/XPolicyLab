"""RoboDojo history-conditioned Video-10k training."""

from omegaconf import OmegaConf


def validate_config(cfg):
    data, model = cfg.data.train, cfg.model
    if cfg.checkpoint_mode != "full":
        raise ValueError("Video-10k requires full video checkpoints")
    if model._target_ != "recoverwam.video_training.create_model":
        raise ValueError("Video-10k requires the standalone video model")
    if data._target_ != "recoverwam.datasets.lerobot.video_event_dataset.VideoEventDataset":
        raise ValueError("Video-10k uses task-balanced video windows")
    if (data.num_frames, data.action_video_freq_ratio, data.condition_history_frames,
            data.condition_history_stride) != (97, 4, 8, 25):
        raise ValueError("Video-10k requires 25 RGB frames, history8 and stride25")
    if not data.condition_include_episode_first or not data.condition_history_target_prefix:
        raise ValueError("Video-10k requires the aligned episode-anchor/history prefix")
    if data.processor is not None or data.pretrained_norm_stats is not None:
        raise ValueError("Video-only training does not normalize actions or robot state")
    if model.video_dit_config.video_attention_mask_mode != "bidirectional":
        raise ValueError("Video-10k preserves bidirectional video attention")
    if cfg.get("joint_learning_rates") or cfg.get("lr_tail") or cfg.get("lr_continuation"):
        raise ValueError("Video-10k uses a single video learning-rate schedule")
    if cfg.get("preflight", {}).get("enabled", False):
        raise ValueError("The joint/action preflight is not a video-only diagnostic")


def create_model(*, video_dit_config, video_scheduler, model_dtype, device, **kwargs):
    from .models.wan21_14b.wan21 import Wan21VideoModel

    def plain(value):
        return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value

    return Wan21VideoModel.from_wan21_14b_pretrained(
        video_dit_config=plain(video_dit_config),
        video_scheduler=plain(video_scheduler),
        torch_dtype=model_dtype,
        device=device,
        **kwargs,
    )


def trainer_class():
    from .trainer import JointTrainer

    class VideoTrainer(JointTrainer):
        supported_modes = {("video_only", "full")}

        def _apply_train_mode(self, model):
            model.eval().requires_grad_(False)
            model.set_train_mode("video_only")
            model.dit.train().requires_grad_(True)

    return VideoTrainer
