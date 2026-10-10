from llava.train.args import TrainingArguments, ModelArguments, DataArguments
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import transformers

@dataclass
class VLAModelArguments(ModelArguments):
    # Action
    action_tokenizer: str = field(default='uniform')
    min_action: float = field(default=-0.06)
    max_action: float = field(default=0.06)
    num_action_bins: int = field(default=256)

    num_action_dims: int = field(default=6)
    num_action_sep_dims: int = field(default=6)
    
    quantile_path: str = field(default="")

    traj_decoder_type: str = field(default="dummy")
    traj_decoder: Optional[str] = field(default=None)

    traj_action_output_dim: int = field(default=2)
    proprio_size: int = field(default=16)
    use_proprio: bool = field(default=True)
    sep_proprio: bool = field(default=False)
    sep_query_token: bool = field(default=False)
    traj_action_output_ee_dim: int = field(default=None)
    traj_action_output_ee_2d_dim: int = field(default=0)
    traj_action_output_ee_rot_dim: int = field(default=0)
    traj_action_output_hand_dim: int = field(default=None)
    cvae_hidden_size: int = field(default=256)
    cvae_latent_dim: int = field(default=256)

    ee_rot_representation: Optional[str] = field(default="quat")

@dataclass 
class VLADataArguments(DataArguments):

    dataset_type: str = field(default="IOAI")

    future_index: int = field(default=1)
    predict_future_step: int = field(default=1)
    hand_input_scale_xy: float = field(default=1)
    hand_input_scale_z: float = field(default=1)

    mask_input: bool = field(default=False)
    mask_ignore: bool = field(default=False)
    relative_hand_pose: bool = field(default=False)

    ignore_language: bool = field(default=False)
    use_short_language_label: bool = field(default=False)
    use_verb_only_short_language_label: bool = field(default=False)
    use_noun_only_short_language_label: bool = field(default=False)
    use_empty_language_label: bool = field(default=False)    
    include_response: bool = field(default=False)
    include_repeat_instruction: bool = field(default=False)
    # data_skip: int = field(default=1)
    # training_data_skip: int = field(default=1)
    # eval_data_skip: int = field(default=1)
    load_depth: bool = field(default=False)
    
    with_aug: bool = field(default=False)

    add_embodiment_discription: bool = field(default=False)
    add_current_language_description: bool = field(default=False)
    add_his_obs_step: int = field(default=0)

    add_his_imgs: bool = field(default=False)
    add_his_img_skip: int = field(default=4)
    add_visual_prompt: bool = field(default=False)
    add_his_obs_prompt: bool = field(default=False)

    raw_action_label: bool = field(default=False)
    
    input_placeholder_diff_index: bool = field(default=False)

    normalization_path: str = field(default="")

    correct_transformation: bool = field(default=False)
    include_2d_label: bool = field(default=False)
    include_rot_label: bool = field(default=False)
    no_norm_ee_label: bool = field(default=False)


    prompt_version: Optional[str] = field(default="v0")
    qa_prompt_version: Optional[str] = field(default="v1")

    mix_language_ratio: float = field(default=0)

    merge_hand: bool = field(default=False)

    use_mano: bool = field(default=False)

    include_handkp: bool = field(default=False)

    use_relative_label: bool = field(default=False)

    input_hand_dof: bool =  field(default=False)

    ee_relative_transformation: bool = field(default=False)

    ee_movement_mask_idx: int = field(default=30)

@dataclass 
class VLATrainingArguments(TrainingArguments):
    invalid_token_weight: float = field(default=1)
    tune_lm_head: bool = field(default=True)

    kl_weight: float = field(default=1e-3)
    ee_2d_loss_coeff: float = field(default=1)
    ee_loss_coeff: float = field(default=1)
    ee_rot_loss_coeff: float = field(default=1)
    loss_use_l1: bool = field(default=False)
    hand_loss_coeff: float = field(default=1)
    hand_loss_dim: int = field(default=15)
    hand_mano_norm: bool = field(default=False)
    hand_kp_loss_coeff: float = field(default=1)

    use_movement_mask: bool = field(default=False)
    ee_movement_loss_weight: float = field(default=1)
    next_token_loss_coeff: float = field(default=0)
    
    hand_decoder_lr: Optional[float] = None