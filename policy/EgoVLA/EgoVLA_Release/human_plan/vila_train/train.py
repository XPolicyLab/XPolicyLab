# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, LlamaForCausalLM, set_seed
from transformers.modeling_utils import unwrap_model

import llava.data.dataset as dataset
import llava.data.datasets_mixture as datasets_mixture
from llava import conversation as conversation_lib
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.data import make_supervised_data_module
from llava.mm_utils import process_image
from llava.model import LlavaLlamaConfig, LlavaLlamaModel
# from llava.train.args import DataArguments, ModelArguments, TrainingArguments
from human_plan.vila_train.args import (
  VLATrainingArguments, VLAModelArguments, VLADataArguments
)
from llava.train.callbacks.autoresume_callback import AutoResumeCallback
from llava.train.llava_trainer import LLaVATrainer, VILADPOTrainer
from llava.train.sequence_parallel import set_pg_manager
from llava.train.utils import (
    get_checkpoint_path,
    mprint,
    prepare_config_for_training,
    unit_test_rope_scaling,
    vision_resolution_elevation,
)
from llava.trl.trainer.utils import DPODataCollatorWithPadding

from llava.train.train import (
  smart_tokenizer_and_embedding_resize,
  safe_save_model_for_hf_trainer
)

local_rank = None

if "WANDB_PROJECT" not in os.environ:
    # Default to WANDB project "VILA".
    os.environ["WANDB_PROJECT"] = "VILA"

from human_plan.utils.action_tokenizer import build_action_tokenizer

def find_all_linear_names(model, lora_llm, lora_vt):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_resampler", "traj_decoder"]
    assert lora_llm or lora_vt, "Not applying LoRA to any of the modules..."

    if not lora_llm:
        multimodal_keywords += ["llm"]
    if not lora_vt:
        multimodal_keywords += ["vision_tower"]

    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            if not "lm_head" in name:
                lora_module_names.add(name)
            # names = name.split(".")
            # lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # if "lm_head" in lora_module_names:  # needed for 16-bit
    #     lora_module_names.remove("lm_head")
    return list(lora_module_names)

def train():
    global local_rank

    parser = HfArgumentParser((VLAModelArguments, VLADataArguments, VLATrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    data_args.sep_query_token = model_args.sep_query_token

    # FIXME(zhijianl): This should be deprecated when we move to the new scripts.
    if os.getenv("RUN_NAME") is not None:
        training_args.run_name = os.getenv("RUN_NAME")
    else:
        training_args.run_name = training_args.output_dir.split("/")[-1]

    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    set_seed(training_args.seed)

    sp_degree = training_args.seq_parallel_size
    ring_degree = training_args.seq_parallel_ring_size
    if sp_degree > 1:
        set_pg_manager(sp_degree, ring_degree, ring_type=training_args.seq_parallel_ring_type)
        print(f"Sequence parallelism is enabled, SP = {sp_degree}")

    resume_path, continue_training = get_checkpoint_path(training_args.output_dir)

    if not continue_training:
        print(f"Models has been ready under {training_args.output_dir}. Skipp training")
        exit(0)

    if resume_path:
        resume_from_checkpoint = True
        if training_args.lora_enable:
            model_cls = LlavaLlamaModel
            config = LlavaLlamaConfig.from_pretrained(model_args.model_name_or_path, resume=resume_from_checkpoint)
            config.resume_path = model_args.model_name_or_path
        else:
            config = AutoConfig.from_pretrained(resume_path, trust_remote_code=True)
            config.resume_path = resume_path
            model_cls = eval(config.architectures[0])
    else:
        ## first time training
        resume_from_checkpoint = False
        model_cls = LlavaLlamaModel
        config = LlavaLlamaConfig.from_pretrained(model_args.model_name_or_path, resume=resume_from_checkpoint)
        if getattr(config, "resume_path", None) is not None:
            config.resume_path = model_args.model_name_or_path

    ## extra configurations
    prepare_config_for_training(config, model_args, training_args, data_args)
    model = model_cls(
        config=config,
        attn_implementation="flash_attention_2",
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
        **bnb_model_from_pretrained_args,
    )

    if not resume_path or training_args.lora_enable:
        if model_args.mlp_path is not None:
            state_dict = torch.load(model_args.mlp_path, map_location="cpu")
            state_dict_new = {}
            for k, v in state_dict.items():
                if k == "0.weight":
                    state_dict_new["layers.1.weight"] = v
                if k == "0.bias":
                    state_dict_new["layers.1.bias"] = v
                if k == "1.weight":
                    state_dict_new["layers.2.weight"] = v
                if k == "1.bias":
                    state_dict_new["layers.2.bias"] = v
                if k == "3.weight":
                    state_dict_new["layers.4.weight"] = v
                if k == "3.bias":
                    state_dict_new["layers.4.bias"] = v
            model.get_mm_projector().load_state_dict(state_dict_new)

    vision_resolution_elevation(model, config)
    # This is an empty func.
    # It would be overwritten by unit test script.
    if unit_test_rope_scaling(model, model.llm.config, training_args):
        return

    # Take a look on model architecture.
    mprint(model)

    model.llm.config.use_cache = False

    ## set tunnable parameters
    logging.warning(
        "You are setting tunable parameters for the model. Previous args include 'freeze_backbone' and 'tune_mm_mlp_adapter' are deprecated.\n Notice: default value of tune_xxx is False, which means you would not tune this part."
    )

    def need_to_modify_do_sample(generation_config):
        if generation_config.do_sample is False:
            if generation_config.temperature is not None and generation_config.temperature != 1.0:
                return True
            if generation_config.top_p is not None and generation_config.top_p != 1.0:
                return True
        return False

    if need_to_modify_do_sample(model.llm.generation_config):
        model.llm.generation_config.do_sample = True

    ## quantize training @yunhao: be careful here
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.llm.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model.llm = prepare_model_for_kbit_training(
            model.llm, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    if training_args.gradient_checkpointing:
        if hasattr(model.llm, "enable_input_require_grads"):
            model.llm.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, PeftModel, get_peft_model

        lora_config = LoraConfig(
            use_dora=training_args.use_dora,
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model, training_args.lora_llm, training_args.lora_vt),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        if resume_from_checkpoint:
            # load non-lora weights
            if os.path.exists(os.path.join(resume_path, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(
                    os.path.join(resume_path, "non_lora_trainables.bin"),
                    map_location="cpu",
                )
                non_lora_trainables = {
                    (k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()
                }
                if any(k.startswith("model.model.") for k in non_lora_trainables):
                    non_lora_trainables = {
                        (k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()
                    }
                model.load_state_dict(non_lora_trainables, strict=False)

            mprint("Resume from checkpoint...", resume_path)
            model = PeftModel.from_pretrained(model, resume_path, is_trainable=True)
        else:
            mprint("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)
        mprint(model)
        model.print_trainable_parameters()

    # currently assume fft for mm projector
    if training_args.lora_enable:
        if not training_args.lora_llm:
            model.get_llm().requires_grad_(training_args.tune_language_model)
        if model.get_vision_tower():
            if training_args.lora_vt:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_vision_tower().vision_tower.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )
            elif training_args.tune_vision_tower:
                model.get_vision_tower().requires_grad_(training_args.tune_vision_tower)
            model.get_mm_projector().requires_grad_(training_args.tune_mm_projector)
            mprint(f"mm projector {training_args.tune_mm_projector}")
            model.print_trainable_parameters()
    else:
        model.get_llm().requires_grad_(training_args.tune_language_model)
        mprint(f"Tunable parameters:\nlanguage model {training_args.tune_language_model}")
        if model.get_vision_tower():
            model.get_vision_tower().requires_grad_(training_args.tune_vision_tower)
            model.get_mm_projector().requires_grad_(training_args.tune_mm_projector)
            mprint(f"vision tower {training_args.tune_vision_tower}")
            mprint(f"mm projector {training_args.tune_mm_projector}")

        if not any(
            [training_args.tune_language_model, training_args.tune_vision_tower, training_args.tune_mm_projector]
        ):
            logging.warning("You are not tuning any part of the model. Please check if this is intended.")

    model.get_traj_decoder().requires_grad_(True)
    # @yunhao: tokenizer instantiation is moved into build_llm
    tokenizer = model.tokenizer

    if tokenizer.bos_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(bos_token="[BOS]"),
            tokenizer=tokenizer,
            model=model.llm,
        )

    # @yunhao: may move this block into method "build_llm"
    tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model.llm,
        )
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # kentang-mit@: It will be useful in on-the-fly packing
    model.llm.pad_token_id = tokenizer.pad_token_id
    model.llm.config.tokenizer_padding_side = tokenizer.padding_side
    model.llm.config.tokenizer_model_max_length = tokenizer.model_max_length
    if training_args.lora_enable:
        model.base_model.model.llm.pad_token_id = tokenizer.pad_token_id

    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        if hasattr(data_args, "num_video_frames") and data_args.num_video_frames != None:
            model.config.num_video_frames = data_args.num_video_frames
        else:
            model.config.num_video_frames = 8

        if hasattr(data_args, "fps"):
            model.config.fps = data_args.fps
        else:
            model.config.fps = 0.0

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        assert not model_args.mm_use_im_patch_token

        model.config.num_time_tokens = data_args.num_time_tokens = model_args.num_time_tokens
        model.config.time_token_format = data_args.time_token_format = model_args.time_token_format
        if model_args.num_time_tokens > 0:
            time_tokens = [model.config.time_token_format.format(t=t) for t in range(model.config.num_time_tokens)]
            num_new_tokens = tokenizer.add_tokens(time_tokens)
            assert len(time_tokens) == num_new_tokens or num_new_tokens == 0
            model.resize_token_embeddings(len(tokenizer))
            model.config.time_token_ids = tokenizer.convert_tokens_to_ids(time_tokens)
        else:
            model.config.time_token_ids = []
        model.config.soft_ce_std = model_args.soft_ce_std

    ## TODO pay attention to quantize
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    model_args.predict_future_step = data_args.predict_future_step
    data_args.action_tokenizer = build_action_tokenizer(
      model_args.action_tokenizer, tokenizer, model_args
    )

    model.config.invalid_token_idx = data_args.action_tokenizer.invalid_token_idx
    model.config.input_placeholder_token_idx = data_args.action_tokenizer.input_placeholder_token_idx
    model.config.input_placeholder_start_token_idx = data_args.action_tokenizer.input_placeholder_start_token_idx
    model.config.input_placeholder_end_token_idx = data_args.action_tokenizer.input_placeholder_end_token_idx

    model.config.merge_hand = data_args.merge_hand

    data_args.traj_action_output_dim = model.traj_decoder.out_dim
    model.config.invalid_token_weight = training_args.invalid_token_weight

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    # Add a training step_end callback to check whether to autosuspend.
    callbacks = [AutoResumeCallback()]

    training_args.logging_loss_keys = [
        "recon_loss", "ee_2d_l2_loss",
        "ee_l2_loss", "hand_l2_loss",
        "ee_rot_loss", "hand_kp_loss", "kl_loss", 
    ]
    trainer = LLaVATrainer(
        model=model, tokenizer=tokenizer, args=training_args, 
        callbacks=callbacks, **data_module
    )
    print(
        "length of dataloader:",
        len(trainer.get_train_dataloader()),
        len(trainer.train_dataset),
        flush=True,
    )
    print(
        "[GPU memory] before trainer",
        torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
        flush=True,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if training_args.debug_e2e:
        exit()

    trainer.save_state()

    model.llm.config.use_cache = True
    model.config.resume_path = model.config._name_or_path = training_args.output_dir
    ## TODO handle lora for new initialization
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
