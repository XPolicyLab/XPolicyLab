import torch

from llava_pythia import conversation as conversation_lib
from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaForCausalLM

import transformers

import logging

from typing import Dict, Optional, Sequence, List
from transformers import CLIPVisionConfig, SiglipVisionConfig, CLIPImageProcessor, SiglipImageProcessor
from llava_pythia.model import *

import os

def find_all_linear_names(model, rank0_print, lora_module=None):
    """
    Identifies all linear module names in the model that are relevant for LoRA (Low-Rank Adaptation).

    Args:
        model: The model to search for linear modules.
        rank0_print: A function for printing messages, typically used for logging.
        lora_module: Optional; specifies which modules are considered for LoRA.

    Returns:
        A list of names of linear modules that are relevant for LoRA.
    """
    cls = torch.nn.Linear
    lora_module_names = set()
    # multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    # multimodal_keywords = ['vision_resampler', 'mm_projector', 'embed_out', 'channel_proj', 'proj_to_unet']
    lang_type = 'phi' if 'phi' in model.name_or_path.lower() else 'pythia'
    multimodal_keywords = ['vision_resampler', 'mm_projector', 'embed_out', 'proj_to_action']
    if 'vit' not in lora_module:
        multimodal_keywords.append("vision_tower")
    rank0_print("##" * 20)

    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue

        # todo phi dont name gpt_neox
        if lang_type == 'pythia':
            if ('embed_out' not in name) and ('llm' not in lora_module) and ('layers' in name) and ('vision' not in name) and ('gpt_neox' in name):
                continue

        elif lang_type == 'phi':
            if ('embed_out' not in name) and ('llm' not in lora_module) and ('layers' in name) and ('vision' not in name) and ('model' in name):
                continue

        if isinstance(module, cls):

            lora_module_names.add(name)

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')

    if 'half' in lora_module:
        new_lora_module_names = set()
        for n in lora_module_names:
            if ('embed_out' not in n) and ('layers' in n) and ('vision' not in n) and ('gpt_neox' in n):
                if int(n.split('.')[2]) % 2 == 0:
                    continue
                else:
                    new_lora_module_names.add(n)
            else:
                new_lora_module_names.add(n)
        lora_module_names = new_lora_module_names


    # rank0_print(lora_module_names)
    return list(lora_module_names)

def load_llava_pythia(config=None, llava_pythia_config=None, rank0_print=print, tokenizer=None):
    """
    Loads the Llava-Pythia model with optional pre-trained weights and configurations.

    Args:
        config: Configuration dictionary containing model, training, and data arguments.
        llava_pythia_config: Specific configuration for the Llava-Pythia model.
        rank0_print: Function for logging, defaults to print.
        tokenizer: Optional tokenizer to be used with the model.

    Returns:
        A tuple containing the loaded model and data arguments.
    """
    model_args = config['model_args']
    training_args = config['training_args']
    data_args = config['data_args']
    model_arch = llava_pythia_config.architectures[0]
    if training_args.load_pretrain:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, \
            GPTNeoXModel, GPTNeoXPreTrainedModel
        kwargs = {"device_map": "cuda"}
        rank0_print("@@@@@@@Loading pretrain weights...@@@@@@@@@@")
        assert config['model_args'].model_pretrain is not "", "load pretrain weights need set the model_pretrain in DataArguments!!!!"
        # model = load_pretrained_model(config['model_args'].model_pretrain, config['model_args'].model_name_or_path, model_name, False, False)
        model_path = config['model_args'].model_pretrain
        model_base = config['model_args'].model_name_or_path
        path = model_path.split('/')[0:-1]
        root_path = '/'.join(path)
        lora_cfg_pretrained = AutoConfig.from_pretrained(root_path)
        config = lora_cfg_pretrained
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)  # default use_fast=False
        print('Loading LLaVA-Pythia from base model...')
        if 'pythia' in model_arch.lower():
            model = LlavaPythiaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained,
                                                           **kwargs)
        elif 'phi' in model_arch.lower():
            model = MiphaPhi15ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained,
                                                           **kwargs)

        print('Loading additional LLaVA-Pythia weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            raise f"there is no non_lora_trainables.bin in {model_path}"


            non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                               non_lora_trainables.items()}
        if any(k.startswith('model.gpt_neox.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}


        # delete lora-related params
        keys_to_del = []
        for k, v in non_lora_trainables.items():
            if 'lora' in k:
                keys_to_del.append(k)
        for key in keys_to_del:
            del non_lora_trainables[key]

        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
    else:
        if 'pythia' in model_arch.lower():
            model = LlavaPythiaForCausalLM.from_pretrained(
                config['model_args'].model_name_or_path,
                config=llava_pythia_config,
                cache_dir=config['training_args'].cache_dir,
                trust_remote_code=True,
                _fast_init=False,
                # attn_implementation="flash_attention_2",
                **config['bnb_model_from_pretrained_args']
            )
        elif 'phi' in model_arch.lower():
            model = MiphaPhi15ForCausalLM.from_pretrained(
                config['model_args'].model_name_or_path,
                config=llava_pythia_config,
                cache_dir=config['training_args'].cache_dir,
                trust_remote_code=True,
                _fast_init=False,
                # attn_implementation="flash_attention_2",
                **config['bnb_model_from_pretrained_args']
            )

    model.config.use_cache = False


    model_args.freeze_backbone = training_args.freeze_backbone
    if model_args.freeze_backbone:
        model.get_model().requires_grad_(False)
    else:
        model.get_model().requires_grad_(True)

    model.get_model().vision_tower.requires_grad_(True) # set to true first
    model.config.freeze_vision_tower = model_args.freeze_vision_tower = training_args.freeze_vision_tower
    if model_args.freeze_vision_tower:
        for n,p in model.get_model().vision_tower.named_parameters():
            if not 'lora' in n.lower():
                p.requires_grad = False
    else:
        for p in model.get_model().vision_tower.parameters():
            p.requires_grad = True


    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # TODO: https://huggingface.co/microsoft/phi-2/discussions/31. But in this code, setting gradient_checkpointing=True, it doesn't raise any error
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # if training_args.lora_enable and (not training_args.load_pretrain):
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model, rank0_print, training_args.lora_module),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type=training_args.lora_task_type,
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("##" * 20)

        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config) # !!!only set lora weights to requires_grad True!!!
        rank0_print(model)
    elif training_args.load_pretrain:
        rank0_print("Already loaded pretrained weights which is based on lora, skipping LoRA initialize...")

    model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter

    if not model_args.tune_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False
    else:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
    # action head需要训练
    model.embed_out.requires_grad_(True)
    model.proj_to_action.requires_grad_(True)

    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["phi-2_v0"]
    rank0_print("default_conversation :")
    rank0_print(conversation_lib.default_conversation)

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    if "clip" in llava_pythia_config.vision_config["vision_tower"]["vision_model_name_or_path"]:
        data_args.image_processor = CLIPImageProcessor.from_pretrained(model_args.model_name_or_path)
    elif "siglip" in llava_pythia_config.vision_config["vision_tower"]["vision_model_name_or_path"]:
        data_args.image_processor = SiglipImageProcessor.from_pretrained(model_args.model_name_or_path)
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    # model.proj_to_unet.requires_grad_(True)

    for k, v in model.named_parameters():
        if v.requires_grad:
            rank0_print(k, v.requires_grad)

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.non_lora_lr = training_args.non_lora_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    return model, data_args

def maybe_zero_3(param, ignore_status=False, name=None):
    """
    Handles parameter gathering for models using DeepSpeed's ZeRO-3 optimization.

    Args:
        param: The parameter to gather.
        ignore_status: If True, ignores the parameter status.
        name: Optional name for logging purposes.

    Returns:
        A detached and cloned version of the parameter on the CPU.
    """
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    """
    Retrieves the state dictionary for PEFT (Parameter-Efficient Fine-Tuning) models, considering ZeRO-3.

    Args:
        named_params: Named parameters of the model.
        bias: Specifies which biases to include ('none', 'all', 'lora_only').

    Returns:
        A dictionary of parameters relevant to PEFT, gathered if necessary.
    """
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    """
    Retrieves non-LoRA parameters for PEFT models, considering ZeRO-3.

    Args:
        named_params: Named parameters of the model.
        require_grad_only: If True, only includes parameters that require gradients.

    Returns:
        A dictionary of non-LoRA parameters, gathered if necessary.
    """
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    """
    Retrieves the state dictionary for multi-modal adapters, considering ZeRO-3.

    Args:
        named_params: Named parameters of the model.
        keys_to_match: Keys to identify relevant parameters.

    Returns:
        A dictionary of parameters for multi-modal adapters, gathered if necessary.
    """
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """
    Safely saves the model state for a Hugging Face Trainer.

    Args:
        trainer: The Hugging Face Trainer instance.
        output_dir: Directory where the model state should be saved.

    Returns:
        None
    """
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """
    Resizes the tokenizer and model embeddings to accommodate new special tokens.

    Args:
        special_tokens_dict: Dictionary of special tokens to add.
        tokenizer: The tokenizer to resize.
        model: The model whose embeddings need resizing.

    Returns:
        None
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

