import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, CLIPImageProcessor, SiglipImageProcessor, \
    GPTNeoXModel, GPTNeoXPreTrainedModel
import torch
from llava_pythia.model import *
from llava_pythia.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="cuda", device="cuda"):
    """
    Loads a pretrained model with optional quantization and device mapping.

    Args:
        - model_path (str): Path to the model directory or file.
        - model_base (str): Base model path, used when loading LoRA models.
        - model_name (str): Name of the model to load.
        - load_8bit (bool): Whether to load the model in 8-bit precision.
        - load_4bit (bool): Whether to load the model in 4-bit precision.
        - device_map (str): Device map for model loading, default is "cuda".
        - device (str): Device to load the model onto, default is "cuda".

    Returns:
        - tokenizer: The tokenizer associated with the model.
        - model: The loaded model.
        - image_processor: The image processor if applicable.
        - context_len (int): The context length of the model.
    """
    kwargs = {"device_map": device_map}
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'pythia' in model_name.lower():
        # Load LLaVA-Phi model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
        if 'lora' in model_name.lower() and model_base is not None:
            
            path = model_path.split('/')[0:-1]
            root_path = '/'.join(path)
            lora_cfg_pretrained = AutoConfig.from_pretrained(root_path)
            config = lora_cfg_pretrained
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True) # default use_fast=False
            print('Loading LLaVA-Pythia from base model...')
            model = LlavaPythiaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            
            # token_num, tokem_dim = model.embed_out.out_features, model.embed_out.in_features
            # if model.embed_out.weight.shape[0] != token_num:
            #     model.embed_out.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            #     model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            
            print('Loading additional LLaVA-Pythia weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.gpt_neox.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            
            # 删除lora相关的参数
            keys_to_del = []
            for k,v in non_lora_trainables.items():
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
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA-Pythia from base model...')
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            model = LlavaPythiaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            print("load llaVA-Pythia MLLM!!!")
            config = LlavaPythiaConfig.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = LlavaPythiaForCausalLM.from_pretrained(
                model_path,
                config=config,
                use_safetensors=True,
                **kwargs).to("cuda")
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    if "clip" in config.vision_config["vision_tower"]["vision_model_name_or_path"]:
        image_processor = CLIPImageProcessor.from_pretrained(model_path)
    elif "siglip" in config.vision_config["vision_tower"]["vision_model_name_or_path"]:
        image_processor = SiglipImageProcessor.from_pretrained(model_path)
    else:
        return NotImplementedError
    # image_processor = CLIPImageProcessor.from_pretrained(model_path)

    if 'pythia' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)

        # TODO: the tokenizer length of phi-2 is 50295, but the output class of lm_head is 51200
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            # model.resize_token_embeddings(len(tokenizer))
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    model.to(device="cuda")
    print(kwargs)
    return tokenizer, model, image_processor, context_len
