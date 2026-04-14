# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import glob
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
 
import torch.distributed as dist
from safetensors import safe_open
from transformers import PretrainedConfig
 
from BeingH.model.llm.qwen2.modeling_qwen2 import Qwen2Config
from BeingH.utils.constants import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
    LLM_MODEL_ARCH,
    VIT_MODEL_ARCH,
    CONNECTOR_ARCH,
    MAP_MLLM_LM,
)
 
logger = logging.getLogger(__name__)


# ==============================================================================
# Special Token Management
# ==============================================================================

def add_special_tokens(tokenizer, model_args) -> Tuple[Any, Dict, int, List[str]]:
    """
    Add VLA-specific special tokens to tokenizer.
    
    Args:
        tokenizer: Tokenizer instance
        model_args: Model arguments containing action_token_num
        
    Returns:
        Tuple of (tokenizer, token_id_dict, num_new_tokens, action_tokens)
    """
    # Add pad token if missing
    if tokenizer.pad_token is None:  # <unk>
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v
    
    new_tokens = []

    conversation_tokens = ['<|im_start|>', '<|im_end|>']
    for token in conversation_tokens:
        if token not in all_special_tokens:
            new_tokens.append(token)

    vision_tokens = ['<vision_start>', '<vision_end>']
    for token in vision_tokens:
        if token not in all_special_tokens:
            new_tokens.append(token)

    action_tokens = ['<|action_start|>', '<|action_end|>', '<|state_start|>', '<|state_end|>']
    for token in action_tokens:
        if token not in all_special_tokens:
            new_tokens.append(token)
    
    spatial_tokens = [
        IMG_CONTEXT_TOKEN,
        QUAD_START_TOKEN,
        QUAD_END_TOKEN,
        REF_START_TOKEN,
        REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
    ]
    new_tokens.extend(spatial_tokens)
         
    # Proprioception tokens
    proprio_context_token = '<PROP_CONTEXT>'
    new_tokens.append(proprio_context_token)

    # Action placeholder tokens
    action_placeholder_tokens = [
        f'<ACTION_TOKEN_{i}>' for i in range(model_args.action_token_num)
    ]
    new_tokens.extend(action_placeholder_tokens)

    # Add all new tokens
    num_new_tokens = tokenizer.add_tokens(new_tokens)

    # Build token ID dictionary
    new_token_ids = {
        'bos_token_id': tokenizer.convert_tokens_to_ids('<|im_start|>'),
        'eos_token_id': tokenizer.convert_tokens_to_ids('<|im_end|>'),
        'start_of_image': tokenizer.convert_tokens_to_ids('<img>'),
        'end_of_image': tokenizer.convert_tokens_to_ids('</img>'),
        'start_of_action': tokenizer.convert_tokens_to_ids('<|action_start|>'),
        'end_of_action': tokenizer.convert_tokens_to_ids('<|action_end|>'),
        'start_of_state': tokenizer.convert_tokens_to_ids('<|state_start|>'),
        'end_of_state': tokenizer.convert_tokens_to_ids('<|state_end|>'),
        'img_context_token_id': tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN),
        'prop_context_token_id': tokenizer.convert_tokens_to_ids(proprio_context_token),
        'action_token_ids': tokenizer.convert_tokens_to_ids(action_placeholder_tokens),
        'newline_token_id': tokenizer.encode('\n')[0],
    }
    
    assert len(tokenizer.encode('\n')) == 1, "Newline should be a single token"

    logger.info(f"Added {num_new_tokens} new special tokens")

    return tokenizer, new_token_ids, num_new_tokens, action_tokens


# ==============================================================================
# Model Loading Utilities
# ==============================================================================

def load_safetensors(path):
    """
    Load all safetensor files from a directory.
    
    Args:
        path: Directory containing .safetensors files
        
    Returns:
        Combined state dict
    """
    safetensor_files = glob.glob(f"{path}/*.safetensors")
    state_dict = {}

    for file_path in safetensor_files:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    return state_dict


def loading_ckpt_from_pretrained(model_args, logger: logging.Logger):
    """
    Load pretrained model components (LLM, ViT, connector).
    
    Args:
        model_args: Model arguments
        logger: Logger instance
        
    Returns:
        Tuple of (language_model, vit_model, connector, llm_config, vit_config, 
                  Tokenizer, mllm_layer_names)
    """

    # Load MLLM or individual LLM
    if model_args.mllm_path:
        logger.info(f"Initializing MLLM from: {model_args.mllm_path}")

        mllm_config = PretrainedConfig.from_json_file(
            os.path.join(model_args.mllm_path, "config.json")
        )
        mllm_state_dict = load_safetensors(model_args.mllm_path)
        mllm_layer_names = [
            k.replace("vision_model", "vit_model") for k in mllm_state_dict.keys()
        ]
        llm_config = mllm_config.llm_config
        vision_config = mllm_config.vision_config
    else:
        logger.info(f"Initializing LLM from: {model_args.llm_path}")
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
        vision_config = None
        mllm_state_dict = None
        mllm_layer_names = []


    # Get model classes
    CustomConfig, CustomForCausalLM, Tokenizer = LLM_MODEL_ARCH[llm_config['architectures'][0]]
    CustomViTConfig, CustomViTModel= VIT_MODEL_ARCH[vision_config['architectures'][0]]
    CustomConnector = CONNECTOR_ARCH[model_args.connector_arch]
    
    if isinstance(llm_config, dict):
        llm_config = CustomConfig.from_dict(llm_config)
    
    llm_config._attn_implementation = 'flash_attention_2'
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.use_mot = model_args.use_expert

    # Configure expert
    if model_args.use_expert:
        expert_config = CustomConfig.from_pretrained(model_args.expert_path)   
        llm_config.expert_config = expert_config
       
        assert llm_config.expert_config.num_hidden_layers == llm_config.num_hidden_layers, \
            "For now, expert and LLM must have same number of layers"
        

    # Initialize models
    if model_args.mllm_path:
        # Load from MLLM checkpoint
        language_model = CustomForCausalLM(config=llm_config)
        mllm_state_dict = language_model.custom_init_pretrained(mllm_state_dict, logger)

        vit_config = CustomViTConfig.from_dict(vision_config)
        vit_config.select_layer = mllm_config.select_layer
        vit_config.downsample_ratio = mllm_config.downsample_ratio

        vit_model = CustomViTModel(vit_config)
        mllm_state_dict = vit_model.custom_init_pretrained(mllm_state_dict, logger)

        connector = CustomConnector(
                llm_hidden_size=llm_config.hidden_size,
                vit_hidden_size=vit_config.hidden_size,
                downsample_ratio=mllm_config.downsample_ratio,
                )
        
        mllm_state_dict = connector.init_weights(mllm_state_dict, logger)

        assert len(mllm_state_dict) == 0, "All weights should be consumed!"
    else:
        # Load from separate LLM and ViT,
        # We abandan it for now, bug may exist
        language_model = CustomForCausalLM.from_pretrained(
            model_args.llm_path, 
            config=llm_config,
        )
        vit_config = CustomViTConfig.from_pretrained(model_args.vit_path)

        if isinstance(CustomViTConfig, SiglipVisionConfig):
            vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
            vit_config.rope = model_args.vit_rope
            vit_config.use_patchify = True
            vit_config.downsample_ratio = 1
            connector = None
    
    # Initialize action expert
    if hasattr(llm_config, 'expert_config'):
        mllm_name = model_args.mllm_path.split("/")[-1]
        expert_name = model_args.expert_path.split("/")[-1]

        if MAP_MLLM_LM.get(mllm_name) == expert_name:
            logger.info("Initializing expert by copying from LLM")
            language_model.init_mot()
        else:
            logger.info("Initializing expert from separate checkpoint")
            language_model.init_expert(model_args.expert_path)
            
    return (
        language_model,
        vit_model,
        connector,
        llm_config,
        vit_config,
        Tokenizer,
        mllm_layer_names
    )

# ==============================================================================
# Embedding Utilities
# ==============================================================================

def check_embedding_freeze_status(model):
    """
    Check and print embedding freeze status.
    
    Args:
        model: Model with language_model attribute
    """
    if not hasattr(model, 'language_model'):
        print("Model does not have 'language_model' attribute.")
        return

    print("\n--- Checking Embedding Freeze Status ---")

    # 1. Check Input Token Embedding
    try:
        input_embeddings = model.language_model.get_input_embeddings()
        is_frozen = not input_embeddings.weight.requires_grad
        status = "Frozen" if is_frozen else "Trainable"
        print(f"Input Token Embeddings: {status}")
    except AttributeError:
        print("Cannot access Input Token Embeddings.")

    # 2. Check Output LM Head
    try:
        output_embeddings = model.language_model.get_output_embeddings()
        is_frozen = not output_embeddings.weight.requires_grad
        status = "Frozen" if is_frozen else "Trainable"
        print(f"Output LM Head:         {status}")
    except AttributeError:
        print("Cannot access Output LM Head.")

    # 3. Check if weights are shared (Tied)
    try:
        input_embeddings = model.language_model.get_input_embeddings()
        output_embeddings = model.language_model.get_output_embeddings()
        # Use `is` operator to check if they are the same object in memory
        if input_embeddings.weight is output_embeddings.weight:
            print("Weight sharing status: Tied (input and output share the same weights)")
        else:
            print("Weight sharing status: Not Tied (input and output have independent weights)")
    except AttributeError:
        print("Cannot determine weight sharing status.")

    print("---------------------------------\n")


def check_selective_embedding_freeze(model, num_old_tokens):
    """
    Detailed check of input Embedding matrix to verify that only newly added Token parts are trainable.

    Args:
        model: Your model instance.
        num_old_tokens (int): Size of the original vocabulary.
    """
    if not hasattr(model, 'language_model'):
        print("Model does not have 'language_model' attribute.")
        return

    print("\n--- Detailed Check of Selectively Frozen Embeddings ---")

    try:
        input_embeddings = model.language_model.get_input_embeddings()
        weight = input_embeddings.weight

        # Check freeze status of old Token Embeddings
        # We sample the last old token
        last_old_token_idx = num_old_tokens - 1
        old_token_grad_status = weight.requires_grad if last_old_token_idx < 0 else weight[last_old_token_idx].requires_grad
        # A more reliable check is to look at the entire slice
        old_slice_requires_grad = weight[:num_old_tokens].requires_grad

        status_old = "Trainable" if old_slice_requires_grad else "Frozen"
        print(f"Original Token Embeddings (index 0 to {last_old_token_idx}): {status_old}")
        if old_slice_requires_grad:
            print("  [Warning] Original Token Embedding is trainable, this may not be the expected result!")

        # Check unfreeze status of new Token Embeddings
        # We sample the first new token
        first_new_token_idx = num_old_tokens
        if first_new_token_idx < len(weight):
            new_slice_requires_grad = weight[first_new_token_idx:].requires_grad
            status_new = "Trainable" if new_slice_requires_grad else "Frozen"
            print(f"New Token Embeddings (index {first_new_token_idx} onwards): {status_new}")
            if not new_slice_requires_grad:
                print("  [Warning] New Token Embedding is frozen, this may not be the expected result!")
        else:
            print("No new Tokens in the vocabulary.")

    except Exception as e:
        print(f"Error occurred during check: {e}")

    print("-----------------------------------------\n")

    
def verify_hook_registration(model, param_name="language_model.model.embed_tokens.weight"):
    """
    Verify whether the parameter with the specified name has successfully registered a backward hook.

    Args:
        model: Your model instance.
        param_name (str): The full name of the parameter you want to check.
    """
    print(f"\n--- Verifying Hook Registration Status ({param_name}) ---")

    target_param = None
    for name, param in model.named_parameters():
        if name == param_name:
            target_param = param
            break

    if target_param is None:
        print(f"[Error] Parameter named '{param_name}' not found.")
        print("--------------------------------------------------\n")
        return

    # Check requires_grad status
    if target_param.requires_grad:
        print("Parameter requires_grad: True (this is required for the Hook method)")
    else:
        print("Parameter requires_grad: False [Warning] Hook method requires this to be True!")

    # Check _backward_hooks attribute
    # This is an internal attribute, but it's safe to inspect during debugging
    if hasattr(target_param, '_backward_hooks') and target_param._backward_hooks:
        num_hooks = len(target_param._backward_hooks)
        print(f"Hook registration status: Success (detected {num_hooks} backward hook(s))")
    else:
        print("Hook registration status: Failed (no backward hooks detected)")

    print("--------------------------------------------------\n")


# ==============================================================================
# Logging Utilities
# ==============================================================================

def create_logger(logging_dir, rank, filename="log"):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0 and logging_dir is not None:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(), 
                logging.FileHandler(f"{logging_dir}/{filename}.txt")
            ]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def get_latest_ckpt(checkpoint_dir: str) -> Optional[str]:
    """
    Get path to latest checkpoint in directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoint subdirectories
        
    Returns:
        Path to latest checkpoint or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    step_dirs = [
        d for d in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.isdigit()
    ]

    if len(step_dirs) == 0:
        return None

    step_dirs = sorted(step_dirs, key=lambda x: int(x))
    latest_step_dir = os.path.join(checkpoint_dir, step_dirs[-1])

    return latest_step_dir


# ==============================================================================
# Helper Functions
# ==============================================================================

def _merge_statistics(statistics_list):
    """
    Merge multiple DatasetStatistics objects into one.

    Merge strategy:
    - mean: average across all statistics
    - std: average across all statistics
    - max: take maximum across all statistics
    - min: take minimum across all statistics
    - q01: take minimum (conservative lower bound)
    - q99: take maximum (conservative upper bound)

    Args:
        statistics_list: List of DatasetStatistics objects

    Returns:
        Merged statistics as a dict (JSON-serializable)
    """
    import numpy as np

    if len(statistics_list) == 1:
        return statistics_list[0].model_dump(mode="json")

    # Get the first statistics as template
    template = statistics_list[0].model_dump(mode="json")
    merged = {"state": {}, "action": {}}

    # Merge each modality (state, action)
    for modality in ["state", "action"]:
        if modality not in template:
            continue

        # Merge each feature key
        for feature_key in template[modality].keys():
            # Collect values from all statistics that have this feature
            all_values = {}
            for stat_key in ["mean", "std", "max", "min", "q01", "q99"]:
                all_values[stat_key] = []

            for stats in statistics_list:
                stats_dict = stats.model_dump(mode="json")
                if modality in stats_dict and feature_key in stats_dict[modality]:
                    feature_stats = stats_dict[modality][feature_key]
                    for stat_key in all_values.keys():
                        if stat_key in feature_stats:
                            all_values[stat_key].append(np.array(feature_stats[stat_key]))

            # Merge values
            merged_feature = {}
            for stat_key, values in all_values.items():
                if not values:
                    continue
                stacked = np.stack(values, axis=0)  # (N, dim)
                if stat_key == "mean":
                    merged_feature[stat_key] = np.mean(stacked, axis=0).tolist()
                elif stat_key == "std":
                    merged_feature[stat_key] = np.mean(stacked, axis=0).tolist()
                elif stat_key == "max":
                    merged_feature[stat_key] = np.max(stacked, axis=0).tolist()
                elif stat_key == "min":
                    merged_feature[stat_key] = np.min(stacked, axis=0).tolist()
                elif stat_key == "q01":
                    merged_feature[stat_key] = np.min(stacked, axis=0).tolist()
                elif stat_key == "q99":
                    merged_feature[stat_key] = np.max(stacked, axis=0).tolist()

            merged[modality][feature_key] = merged_feature

    return merged


def save_dataset_metadata(train_dataset, training_args, dataset_meta, logger):
    """
    Saves the metadata of the training dataset(s) to a JSON file.
    This function can handle raw datasets, or datasets wrapped in a ConcatDataset.

    Supports three formats:
    1. Default (nested): Preserve individual sub-dataset statistics
    2. Optional (merged): Save merged statistics + individual variants
    3. NEW (hierarchical): Save 2-level hierarchical statistics (embodiment → task)
       with {dataset_name}_variants key for inference-time variant selection
    """
    output_dir = Path(training_args.output_dir)
    exp_cfg_dir = output_dir / "experiment_cfg"
    exp_cfg_dir.mkdir(parents=True, exist_ok=True)
    if dist.get_rank() != 0:
        return

    grouped_dataset_meta_list = list(dataset_meta.items())

    for idx, grouped_dataset in enumerate(train_dataset.grouped_datasets):
        grouped_dataset_name = grouped_dataset_meta_list[idx][0]
        if not hasattr(grouped_dataset, 'dataset_metadatas'):
            continue

        specific_metadatas = grouped_dataset.dataset_metadatas

        # Serialize individual sub-dataset metadatas
        serializable_metadatas = {
            sub_dataset_name: metadata_obj.model_dump(mode="json")
            for sub_dataset_name, metadata_obj in specific_metadatas.items()
        }

        metadata_filepath = exp_cfg_dir / f"{grouped_dataset_name}_metadata.json"
        metadata_json = {}

        if os.path.exists(metadata_filepath):
            try:
                with open(metadata_filepath, "r") as f:
                    metadata_json = json.load(f)
            except json.JSONDecodeError:
                logger.info(f"Warning: Could not decode existing metadata file at {metadata_filepath}.")

        # Check for hierarchical stats or stats_sources (new format)
        has_hierarchical_stats = (
            hasattr(grouped_dataset, 'hierarchical_stats') and
            grouped_dataset.hierarchical_stats is not None
        )
        has_stats_sources = (
            hasattr(grouped_dataset, 'stats_sources') and
            grouped_dataset.stats_sources
        )

        if has_hierarchical_stats:
            # NEW: Save hierarchical metadata
            logger.info(f"Saving hierarchical metadata for '{grouped_dataset_name}'")

            hierarchical_stats = grouped_dataset.hierarchical_stats
            grouping_structure = getattr(grouped_dataset, 'grouping_structure', {})

            # Get merged metadata (use first dataset's modalities as template)
            merged_metadata = list(specific_metadatas.values())[0]

            # Build hierarchical metadata structure
            metadata_json.update({
                grouped_dataset_name: {
                    "statistics": hierarchical_stats['total']['statistics'],
                    "modalities": merged_metadata.modalities.model_dump(mode="json") if hasattr(merged_metadata, 'modalities') else {},
                    "embodiment_tag": getattr(merged_metadata, 'embodiment_tag', 'multi_embodiment'),
                    "hierarchical_stats": hierarchical_stats,
                    "grouping_structure": grouping_structure
                }
            })

            logger.info(f"✓ Hierarchical metadata saved:")
            logger.info(f"  - Total datasets: {hierarchical_stats['total']['dataset_count']}")
            logger.info(f"  - Embodiment groups: {list(hierarchical_stats['embodiment_groups'].keys())}")
            logger.info(f"  - Task datasets: {len(hierarchical_stats['task_datasets'])}")

        elif has_stats_sources:
            # NEW: Save hierarchical stats metadata (Level 1 + Level 2)
            logger.info(f"Saving hierarchical stats metadata for '{grouped_dataset_name}'")

            from configs.dataset_info import DATASET_INFO

            stats_sources = grouped_dataset.stats_sources
            stats_level = getattr(grouped_dataset, 'stats_level', 'auto')

            # === Step 1: Collect embodiment information ===
            embodiment_groups = {}  # {embodiment_name: [sub_names]}
            embodiment_tags = {}  # {embodiment_name: embodiment_tag}

            # Define priority order for registry search (prefer uni_posttrain for hierarchical datasets)
            REGISTRY_PRIORITY = ['uni_posttrain']

            for sub_name in specific_metadatas.keys():
                # Find embodiment info from DATASET_INFO with priority search
                matched_registry = None
                dataset_meta_info = None

                # First, try priority registries (e.g., uni_posttrain for hierarchical structure)
                for priority_reg in REGISTRY_PRIORITY:
                    if priority_reg in DATASET_INFO:
                        registry_datasets = DATASET_INFO[priority_reg]
                        if isinstance(registry_datasets, dict) and sub_name in registry_datasets:
                            matched_registry = priority_reg
                            dataset_meta_info = registry_datasets[sub_name]
                            break

                # If not found in priority registries, search all registries
                if not dataset_meta_info:
                    for registry_name, registry_datasets in DATASET_INFO.items():
                        if isinstance(registry_datasets, dict) and sub_name in registry_datasets:
                            matched_registry = registry_name
                            dataset_meta_info = registry_datasets[sub_name]
                            break

                # Process embodiment info if found
                if dataset_meta_info:
                    embodiment_name = dataset_meta_info.get('embodiment')
                    embodiment_tag = dataset_meta_info.get('embodiment_tag')

                    if embodiment_name:
                        embodiment_groups.setdefault(embodiment_name, []).append(sub_name)

                        # Store embodiment_tag mapping
                        if embodiment_tag:
                            embodiment_tags[embodiment_name] = embodiment_tag

            # === Step 2: Build variants (Level 1 + Level 2) ===
            variants = {}

            # Add Level 1: Task-specific variants (directly from training)
            for sub_name, meta in specific_metadatas.items():
                variants[sub_name] = {
                    **meta.model_dump(mode="json"),
                    "stats_level": "task"
                }

            # Add Level 2: Embodiment-merged variants
            for embodiment_name in embodiment_groups.keys():
                try:
                    subtask_names = embodiment_groups[embodiment_name]
                    template_meta = specific_metadatas[subtask_names[0]]

                    # Merge statistics from all subtasks under this embodiment
                    subtask_metas = [specific_metadatas[name] for name in subtask_names]
                    merged_statistics = _merge_statistics([m.statistics for m in subtask_metas])

                    # Get embodiment_tag from mapping (use template's tag if not found)
                    embodiment_tag_value = embodiment_tags.get(embodiment_name, template_meta.embodiment_tag.value)

                    variants[embodiment_name] = {
                        "statistics": merged_statistics,
                        "modalities": template_meta.modalities.model_dump(mode="json"),
                        "embodiment_tag": embodiment_tag_value,
                        "stats_level": "embodiment",
                    }

                    logger.info(f"  ✓ Level 2 (embodiment): {embodiment_name}")
                    logger.info(f"    Merged from {len(subtask_names)} subtasks: {subtask_names}")

                except Exception as e:
                    logger.warning(f"  ⚠ Failed to create embodiment variant for {embodiment_name}: {e}")

            # === Step 3: Set top-level default (first variant for backward compatibility) ===
            first_variant_name = list(specific_metadatas.keys())[0]
            first_variant_meta = specific_metadatas[first_variant_name]

            metadata_json.update({
                grouped_dataset_name: {
                    **first_variant_meta.model_dump(mode="json"),
                    "default_variant": first_variant_name,
                    "stats_level": stats_level,
                },
                f"{grouped_dataset_name}_variants": variants
            })

            logger.info(f"✓ Hierarchical stats saved:")
            logger.info(f"  - Default: {first_variant_name}")
            logger.info(f"  - Level 1 (task): {list(specific_metadatas.keys())}")
            logger.info(f"  - Level 2 (embodiment): {list(embodiment_groups.keys())}")
            logger.info(f"  - Total: {len(variants)} variants")

        else:
            # Legacy formats
            save_merged = getattr(training_args, 'save_merged_metadata', False)

            if save_merged:
                # User explicitly enabled merged metadata
                # Save both merged and individual variants (even for single dataset)
                from copy import deepcopy
                import numpy as np

                # Merge statistics
                merged_metadata = deepcopy(list(specific_metadatas.values())[0])

                # Get all modality keys (state, action, etc.)
                statistics_dict = vars(merged_metadata.statistics) if hasattr(merged_metadata.statistics, '__dict__') else {}

                for modality_key in statistics_dict.keys():
                    if modality_key.startswith('_'):
                        continue

                    modality_stats = getattr(merged_metadata.statistics, modality_key, None)
                    if modality_stats is None:
                        continue

                    # Get all feature keys (joint_position, etc.)
                    modality_stats_dict = vars(modality_stats) if hasattr(modality_stats, '__dict__') else {}

                    for feature_key in modality_stats_dict.keys():
                        if feature_key.startswith('_'):
                            continue
                        feature_stats = getattr(modality_stats, feature_key, None)
                        if feature_stats is None or not hasattr(feature_stats, 'min'):
                            continue

                        # Collect stats from all variants
                        all_mins, all_maxes = [], []
                        for metadata in specific_metadatas.values():
                            mod_stats = getattr(metadata.statistics, modality_key, None)
                            if mod_stats is None:
                                continue
                            feat_stats = getattr(mod_stats, feature_key, None)
                            if feat_stats is None:
                                continue
                            if hasattr(feat_stats, 'min') and feat_stats.min:
                                all_mins.append(np.array(feat_stats.min))
                            if hasattr(feat_stats, 'max') and feat_stats.max:
                                all_maxes.append(np.array(feat_stats.max))

                        # Merge: min of mins, max of maxes
                        if all_mins:
                            feature_stats.min = np.minimum.reduce(all_mins).tolist()
                        if all_maxes:
                            feature_stats.max = np.maximum.reduce(all_maxes).tolist()

                # Save merged + variants
                metadata_json.update({
                    grouped_dataset_name: merged_metadata.model_dump(mode="json"),
                    f"{grouped_dataset_name}_variants": serializable_metadatas
                })
                logger.info(f"Saved merged metadata + {len(specific_metadatas)} variant metadatas")
            else:
                # Default: Save individual sub-dataset metadatas (backward compatible)
                metadata_json.update({
                    grouped_dataset_name: serializable_metadatas
                })
                logger.info(f"Saved {len(specific_metadatas)} individual sub-dataset metadata(s)")

        with open(metadata_filepath, "w") as f:
            json.dump(metadata_json, f, indent=4)
        logger.info(f"Successfully saved dataset metadata to {metadata_filepath}")

