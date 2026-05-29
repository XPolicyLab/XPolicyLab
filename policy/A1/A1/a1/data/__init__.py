import os
import logging

import numpy as np
from torch.utils.data import DataLoader, DistributedSampler

from a1.data.A1_datasets import SRPlanning, SRAffordance, SRTrajectory, AGD20K, BlipLaionCC, \
    DroidCotrackPlanning, DroidMolmoSam2Planning, DroidCotrackTrajectory, \
    ManiskillTrajectory, DroidMolmoSam2Trajectory, ManiskillPlanning, Hoi4DPlanning, RoboVQA, \
    CleverMath, SuperCLEVR, TRANCE, OXEFAST, AgibotFAST
from a1.config import DataConfig, TrainConfig, ModelConfig
from a1.data.academic_datasets import ChartQa, ScienceQAImageOnly, TextVqa, OkVqa, DocQa, \
    InfoQa, AOkVqa, Vqa2, PlotQa, FigureQa, DvQa, SceneTextQa, TabWMPDirectAnswer, \
    AndroidControl, TallyQa, AI2D, CountBenchQa, RealWorldQa, MathVista, MMMU, ClockBench
from a1.data.collator import MMCollator,MMCollatorForAction
from a1.data.data_formatter import DataFormatter
from a1.data.dataset import DeterministicDataset,IterableDatasetWrapper
from a1.data.iterable_dataset_mixture import IterableDatasetMixture,MultiSourceIterableDataset,SimpleMultiSourceIterableDataset
from a1.data.model_preprocessor import Preprocessor, MultiModalPreprocessor
from a1.data.pixmo_datasets import PixMoPointExplanations as PixMoPointExplanationHF, \
    PixMoDocs, PixMoCount, PixMoPoints, PixMoCapQa, PixMoCap, PixMoPointExplanations, \
    PixMoAskModelAnything, PixMoPointsEval
from a1.torch_util import get_global_rank, get_world_size
from a1.vla.config_loader import read_vla_yaml_config

from a1.data.vla.utils import NormalizationType



log = logging.getLogger(__name__)


def build_mm_preprocessor(
    model_config: ModelConfig,
    for_inference=False,
    shuffle_messages=True,
    is_training=False,
    require_image_features=False
):
    v_cfg = model_config.vision_backbone
    h, w = model_config.llm_patches_per_crop()
    if not model_config.image_padding_embed:
        image_padding_mask = None
    elif model_config.fix_image_padding:
        image_padding_mask = 2
    else:
        image_padding_mask = 1

    return Preprocessor(
        DataFormatter(
            prompt_templates=model_config.prompt_type,
            message_format=model_config.message_formatting,
            system_prompt=model_config.system_prompt_kind,
            always_start_with_space=model_config.always_start_with_space,
            default_inference_len=model_config.default_inference_len
        ),
        MultiModalPreprocessor(
            tokenizer=model_config.get_tokenizer(),
            normalize=str(v_cfg.image_model_type),
            crop_mode=model_config.crop_mode,
            max_crops=model_config.max_crops,
            overlap_margins=model_config.overlap_margins,
            resize=v_cfg.resize_mode,
            use_col_tokens=model_config.use_col_tokens,
            base_image_input_size=v_cfg.image_default_input_size,
            image_pooling_w=model_config.image_pooling_w,
            image_pooling_h=model_config.image_pooling_h,
            image_token_length_w=w,
            image_token_length_h=h,
            image_patch_size=v_cfg.image_patch_size,
            image_padding_mask=image_padding_mask,
            pad_value=model_config.pad_value,
            loss_token_weighting=model_config.multi_annotation_weighting,
        ),
        for_inference=for_inference,
        shuffle_messages=shuffle_messages,
        is_training=is_training,
        require_image_features=require_image_features,
    )


def build_torch_mm_eval_dataloader(
    batch_size, seed, model_config, data_config, pad_batches, max_steps=None, train_config=None
):
    preprocessor = build_mm_preprocessor(
        model_config, for_inference=data_config.for_inference, shuffle_messages=data_config.shuffle_messages,
        require_image_features=pad_batches
    )
    logging.info(f"Loading eval dataset: {data_config.dataset}/{data_config.split}")
    if data_config.dataset == "rlds_dataset":
        from a1.data.vla.rlds_datasets import RLDSDataset, RLDSBatchTransform
        batch_transform = RLDSBatchTransform(
            use_wrist_image=data_config.use_wrist_image,
            use_proprio=data_config.use_proprio,
            fixed_action_dim=model_config.fixed_action_dim,
        )

        dataset = RLDSDataset(
            data_root_dir=data_config.rlds_data_root_dir, 
            data_mix = data_config.rlds_dataset_name,
            batch_transform=batch_transform,
            resize_resolution=model_config.vision_backbone.image_default_input_size,
            train=data_config.split == "train",
            image_aug=True)
        log.info(f"length of the eval dataset: {len(dataset)}")
    else:
        dataset = get_dataset_by_name(data_config.dataset, data_config.split, train_config=train_config)
    n_pad = 0
    if pad_batches:
        global_batch_size = batch_size*get_world_size()
        n_steps = (len(dataset) + global_batch_size - 1) // global_batch_size
        if max_steps:
            n_steps = min(n_steps, max_steps)
        if n_steps*global_batch_size > len(dataset):
            # Pad the dataset so that it can produce enough batches of `global_batch_size` size
            # to cover the entire dataset without dropping any examples
            # We need this if evaluating FSDP models since they will need all devices to get
            # exactly the same number of batches
            n_pad = (n_steps*global_batch_size) - len(dataset)

    dataset = DeterministicDataset(
        dataset=dataset,
        seed=seed,
        preprocessor=preprocessor,
        n_pad=n_pad
    )

    sampler = DistributedSampler(
        dataset,
        drop_last=data_config.drop_last,
        shuffle=data_config.shuffle,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        seed=seed,
    )
    num_workers = 0 if data_config.dataset == "rlds_dataset" else data_config.num_workers
    if data_config.dataset == "rlds_dataset":
        collate_f = MMCollatorForAction(
            model_config=model_config,
            max_sequence_length=data_config.sequence_length,
            max_crops=model_config.get_max_crops(),
            include_metadata=True,
            pad=data_config.pad,
        )
    else:
        collate_f = MMCollator(
            data_config.sequence_length,
            max_crops=model_config.get_max_crops(),
            include_metadata=True,
            pad=data_config.pad,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_f ,
        num_workers=num_workers,
        # sampler=sampler,
        sampler=None,
        pin_memory=data_config.pin_memory,
        prefetch_factor=None if data_config.num_workers == 0 else data_config.prefetch_factor,
        persistent_workers=False if data_config.num_workers == 0 else data_config.persistent_workers,
        timeout=data_config.timeout,
    )


def build_eval_dataloader(
    train_config: TrainConfig,
    data_config: DataConfig,
    batch_size: int,
    max_steps=None
) -> DataLoader:
    seed = data_config.seed if data_config.seed is not None else train_config.seed
    if data_config.multi_modal in ["torch"]:
        log.info(f'train_config: {train_config}')
        return build_torch_mm_eval_dataloader(
            batch_size, seed, train_config.model, data_config,
            pad_batches=train_config.fsdp is not None and not data_config.drop_last,
            max_steps=max_steps, train_config=train_config
        )
    else:
        raise NotImplementedError(data_config.multi_modal)


def build_train_dataloader(train_config: TrainConfig, device=None) -> DataLoader:
    
    if device is None:
        device = "cpu"
    assert train_config.device_train_batch_size is not None
    seed = train_config.data.seed if train_config.data.seed is not None else train_config.seed
    data_config = train_config.data

    if train_config.data.multi_modal in ["torch", "torch_hf"]:
        preprocessor = build_mm_preprocessor( # preprocessor includes the DataFormatter and MultiModalPreprocessor
            train_config.model, shuffle_messages=data_config.shuffle, is_training=True, require_image_features=True)
        if data_config.dataset:
            datasets = [get_dataset_by_name(
                data_config.dataset, data_config.split)]
            rates = [1]
        else:
            if data_config.mixture:
                mixture = {}
                for name, rate in data_config.mixture.items():
                    logging.info(f"Loading train dataset {name}/{data_config.split}")
                    mixture[name] = (get_dataset_by_name(name, data_config.split), rate)
            else:
                mixture = {}
                for root_size_mixture in data_config.root_size_mixture:
                    group_datasets = {}
                    for name, as_size in root_size_mixture.mixture.items():
                        logging.info(f"Loading train dataset {name}/{data_config.split}")
                        dataset = get_dataset_by_name(name, data_config.split, train_config=train_config)
                        if as_size is not None:
                            size = as_size
                        else:
                            size = len(dataset)
                        group_datasets[name] = (dataset, np.sqrt(size))
                    total_rate = sum(x[1] for x in group_datasets.values())
                    mixture.update({name: (ds, r/total_rate*root_size_mixture.rate)
                                     for name, (ds, r) in group_datasets.items()})
            total_rate = sum(x[1] for x in mixture.values())
            mixture = sorted(mixture.items(), key=lambda x: x[0])
            # assert False, f"mixture: {mixture}"
            rates = [rate/total_rate for (_, (_, rate)) in mixture]
            datasets = [ds for (_, (ds, _)) in mixture]
            logging.info("Sampling rates:")
            names = list(x[0] for x in mixture)
            for ix in np.argsort(rates)[::-1]:
                logging.info(f"{names[ix]}: {100*rates[ix]:0.2f}")
        
        # each dataset is wrapped in a DeterministicDataset
        datasets = [DeterministicDataset(ds, preprocessor, data_config.seed) for ds in datasets]
        assert train_config.epoch == 0 or train_config.epoch is None

        # all DeterministicDatasets are wrapped in an IterableDatasetMixture
        dataset = IterableDatasetMixture(
            datasets=datasets,
            mixture_rates=rates,
            global_batch_size=train_config.global_train_batch_size,
            seed=data_config.seed,
            shuffle=data_config.shuffle,
        )
        return DataLoader(
            dataset,
            batch_size=train_config.device_train_batch_size,
            drop_last=train_config.data.drop_last,
            collate_fn=MMCollator(
                train_config.data.sequence_length, 
                False,
                pad=data_config.pad, max_crops=train_config.model.get_max_crops()),
            num_workers=train_config.data.num_workers,
            pin_memory=train_config.data.pin_memory,
            prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
            persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
            timeout=train_config.data.timeout,
        )
    else:
        raise NotImplementedError(train_config.data.multi_modal)

def build_vla_train_dataloader(train_config: TrainConfig, device=None):
    """Build a mixed VLA dataloader from YAML config using factory pattern.

    Supports:
    - datasets.rlds: {name, path} (weight defaults to 1.0)
    - datasets.lerobot: list of [path, weight] or {path, weight}
    - datasets.droid: list of [path, weight] or {path, weight}
    - datasets.robochallenge: list of [path, embodiment, task_name, weight, ...]
    - datasets.robomind: list of [path, embodiment, weight] or {path, embodiment, weight}
    - datasets.robocoin: list of [path, weight] or {path, weight}
    Returns a DataLoader over a MultiSourceIterableDataset. num_workers is set to 0 because
    RLDS dataset performs its own internal parallelism.
    """
    from a1.data.vla.dataset_builders import DatasetBuilderFactory

    yaml_name = os.getenv("VLA_CONFIG_YAML")
    vla_cfg = read_vla_yaml_config(yaml_name)
    log.info(f"Loaded vla config: {yaml_name}")

    datasets_config = vla_cfg['datasets']

    iterable_sources = []
    source_weights = []

    # Define dataset sections to process: (section_name, builder_name, is_list)
    sections = [
        # Main datasets
        ("rlds", "rlds", True),           # List config
        ("lerobot", "lerobot", True),        # List config
        ("droid", "droid", True),            # List config
        ("robochallenge", "robochallenge", True),  # List config with special params
        ("robomind", "robomind", True),      # List config
        ("robocoin", "robocoin", True),      # List config
    ]
    for section_path, builder_name, is_list in sections:
        section_config = datasets_config.get(section_path)

        if not section_config:
            continue
        # Handle list or single config
        configs = section_config if is_list else [section_config]

        for cfg_item in configs:
            # Build dataset using factory
            result = DatasetBuilderFactory.build_from_config(
                builder_name,
                cfg_item,
                train_config,
                device or "cpu"
            )

            if result:
                ds, weight = result
                iterable_sources.append(ds)
                source_weights.append(weight)
            else:
                log.error(f'building {section_path} {builder_name} error in {cfg_item}')

    assert len(iterable_sources) > 0, "No datasets configured in vla_config.yaml"
    log.info(f"Built {len(iterable_sources)} datasets: {[type(ds).__name__ for ds in iterable_sources]}")

    # Check if any RLDS dataset is used (for num_workers setting)
    from a1.data.vla.rlds_datasets import RLDSDataset
    has_rlds = any(isinstance(ds, RLDSDataset) or
                   (hasattr(ds, 'dataset') and isinstance(ds.dataset, RLDSDataset))
                   for ds in iterable_sources)

    # 从配置中读取图像增强参数（如果未配置则使用默认值）
    image_aug_config = vla_cfg.get("image_augmentation", {})
    enable_image_augmentation = image_aug_config.get("enable", True)
    enable_random_erasing = image_aug_config.get("enable_random_erasing", True)
    enable_sharpening = image_aug_config.get("enable_sharpening", True)
    augmentation_prob = image_aug_config.get("augmentation_prob", 0.5)

    # Build mixed iterable dataset
    mixed = SimpleMultiSourceIterableDataset(
        iterable_sources,
        weights=source_weights,
        seed=train_config.data.seed or 0,
        enable_image_augmentation=enable_image_augmentation,
        enable_random_erasing=enable_random_erasing,
        enable_sharpening=enable_sharpening,
        augmentation_prob=augmentation_prob,
    )

    # Collator for action-style samples
    collate_f = MMCollatorForAction(
        model_config=train_config.model,
        use_proprio=train_config.data.use_proprio,
        max_sequence_length=train_config.data.sequence_length,
        include_metadata=False,
        pad=train_config.data.pad,
        max_crops=train_config.model.get_max_crops(),
    )

    log.info("Build vla train dataloader successfully!")
    return DataLoader(
        mixed,
        batch_size=train_config.device_train_batch_size,
        drop_last=train_config.data.drop_last,
        collate_fn=collate_f,
        num_workers=train_config.data.num_workers if not has_rlds else 0,
        pin_memory=train_config.data.pin_memory,
        prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
        persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
        timeout=train_config.data.timeout,
    )

# for early exit inference, get value matrix from rlds dataset
def build_rlds_train_dataloader(train_config: TrainConfig, device=None):
    if device is None:
        device = "cpu"
    assert train_config.device_train_batch_size is not None
    data_config = train_config.data

    preprocessor = build_mm_preprocessor(
        train_config.model, shuffle_messages=data_config.shuffle, is_training=True, require_image_features=True)
    if data_config.dataset:
        logging.info(f"Loading train dataset: {data_config.dataset}/{data_config.split}")
        from a1.data.vla.rlds_datasets import RLDSDataset, RLDSBatchTransform
        batch_transform = RLDSBatchTransform(
            use_wrist_image=data_config.use_wrist_image,
            use_proprio=data_config.use_proprio,
        )
        dataset = RLDSDataset(
            data_root_dir=data_config.rlds_data_root_dir, 
            data_mix = data_config.rlds_dataset_name,
            batch_transform=batch_transform,
            resize_resolution=train_config.model.vision_backbone.image_default_input_size,
            shuffle_buffer_size=100_000,
            train=data_config.split == "train",
            image_aug=False,
            # single_sample_mode=True,# just for Model convergence  of test some samples
            )
        log.info(f"length of the train dataset: {len(dataset)}")
        rates = [1]
    else:
        raise ValueError("RLDS dataset requires a dataset name to be specified in the data config.")
    # datasets = [DeterministicDataset(ds, preprocessor, data_config.seed) for ds in datasets]
    
    dataset = IterableDatasetWrapper(dataset, preprocessor, data_config.seed) 
    
    assert train_config.epoch == 0 or train_config.epoch is None

    return DataLoader(
        dataset,
        batch_size=train_config.device_train_batch_size,
        # shuffle=data_config.shuffle,
        drop_last=train_config.data.drop_last,
        collate_fn=MMCollatorForAction(
                model_config=train_config.model,
                use_proprio=data_config.use_proprio,
                max_sequence_length=train_config.data.sequence_length, 
                include_metadata=False,
            pad=data_config.pad, max_crops=train_config.model.get_max_crops()),
        # num_workers=train_config.data.num_workers,
        num_workers=0, # use dataset internal workers
        pin_memory=train_config.data.pin_memory,
        prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
        persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
        timeout=train_config.data.timeout,
    )



def build_rlds_dit_action_train_dataloader(train_config: TrainConfig,text_model_path,vision_model_path, device=None):
    """专门用于 DiffusionTransformerAction 模型的 RLDS 训练数据加载器"""
    if device is None:
        device = "cpu"
    assert train_config.device_train_batch_size is not None
    data_config = train_config.data

    # 不使用 build_mm_preprocessor，因为 DiffusionTransformerAction 不需要这个处理
    
    if data_config.dataset:
        logging.info(f"Loading train dataset: {data_config.dataset}/{data_config.split}")
        from a1.data.vla.rlds_datasets import RLDSDataset, DiTActionRLDSBatchTransform
        from a1.data.collator import DiTActionCollator
        from transformers import AutoTokenizer, AutoProcessor
        
        # 创建 DiT Action 专用的 batch transform
        # siglip_ckpt_path= "/mnt/data/zhangjian/google/siglip-so400m-patch14-384"
        # text_model_ckpt_path = "/mnt/data/zhangjian/Qwen3/Qwen3-1.7B"
        tokenizer = AutoTokenizer.from_pretrained(text_model_path)
        processor = AutoProcessor.from_pretrained(vision_model_path)

        batch_transform = DiTActionRLDSBatchTransform(
            tokenizer=tokenizer,
            processor=processor,
            use_wrist_image=data_config.use_wrist_image,
            use_proprio=data_config.use_proprio,
            max_text_length=data_config.sequence_length,
        )
        
        dataset = RLDSDataset(
            data_root_dir=data_config.rlds_data_root_dir, 
            data_mix = data_config.rlds_dataset_name,
            batch_transform=batch_transform,
            resize_resolution=(processor.image_processor.size["height"], processor.image_processor.size["width"]),
            train=data_config.split == "train",
            image_aug=False,
            # single_sample_mode=True,
            )
        if get_global_rank() == 0:
            log.info(f"length of the {data_config.rlds_dataset_name} train dataset: {len(dataset)}")
        # dataset = IterableDatasetWrapper(dataset,None, data_config.seed) 

        # print("******", "Start creating DiT Action collator...")
        # 创建 DiT Action 专用的 collator
        collate_fn = DiTActionCollator(
            include_metadata=False,
            use_proprio=data_config.use_proprio,
            pad=data_config.pad,
            max_sequence_length=train_config.data.sequence_length,
            pad_value=tokenizer.pad_token_id
            # pad_value=-1
        )
        # print("******", "DiT Action collator created successfully.")
        
    else:
        raise ValueError("DiT Action RLDS dataset requires a dataset name to be specified in the data config.")
    
    # 直接使用 dataset，不需要 preprocessor 包装
    assert train_config.epoch == 0 or train_config.epoch is None
    
    dataloader =  DataLoader(
        dataset,
        batch_size=train_config.device_train_batch_size,
        drop_last=train_config.data.drop_last,
        collate_fn=collate_fn,
        num_workers=0,  # 使用 dataset 内部的 workers
        pin_memory=train_config.data.pin_memory,
        prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
        persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
        timeout=train_config.data.timeout,
    )
    # print("******", "DiT Action train dataloader created successfully.")
    return dataloader


def get_dataset_by_name(dataset_name, split, train_config=None):
    if dataset_name in ["oxe_magic_soup_plus_minus","oxe_magic_soup_plus","oxe_magic_soup","oxe_magic_soup_plus_minus_A1","oxe_magic_soup_plus_minus_A1_debug", "libero_spatial_no_noops"]:
        assert train_config is not None
        return OXEFAST(split, data_name=dataset_name, train_config=train_config)
    if 'oxe' in dataset_name:
        assert train_config is not None
        return OXEFAST(split, data_name=dataset_name, train_config=train_config)
    if 'agibot' in dataset_name:
        assert train_config is not None
        return AgibotFAST(split, data_name=dataset_name, train_config=train_config)
    if dataset_name in ["scifi_document_qa", "pixmo_docs_other"]:
        return PixMoDocs("other", split=split)
    elif dataset_name in ["scifi_table_qa", "pixmo_docs_tables"]:
        return PixMoDocs("tables", split=split)
    elif dataset_name in ["scifi_diagram_qa", "pixmo_docs_diagrams"]:
        return PixMoDocs("diagrams", split=split)
    elif dataset_name in ["scifi_charts_qa", "pixmo_docs_charts"]:
        return PixMoDocs("charts", split=split)

    # A1 datasets
    elif dataset_name in ["sr_planning", "sharerobot_planning"]:
        return SRPlanning(split)
    elif dataset_name in ["sr_affordance", "sharerobot_affordance"]:
        return SRAffordance(split)
    elif dataset_name in ["sr_trajectory", "sharerobot_trajectory"]:
        return SRTrajectory(split)
    elif dataset_name in ["agd20k", "AGD20K"]:
        return AGD20K(split)
    elif dataset_name in ["blip_laion_cc", "laion_cc"]:
        return BlipLaionCC(split)
    elif dataset_name in ["droid_cotrack_planning"]:
        return DroidCotrackPlanning(split)
    elif dataset_name in ["droid_molmo_sam2_planning"]:
        return DroidMolmoSam2Planning(split)
    elif dataset_name in ["maniskill_molmo_planning", "maniskill_planning"]:
        return ManiskillPlanning(split)
    elif dataset_name in ["droid_cotrack_trajectory"]:
        return DroidCotrackTrajectory(split)
    elif dataset_name in ["droid_molmo_sam2_trajectory"]:
        return DroidMolmoSam2Trajectory(split)
    elif dataset_name in ["maniskill_molmo_trajectory", "maniskill_trajectory"]:
        return ManiskillTrajectory(split)
    elif dataset_name in ["hoi4d_planning", "hoi4d"]:
        return Hoi4DPlanning(split)
    elif dataset_name in ["robovqa",  "RoboVQA"]:
        return RoboVQA(split)
    elif dataset_name in ["clevrmath", "clever_math", "clevr_math", "CleverMath", "ClevrMath"]:
        return CleverMath(split)
    elif dataset_name in ["super_clevr", "Super_Clevr", "SuperCLEVR", "superclevr"]:
        return SuperCLEVR(split)
    ## Trance
    elif dataset_name in ["trance", "trance_train", "Trance", "TRANCE"]:
        return TRANCE(split='train')
    elif dataset_name in ["trance_test_id"]:
        return TRANCE(split='test_id')
    elif dataset_name in ["trance_test_ood_left"]:
        return TRANCE(split='test_ood_left')
    elif dataset_name in ["trance_test_ood_right"]:
        return TRANCE(split='test_ood_right')


    # PixMo-Pointing
    elif dataset_name in ["pointing_high_freq", "pixmo_points_high_freq"]:
        return PixMoPoints(kind="high_frequency", split=split, counting=False)
    elif dataset_name in ["point_count_high_freq", "pixmo_points_high_freq_counting"]:
        return PixMoPoints(kind="high_frequency", split=split, counting=True)
    elif dataset_name in ["pointing", "pixmo_points"]:
        return PixMoPoints(kind="basic", split=split, counting=False)
    elif dataset_name in ["point_count", "pixmo_points_counting"]:
        return PixMoPoints(kind="basic", split=split, counting=True)

    # PixMo-Point-Explanations
    elif dataset_name in ["point_qa", "pixmo_pointing_explanations"]:
        return PixMoPointExplanations(split=split, split_groups=True)

    # PixMo-Count
    elif dataset_name in ["fast_flickr_count_qa_point_count", "pixmo_count_counting"]:
        return PixMoCount(split=split, counting=True)
    elif dataset_name in ["fast_flickr_count_qa_pointing", "pixmo_count"]:
        return PixMoCount(split=split, counting=False)

    # PixMo-AskModelAnything
    elif dataset_name in ["user_qa", "pixmo_ask_model_anything"]:
        return PixMoAskModelAnything(split=split)

    # PixMo-CapQa
    elif dataset_name in ["synthetic_qa_v3_as_user_qa", "pixmo_cap_qa"]:
        return PixMoCapQa(split=split)

    # PixMo-Cap
    if dataset_name in ["cockatoo_and_transcript_712k_sept6", "pixmo_cap_with_transcripts"]:
        return PixMoCap(split, mode="transcript_and_caption")
    if dataset_name in ["cockatoo_712k_sept6", "pixmo_cap"]:
        return PixMoCap(split, mode="captions")

    if dataset_name == "pointing_eval":
        assert split == "test"
        return PixMoPointsEval()

    # Academic datasets
    if dataset_name == "android_control":
        return AndroidControl(split)
    if dataset_name == "android_control_ll":
        return AndroidControl(split, mode="ll")
    if dataset_name == "chart_qa":
        return ChartQa(split, weighted=False)
    if dataset_name == "real_world_qa_no_instruction":
        assert split == "test"
        return RealWorldQa("no_instruction")
    if dataset_name == "chart_qa_weighted":
        return ChartQa(split, weighted=True)
    if dataset_name == "info_qa":
        return InfoQa(split)
    if dataset_name == "doc_qa":
        return DocQa(split)
    if dataset_name == "science_qa_img":
        return ScienceQAImageOnly(split)
    if dataset_name == "coco_2014_vqa_multi":
        return Vqa2(split, multi_question=True)
    if dataset_name == "coco_2014_vqa":
        return Vqa2(split, multi_question=False)
    if dataset_name == "text_vqa":
        return TextVqa(split)
    if dataset_name == "plot_qa":
        return PlotQa(split, in_memory=False)
    if dataset_name == "figure_qa":
        return FigureQa(dict(train="train", validation="validation1")[split])
    if dataset_name == "dv_qa":
        return DvQa(split, in_memory=False)
    if dataset_name == "okvqa":
        return OkVqa(split)
    if dataset_name in ["mmmu"]:
        return MMMU(split)
    if dataset_name in ["mmmu_test"]:
        return MMMU(split)
    if dataset_name == "a_okvqa_da":
        return AOkVqa(split=split, direct_answer=True)
    if dataset_name == "a_okvqa_mc":
        return AOkVqa(split=split, direct_answer=False)
    if dataset_name == "st_qa":
        return SceneTextQa(split=split)
    if dataset_name == "tabwmp_da":
        return TabWMPDirectAnswer(split=split, include_options=False)
    if dataset_name == "countbench_qa":
        assert split == "huggingface"
        return CountBenchQa()
    if dataset_name == "tally_qa":
        return TallyQa(split=split)
    if dataset_name == "ai2_diagram_v2_mix_transparent":
        return AI2D(split=split, boxes="both")
    if dataset_name == "clock_bench":
        return ClockBench(split=split)
    
    if dataset_name == "dummy_rlds":  
        from a1.data.vla.dummy_datasets import DummyRLDS
        return DummyRLDS(split)

    elif dataset_name == "math_vista_v2":
        if split == "validation":
            split = "testmini"
        return MathVista(split)
    raise NotImplementedError(dataset_name, split)