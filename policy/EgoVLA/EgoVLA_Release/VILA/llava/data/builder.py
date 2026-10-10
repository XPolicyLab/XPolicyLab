import os
from itertools import chain
from typing import Any, List, Optional

from hydra.utils import instantiate
from torch.utils.data import ConcatDataset, Dataset
from transformers import PreTrainedTokenizer

from llava.data.datasets_mixture import DATASETS_LEGACY
from llava.train.args import DataArguments, TrainingArguments
from llava.utils import distributed as dist
from llava.utils import io
from llava.utils.logging import logger

__all__ = ["DATASETS", "MIXTURES", "register_datasets", "register_mixtures", "parse_mixture", "build_dataset"]


def register_datasets(name: Optional[str] = None):
    if name is None:
        name = os.environ.get("VILA_DATASETS", "default")
        logger.info(f"Registering datasets from '{name}'.")
    return io.load(os.path.join(os.path.dirname(__file__), "registry", "datasets", f"{name}.yaml"))


def register_mixtures():
    return io.load(os.path.join(os.path.dirname(__file__), "registry", "mixtures.yaml"))


DATASETS = register_datasets()
MIXTURES = register_mixtures()


def parse_mixture(mixture: str) -> List[str]:
    names = mixture.split("+") if "+" in mixture else [mixture]
    while any(name in MIXTURES for name in names):
        names = list(chain(*[MIXTURES.get(name, [name]) for name in names]))
    return sorted(names)


class RepeatedDataset(Dataset):
    def __init__(self, dataset: Dataset, times: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.times = times

    def __len__(self) -> int:
        return len(self.dataset) * self.times

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index % len(self.dataset)]


def build_dataset(
    mixture: str,
    data_args: DataArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    datasets = []
    for name in parse_mixture(mixture):
        if "*" in name:
            name, times = name.split("*")
            times = int(times)
        else:
            times = 1

        if DATASETS is not None and name in DATASETS:
            if name in DATASETS_LEGACY:
                logger.warning(f"Dataset '{name}' exists in both new and legacy registries. Using the new one.")
            dataset = instantiate(DATASETS[name], _partial_=True)(
                tokenizer=tokenizer,
                data_args=data_args,
            )
        elif name in DATASETS_LEGACY:
            logger.warning(f"Dataset '{name}' is from the legacy registry. Please consider migrating it.")
            dataset = build_dataset_legacy(
                name,
                data_args=data_args,
                training_args=training_args,
                tokenizer=tokenizer,
            )
        else:
            raise ValueError(f"Dataset '{name}' is not found in the registries.")

        if times > 1:
            dataset = RepeatedDataset(dataset, times)
        datasets.append(dataset)
    return ConcatDataset(datasets)


def build_dataset_legacy(
    name: str,
    data_args: DataArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    from llava.data.dataset import (
        LazyCCSWebDataset,
        LazyCoyoDataset,
        LazyCoyoWebDataset,
        LazyMMC4Dataset,
        LazySupervisedDataset,
        LazyVideoWebDataset,
        LazyWDSDataset,
        # VLA datasets
        LazyVLAHoloAssistDataset,
        LazyVLAHoloAssistHFAbsDataset,
        LazyVLAHoloAssistHFDataset,
        LazyVLAHoloAssistHFEEDataset,
        LazyVLAHOT3DHFAbsDataset,
        LazyVLAHOI4DHFAbsDataset,
        LazyVLATACOHFAbsDataset,
    )
    from llava.data.opentelevision_dataset import (
        LazyVLAOTVSimHFAbsDataset
    )
    from llava.data.dataset_impl.coyo_recap import LazyCoyoWebRecapDataset
    from llava.data.dataset_impl.general_img_text import LazyImageTextWebDataset
    from llava.data.dataset_impl.hiertext import VILAHierText
    from llava.data.dataset_impl.panda70m import VILAPanda70m
    from llava.data.dataset_impl.sam import LazySAMWebDataset
    from llava.data.dataset_impl.textocr import VILATextOCR

    dataset = DATASETS_LEGACY[name]
    dataset_type = dataset.dataset_type
    additional_kwargs = {}
    if dataset_type == "torch":
        dataset_cls = LazySupervisedDataset
    elif dataset_type == "wds":
        dataset_cls = LazyWDSDataset
    elif dataset_type == "mmc4":
        dataset_cls = LazyMMC4Dataset
    elif dataset_type == "coyo":
        dataset_cls = LazyCoyoDataset
    elif dataset_type == "sam-wds":
        dataset_cls = LazySAMWebDataset
    elif dataset_type == "coyo-wds":
        dataset_cls = LazyCoyoWebDataset
    elif dataset_type == "coyo-wds-qas":
        print("dataset.py: Loading coyo-wds-qas class")
        from llava.data.dataset_impl.coyo_qa import LazyCoyoWebQADataset

        dataset_cls = LazyCoyoWebQADataset
    elif dataset_type == "coyo-wds-recap":
        dataset_cls = LazyCoyoWebRecapDataset
    elif dataset_type == "textocr":
        dataset_cls = VILATextOCR
    elif dataset_type == "hiertext":
        dataset_cls = VILAHierText
    elif dataset_type == "panda70m":
        dataset_cls = VILAPanda70m
    elif dataset_type == "ccs-wds":
        dataset_cls = LazyCCSWebDataset
    elif dataset_type == "video-wds":
        dataset_cls = LazyVideoWebDataset
    elif dataset_type == "imgtxt-wds":
        dataset_cls = LazyImageTextWebDataset
    elif dataset_type == "ioai_ee":
      dataset_cls = LazyVLAIOAIEEDataset
    elif dataset_type == "holoassist":
      dataset_cls = LazyVLAHoloAssistDataset
    elif dataset_type == "holoassist_hf":
      dataset_cls = LazyVLAHoloAssistHFDataset
      additional_kwargs["data_skip"] = getattr(dataset, "data_skip", 1)
    elif dataset_type == "holoassist_hf_ee":
      dataset_cls = LazyVLAHoloAssistHFEEDataset
      additional_kwargs["data_skip"] = getattr(dataset, "data_skip", 1)
    elif dataset_type == "epic_kitchen_ee_2d":
      dataset_cls = LazyVLAEpicKitchenDataset
    elif dataset_type == "epic_kitchen_ee_2d_hf":
      dataset_cls = LazyVLAEpicKitchenHFDataset
    elif dataset_type == "holoassist_hf_abs_hand":
      dataset_cls = LazyVLAHoloAssistHFAbsDataset
      additional_kwargs["data_skip"] = getattr(dataset, "data_skip", 1)
    # elif dataset_type == "holoassist_hf_rel_hand":
    #   dataset_cls = LazyVLAHoloAssistHFRelDataset
    #   additional_kwargs["data_skip"] = getattr(dataset, "data_skip", 1)
    elif dataset_type == "holoassist_hf_abs_qa":
      dataset_cls = LazyVLAHoloAssistHFQADataset
      additional_kwargs["data_skip"] = getattr(dataset, "data_skip", 1)
    elif dataset_type == "hot3d_hf_abs_hand":
      dataset_cls = LazyVLAHOT3DHFAbsDataset
      additional_kwargs["data_skip"] = getattr(dataset, "data_skip", 1)
    elif dataset_type == "hoi4d_hf_abs_hand":
      dataset_cls = LazyVLAHOI4DHFAbsDataset
      additional_kwargs["data_skip"] = getattr(dataset, "data_skip", 1)
    elif dataset_type == "taco_hf_abs_hand":
      dataset_cls = LazyVLATACOHFAbsDataset
      additional_kwargs["data_skip"] = getattr(dataset, "data_skip", 1)
    elif dataset_type == "otv_sim":
      dataset_cls = LazyVLAOTVSimHFAbsDataset
      additional_kwargs["data_skip"] = getattr(dataset, "data_skip", 1)
    else:
        raise NotImplementedError(f"{dataset_type} is not supported.")

    data_args.meta_path = getattr(dataset, "meta_path", None)
    data_args.caption_choice = getattr(dataset, "caption_choice", None)
    data_args.caption_choice_2 = getattr(dataset, "caption_choice_2", None)
    data_args.start_idx = getattr(dataset, "start_idx", None)
    data_args.end_idx = getattr(dataset, "end_idx", None)

    # Abs dataset
    data_args.image_mapping_path = getattr(dataset, "image_mapping_path", None)
    data_args.stats_path = getattr(dataset, "stats_path", None)

    return dataset_cls(
        tokenizer=tokenizer,
        data_path=dataset.data_path,
        image_folder=getattr(dataset, "image_path"),
        data_args=data_args,
        training_args=training_args,
        **additional_kwargs
    )
